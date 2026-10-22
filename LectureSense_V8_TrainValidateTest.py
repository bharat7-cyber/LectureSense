import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the stable V7 feature pipeline and V4/visual caches.
V7_PATH = ROOT / "LectureSense_V7_FullDevelopment_GatedFusion.py"
if not V7_PATH.exists():
    raise FileNotFoundError(
        f"Required backbone file not found: {V7_PATH}\n"
        "Keep LectureSense_V7_FullDevelopment_GatedFusion.py beside this script."
    )

import importlib.util
spec = importlib.util.spec_from_file_location("lecturesense_v7", V7_PATH)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

DEVICE = v7.DEVICE

TRAIN = list(v7.TRAIN_LECTURES)
VAL = list(v7.FORMER_VAL_LECTURES)
TEST = list(v7.TEST)

OUT = ROOT / "runs" / "v8_train_validate_test"
CACHE_ROOT = OUT / "cache"
MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = MODEL_DIR / "v8_train_validate_test_ensemble.pt"
CONFIG_PATH = MODEL_DIR / "v8_train_validate_test_config.json"

DEFAULT_SEEDS = [11, 29, 47, 71, 103]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lecture_cache_path(lecture):
    return CACHE_ROOT / lecture / "features.npz"


def build_or_load_v8_lecture(lecture):
    cache = lecture_cache_path(lecture)
    if cache.exists():
        z = np.load(cache)
        return {
            "lecture": lecture,
            "X": np.asarray(z["X"], dtype=np.float32),
            "Y": np.asarray(z["Y"], dtype=np.int64),
            "n_slides": int(z["n_slides"]),
        }

    # Build using the exact same proven V4+visual feature construction.
    d = v7.build_or_load_lecture(lecture, save=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=np.int64)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X=X,
        Y=Y,
        n_slides=np.int32(X.shape[1]),
    )
    return {"lecture": lecture, "X": X, "Y": Y, "n_slides": X.shape[1]}


class V8TemporalFusion(nn.Module):
    """V8 candidate reranker.

    Keeps V4 rank as a strong anchor, models slide candidates jointly with
    self-attention, and adds separate transition/context residual heads.
    """

    def __init__(self, feature_dim=28, d_model=96, nhead=4, layers=2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, d_model),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

        self.gate_head = nn.Sequential(
            nn.Linear(d_model, 48), nn.GELU(), nn.Dropout(0.10), nn.Linear(48, 1)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, 48), nn.GELU(), nn.Dropout(0.10), nn.Linear(48, 1)
        )
        self.transition_head = nn.Sequential(
            nn.Linear(d_model, 48), nn.GELU(), nn.Dropout(0.10), nn.Linear(48, 1)
        )

        self.anchor_raw = nn.Parameter(torch.tensor(2.4))
        self.visual_raw = nn.Parameter(torch.tensor(0.10))
        self.transition_raw = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        h = self.norm(self.transformer(self.input(x)))
        gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        residual = self.residual_head(h).squeeze(-1)
        transition = torch.sigmoid(self.transition_head(h).squeeze(-1))

        anchor_scale = F.softplus(self.anchor_raw) + 0.25
        visual_scale = F.softplus(self.visual_raw)
        transition_scale = F.softplus(self.transition_raw)

        v4_rank = x[:, :, 2]
        vis_rank = x[:, :, 5]
        boundary = x[:, :, 20]

        logits = (
            anchor_scale * v4_rank
            + gate * visual_scale * vis_rank
            + 0.65 * residual
            + transition_scale * transition * boundary
        )
        return logits, gate, transition


def hard_negative_loss(logits, target, margin=0.12, hard_k=6):
    ce = F.cross_entropy(logits, target)
    with torch.no_grad():
        order = torch.argsort(logits, dim=1, descending=True)
    losses = []
    for i in range(logits.shape[0]):
        pos = int(target[i])
        negs = [int(x) for x in order[i].tolist() if int(x) != pos][:min(hard_k, logits.shape[1] - 1)]
        if negs:
            losses.append(F.relu(margin - logits[i, pos] + logits[i, negs]).mean())
    hn = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)
    return ce + 0.35 * hn


def make_batches(rows, batch_size=24, seed=20260928):
    groups = {}
    for x, y, w, _ in rows:
        groups.setdefault(x.shape[0], []).append((x, y, w, None))
    rng = random.Random(seed)
    batches = []
    for g in groups.values():
        rng.shuffle(g)
        for start in range(0, len(g), batch_size):
            part = g[start:start + batch_size]
            batches.append((
                torch.from_numpy(np.stack([p[0] for p in part])).float(),
                torch.from_numpy(np.asarray([p[1] for p in part], dtype=np.int64)).long(),
                torch.from_numpy(np.asarray([p[2] for p in part], dtype=np.float32)).float(),
            ))
    rng.shuffle(batches)
    return batches


def rows_from_data(data_list):
    rows = []
    for d in data_list:
        n = len(d["Y"])
        w = 1.0 / max(1, n)
        for i in range(n):
            rows.append((d["X"][i], int(d["Y"][i]), w, d["lecture"]))
    return rows


def train_one_seed(train_data, seed, epochs=36, batch_size=24):
    set_seed(seed)
    model = V8TemporalFusion(feature_dim=28).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rows = rows_from_data(train_data)
    best_state = None
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_total = 0
        for xb, yb, wb in make_batches(rows, batch_size=batch_size, seed=seed + epoch):
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            logits, gate, transition = model(xb)
            ce = F.cross_entropy(logits, yb, reduction="none")
            wbn = wb / (wb.mean() + 1e-8)
            loss_ce = (ce * wbn).mean()
            loss_rank = hard_negative_loss(logits, yb)

            teacher = F.softmax(xb[:, :, 2], dim=1)
            kl = F.kl_div(F.log_softmax(logits, dim=1), teacher, reduction="batchmean")

            # Encourage boundary-sensitive transitions without forcing them everywhere.
            boundary = xb[:, :, 20].clamp(0.0, 1.0)
            transition_loss = F.binary_cross_entropy(transition, boundary)

            agreement = xb[:, :, 2] * xb[:, :, 5]
            gate_penalty = (gate * F.relu(-agreement)).mean()

            loss = (
                0.75 * loss_ce
                + 0.35 * loss_rank
                + 0.030 * kl
                + 0.035 * transition_loss
                + 0.010 * gate_penalty
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(yb)
            n_total += len(yb)

        sched.step()
        avg = total / max(1, n_total)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[V8 seed={seed}] epoch={epoch:02d} loss={avg:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for start in range(0, len(X), 32):
        xb = torch.from_numpy(X[start:start + 32]).float().to(DEVICE)
        logits, _, _ = model(xb)
        out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run --mode train first.")
    ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    models = []
    for state in ck["models"]:
        m = V8TemporalFusion(feature_dim=int(ck["feature_dim"])).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models, dict(ck["decoder"])


def evaluate_lecture(models, decoder, lecture, split_name):
    srt, pdf, gt_path, video = v7.asset_paths(lecture)
    if any(x is None for x in (srt, pdf, gt_path, video)):
        raise RuntimeError("Missing exact asset(s)")

    # Make sure V4 and visual evidence exist.
    v4_dir = v7.ensure_v4(lecture, srt, pdf, video)
    v4 = np.load(v4_dir / "v4_final_score_matrix.npy").astype(np.float32)
    bp = v4_dir / "v4_boundary_probability.npy"
    boundary = np.load(bp).astype(np.float32) if bp.exists() else np.zeros(v4.shape[0], dtype=np.float32)

    segments = v7.get_segments(srt)
    mids = v7.segment_mids(segments)
    visual_times, visual_scores = v7.get_visual_scores(lecture, pdf, video)
    gt = v7.load_gt(gt_path)

    X, _ = v7.build_features_from_mids(
        v4=v4,
        boundary=boundary,
        mids=mids,
        visual_times=visual_times,
        visual_scores=visual_scores,
        times=mids.astype(np.float32),
        gt_slides=None,
    )
    X = v7.normalize_rows(X)
    emissions = np.mean([predict(m, X) for m in models], axis=0)

    path = v7.decode_monotonic(
        emissions,
        boundaries=boundary,
        max_jump=int(decoder["max_jump"]),
        jump_penalty=float(decoder["jump_penalty"]),
        boundary_reward=float(decoder["boundary_reward"]),
    )

    preds, truths = [], []
    for t, g in zip(gt["time"].to_numpy(dtype=np.float32), gt["gt"].to_numpy(dtype=np.int64)):
        si = int(np.searchsorted(mids, float(t), side="right") - 1)
        si = max(0, min(si, len(path) - 1))
        preds.append(int(path[si]) + 1)
        truths.append(int(g))
    preds = np.asarray(preds)
    truths = np.asarray(truths)
    acc = float((preds == truths).mean())
    return {"split": split_name, "lecture": lecture, "points": len(truths), "accuracy": acc}


def validate_and_choose_decoder(models):
    candidates = []
    for max_jump in [1, 2, 3, 4]:
        for penalty in [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]:
            for boundary_reward in [0.00, 0.08, 0.12, 0.18, 0.25]:
                decoder = {
                    "max_jump": max_jump,
                    "jump_penalty": penalty,
                    "boundary_reward": boundary_reward,
                }
                vals = []
                for lecture in VAL:
                    try:
                        r = evaluate_lecture(models, decoder, lecture, "validation")
                        vals.append(r["accuracy"])
                    except Exception as exc:
                        print(f"[VAL SKIP] {lecture}: {type(exc).__name__}: {exc}")
                if vals:
                    candidates.append((float(np.mean(vals)), decoder))

    if not candidates:
        raise RuntimeError("No validation lecture could be evaluated.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_macro, best_decoder = candidates[0]
    print("\nBest validation decoder:")
    print(best_decoder)
    print(f"Validation macro accuracy: {best_macro:.4f} ({best_macro*100:.2f}%)")
    return best_decoder, best_macro


def train_mode(args):
    print("=" * 78)
    print("LECTURESENSE V8 — TRAIN / VALIDATE / TEST")
    print("=" * 78)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_data = []
    for lecture in TRAIN:
        try:
            d = build_or_load_v8_lecture(lecture)
            train_data.append(d)
        except Exception as exc:
            print(f"[TRAIN SKIP] {lecture}: {type(exc).__name__}: {exc}")

    print("\nTrain lectures loaded:", len(train_data), "/", len(TRAIN))
    print("Train points:", sum(len(d["Y"]) for d in train_data))
    print("Validation lectures:", len(VAL))
    print("Test lectures:", len(TEST))
    print("Test labels used during training: NO")
    print("Test labels used during validation/model selection: NO")

    if len(train_data) < 8:
        raise RuntimeError("Too few training lectures available.")

    models = []
    for seed in args.seeds:
        print("\n" + "-" * 72)
        print(f"TRAINING V8 ENSEMBLE MEMBER seed={seed}")
        print("-" * 72)
        models.append(train_one_seed(train_data, seed, epochs=args.epochs, batch_size=args.batch_size))

    # Choose decoder strictly on validation data; no test labels are inspected.
    best_decoder, val_macro = validate_and_choose_decoder(models)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "LectureSense-V8-TrainValidateTest",
        "feature_dim": 28,
        "models": [m.state_dict() for m in models],
        "train_lectures": [d["lecture"] for d in train_data],
        "validation_lectures": VAL,
        "test_lectures": TEST,
        "decoder": best_decoder,
        "validation_macro_accuracy": val_macro,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }
    torch.save(payload, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({
        "version": payload["version"],
        "train_lectures": payload["train_lectures"],
        "validation_lectures": VAL,
        "test_lectures": TEST,
        "decoder": best_decoder,
        "validation_macro_accuracy": val_macro,
        "ensemble_members": len(models),
        "feature_dim": 28,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }, indent=2), encoding="utf-8")

    print("\nSaved model:", MODEL_PATH)
    print("Saved config:", CONFIG_PATH)


def validate_mode():
    models, decoder = load_models()
    rows = []
    for lecture in VAL:
        try:
            rows.append(evaluate_lecture(models, decoder, lecture, "validation"))
        except Exception as exc:
            print(f"[VAL SKIP] {lecture}: {type(exc).__name__}: {exc}")
    if not rows:
        raise RuntimeError("No validation lecture evaluated.")
    result = pd.DataFrame(rows)
    macro = float(result["accuracy"].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "validation_results.csv", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps({
        "validation_macro_accuracy": macro,
        "results": rows,
        "decoder": decoder,
        "test_used_for_training": False,
        "test_used_for_model_selection": True,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("V8 VALIDATION RESULT")
    print("=" * 78)
    print(result.to_string(index=False))
    print(f"\nValidation macro accuracy: {macro:.4f} ({macro*100:.2f}%)")
    print("Decoder:", decoder)


def test_mode():
    models, decoder = load_models()
    rows = []
    for lecture in TEST:
        try:
            rows.append(evaluate_lecture(models, decoder, lecture, "test"))
        except Exception as exc:
            print(f"[TEST SKIP] {lecture}: {type(exc).__name__}: {exc}")
    if not rows:
        raise RuntimeError("No test lecture evaluated.")
    result = pd.DataFrame(rows)
    macro = float(result["accuracy"].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "test_results.csv", index=False)
    (OUT / "test_summary.json").write_text(json.dumps({
        "test_macro_accuracy": macro,
        "results": rows,
        "decoder": decoder,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("V8 FINAL HELD-OUT TEST RESULT")
    print("=" * 78)
    print(result.to_string(index=False))
    print(f"\nMacro accuracy: {macro:.4f} ({macro*100:.2f}%)")
    print("Test labels were not used for training or model selection.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "validate", "test"], default="train")
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = ap.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "validate":
        validate_mode()
    else:
        test_mode()


if __name__ == "__main__":
    main()
