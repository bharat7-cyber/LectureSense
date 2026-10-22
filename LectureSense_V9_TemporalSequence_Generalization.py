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

V8_PATH = ROOT / "LectureSense_V8_TrainValidateTest.py"
if not V8_PATH.exists():
    raise FileNotFoundError(
        f"Required backbone file not found: {V8_PATH}\n"
        "Keep LectureSense_V8_TrainValidateTest.py beside this script."
    )

import importlib.util
spec = importlib.util.spec_from_file_location("lecturesense_v8", V8_PATH)
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)

DEVICE = v8.DEVICE
TRAIN = list(v8.TRAIN)
VAL = list(v8.VAL)
TEST = list(v8.TEST)

OUT = ROOT / "runs" / "v9_temporal_sequence_generalization"
MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = MODEL_DIR / "v9_temporal_sequence_ensemble.pt"
CONFIG_PATH = MODEL_DIR / "v9_temporal_sequence_config.json"

DEFAULT_SEEDS = [13, 31, 59, 83, 107]
WINDOW = 5
HALF = WINDOW // 2
FEATURE_DIM = 28


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lecture(lecture):
    d = v8.build_or_load_v8_lecture(lecture)
    return {
        "lecture": lecture,
        "X": np.asarray(d["X"], dtype=np.float32),
        "Y": np.asarray(d["Y"], dtype=np.int64),
        "n_slides": int(d["n_slides"]),
    }


def pad_context(X, i, half=HALF):
    n = len(X)
    idx = [min(n - 1, max(0, i + k)) for k in range(-half, half + 1)]
    return X[np.asarray(idx, dtype=np.int64)]


def augment_context(x, rng, noise_std=0.012, feature_drop=0.06):
    # Mild feature corruption improves robustness without changing labels.
    y = x.copy()
    scale = rng.normal(1.0, noise_std, size=y.shape).astype(np.float32)
    y *= scale
    mask = rng.random(size=(1, y.shape[1], y.shape[2])) < feature_drop
    y[mask.repeat(y.shape[0], axis=0)] = 0.0
    return y


def build_training_index(train_data):
    rows = []
    for li, d in enumerate(train_data):
        n = len(d["Y"])
        for i in range(n):
            rows.append((li, i, 1.0 / max(1, n)))
    return rows


class V9SequenceFusion(nn.Module):
    """Temporal-context candidate reranker.

    Stage 1: small GRU reads a short temporal window for every candidate slide.
    Stage 2: self-attention lets candidate slides compete at the same timestamp.
    Stage 3: residual scoring is anchored by V4 / visual rank features.
    """
    def __init__(self, feature_dim=FEATURE_DIM, temporal_hidden=64, candidate_dim=96, heads=4, layers=2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim, temporal_hidden),
            nn.LayerNorm(temporal_hidden),
            nn.GELU(),
        )
        self.temporal_gru = nn.GRU(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(temporal_hidden * 2, candidate_dim),
            nn.LayerNorm(candidate_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=candidate_dim,
            nhead=heads,
            dim_feedforward=candidate_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.candidate_encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(candidate_dim)

        self.score_head = nn.Sequential(
            nn.Linear(candidate_dim + 4, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        self.anchor_scale_raw = nn.Parameter(torch.tensor(1.8))
        self.visual_scale_raw = nn.Parameter(torch.tensor(0.25))
        self.transition_scale_raw = nn.Parameter(torch.tensor(0.20))

    def forward(self, x):
        # x: [B, W, S, F]
        B, W, S, Fdim = x.shape
        z = x.permute(0, 2, 1, 3).contiguous().reshape(B * S, W, Fdim)
        z = self.in_proj(z)
        z, _ = self.temporal_gru(z)
        z = z[:, -1, :]
        z = self.candidate_proj(z).reshape(B, S, -1)
        z = self.norm(self.candidate_encoder(z))

        center = x[:, HALF, :, :]
        anchor = center[:, :, 2]
        visual = center[:, :, 5]
        boundary = center[:, :, 20].clamp(0.0, 1.0)
        local_v4 = center[:, :, 6]

        aux = torch.stack([anchor, visual, boundary, local_v4], dim=-1)
        residual = self.score_head(torch.cat([z, aux], dim=-1)).squeeze(-1)

        anchor_scale = F.softplus(self.anchor_scale_raw) + 0.35
        visual_scale = F.softplus(self.visual_scale_raw)
        transition_scale = F.softplus(self.transition_scale_raw)

        # V4 rank is deliberately kept as a strong prior; the network learns
        # when temporal/context evidence should move away from it.
        logits = (
            anchor_scale * anchor
            + visual_scale * visual
            + 0.75 * residual
            + transition_scale * boundary
            + 0.15 * local_v4
        )
        return logits


def mixed_loss(logits, target, margin=0.10, label_smoothing=0.04, hard_k=5):
    ce = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    order = torch.argsort(logits.detach(), dim=1, descending=True)
    rank_terms = []
    for i in range(logits.shape[0]):
        pos = int(target[i])
        negs = [int(j) for j in order[i].tolist() if int(j) != pos][:hard_k]
        if negs:
            rank_terms.append(F.relu(margin - logits[i, pos] + logits[i, negs]).mean())
    rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=logits.device)
    return ce + 0.30 * rank


def make_batch(train_data, items, rng, augment=True):
    contexts, targets, weights = [], [], []
    for li, i, w in items:
        x = pad_context(train_data[li]["X"], i)
        if augment:
            x = augment_context(x, rng)
        contexts.append(x)
        targets.append(int(train_data[li]["Y"][i]))
        weights.append(float(w))
    return (
        torch.from_numpy(np.stack(contexts)).float(),
        torch.from_numpy(np.asarray(targets, dtype=np.int64)).long(),
        torch.from_numpy(np.asarray(weights, dtype=np.float32)).float(),
    )


def make_grouped_batches(index, train_data, batch_size, seed, augment=True):
    groups = {}
    for item in index:
        groups.setdefault(train_data[item[0]]["n_slides"], []).append(item)
    rng = random.Random(seed)
    all_batches = []
    for group in groups.values():
        group = list(group)
        rng.shuffle(group)
        for start in range(0, len(group), batch_size):
            all_batches.append(group[start:start + batch_size])
    rng.shuffle(all_batches)
    np_rng = np.random.default_rng(seed)
    return [make_batch(train_data, b, np_rng, augment=augment) for b in all_batches]


def train_one_seed(train_data, seed, epochs, batch_size):
    set_seed(seed)
    model = V9SequenceFusion().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    index = build_training_index(train_data)
    best_state = None
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        batches = make_grouped_batches(index, train_data, batch_size, seed + epoch, augment=True)
        running = 0.0
        seen = 0
        for xb, yb, wb in batches:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            per = F.cross_entropy(logits, yb, reduction="none", label_smoothing=0.04)
            wb = wb / (wb.mean() + 1e-8)
            loss = (per * wb).mean() + 0.30 * (mixed_loss(logits, yb) - F.cross_entropy(logits, yb, label_smoothing=0.04))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            running += float(loss.detach().cpu()) * len(yb)
            seen += len(yb)
        sched.step()
        avg = running / max(1, seen)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[V9 seed={seed}] epoch={epoch:02d} loss={avg:.5f}")

    if best_state is None:
        raise RuntimeError("No model checkpoint produced.")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_lecture(model, X):
    model.eval()
    outputs = []
    B = 8
    for start in range(0, len(X), B):
        batch = []
        for i in range(start, min(len(X), start + B)):
            batch.append(pad_context(X, i))
        xb = torch.from_numpy(np.stack(batch)).float().to(DEVICE)
        outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_ensemble(models, X):
    return np.mean([predict_lecture(m, X) for m in models], axis=0)


def evaluate_lecture(models, decoder, lecture, split_name):
    srt, pdf, gt_path, video = v8.v7.asset_paths(lecture)
    if any(x is None for x in (srt, pdf, gt_path, video)):
        raise RuntimeError("Missing exact asset(s)")

    v4_dir = v8.v7.ensure_v4(lecture, srt, pdf, video)
    v4 = np.load(v4_dir / "v4_final_score_matrix.npy").astype(np.float32)
    bp = v4_dir / "v4_boundary_probability.npy"
    boundary = np.load(bp).astype(np.float32) if bp.exists() else np.zeros(v4.shape[0], dtype=np.float32)
    segments = v8.v7.get_segments(srt)
    mids = v8.v7.segment_mids(segments)
    visual_times, visual_scores = v8.v7.get_visual_scores(lecture, pdf, video)
    gt = v8.v7.load_gt(gt_path)
    X, _ = v8.v7.build_features_from_mids(
        v4=v4,
        boundary=boundary,
        mids=mids,
        visual_times=visual_times,
        visual_scores=visual_scores,
        times=mids.astype(np.float32),
        gt_slides=None,
    )
    X = v8.v7.normalize_rows(X)
    emissions = predict_ensemble(models, X)
    path = v8.v7.decode_monotonic(
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
    acc = float((np.asarray(preds) == np.asarray(truths)).mean())
    return {"split": split_name, "lecture": lecture, "points": len(truths), "accuracy": acc}


def decoder_sweep(models):
    candidates = []
    for max_jump in (1, 2, 3, 4):
        for penalty in (0.0, 0.01, 0.03, 0.05, 0.08):
            for boundary_reward in (0.0, 0.08, 0.16):
                dec = {"max_jump": max_jump, "jump_penalty": penalty, "boundary_reward": boundary_reward}
                vals = []
                rows = []
                for lecture in VAL:
                    try:
                        r = evaluate_lecture(models, dec, lecture, "validation")
                        rows.append(r)
                        vals.append(r["accuracy"])
                    except Exception as exc:
                        print(f"[VAL SKIP] {lecture}: {type(exc).__name__}: {exc}")
                if vals:
                    candidates.append((float(np.mean(vals)), dec, rows))
    if not candidates:
        raise RuntimeError("No validation configuration evaluated.")
    candidates.sort(key=lambda z: z[0], reverse=True)
    return candidates[0]


def train_mode(args):
    print("=" * 78)
    print("LECTURESENSE V9 — TEMPORAL SEQUENCE GENERALIZATION")
    print("=" * 78)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_data = []
    for lecture in TRAIN:
        try:
            d = load_lecture(lecture)
            train_data.append(d)
        except Exception as exc:
            print(f"[TRAIN SKIP] {lecture}: {type(exc).__name__}: {exc}")

    print(f"Train lectures loaded: {len(train_data)} / {len(TRAIN)}")
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
        print(f"TRAINING V9 ENSEMBLE MEMBER seed={seed}")
        print("-" * 72)
        models.append(train_one_seed(train_data, seed, args.epochs, args.batch_size))

    print("\nRunning validation-only decoder sweep...")
    best_macro, best_decoder, best_rows = decoder_sweep(models)
    print("Best validation decoder:", best_decoder)
    print(f"Validation macro accuracy: {best_macro:.4f} ({best_macro*100:.2f}%)")
    print(pd.DataFrame(best_rows).to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "LectureSense-V9-TemporalSequence-Generalization",
        "feature_dim": FEATURE_DIM,
        "window": WINDOW,
        "models": [m.state_dict() for m in models],
        "train_lectures": [d["lecture"] for d in train_data],
        "validation_lectures": VAL,
        "test_lectures": TEST,
        "decoder": best_decoder,
        "validation_macro_accuracy": best_macro,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }
    torch.save(payload, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best_rows).to_csv(OUT / "validation_results_at_train.csv", index=False)
    (OUT / "validation_summary_at_train.json").write_text(json.dumps({
        "validation_macro_accuracy": best_macro,
        "decoder": best_decoder,
        "test_used": False,
    }, indent=2), encoding="utf-8")
    print("Saved model:", MODEL_PATH)
    print("Saved config:", CONFIG_PATH)


def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run --mode train first.")
    ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    models = []
    for state in ck["models"]:
        m = V9SequenceFusion(feature_dim=int(ck["feature_dim"])).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models, dict(ck["decoder"])


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
    df = pd.DataFrame(rows)
    macro = float(df["accuracy"].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "validation_results.csv", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps({
        "validation_macro_accuracy": macro,
        "decoder": decoder,
        "test_used": False,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("V9 VALIDATION RESULT")
    print("=" * 78)
    print(df.to_string(index=False))
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
    df = pd.DataFrame(rows)
    macro = float(df["accuracy"].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "test_results.csv", index=False)
    (OUT / "test_summary.json").write_text(json.dumps({
        "test_macro_accuracy": macro,
        "results": rows,
        "decoder": decoder,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("V9 FINAL HELD-OUT TEST RESULT")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nMacro accuracy: {macro:.4f} ({macro*100:.2f}%)")
    print("Test labels were not used for training or model selection.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "validate", "test"], default="train")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
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
