
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# LectureSense V6
# Candidate-level multimodal reranker with:
#   - V4 learned temporal scores
#   - Visual-First scores
#   - hard-negative mining
#   - residual/skip connection from V4 + visual evidence
#   - 3-seed ensemble
#   - validation-only decoder search
#
# Split:
#   12 official train lectures (one currently missing its exact video)
#   4 validation lectures
#   4 held-out test lectures
#
# IMPORTANT:
#   This script never uses test labels for training or model selection.
#   It does not guarantee 95%; it reports measured performance.
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN = [
    "cities_and_decarbonization",
    "climate_and_cities",
    "climate_science_policy_MIT2",
    "cognitive_robotics_MIT",
    "computer_vision_2_2",
    "creating_breakthrough_products_MIT",
    "cryptocurrency_MIT",
    "deeplearning",
    "image_processing",
    "ML_for_health_MIT",
    "numerics",
    "phonetics",
]

VAL = [
    "physics",
    "psychology",
    "reinforcement_learning",
    "sensory_systems",
]

TEST = [
    "short_range",
    "solar_resource",
    "team_dynamics_game_design_MIT",
    "theory_of_computation",
]

DATA = ROOT / "data" / "MaViLS" / "data"
VIDEO_DIR = Path(r"C:\Users\Bharat\Downloads\archive\video")
V4_ROOT = ROOT / "runs" / "v5_gate_train_v4"

OUT = ROOT / "runs" / "v6_hardnegative_fusion_FINAL"
CACHE_ROOT = OUT / "lecture_cache"

MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = MODEL_DIR / "v6_hardnegative_ensemble.pt"
CONFIG_PATH = MODEL_DIR / "v6_hardnegative_config.json"

SRT = {
    "cities_and_decarbonization": "cities_and_decarbonization.srt",
    "climate_and_cities": "climate_and_cities.srt",
    "climate_science_policy_MIT2": "climate_science_policy_MIT.srt",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT.srt",
    "computer_vision_2_2": "computer_vision_2_2_high_res.srt",
    "creating_breakthrough_products_MIT": "creating_breakthrough_products_MIT.srt",
    "cryptocurrency_MIT": "cryptocurrency.srt",
    "deeplearning": "deeplearning_goodfellow.srt",
    "image_processing": "image_processing_high_res.srt",
    "ML_for_health_MIT": "ML_for_health_MIT.srt",
    "numerics": "numerics_hennig.srt",
    "phonetics": "phonetics.srt",
    "physics": "physics_intro_02.srt",
    "psychology": "psychology_MIT.srt",
    "reinforcement_learning": "reinforcement_learning_silver.srt",
    "sensory_systems": "visual_system_MIT.srt",
    "short_range": "short_range_mit.srt",
    "solar_resource": "solar_resource.srt",
    "team_dynamics_game_design_MIT": "team_dynamics_game_design_MIT.srt",
    "theory_of_computation": "theory_of_computation.srt",
}

PDF = {
    "cities_and_decarbonization": "cities&decarbonization.pdf",
    "climate_and_cities": "cities&climate.pdf",
    "climate_science_policy_MIT2": "climate_science_policy_MIT.pdf",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT.pdf",
    "computer_vision_2_2": "computer_vision_2_2_geiger.pdf",
    "creating_breakthrough_products_MIT": "creating_breakthrough_products.pdf",
    "cryptocurrency_MIT": "cryptocurrency.pdf",
    "deeplearning": "deeplearning_goodfellow.pdf",
    "image_processing": "image_processing.pdf",
    "ML_for_health_MIT": "ML_for_health_care_MIT.pdf",
    "numerics": "numerics.pdf",
    "phonetics": "phonetics.pdf",
    "physics": "physics_intro_02.pdf",
    "psychology": "psychology.pdf",
    "reinforcement_learning": "reinforcement_learning.pdf",
    "sensory_systems": "sensory_systems.pdf",
    "short_range": "short_range_mit.pdf",
    "solar_resource": "solar_resource.pdf",
    "team_dynamics_game_design_MIT": "team_dynamics_game_design_mit.pdf",
    "theory_of_computation": "theory_of_computation.pdf",
}

VIDEO = {
    "cities_and_decarbonization": "cities_and_decarbonization_standard_res.mp4",
    "climate_and_cities": "cities_and_climate_high_res.mp4",
    # Intentionally left unresolved: no exact filename was identified.
    "climate_science_policy_MIT2": None,
    "cognitive_robotics_MIT": "cognitive_robotics_high_res.mp4",
    "computer_vision_2_2": "computer_vision_2_2_high_res.mp4",
    "creating_breakthrough_products_MIT": "creating_breakthrough_products_MIT.mp4",
    "cryptocurrency_MIT": "cryptocurrency_high_res.mp4",
    "deeplearning": "deep_learning_high_res.mp4",
    "image_processing": "image_processing_high_res.mp4",
    "ML_for_health_MIT": "ML_for_health_high_res.mp4",
    "numerics": "numerics_high_res.mp4",
    "phonetics": "phonetics_high_res.mp4",
    "physics": "physics_high_res.mp4",
    "psychology": "psychology_high_res.mp4",
    "reinforcement_learning": "reinforcement_learning_high_res.mp4",
    "sensory_systems": "sensory_systems_high_res.mp4",
    "short_range": "short_range_MIT.mp4",
    "solar_resource": "solar_resource_high_res.mp4",
    "team_dynamics_game_design_MIT": "team_dynamics_high_res.mp4",
    "theory_of_computation": "theory_of_computation_high_res.mp4",
}

GT = {x: f"ground_truth_{x}.xlsx" for x in TRAIN + VAL + TEST}
GT["team_dynamics_game_design_MIT"] = "ground_truth_team_dynamics_game_design_MIT.xlsx"
GT["sensory_systems"] = "ground_truth_sensory_systems.xlsx"


def asset_paths(lecture):
    srt = DATA / "audioscripts" / SRT[lecture]
    pdf = DATA / "lectures" / PDF[lecture]
    gt = DATA / "ground_truth_files" / GT[lecture]
    video = VIDEO_DIR / VIDEO[lecture] if VIDEO.get(lecture) else None
    return (
        srt if srt.exists() else None,
        pdf if pdf.exists() else None,
        gt if gt.exists() else None,
        video if video is not None and video.exists() else None,
    )


def load_gt(path):
    df = pd.read_excel(path)
    df.columns = [
        str(c).strip().lower().replace(" ", "").replace("-", "_")
        for c in df.columns
    ]

    time_col = next(
        (c for c in ("time", "timestamp", "start", "key") if c in df.columns),
        None,
    )
    slide_col = next(
        (
            c for c in
            ("slidenumber", "slide_number", "slide", "slideid", "slide_id")
            if c in df.columns
        ),
        None,
    )

    if time_col is None or slide_col is None:
        raise ValueError(
            f"GT columns not understood: {list(df.columns)}"
        )

    out = df[[time_col, slide_col]].copy()
    out.columns = ["time", "gt"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["gt"] = pd.to_numeric(out["gt"], errors="coerce")
    out = out.dropna()
    out = out[out["gt"] > 0]
    out["gt"] = out["gt"].astype(int)

    if out.empty:
        raise ValueError(f"No valid GT rows in {path}")

    return out.sort_values("time").reset_index(drop=True)


def get_segments(srt):
    from inference import read_mavils_srt
    segs = read_mavils_srt(str(srt))
    if not segs:
        raise RuntimeError("No transcript segments")
    return segs


def segment_mids(segments):
    mids = []
    for s in segments:
        if isinstance(s, dict):
            a = s.get("start", s.get("start_time"))
            b = s.get("end", s.get("end_time", a))
        else:
            a = getattr(s, "start")
            b = getattr(s, "end", a)
        if a is None and b is None:
            raise ValueError("Transcript segment has no timestamp")
        if a is None:
            a = b
        if b is None:
            b = a
        mids.append((float(a) + float(b)) / 2.0)
    return np.asarray(mids, dtype=np.float32)


def ensure_v4(lecture, srt, pdf, video):
    d = V4_ROOT / lecture
    score_path = d / "v4_final_score_matrix.npy"
    if score_path.exists():
        print(f"[V4] Reusing: {d}")
        return d

    if video is None:
        raise RuntimeError(
            f"No exact video available for {lecture}; refusing to guess."
        )

    ck = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
    if not ck.exists():
        raise FileNotFoundError(ck)

    d.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "src" / "inference_temporal_v4.py"),
        "--transcript", str(srt),
        "--slides", str(pdf),
        "--video", str(video),
        "--checkpoint", str(ck),
        "--out", str(d / "timeline.json"),
        "--render-dir", str(d / "rendered_slides"),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", "0.4",
        "--video-sample-seconds", "1.0",
    ]

    print("[V4] Running official V4...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return d


def visual_cache(lecture, pdf, video):
    new_cache = OUT / "_visual_cache" / f"{lecture}.npz"
    old_cache = (
        ROOT / "runs" / "v5_candidate_fusion_FINAL"
        / "_visual_cache" / f"{lecture}.npz"
    )

    if new_cache.exists():
        z = np.load(new_cache)
        print(f"[VISUAL] Reusing V6 cache: {lecture}")
        return (
            np.asarray(z["times"], dtype=np.float32),
            np.asarray(z["scores"], dtype=np.float32),
            np.asarray(z["path"], dtype=np.int32),
        )

    if old_cache.exists():
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_cache, new_cache)
        z = np.load(new_cache)
        print(f"[VISUAL] Reusing previous V5 cache: {lecture}")
        return (
            np.asarray(z["times"], dtype=np.float32),
            np.asarray(z["scores"], dtype=np.float32),
            np.asarray(z["path"], dtype=np.int32),
        )

    # Also reuse the V5 training-data visual cache when it exists.
    candidates = [
        ROOT / "runs" / "v5_train_gate_dataset_FINAL_V2"
        / "_visual_cache" / f"{lecture}.npz",
        ROOT / "runs" / "v5_train_gate_dataset_FINAL"
        / "_visual_cache" / f"{lecture}.npz",
    ]

    for c in candidates:
        if c.exists():
            new_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c, new_cache)
            z = np.load(new_cache)
            print(f"[VISUAL] Reusing existing training cache: {lecture}")
            return (
                np.asarray(z["times"], dtype=np.float32),
                np.asarray(z["scores"], dtype=np.float32),
                np.asarray(z["path"], dtype=np.int32),
            )

    if video is None:
        raise RuntimeError(
            f"No exact video available for {lecture}; refusing to guess."
        )

    from visual_first_alignment import run_visual_first, viterbi_monotonic

    print(f"[VISUAL] Running Visual-First: {lecture}")
    _, scores, times, _ = run_visual_first(
        Path(video),
        Path(pdf),
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )

    scores = np.asarray(scores, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)

    path, _ = viterbi_monotonic(
        scores,
        max_jump=2,
        jump_penalty=0.05,
    )
    path = np.asarray(path, dtype=np.int32)

    new_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        new_cache,
        times=times,
        scores=scores,
        path=path,
    )

    return times, scores, path


def row_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def row_rank(x):
    order = np.argsort(-x, axis=-1)
    rank = np.empty_like(order, dtype=np.float32)
    for i in range(x.shape[0]):
        rank[i, order[i]] = np.arange(x.shape[1], dtype=np.float32)
    if x.shape[1] > 1:
        rank /= float(x.shape[1] - 1)
    return 1.0 - rank


def entropy(p):
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def build_features(v4, visual, boundary, mids, visual_times, gt):
    """
    Creates one candidate feature matrix per GT timestamp.

    Shape:
        X = [N_GT, N_SLIDES, F]

    Positive candidate is the GT slide.
    Hard negatives are mined from top V4 / visual / neighboring candidates.
    """
    n_seg, n_slides = v4.shape
    vp = row_softmax(v4)
    vr = row_rank(v4)
    wp = row_softmax(visual)
    wr = row_rank(visual)

    gt_times = gt["time"].to_numpy(np.float32)
    gt_slides = gt["gt"].to_numpy(np.int64) - 1

    X = []
    Y = []
    T = []

    for t, truth in zip(gt_times, gt_slides):
        si = int(np.searchsorted(mids, float(t), side="right") - 1)
        si = max(0, min(si, n_seg - 1))
        vi = int(np.argmin(np.abs(visual_times - float(t))))

        v_cur = v4[si]
        v_prev = v4[max(0, si - 1)]
        v_next = v4[min(n_seg - 1, si + 1)]

        w_cur = visual[vi]
        w_prev = visual[max(0, vi - 1)]
        w_next = visual[min(len(visual_times) - 1, vi + 1)]

        v_top = int(np.argmax(v_cur))
        w_top = int(np.argmax(w_cur))

        v_sorted = np.sort(v_cur)
        w_sorted = np.sort(w_cur)
        v_margin = float(v_sorted[-1] - v_sorted[-2]) if n_slides > 1 else 0.0
        w_margin = float(w_sorted[-1] - w_sorted[-2]) if n_slides > 1 else 0.0

        base = []
        for s in range(n_slides):
            same_v4 = float(s == v_top)
            same_vis = float(s == w_top)

            # Strong residual features.
            f = [
                float(v_cur[s]),
                float(vp[si, s]),
                float(vr[si, s]),
                float(w_cur[s]),
                float(wp[vi, s]),
                float(wr[vi, s]),

                # Local temporal support.
                float((v_prev[s] + v_cur[s] + v_next[s]) / 3.0),
                float((w_prev[s] + w_cur[s] + w_next[s]) / 3.0),
                float(v_cur[s] - v_prev[s]),
                float(v_cur[s] - v_next[s]),
                float(w_cur[s] - w_prev[s]),
                float(w_cur[s] - w_next[s]),

                # Cross-modal relation.
                float(v_cur[s] * w_cur[s]),
                float(v_cur[s] - w_cur[s]),
                float(w_cur[s] - v_cur[s]),
                float(v_cur[s] + w_cur[s]),

                # Confidence.
                v_margin,
                w_margin,
                float(entropy(vp[si])),
                float(entropy(wp[vi])),

                # Boundary / temporal context.
                float(boundary[si]) if si < len(boundary) else 0.0,
                float(abs(s - v_top)),
                float(abs(s - w_top)),
                same_v4,
                same_vis,
                float(s == v_top == w_top),

                # Slide/time position.
                float(s / max(1, n_slides - 1)),
                float(si / max(1, n_seg - 1)),
            ]
            base.append(f)

        X.append(np.asarray(base, dtype=np.float32))
        Y.append(int(truth))
        T.append(float(t))

    return (
        np.stack(X, axis=0),
        np.asarray(Y, dtype=np.int64),
        np.asarray(T, dtype=np.float32),
    )


def build_or_load_lecture(lecture):
    cache = CACHE_ROOT / lecture / "candidate_features.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"[DATA] Reusing V6 candidate cache: {lecture}")
        return {
            "lecture": lecture,
            "X": np.asarray(z["X"], dtype=np.float32),
            "Y": np.asarray(z["Y"], dtype=np.int64),
            "T": np.asarray(z["T"], dtype=np.float32),
            "n_slides": int(z["n_slides"]),
        }

    srt, pdf, gt_path, video = asset_paths(lecture)

    print("\n" + "=" * 78)
    print("PREPARING", lecture)
    print("=" * 78)
    print("SRT  :", srt)
    print("PDF  :", pdf)
    print("GT   :", gt_path)
    print("VIDEO:", video)

    if any(x is None for x in (srt, pdf, gt_path, video)):
        raise RuntimeError("Missing exact asset(s)")

    v4_dir = ensure_v4(lecture, srt, pdf, video)

    v4 = np.load(
        v4_dir / "v4_final_score_matrix.npy"
    ).astype(np.float32)

    boundary_path = v4_dir / "v4_boundary_probability.npy"
    boundary = (
        np.load(boundary_path).astype(np.float32)
        if boundary_path.exists()
        else np.zeros(v4.shape[0], dtype=np.float32)
    )

    segments = get_segments(srt)
    mids = segment_mids(segments)

    if v4.shape[0] != len(mids):
        raise ValueError(
            f"V4 rows={v4.shape[0]} != transcript segments={len(mids)}"
        )

    visual_times, visual_scores, _ = visual_cache(
        lecture,
        pdf,
        video,
    )

    if visual_scores.shape[1] != v4.shape[1]:
        raise ValueError(
            f"V4 slide count {v4.shape[1]} != visual slide count "
            f"{visual_scores.shape[1]}"
        )

    gt = load_gt(gt_path)

    X, Y, T = build_features(
        v4,
        visual_scores,
        boundary,
        mids,
        visual_times,
        gt,
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X=X,
        Y=Y,
        T=T,
        n_slides=np.int32(v4.shape[1]),
    )

    print(
        f"[DATA] {lecture}: timestamps={len(Y)}, "
        f"slides={v4.shape[1]}, features={X.shape[-1]}"
    )

    return {
        "lecture": lecture,
        "X": X,
        "Y": Y,
        "T": T,
        "n_slides": int(v4.shape[1]),
    }


def standardize(train_data):
    flat = np.concatenate(
        [
            d["X"].reshape(-1, d["X"].shape[-1])
            for d in train_data
        ],
        axis=0,
    )

    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    return mean, std


def apply_standardization(d, mean, std):
    return {
        **d,
        "X": ((d["X"] - mean) / std).astype(np.float32),
    }


class HardNegativeReranker(nn.Module):
    """
    Residual scorer:
        learned residual + stable V4/visual evidence.

    The first four input channels are V4 score/prob/rank and visual
    score/prob/rank. A small skip connection keeps strong V4 evidence
    from being destroyed by the new model.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(160, 80),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(80, 1),
        )

        self.v4_skip = nn.Parameter(torch.tensor(0.70))
        self.visual_skip = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, s, f = x.shape

        residual = self.mlp(
            x.reshape(b * s, f)
        ).reshape(b, s)

        # x[:, :, 0] is standardized V4 raw score.
        # x[:, :, 3] is standardized Visual raw score.
        base = (
            self.v4_skip * x[:, :, 0]
            + self.visual_skip * x[:, :, 3]
            + self.bias
        )

        return base + residual


def hard_negative_loss(logits, target, hard_k=8, margin=0.15):
    """
    Listwise CE + explicit margin against top hard negatives.

    `target` is slide index within the candidate row.
    """
    ce = F.cross_entropy(logits, target)

    with torch.no_grad():
        order = torch.argsort(logits, dim=1, descending=True)

    hard_losses = []

    for i in range(logits.shape[0]):
        pos = target[i]

        negatives = [
            int(j)
            for j in order[i].tolist()
            if int(j) != int(pos)
        ][:hard_k]

        if not negatives:
            continue

        pos_logit = logits[i, pos]
        neg_logits = logits[i, negatives]

        ranking = F.relu(
            margin - pos_logit + neg_logits
        ).mean()

        hard_losses.append(ranking)

    if hard_losses:
        hn = torch.stack(hard_losses).mean()
    else:
        hn = torch.tensor(0.0, device=logits.device)

    return ce + 0.35 * hn


def train_one_seed(train_data, mean, std, seed, epochs=45, batch_size=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = [
        apply_standardization(d, mean, std)
        for d in train_data
    ]

    order = list(range(len(data)))
    random.Random(seed).shuffle(order)

    dev_n = max(2, int(round(0.25 * len(order))))
    dev_ids = order[:dev_n]
    fit_ids = order[dev_n:]

    model = HardNegativeReranker(
        data[0]["X"].shape[-1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=1e-4,
    )

    best_dev = -1.0
    best_state = None
    patience = 10
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(seed + epoch).shuffle(fit_ids)

        losses = []

        for idx in fit_ids:
            d = data[idx]

            x = torch.from_numpy(d["X"]).to(DEVICE)
            y = torch.from_numpy(d["Y"]).to(DEVICE)

            optimizer.zero_grad()

            for start in range(0, len(y), batch_size):
                xb = x[start:start + batch_size]
                yb = y[start:start + batch_size]

                logits = model(xb)
                loss = hard_negative_loss(
                    logits,
                    yb,
                    hard_k=min(8, logits.shape[1] - 1),
                    margin=0.15,
                )

                # Mild adjacent-segment consistency.
                if len(logits) > 1:
                    p = F.log_softmax(logits[:-1], dim=1)
                    q = F.softmax(logits[1:], dim=1)
                    smooth = F.kl_div(
                        p,
                        q,
                        reduction="batchmean",
                    )
                    loss = loss + 0.015 * smooth

                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )
                optimizer.step()

                losses.append(float(loss.detach().cpu()))

        model.eval()
        dev_acc = []

        with torch.no_grad():
            for idx in dev_ids:
                d = data[idx]
                x = torch.from_numpy(d["X"]).to(DEVICE)
                pred = model(x).argmax(dim=1).cpu().numpy()
                dev_acc.append(
                    float((pred == d["Y"]).mean())
                )

        score = float(np.mean(dev_acc))

        if score > best_dev:
            best_dev = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[SEED {seed}] epoch={epoch:02d} "
                f"loss={np.mean(losses):.4f} "
                f"group-dev={score:.4f}"
            )

        if stale >= patience:
            print(
                f"[SEED {seed}] early stop at epoch {epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_dev


def train_ensemble(train_data, mean, std):
    models = []
    dev_scores = []

    for seed in (17, 42, 91):
        print("\n" + "-" * 72)
        print(f"TRAINING ENSEMBLE MEMBER seed={seed}")
        print("-" * 72)

        model, dev = train_one_seed(
            train_data,
            mean,
            std,
            seed,
        )

        models.append(model)
        dev_scores.append(dev)

    return models, dev_scores


@torch.no_grad()
def predict_matrix(models, d, mean, std):
    dd = apply_standardization(d, mean, std)
    x = torch.from_numpy(dd["X"]).to(DEVICE)

    logits = []
    for model in models:
        model.eval()
        logits.append(
            model(x).cpu().numpy()
        )

    return np.mean(logits, axis=0)


def greedy_viterbi(score, max_jump=3, jump_penalty=0.05):
    score = np.asarray(score, dtype=np.float32)
    n, s = score.shape

    dp = np.full((n, s), -np.inf, dtype=np.float32)
    back = np.full((n, s), -1, dtype=np.int16)

    first = min(s, max_jump + 1)
    dp[0, :first] = score[0, :first]

    for t in range(1, n):
        for j in range(s):
            lo = max(0, j - max_jump)
            best = -np.inf
            bestp = -1

            for p in range(lo, j + 1):
                if not np.isfinite(dp[t - 1, p]):
                    continue

                trans = (
                    0.0
                    if p == j
                    else -jump_penalty * (j - p)
                )

                val = dp[t - 1, p] + trans

                if val > best:
                    best = val
                    bestp = p

            dp[t, j] = best + score[t, j]
            back[t, j] = bestp

    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))

    for t in range(n - 1, 0, -1):
        p = int(back[t, path[t]])
        path[t - 1] = path[t] if p < 0 else p

    return path


def eval_lecture(models, mean, std, lecture, max_jump, penalty):
    srt, pdf, gt_path, video = asset_paths(lecture)

    print("\n" + "=" * 78)
    print("EVALUATING", lecture)
    print("=" * 78)
    print("SRT  :", srt)
    print("PDF  :", pdf)
    print("GT   :", gt_path)
    print("VIDEO:", video)

    if any(x is None for x in (srt, pdf, gt_path, video)):
        raise RuntimeError("Missing exact asset(s)")

    # Make sure V4/Visual caches exist. Do not use GT beyond scoring.
    v4_dir = ensure_v4(
        lecture,
        srt,
        pdf,
        video,
    )

    v4 = np.load(
        v4_dir / "v4_final_score_matrix.npy"
    ).astype(np.float32)

    bpath = v4_dir / "v4_boundary_probability.npy"
    boundary = (
        np.load(bpath).astype(np.float32)
        if bpath.exists()
        else np.zeros(v4.shape[0], dtype=np.float32)
    )

    segments = get_segments(srt)
    mids = segment_mids(segments)

    visual_times, visual_scores, _ = visual_cache(
        lecture,
        pdf,
        video,
    )

    # Build an inference feature row for every transcript segment.
    dummy_gt = pd.DataFrame({
        "time": mids,
        "gt": np.ones(len(mids), dtype=int),
    })

    X, _, T = build_features(
        v4,
        visual_scores,
        boundary,
        mids,
        visual_times,
        dummy_gt,
    )

    d = {
        "lecture": lecture,
        "X": X,
        "Y": np.zeros(len(X), dtype=np.int64),
        "T": T,
        "n_slides": v4.shape[1],
    }

    emissions = predict_matrix(
        models,
        d,
        mean,
        std,
    )

    path = greedy_viterbi(
        emissions,
        max_jump=max_jump,
        jump_penalty=penalty,
    )

    gt = load_gt(gt_path)

    preds = []
    truths = []

    for t, g in zip(gt["time"], gt["gt"]):
        si = int(
            np.searchsorted(
                mids,
                float(t),
                side="right",
            ) - 1
        )
        si = max(0, min(si, len(path) - 1))

        preds.append(int(path[si]) + 1)
        truths.append(int(g))

    preds = np.asarray(preds)
    truths = np.asarray(truths)

    acc = float((preds == truths).mean())

    return {
        "lecture": lecture,
        "points": len(truths),
        "v6_accuracy": acc,
    }


def save_bundle(models, mean, std, dev_scores, decoder):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "V6-hardnegative-ensemble",
        "input_dim": int(len(mean)),
        "feature_mean": mean,
        "feature_std": std,
        "dev_scores": np.asarray(dev_scores, dtype=np.float32),
        "decoder": decoder,
        "models": [
            m.state_dict()
            for m in models
        ],
    }

    torch.save(
        payload,
        MODEL_PATH,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "version": "V6-hardnegative-ensemble",
                "members": 3,
                "feature_dim": int(len(mean)),
                "dev_scores": [float(x) for x in dev_scores],
                "decoder": decoder,
                "test_used_for_training": False,
                "test_used_for_model_selection": False,
                "note": (
                    "95 percent is a target, not a guaranteed outcome."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved model:", MODEL_PATH)
    print("Saved config:", CONFIG_PATH)


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run --mode train first."
        )

    ck = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    mean = np.asarray(
        ck["feature_mean"],
        dtype=np.float32,
    )
    std = np.asarray(
        ck["feature_std"],
        dtype=np.float32,
    )

    models = []
    for state in ck["models"]:
        m = HardNegativeReranker(
            int(ck["input_dim"])
        ).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    decoder = dict(
        ck.get(
            "decoder",
            {
                "max_jump": 3,
                "jump_penalty": 0.05,
            },
        )
    )

    return models, mean, std, decoder


def train_mode():
    print("=" * 78)
    print("LECTURESENSE V6 — HARD-NEGATIVE MULTIMODAL ENSEMBLE")
    print("=" * 78)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Training lectures:", len(TRAIN))
    print("Validation lectures:", len(VAL))
    print("Test lectures:", len(TEST))
    print("Test used for training: NO")
    print("=" * 78)

    train_data = []

    for lecture in TRAIN:
        try:
            train_data.append(
                build_or_load_lecture(lecture)
            )
        except Exception as exc:
            print(
                f"[TRAIN SKIP] {lecture}: "
                f"{type(exc).__name__}: {exc}"
            )

    if len(train_data) < 5:
        raise RuntimeError(
            f"Only {len(train_data)} usable training lectures remain."
        )

    mean, std = standardize(train_data)

    models, dev_scores = train_ensemble(
        train_data,
        mean,
        std,
    )

    # A safe default decoder is saved first.
    decoder = {
        "max_jump": 3,
        "jump_penalty": 0.05,
    }

    save_bundle(
        models,
        mean,
        std,
        dev_scores,
        decoder,
    )

    rows = []

    for d in train_data:
        em = predict_matrix(
            models,
            d,
            mean,
            std,
        )
        pred = em.argmax(axis=1)

        rows.append(
            {
                "lecture": d["lecture"],
                "points": int(len(d["Y"])),
                "candidate_top1_accuracy": float(
                    (pred == d["Y"]).mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUT / "train_results.csv",
        index=False,
    )

    summary = {
        "lectures_used": len(train_data),
        "total_training_points": int(
            sum(len(d["Y"]) for d in train_data)
        ),
        "macro_candidate_top1": float(
            result["candidate_top1_accuracy"].mean()
        ),
        "ensemble_internal_group_dev": float(
            np.mean(dev_scores)
        ),
        "test_used": False,
    }

    (OUT / "train_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTRAIN RESULTS")
    print(result.to_string(index=False))
    print("\nTRAIN SUMMARY")
    print(json.dumps(summary, indent=2))


def validate_mode():
    models, mean, std, _ = load_bundle()

    # Validation-only decoder sweep.
    configs = []
    for max_jump in (1, 2, 3, 4):
        for penalty in (0.0, 0.01, 0.03, 0.05, 0.08, 0.12):
            configs.append(
                {
                    "max_jump": max_jump,
                    "jump_penalty": penalty,
                }
            )

    best_config = None
    best_score = -1.0
    best_rows = None

    for cfg in configs:
        rows = []

        for lecture in VAL:
            try:
                row = eval_lecture(
                    models,
                    mean,
                    std,
                    lecture,
                    cfg["max_jump"],
                    cfg["jump_penalty"],
                )
                rows.append(row)
            except Exception as exc:
                print(
                    f"[VAL SKIP] {lecture}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if len(rows) < 2:
            continue

        macro = float(
            np.mean(
                [r["v6_accuracy"] for r in rows]
            )
        )

        print(
            f"[VAL CONFIG] max_jump={cfg['max_jump']} "
            f"penalty={cfg['jump_penalty']:.2f} "
            f"macro={macro:.4f}"
        )

        if macro > best_score:
            best_score = macro
            best_config = cfg
            best_rows = rows

    if best_config is None:
        raise RuntimeError("No validation configuration evaluated.")

    # Lock the selected validation decoder for future TEST mode.
    # Models themselves are not retrained on validation labels.
    ck = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    ck["decoder"] = best_config

    torch.save(
        ck,
        MODEL_PATH,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "version": "V6-hardnegative-ensemble",
                "decoder": best_config,
                "validation_macro_accuracy": best_score,
                "validation_only_selection": True,
                "test_used_for_model_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = pd.DataFrame(best_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUT / "validation_results.csv",
        index=False,
    )

    summary = {
        "validation_macro_accuracy": best_score,
        "decoder": best_config,
        "test_used": False,
    }

    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("BEST VALIDATION RESULT")
    print("=" * 78)
    print(result.to_string(index=False))
    print("\nMacro accuracy:", f"{best_score:.4f}")
    print("Locked decoder:", best_config)
    print("Test was not used.")


def test_mode():
    models, mean, std, decoder = load_bundle()

    rows = []

    for lecture in TEST:
        try:
            row = eval_lecture(
                models,
                mean,
                std,
                lecture,
                int(decoder["max_jump"]),
                float(decoder["jump_penalty"]),
            )
            rows.append(row)
        except Exception as exc:
            print(
                f"[TEST SKIP] {lecture}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not rows:
        raise RuntimeError("No test lecture evaluated.")

    result = pd.DataFrame(rows)
    macro = float(result["v6_accuracy"].mean())

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUT / "test_results.csv",
        index=False,
    )

    summary = {
        "test_macro_accuracy": macro,
        "decoder": decoder,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }

    (OUT / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL HELD-OUT TEST RESULT")
    print("=" * 78)
    print(result.to_string(index=False))
    print("\nMacro accuracy:", f"{macro:.4f}")
    print("Test labels were not used for training or selection.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "validate", "test"],
        default="train",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    elif args.mode == "validate":
        validate_mode()
    else:
        test_mode()


if __name__ == "__main__":
    main()
