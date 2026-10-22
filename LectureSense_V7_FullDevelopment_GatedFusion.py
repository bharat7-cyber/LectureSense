
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
from torch.utils.data import Dataset, DataLoader

# ============================================================
# LectureSense V7
# Full-development-data gated multimodal fusion.
#
# Development data:
#   Official train lectures + the four former validation lectures
#   are used for final development training.
#
# Held-out test:
#   short_range
#   solar_resource
#   team_dynamics_game_design_MIT
#   theory_of_computation
#
# The test labels are never used in training or model selection.
#
# Main changes from V6:
#   1) V4 is an explicit anchor rather than a free competing score.
#   2) A learned gate decides how much visual evidence to trust.
#   3) Candidate interactions are modeled with a Transformer encoder.
#   4) Training uses all available development lectures.
#   5) Per-lecture loss weighting prevents long lectures dominating.
#   6) Five-seed ensemble.
#   7) Monotonic temporal decoder with boundary-aware transition support.
#
# 90% is a target, not a guaranteed outcome.
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_LECTURES = [
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

FORMER_VAL_LECTURES = [
    "physics",
    "psychology",
    "reinforcement_learning",
    "sensory_systems",
]

DEVELOPMENT = TRAIN_LECTURES + FORMER_VAL_LECTURES

TEST = [
    "short_range",
    "solar_resource",
    "team_dynamics_game_design_MIT",
    "theory_of_computation",
]

DATA = ROOT / "data" / "MaViLS" / "data"
VIDEO_DIR = Path(r"C:\Users\Bharat\Downloads\archive\video")
V4_ROOT = ROOT / "runs" / "v5_gate_train_v4"

OUT = ROOT / "runs" / "v7_full_development_gated"
CACHE_ROOT = OUT / "candidate_cache"

MODEL_DIR = ROOT / "checkpoints"
MODEL_PATH = MODEL_DIR / "v7_full_development_gated_ensemble.pt"
CONFIG_PATH = MODEL_DIR / "v7_full_development_gated_config.json"

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

GT = {
    x: f"ground_truth_{x}.xlsx"
    for x in DEVELOPMENT + TEST
}
GT["team_dynamics_game_design_MIT"] = (
    "ground_truth_team_dynamics_game_design_MIT.xlsx"
)
GT["sensory_systems"] = "ground_truth_sensory_systems.xlsx"


def resolve_srt(lecture):
    expected = DATA / "audioscripts" / SRT.get(lecture, "")
    if expected.exists():
        return expected

    # Robust fallback for filenames that differ only in suffix/version.
    key = lecture.lower().replace("_mit2", "")
    candidates = list((DATA / "audioscripts").glob("*.srt"))

    scored = []
    for p in candidates:
        name = p.stem.lower()
        score = 0
        for token in key.replace("_", " ").split():
            if len(token) >= 4 and token in name:
                score += 1
        if lecture == "theory_of_computation" and (
            "theory" in name and "computation" in name
        ):
            score += 5
        if lecture == "climate_science_policy_MIT2" and (
            "climate" in name and "policy" in name
        ):
            score += 5

        if score:
            scored.append((score, p))

    if scored:
        scored.sort(
            key=lambda x: (-x[0], x[1].name.lower())
        )
        if scored[0][0] >= 2:
            return scored[0][1]

    return None


def asset_paths(lecture):
    srt = resolve_srt(lecture)
    pdf = DATA / "lectures" / PDF[lecture]
    gt = DATA / "ground_truth_files" / GT[lecture]

    video = (
        VIDEO_DIR / VIDEO[lecture]
        if VIDEO.get(lecture)
        else None
    )

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
        (
            c for c in
            ("time", "timestamp", "start", "key")
            if c in df.columns
        ),
        None,
    )

    slide_col = next(
        (
            c for c in
            (
                "slidenumber",
                "slide_number",
                "slide",
                "slideid",
                "slide_id",
            )
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

        mids.append(
            (float(a) + float(b)) / 2.0
        )

    return np.asarray(mids, dtype=np.float32)


def ensure_v4(lecture, srt, pdf, video):
    d = V4_ROOT / lecture
    score_path = d / "v4_final_score_matrix.npy"

    if score_path.exists():
        return d

    if video is None:
        raise RuntimeError(
            f"No exact video available for {lecture}"
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

    import subprocess

    subprocess.run(
        cmd,
        check=True,
        cwd=ROOT,
    )

    return d


def get_visual_scores(lecture, pdf, video):
    cache = (
        OUT / "_visual_cache"
        / f"{lecture}.npz"
    )

    # Reuse caches made by V6 if available.
    old_candidates = [
        ROOT / "runs"
        / "v6_hardnegative_fusion_FINAL"
        / "_visual_cache"
        / f"{lecture}.npz",

        ROOT / "runs"
        / "v5_candidate_fusion_FINAL"
        / "_visual_cache"
        / f"{lecture}.npz",

        ROOT / "runs"
        / "v5_train_gate_dataset_FINAL_V2"
        / "_visual_cache"
        / f"{lecture}.npz",
    ]

    if not cache.exists():
        for c in old_candidates:
            if c.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(c.read_bytes())
                break

    if cache.exists():
        z = np.load(cache)
        return (
            np.asarray(z["times"], dtype=np.float32),
            np.asarray(z["scores"], dtype=np.float32),
        )

    if video is None:
        raise RuntimeError(
            f"No exact video available for {lecture}"
        )

    from visual_first_alignment import run_visual_first

    _, scores, times, _ = run_visual_first(
        Path(video),
        Path(pdf),
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )

    times = np.asarray(times, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache,
        times=times,
        scores=scores,
    )

    return times, scores


def row_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def row_rank(x):
    order = np.argsort(-x, axis=-1)
    rank = np.empty_like(order, dtype=np.float32)

    for i in range(x.shape[0]):
        rank[i, order[i]] = np.arange(
            x.shape[1],
            dtype=np.float32,
        )

    if x.shape[1] > 1:
        rank /= float(x.shape[1] - 1)

    return 1.0 - rank


def entropy(p):
    return -np.sum(
        p * np.log(p + 1e-12),
        axis=-1,
    )


def build_features_from_mids(
    v4,
    boundary,
    mids,
    visual_times,
    visual_scores,
    times,
    gt_slides=None,
):
    """
    Builds candidate-level features for arbitrary transcript times.

    The cache generated by this code has 28 features per candidate.
    Feature conventions match V6 so existing caches remain reusable.
    """
    n_seg, n_slides = v4.shape

    vp = row_softmax(v4)
    vr = row_rank(v4)

    wp = row_softmax(visual_scores)
    wr = row_rank(visual_scores)

    X = []
    Y = []

    for t in times:
        si = int(
            np.searchsorted(
                mids,
                float(t),
                side="right",
            ) - 1
        )
        si = max(
            0,
            min(si, n_seg - 1),
        )

        vi = int(
            np.argmin(
                np.abs(visual_times - float(t))
            )
        )

        v_cur = v4[si]
        v_prev = v4[max(0, si - 1)]
        v_next = v4[min(n_seg - 1, si + 1)]

        w_cur = visual_scores[vi]
        w_prev = visual_scores[
            max(0, vi - 1)
        ]
        w_next = visual_scores[
            min(len(visual_times) - 1, vi + 1)
        ]

        v_top = int(np.argmax(v_cur))
        w_top = int(np.argmax(w_cur))

        v_sorted = np.sort(v_cur)
        w_sorted = np.sort(w_cur)

        v_margin = (
            float(v_sorted[-1] - v_sorted[-2])
            if n_slides > 1
            else 0.0
        )
        w_margin = (
            float(w_sorted[-1] - w_sorted[-2])
            if n_slides > 1
            else 0.0
        )

        row = []

        for s in range(n_slides):
            row.append(
                [
                    float(v_cur[s]),
                    float(vp[si, s]),
                    float(vr[si, s]),
                    float(w_cur[s]),
                    float(wp[vi, s]),
                    float(wr[vi, s]),

                    float(
                        (
                            v_prev[s]
                            + v_cur[s]
                            + v_next[s]
                        ) / 3.0
                    ),

                    float(
                        (
                            w_prev[s]
                            + w_cur[s]
                            + w_next[s]
                        ) / 3.0
                    ),

                    float(v_cur[s] - v_prev[s]),
                    float(v_cur[s] - v_next[s]),
                    float(w_cur[s] - w_prev[s]),
                    float(w_cur[s] - w_next[s]),

                    float(v_cur[s] * w_cur[s]),
                    float(v_cur[s] - w_cur[s]),
                    float(w_cur[s] - v_cur[s]),
                    float(v_cur[s] + w_cur[s]),

                    v_margin,
                    w_margin,

                    float(entropy(vp[si])),
                    float(entropy(wp[vi])),

                    float(
                        boundary[si]
                        if si < len(boundary)
                        else 0.0
                    ),

                    float(abs(s - v_top)),
                    float(abs(s - w_top)),

                    float(s == v_top),
                    float(s == w_top),
                    float(s == v_top == w_top),

                    float(
                        s / max(1, n_slides - 1)
                    ),

                    float(
                        si / max(1, n_seg - 1)
                    ),
                ]
            )

        X.append(
            np.asarray(row, dtype=np.float32)
        )

        if gt_slides is not None:
            Y.append(
                int(
                    gt_slides[
                        len(Y)
                    ]
                )
            )

    X = np.stack(X, axis=0)

    if Y:
        return X, np.asarray(Y, dtype=np.int64)

    return X, None


def normalize_rows(X):
    """
    Robust per-row normalization for raw V4/visual scores.
    Rank/probability features are already normalized and are retained.
    """
    X = X.copy()

    for channel in (0, 3):
        a = X[:, :, channel]
        mean = a.mean(axis=1, keepdims=True)
        std = a.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        X[:, :, channel] = (
            a - mean
        ) / std

    # Distance features are compressed to avoid excessive scale.
    X[:, :, 21] = np.tanh(
        X[:, :, 21]
    )
    X[:, :, 22] = np.tanh(
        X[:, :, 22]
    )

    return X.astype(np.float32)


def build_or_load_lecture(lecture, save=True):
    cache = (
        CACHE_ROOT
        / lecture
        / "features.npz"
    )

    if cache.exists():
        z = np.load(cache)
        return {
            "lecture": lecture,
            "X": np.asarray(
                z["X"],
                dtype=np.float32,
            ),
            "Y": np.asarray(
                z["Y"],
                dtype=np.int64,
            ),
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

    if any(
        x is None
        for x in (
            srt,
            pdf,
            gt_path,
            video,
        )
    ):
        raise RuntimeError(
            "Missing exact asset(s)"
        )

    v4_dir = ensure_v4(
        lecture,
        srt,
        pdf,
        video,
    )

    v4 = np.load(
        v4_dir
        / "v4_final_score_matrix.npy"
    ).astype(np.float32)

    boundary_path = (
        v4_dir
        / "v4_boundary_probability.npy"
    )

    boundary = (
        np.load(boundary_path).astype(
            np.float32
        )
        if boundary_path.exists()
        else np.zeros(
            v4.shape[0],
            dtype=np.float32,
        )
    )

    segments = get_segments(srt)
    mids = segment_mids(segments)

    if len(mids) != v4.shape[0]:
        raise ValueError(
            f"{lecture}: V4 rows={v4.shape[0]} "
            f"but transcript segments={len(mids)}"
        )

    visual_times, visual_scores = (
        get_visual_scores(
            lecture,
            pdf,
            video,
        )
    )

    gt = load_gt(gt_path)

    X, Y = build_features_from_mids(
        v4=v4,
        boundary=boundary,
        mids=mids,
        visual_times=visual_times,
        visual_scores=visual_scores,
        times=gt["time"].to_numpy(
            dtype=np.float32
        ),
        gt_slides=(
            gt["gt"].to_numpy(
                dtype=np.int64
            ) - 1
        ),
    )

    X = normalize_rows(X)

    if save:
        cache.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.savez_compressed(
            cache,
            X=X,
            Y=Y,
            n_slides=np.int32(
                X.shape[1]
            ),
        )

    return {
        "lecture": lecture,
        "X": X,
        "Y": Y,
        "n_slides": X.shape[1],
    }


class LectureDataset(Dataset):
    def __init__(self, lectures):
        self.rows = []

        for d in lectures:
            n = len(d["Y"])
            weight = 1.0 / max(1, n)

            for i in range(n):
                self.rows.append(
                    (
                        d["X"][i],
                        int(d["Y"][i]),
                        weight,
                        d["lecture"],
                    )
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y, w, lecture = self.rows[i]
        return (
            torch.from_numpy(
                x
            ).float(),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
            torch.tensor(
                w,
                dtype=torch.float32,
            ),
        )


class GatedCandidateTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=28,
        d_model=96,
        nhead=4,
        layers=2,
    ):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(
                feature_dim,
                128,
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                128,
                d_model,
            ),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=192,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.gate_head = nn.Sequential(
            nn.Linear(
                d_model,
                48,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                48,
                1,
            ),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                d_model,
                48,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                48,
                1,
            ),
        )

        # Positive scales, initialized to preserve V4 anchor.
        self.anchor_raw = nn.Parameter(
            torch.tensor(2.2)
        )
        self.visual_raw = nn.Parameter(
            torch.tensor(0.15)
        )

    def forward(self, x):
        h = self.input(x)

        # No padding is used: every batch here contains rows
        # with the same slide count only in the collate strategy.
        # The fallback handles variable slide counts by processing
        # one row at a time in train/predict.
        h = self.transformer(h)
        h = self.norm(h)

        gate = torch.sigmoid(
            self.gate_head(h).squeeze(-1)
        )

        residual = self.residual_head(
            h
        ).squeeze(-1)

        anchor_scale = (
            F.softplus(
                self.anchor_raw
            ) + 0.20
        )

        visual_scale = (
            F.softplus(
                self.visual_raw
            )
        )

        # V4 rank = feature 2 after normalization.
        # Visual rank = feature 5.
        v4_rank = x[:, :, 2]
        vis_rank = x[:, :, 5]

        # Explicit V4 anchor plus learned visual gate and residual.
        logits = (
            anchor_scale * v4_rank
            + gate
            * visual_scale
            * vis_rank
            + 0.70 * residual
        )

        return (
            logits,
            gate,
        )


def hard_negative_loss(
    logits,
    target,
    margin=0.12,
    hard_k=6,
):
    ce = F.cross_entropy(
        logits,
        target,
    )

    with torch.no_grad():
        order = torch.argsort(
            logits,
            dim=1,
            descending=True,
        )

    losses = []

    for i in range(
        logits.shape[0]
    ):
        pos = int(
            target[i]
        )

        negs = [
            int(x)
            for x in order[i].tolist()
            if int(x) != pos
        ][
            :min(
                hard_k,
                logits.shape[1] - 1,
            )
        ]

        if not negs:
            continue

        p = logits[i, pos]
        n = logits[
            i,
            negs,
        ]

        losses.append(
            F.relu(
                margin
                - p
                + n
            ).mean()
        )

    hn = (
        torch.stack(losses).mean()
        if losses
        else torch.tensor(
            0.0,
            device=logits.device,
        )
    )

    return ce + 0.35 * hn


def make_batches(rows, batch_size):
    """
    Rows are grouped by slide count so a standard tensor batch can
    be passed through the Transformer without padding.
    """
    groups = {}

    for item in rows:
        s = item[0].shape[0]
        groups.setdefault(
            s,
            []
        ).append(item)

    batches = []

    rng = random.Random(
        20260928
    )

    for group in groups.values():
        rng.shuffle(group)

        for start in range(
            0,
            len(group),
            batch_size,
        ):
            part = group[
                start:start + batch_size
            ]

            xs = np.stack(
                [
                    p[0]
                    for p in part
                ],
                axis=0,
            )

            ys = np.asarray(
                [
                    p[1]
                    for p in part
                ],
                dtype=np.int64,
            )

            ws = np.asarray(
                [
                    p[2]
                    for p in part
                ],
                dtype=np.float32,
            )

            batches.append(
                (
                    torch.from_numpy(
                        xs
                    ).float(),
                    torch.from_numpy(
                        ys
                    ).long(),
                    torch.from_numpy(
                        ws
                    ).float(),
                )
            )

    rng.shuffle(batches)
    return batches


def train_one_seed(
    development,
    seed,
    epochs=36,
    batch_size=24,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows = LectureDataset(
        development
    ).rows

    model = GatedCandidateTransformer(
        feature_dim=28
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=2e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    best_state = None
    best_loss = float("inf")

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        total_loss = 0.0
        total_n = 0

        batches = make_batches(
            rows,
            batch_size=batch_size,
        )

        for xb, yb, wb in batches:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            wb = wb.to(DEVICE)

            opt.zero_grad()

            logits, gate = model(
                xb
            )

            ce = F.cross_entropy(
                logits,
                yb,
                reduction="none",
            )

            # Lecture-balancing weights.
            # Normalize weights inside each batch.
            wbn = (
                wb / (
                    wb.mean()
                    + 1e-8
                )
            )

            loss_ce = (
                ce * wbn
            ).mean()

            loss_rank = hard_negative_loss(
                logits,
                yb,
                margin=0.12,
                hard_k=6,
            )

            # Keep the learned distribution reasonably close to
            # the V4 anchor unless evidence justifies moving it.
            v4_teacher = F.softmax(
                xb[:, :, 2],
                dim=1,
            )

            student_log = F.log_softmax(
                logits,
                dim=1,
            )

            kl = F.kl_div(
                student_log,
                v4_teacher,
                reduction="batchmean",
            )

            # Encourage decisive visual gating only where V4 and
            # visual evidence broadly agree. This is a weak regularizer.
            agreement = (
                xb[:, :, 2]
                * xb[:, :, 5]
            )

            agreement_penalty = (
                gate
                * F.relu(
                    -agreement
                )
            ).mean()

            loss = (
                0.75 * loss_ce
                + 0.35 * loss_rank
                + 0.035 * kl
                + 0.01 * agreement_penalty
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                4.0,
            )

            opt.step()

            total_loss += float(
                loss.detach().cpu()
            ) * len(yb)

            total_n += len(yb)

        scheduler.step()

        avg_loss = (
            total_loss
            / max(
                1,
                total_n,
            )
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):
            print(
                f"[V7 seed={seed}] "
                f"epoch={epoch:02d} "
                f"loss={avg_loss:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model


@torch.no_grad()
def predict(model, X):
    model.eval()

    outputs = []

    # X can contain different slide counts across lectures,
    # but within a lecture all rows share the same count.
    for start in range(
        0,
        len(X),
        32,
    ):
        xb = torch.from_numpy(
            X[start:start + 32]
        ).float().to(DEVICE)

        logits, gate = model(
            xb
        )

        outputs.append(
            logits.cpu().numpy()
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def decode_monotonic(
    emissions,
    boundaries=None,
    max_jump=3,
    jump_penalty=0.02,
    boundary_reward=0.12,
):
    emissions = np.asarray(
        emissions,
        dtype=np.float32,
    )

    n, s = emissions.shape

    dp = np.full(
        (n, s),
        -np.inf,
        dtype=np.float32,
    )

    back = np.full(
        (n, s),
        -1,
        dtype=np.int16,
    )

    first = min(
        s,
        max_jump + 1,
    )

    dp[0, :first] = (
        emissions[0, :first]
    )

    for t in range(
        1,
        n,
    ):
        b = (
            float(boundaries[t])
            if boundaries is not None
            and t < len(boundaries)
            else 0.0
        )

        for j in range(s):
            lo = max(
                0,
                j - max_jump,
            )

            best = -np.inf
            best_p = -1

            for p in range(
                lo,
                j + 1,
            ):
                if not np.isfinite(
                    dp[t - 1, p]
                ):
                    continue

                delta = j - p

                transition = (
                    0.0
                    if delta == 0
                    else -jump_penalty * delta
                    + boundary_reward * b
                )

                value = (
                    dp[t - 1, p]
                    + transition
                )

                if value > best:
                    best = value
                    best_p = p

            dp[t, j] = (
                best
                + emissions[t, j]
            )

            back[t, j] = best_p

    path = np.empty(
        n,
        dtype=np.int32,
    )

    path[-1] = int(
        np.argmax(
            dp[-1]
        )
    )

    for t in range(
        n - 1,
        0,
        -1,
    ):
        p = int(
            back[
                t,
                path[t],
            ]
        )
        path[t - 1] = (
            path[t]
            if p < 0
            else p
        )

    return path


def train_mode():
    print("=" * 78)
    print("LECTURESENSE V7 — FULL DEVELOPMENT GATED FUSION")
    print("=" * 78)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Development lectures:",
        len(DEVELOPMENT),
    )

    print(
        "Held-out test lectures:",
        len(TEST),
    )

    print(
        "Test labels used for training:",
        "NO",
    )

    development = []

    for lecture in DEVELOPMENT:
        try:
            development.append(
                build_or_load_lecture(
                    lecture,
                    save=True,
                )
            )
        except Exception as exc:
            print(
                f"[DEVELOPMENT SKIP] "
                f"{lecture}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if len(development) < 10:
        raise RuntimeError(
            f"Only {len(development)} "
            "development lectures available."
        )

    print("\nDevelopment summary:")
    total = 0

    for d in development:
        total += len(
            d["Y"]
        )
        print(
            f"  {d['lecture']:<42}"
            f" {len(d['Y']):>5}"
        )

    print(
        "Total points:",
        total,
    )

    models = []

    for seed in (
        11,
        29,
        47,
        71,
        103,
    ):
        print(
            "\n" + "-" * 72
        )
        print(
            f"TRAINING V7 ENSEMBLE "
            f"MEMBER seed={seed}"
        )
        print(
            "-" * 72
        )

        models.append(
            train_one_seed(
                development,
                seed,
                epochs=36,
                batch_size=24,
            )
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version":
            "LectureSense-V7-FullDevelopment-Gated",
        "feature_dim": 28,
        "models": [
            m.state_dict()
            for m in models
        ],
        "development_lectures": [
            d["lecture"]
            for d in development
        ],
        "test_lectures": TEST,
        "decoder": {
            "max_jump": 3,
            "jump_penalty": 0.02,
            "boundary_reward": 0.12,
        },
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
    }

    torch.save(
        payload,
        MODEL_PATH,
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "version":
                    "LectureSense-V7-FullDevelopment-Gated",
                "development_lectures": [
                    d["lecture"]
                    for d in development
                ],
                "test_used_for_training": False,
                "test_used_for_model_selection": False,
                "decoder": payload["decoder"],
                "ensemble_members": 5,
                "feature_dim": 28,
                "note":
                    "Full-development training was used after "
                    "architecture experimentation. "
                    "90 percent is a target, not a guarantee.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved model:",
        MODEL_PATH,
    )
    print(
        "Saved config:",
        CONFIG_PATH,
    )

    # Development-set descriptive diagnostic only.
    # This is not presented as an unbiased validation score.
    rows = []

    for d in development:
        em = np.mean(
            [
                predict(
                    m,
                    d["X"],
                )
                for m in models
            ],
            axis=0,
        )

        pred = em.argmax(
            axis=1
        )

        rows.append(
            {
                "lecture": d["lecture"],
                "points": int(
                    len(d["Y"])
                ),
                "descriptive_top1":
                    float(
                        (
                            pred
                            == d["Y"]
                        ).mean()
                    ),
            }
        )

    result = pd.DataFrame(rows)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT / "development_results.csv",
        index=False,
    )

    print("\nDevelopment descriptive results:")
    print(
        result.to_string(
            index=False
        )
    )


def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. "
            "Run --mode train first."
        )

    ck = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    models = []

    for state in ck["models"]:
        m = GatedCandidateTransformer(
            feature_dim=int(
                ck["feature_dim"]
            )
        ).to(DEVICE)

        m.load_state_dict(
            state
        )

        m.eval()
        models.append(m)

    decoder = dict(
        ck.get(
            "decoder",
            {
                "max_jump": 3,
                "jump_penalty": 0.02,
                "boundary_reward": 0.12,
            },
        )
    )

    return (
        models,
        decoder,
    )


def evaluate_test_lecture(
    models,
    decoder,
    lecture,
):
    srt, pdf, gt_path, video = (
        asset_paths(lecture)
    )

    print("\n" + "=" * 78)
    print(
        "FINAL HELD-OUT TEST:",
        lecture,
    )
    print("=" * 78)
    print("SRT  :", srt)
    print("PDF  :", pdf)
    print("GT   :", gt_path)
    print("VIDEO:", video)

    if any(
        x is None
        for x in (
            srt,
            pdf,
            gt_path,
            video,
        )
    ):
        raise RuntimeError(
            "Missing exact asset(s)"
        )

    v4_dir = ensure_v4(
        lecture,
        srt,
        pdf,
        video,
    )

    v4 = np.load(
        v4_dir
        / "v4_final_score_matrix.npy"
    ).astype(np.float32)

    boundary_path = (
        v4_dir
        / "v4_boundary_probability.npy"
    )

    boundary = (
        np.load(
            boundary_path
        ).astype(np.float32)
        if boundary_path.exists()
        else np.zeros(
            v4.shape[0],
            dtype=np.float32,
        )
    )

    segments = get_segments(
        srt
    )
    mids = segment_mids(
        segments
    )

    visual_times, visual_scores = (
        get_visual_scores(
            lecture,
            pdf,
            video,
        )
    )

    gt = load_gt(
        gt_path
    )

    test_times = mids.astype(
        np.float32
    )

    X, _ = build_features_from_mids(
        v4=v4,
        boundary=boundary,
        mids=mids,
        visual_times=visual_times,
        visual_scores=visual_scores,
        times=test_times,
        gt_slides=None,
    )

    X = normalize_rows(
        X
    )

    emissions = np.mean(
        [
            predict(
                m,
                X,
            )
            for m in models
        ],
        axis=0,
    )

    path = decode_monotonic(
        emissions,
        boundaries=boundary,
        max_jump=int(
            decoder["max_jump"]
        ),
        jump_penalty=float(
            decoder["jump_penalty"]
        ),
        boundary_reward=float(
            decoder["boundary_reward"]
        ),
    )

    preds = []
    truths = []

    for t, g in zip(
        gt["time"].to_numpy(
            dtype=np.float32
        ),
        gt["gt"].to_numpy(
            dtype=np.int64
        ),
    ):
        si = int(
            np.searchsorted(
                mids,
                float(t),
                side="right",
            ) - 1
        )

        si = max(
            0,
            min(
                si,
                len(path) - 1,
            ),
        )

        preds.append(
            int(path[si]) + 1
        )
        truths.append(
            int(g)
        )

    preds = np.asarray(
        preds,
        dtype=np.int64
    )
    truths = np.asarray(
        truths,
        dtype=np.int64
    )

    accuracy = float(
        (
            preds == truths
        ).mean()
    )

    return {
        "lecture": lecture,
        "points": int(
            len(truths)
        ),
        "v7_accuracy": accuracy,
    }


def test_mode():
    models, decoder = load_models()

    rows = []

    for lecture in TEST:
        try:
            rows.append(
                evaluate_test_lecture(
                    models,
                    decoder,
                    lecture,
                )
            )
        except Exception as exc:
            print(
                f"[TEST SKIP] "
                f"{lecture}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if not rows:
        raise RuntimeError(
            "No test lecture evaluated."
        )

    result = pd.DataFrame(
        rows
    )

    macro = float(
        result[
            "v7_accuracy"
        ].mean()
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT / "test_results.csv",
        index=False,
    )

    summary = {
        "test_macro_accuracy":
            macro,
        "results":
            rows,
        "test_used_for_training":
            False,
        "test_used_for_model_selection":
            False,
        "decoder":
            decoder,
    }

    (OUT / "test_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(
        "V7 FINAL HELD-OUT TEST RESULT"
    )
    print("=" * 78)

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nMacro accuracy:",
        f"{macro:.4f}",
        f"({macro * 100:.2f}%)",
    )

    print(
        "Test labels were not used for "
        "training or model selection."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "test",
        ],
        default="train",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    else:
        test_mode()


if __name__ == "__main__":
    main()
