from __future__ import annotations

"""
LectureSense V4.2 - validation-only decoder / fusion tuner
===========================================================

This is a post-model tuning workflow, not a newly trained alignment network.
It may improve the current alignment through residual/video score fusion and
boundary-aware decoding, but it cannot guarantee a particular test accuracy.

Tunes ONLY the post-model stages (residual scaling, video fusion, score
temperature and the boundary-aware DP decoder). The V4 checkpoint is never
modified or retrained, and no embeddings / video features are recomputed:
everything is read from the cached matrices written by
src/inference_temporal_v4.py:

    v4_base_score_matrix.npy      v4_adjusted_score_matrix.npy
    v4_final_score_matrix.npy     v4_video_score_matrix.npy
    v4_boundary_probability.npy   v4_next_slide_logits.npy

Modes
-----
  python tune_v4_decoder_only.py                  # validate (default): search on
                                                  # physics, psychology, RL, sensory
  python tune_v4_decoder_only.py --mode test      # evaluate the SELECTED config on
                                                  # the untouched test lectures

Guarantees
----------
  * validation mode loads ONLY validation lectures; test labels are never read by this tuner.
  * test mode never writes the config and never searches.
  * the DP used for the sweep is a vectorised re-implementation of
    boundary_dp_v4.boundary_aware_dp; it is cross-checked against the real
    function on every validation lecture at start-up (falls back to the real,
    slow DP automatically if any check disagrees).

Outputs (kept separate from the V4.1 files)
---------------------------------------------
  checkpoints/v4_2_decoder_tuned_config.json
  runs/v4_2_decoder_tuning/validation_sweep.csv
  runs/v4_2_decoder_tuning/test_report.json      (test mode only)

Validation mode reads only validation lecture ground truth. This script does
not use test labels for tuning. Note that the project's existing test lectures
have been evaluated by earlier model versions, so they are not a fresh,
independent benchmark for a new generalization claim.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS / IMPORTS  (src must be on sys.path BEFORE project imports)
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import inference_temporal_v4 as infer_v4          # load_v4_checkpoint lives here
    from boundary_dp_v4 import boundary_aware_dp
except Exception as exc:                              # pragma: no cover
    sys.exit(
        "Could not import the V4 project modules from src/.\n"
        f"  {type(exc).__name__}: {exc}\n"
        "Run this script from the project root (the folder that contains src/)."
    )

try:
    from video_visual_alignment import fuse_model_and_video_scores
except Exception:                                     # pragma: no cover
    fuse_model_and_video_scores = None

DATA_ROOT = ROOT / "data" / "MaViLS" / "data"
DEFAULT_V4_ROOT = ROOT / "runs" / "v5_gate_train_v4"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
OUTPUT_ROOT = ROOT / "runs" / "v4_2_decoder_tuning"
CONFIG_PATH = ROOT / "checkpoints" / "v4_2_decoder_tuned_config.json"
CSV_PATH = OUTPUT_ROOT / "validation_sweep.csv"
TEST_REPORT_PATH = OUTPUT_ROOT / "test_report.json"


def _spec(srt, gt, folder, v4_root):
    return {
        "srt": DATA_ROOT / "audioscripts" / srt,
        "gt": DATA_ROOT / "ground_truth_files" / gt,
        "v4_dir": v4_root / folder,
    }


def lecture_specs(split: str, v4_root: Path) -> dict:
    if split == "validation":
        return {
            "physics": _spec("physics_intro_02.srt", "ground_truth_physics.xlsx", "physics", v4_root),
            "psychology": _spec("psychology_MIT.srt", "ground_truth_psychology.xlsx", "psychology", v4_root),
            "reinforcement_learning": _spec(
                "reinforcement_learning_silver.srt", "ground_truth_reinforcement_learning.xlsx",
                "reinforcement_learning", v4_root),
            "sensory_systems": _spec(
                "visual_system_MIT.srt", "ground_truth_sensory_systems.xlsx", "sensory_systems", v4_root),
        }
    if split == "test":
        return {
            "short_range": _spec("short_range_mit.srt", "ground_truth_short_range.xlsx", "short_range", v4_root),
            "solar_resource": _spec("solar_resource.srt", "ground_truth_solar_resource.xlsx", "solar_resource", v4_root),
            "team_dynamics_game_design_MIT": _spec(
                "team_dynamics_game_design_MIT.srt", "ground_truth_team_dynamics_game_design_MIT.xlsx",
                "team_dynamics_game_design_MIT", v4_root),
            "theory_of_computation": _spec(
    "theory_of_computation_MIT.srt",
    "ground_truth_theory_of_computation.xlsx",
    "theory_of_computation", v4_root
),
        }
    raise ValueError(split)


# ============================================================
# CONFIG SCHEMA
# ============================================================

FUSION_KEYS = ("residual_scale", "video_weight", "score_temperature")
DECODER_KEYS = (
    "jump_penalty", "max_forward_jump", "max_duration_seconds", "duration_penalty",
    "max_run_seconds", "boundary_reward", "boundary_stay_penalty",
    "transition_logits_weight", "premature_transition_penalty", "boundary_threshold",
)
ALL_KEYS = FUSION_KEYS + DECODER_KEYS

# Same defaults infer_v4.load_v4_checkpoint() falls back to.
DEFAULT_DECODER = {
    "jump_penalty": 0.05, "max_forward_jump": 1, "max_duration_seconds": 120.0,
    "duration_penalty": 0.001, "max_run_seconds": 1200.0, "boundary_reward": 1.0,
    "boundary_stay_penalty": 0.50, "transition_logits_weight": 0.35,
    "premature_transition_penalty": 0.50, "boundary_threshold": 0.20,
}

# What app.py actually passes to inference_temporal_v4.py (overrides the
# checkpoint's decoder_config). These define the CURRENT production baseline.
APP_DECODER_OVERRIDES = {"jump_penalty": 0.05, "max_duration_seconds": 120.0, "duration_penalty": 0.003}
APP_VIDEO_WEIGHT = 0.40

# Discrete search grid. Baseline values are injected automatically.
SEARCH_SPACE = {
    "residual_scale":               [0.50, 0.75, 1.00, 1.25, 1.50],
    "video_weight":                 [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
    "score_temperature":            [0.50, 0.75, 1.00, 1.50, 2.00, 3.00],
    "boundary_reward":              [0.00, 0.25, 0.50, 1.00, 1.50, 2.00],
    "boundary_stay_penalty":        [0.00, 0.25, 0.50, 1.00, 2.00],
    "transition_logits_weight":     [0.00, 0.10, 0.20, 0.35, 0.50, 0.75],
    "premature_transition_penalty": [0.00, 0.25, 0.50, 1.00, 1.50],
    "jump_penalty":                 [0.00, 0.02, 0.05, 0.10, 0.20, 0.40],
    "max_forward_jump":             [1, 2, 3],
    "duration_penalty":             [0.0005, 0.001, 0.003, 0.01, 0.03],
    "max_duration_seconds":         [60.0, 90.0, 120.0, 180.0, 240.0, 360.0],
    "max_run_seconds":              [600.0, 1200.0, 1800.0],
    "boundary_threshold":           [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
}


def pct(x) -> str:
    return f"{100.0 * float(x):6.2f}%"


# ============================================================
# GROUND TRUTH / TRANSCRIPT
# ============================================================

def _norm_col(value) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def _find_col(columns, candidates, contains=None):
    for c in candidates:
        if c in columns:
            return c
    if contains:
        for c in columns:
            if contains in c:
                return c
    return None


def _to_seconds(value) -> float:
    """Seconds as float from numbers, 'HH:MM:SS(.mmm)', 'MM:SS', time, timedelta."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, timedelta):
        return float(value.total_seconds())
    if isinstance(value, dtime):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, datetime):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    text = str(value).strip().replace(",", ".")
    if not text:
        return float("nan")
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return float("nan")
    total = 0.0
    for p in parts:
        total = total * 60.0 + p
    return total


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    df = pd.read_excel(path)
    df.columns = [_norm_col(c) for c in df.columns]

    time_col = _find_col(df.columns, ["time", "timestamp", "start", "key"], contains="time")
    slide_col = _find_col(
        df.columns, ["slidenumber", "slide_number", "slide", "slideid", "slide_id"], contains="slide"
    )
    if time_col is None or slide_col is None:
        raise RuntimeError(f"Could not identify GT columns in {path}. Columns={list(df.columns)}")

    out = df[[time_col, slide_col]].copy()
    out.columns = ["time", "gt"]
    out["time"] = out["time"].map(_to_seconds)
    out["gt"] = pd.to_numeric(out["gt"], errors="coerce")
    out = out.dropna()
    out = out[out["gt"] > 0]                     # project convention: slide ids <= 0 invalid
    out["gt"] = out["gt"].astype(int)
    out = out.sort_values("time")
    if out.empty:
        raise RuntimeError(f"No valid GT rows found in {path}")
    return out.reset_index(drop=True)


def load_transcript(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"SRT not found: {path}")
    segments = infer_v4.base.read_mavils_srt(str(path))
    if not segments:
        raise RuntimeError(f"No transcript segments found in {path}")
    starts = np.asarray([float(x.get("start", x.get("start_time", 0.0))) for x in segments], dtype=np.float32)
    ends = np.asarray([float(x.get("end", x.get("end_time", 0.0))) for x in segments], dtype=np.float32)
    return starts, ends


def map_gt_to_transcript(gt_df, starts, ends):
    n = len(starts)
    idx_list, labels, times = [], [], []
    for t, g in zip(gt_df["time"].to_numpy(dtype=float), gt_df["gt"].to_numpy(dtype=int)):
        idx = int(np.searchsorted(starts, t, side="right") - 1)
        idx = min(max(idx, 0), n - 1)
        if t > float(ends[idx]) and idx + 1 < n:
            if abs(float(starts[idx + 1]) - t) < abs(t - float(ends[idx])):
                idx += 1
        idx_list.append(idx)
        labels.append(g - 1)
        times.append(t)
    return (np.asarray(idx_list, dtype=np.int64), np.asarray(labels, dtype=np.int64),
            np.asarray(times, dtype=np.float32))


# ============================================================
# CACHED MATRIX LOADING (safe: every matrix except one is optional)
# ============================================================

def _load_npy(folder: Path, names):
    for name in names:
        p = folder / name
        if p.exists():
            try:
                return np.load(p).astype(np.float32)
            except Exception as exc:
                print(f"  Warning: could not read {p.name}: {exc}")
    return None


def load_lecture(name: str, spec: dict) -> dict:
    folder = spec["v4_dir"]
    if not folder.exists():
        raise FileNotFoundError(
            f"Cached V4 outputs not found for '{name}': {folder}\n"
            "Pass --v4-root if they live somewhere else."
        )
    print(f"\n{'=' * 78}\nLOADING {name}\n{'=' * 78}\n  V4 dir: {folder}")

    starts, ends = load_transcript(spec["srt"])
    gt_idx, gt_lab, gt_time = map_gt_to_transcript(load_ground_truth(spec["gt"]), starts, ends)

    base = _load_npy(folder, ["v4_base_score_matrix.npy", "base_score_matrix.npy"])
    adjusted = _load_npy(folder, ["v4_adjusted_score_matrix.npy", "adjusted_score_matrix.npy"])
    final = _load_npy(folder, ["v4_final_score_matrix.npy", "final_score_matrix.npy", "fused_score_matrix.npy"])
    video = _load_npy(folder, ["v4_video_score_matrix.npy", "video_score_matrix.npy"])
    boundary = _load_npy(folder, ["v4_boundary_probability.npy", "boundary_probability.npy"])
    next_logits = _load_npy(folder, ["v4_next_slide_logits.npy", "next_slide_logits.npy"])

    # ---- select a pre-video score matrix safely ----
    adjusted_is_real = adjusted is not None
    using_base_fallback = False
    using_final_fallback = False
    if adjusted is None:
        if base is not None:
            adjusted = base.copy()
            using_base_fallback = True
            print("  Note: adjusted matrix missing -> using base scores; residual scaling is disabled.")
        elif final is not None:
            adjusted = final.copy()
            using_final_fallback = True
            print("  Note: adjusted/base matrices missing -> using cached final scores; re-fusion is disabled.")
        else:
            found = [p.name for p in folder.glob("*.npy")]
            raise RuntimeError(f"No usable score matrix in {folder}. Found: {found}")

    # Score matrices must be rank-2 and finite before DP consumes them.
    if adjusted.ndim != 2 or adjusted.size == 0:
        raise RuntimeError(f"Adjusted score matrix must be non-empty 2-D data for {name}: {adjusted.shape}")
    adjusted = np.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n, s = adjusted.shape
    if len(starts) != n:
        raise RuntimeError(f"Transcript segments ({len(starts)}) != matrix rows ({n}) for {name}")

    can_residual = adjusted_is_real and base is not None and base.shape == adjusted.shape
    if base is not None and base.shape != adjusted.shape:
        print("  Warning: base/adjusted shapes differ -> residual scaling disabled.")
    elif adjusted_is_real and base is None:
        print("  Note: base score matrix missing -> residual scaling disabled; using adjusted V4 scores.")
    if not can_residual:
        # Missing base artifacts are expected in some cached runs. Do not crash;
        # hold the available V4 matrix fixed and tune only supported parameters.
        base = None

    # Re-fuse only if the matrix being fused is known to be pre-video evidence.
    # A base-only fallback is suitable when no cached final matrix exists. If a
    # final matrix exists, its original pre-video adjusted matrix is required to
    # compare alternate visual weights fairly.
    can_rebuild_pre_video = adjusted_is_real or (using_base_fallback and final is None)
    can_video = can_rebuild_pre_video and not using_final_fallback and video is not None and video.shape == adjusted.shape
    if video is not None and video.shape != adjusted.shape:
        print("  Warning: video/V4 shapes differ -> video fusion disabled.")
    elif not can_video:
        print("  Note: no compatible pre-video/video matrix pair -> video-weight search disabled.")
    if not can_video:
        video = None

    if final is not None and final.shape != adjusted.shape:
        print("  Warning: final matrix shape differs -> ignoring cached final matrix.")
        final = None
    elif final is not None:
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if boundary is None or len(boundary.reshape(-1)) != n:
        print("  WARNING: boundary probabilities missing/mismatched -> zeros "
              "(boundary parameters will have little meaning for this lecture).")
        boundary = np.zeros(n, dtype=np.float32)
    boundary = np.nan_to_num(boundary.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if next_logits is None or next_logits.shape != (n, s):
        print("  WARNING: next-slide logits missing/mismatched -> zeros.")
        next_logits = np.zeros((n, s), dtype=np.float32)
    next_logits = np.nan_to_num(next_logits, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    keep = (gt_lab >= 0) & (gt_lab < s) & (gt_idx < n)
    gt_idx, gt_lab, gt_time = gt_idx[keep], gt_lab[keep], gt_time[keep]
    if len(gt_lab) == 0:
        raise RuntimeError(f"No usable ground-truth rows for {name}")

    # ---- invariants reused by every DP call ----
    nl = next_logits.astype(np.float64)
    nl = nl - np.max(nl, axis=1, keepdims=True)
    nl = nl - np.log(np.maximum(np.sum(np.exp(nl), axis=1, keepdims=True), 1e-12))

    print(f"  segments={n}  slides={s}  GT rows={len(gt_lab)}  "
          f"base={'Y' if base is not None else 'N'}  video={'Y' if video is not None else 'N'}  "
          f"final={'Y' if final is not None else 'N'}")

    return {
        "name": name, "starts": starts, "ends": ends,
        "gt_indices": gt_idx, "gt_labels": gt_lab, "gt_times": gt_time,
        "base": base, "adjusted": adjusted, "final": final, "video": video,
        "boundary": boundary, "next_logits": next_logits, "next_logprob": nl,
        "slides": s, "can_residual": can_residual, "can_video": can_video,
        "_fuse_cache": {},
    }


def load_dataset(specs: dict) -> dict:
    return {name: load_lecture(name, spec) for name, spec in specs.items()}


# ============================================================
# SCORE CONSTRUCTION (residual scaling -> real video fusion -> temperature)
# ============================================================

FUSE_FAITHFUL = None      # set by check_fusion_fidelity()


def _fuse(v4_scores, video_scores, weight):
    if fuse_model_and_video_scores is not None:
        return np.asarray(
            fuse_model_and_video_scores(v4_scores, video_scores, video_weight=float(weight)),
            dtype=np.float32,
        )
    w = float(weight)                                  # fallback only if the module is unavailable
    return ((1.0 - w) * v4_scores + w * video_scores).astype(np.float32)


def fused_scores(item, residual_scale, video_weight, baseline_fusion=None):
    """V4 scores with residual scaling and video fusion applied (no temperature)."""
    key = (round(float(residual_scale), 6), round(float(video_weight), 6))

    # Exact production configuration -> use the cached final matrix verbatim.
    if baseline_fusion is not None and item["final"] is not None and key == baseline_fusion:
        return item["final"]

    cache = item["_fuse_cache"]
    if key in cache:
        return cache[key]

    adjusted = item["adjusted"]
    if item["can_residual"]:
        v4 = (item["base"] + float(residual_scale) * (adjusted - item["base"])).astype(np.float32)
    else:
        v4 = adjusted

    out = _fuse(v4, item["video"], video_weight) if item["can_video"] else v4

    if len(cache) > 48:
        cache.clear()
    cache[key] = out
    return out


def build_scores(item, cfg, baseline_fusion=None):
    scores = fused_scores(item, cfg["residual_scale"], cfg["video_weight"], baseline_fusion)
    temperature = max(1e-3, float(cfg["score_temperature"]))
    if abs(temperature - 1.0) < 1e-12:
        return scores
    return (scores * np.float32(1.0 / temperature)).astype(np.float32)


def check_fusion_fidelity(dataset, baseline_weight):
    """Does the (real) fusion function reproduce the cached final matrix?"""
    global FUSE_FAITHFUL
    diffs = []
    for item in dataset.values():
        if item["final"] is None or not item["can_video"]:
            continue
        rebuilt = fused_scores(item, 1.0, baseline_weight, baseline_fusion=None)
        scale = max(1e-6, float(np.std(item["final"])))
        diffs.append(float(np.max(np.abs(rebuilt - item["final"]))) / scale)
    if not diffs:
        FUSE_FAITHFUL = None
        print("Fusion check: no lecture has both video + final matrices; skipped.")
        return
    worst = max(diffs)
    FUSE_FAITHFUL = worst < 1e-3
    if FUSE_FAITHFUL:
        print(f"Fusion check: fuse_model_and_video_scores(w={baseline_weight}) reproduces the cached "
              f"final matrices (max rel. diff {worst:.2e}).")
    else:
        print(f"WARNING: re-fusing at w={baseline_weight} differs from the cached final matrices "
              f"(max rel. diff {worst:.3g}). The baseline is still measured on the cached matrices, "
              "but video-weight candidates are only approximate (were the caches created with "
              "different video settings?).")


# ============================================================
# FAST, EXACT RE-IMPLEMENTATION OF boundary_dp_v4.boundary_aware_dp
# (vectorised over run-start k and slide; cross-checked at start-up)
# ============================================================

USE_FAST_DP = True


def fast_boundary_dp(
    score_matrix, boundary_prob, next_logprob, segment_starts, segment_ends,
    jump_penalty, max_forward_jump, max_duration_seconds, duration_penalty, max_run_seconds,
    boundary_reward, boundary_stay_penalty, transition_logits_weight,
    premature_transition_penalty, boundary_threshold,
):
    """Same recurrence, tie-breaking and constraints as boundary_aware_dp.
    `next_logprob` is the already log-softmaxed (float64) next-slide logits.
    Returns a list path, or None when no valid alignment exists."""
    scores = np.asarray(score_matrix, dtype=np.float32)
    n, s = scores.shape
    if n == 0 or s == 0:
        return None

    starts = np.asarray(segment_starts, dtype=np.float32)
    ends = np.asarray(segment_ends, dtype=np.float32)
    if np.any(ends < starts) or np.any(np.diff(starts) < 0):
        return None

    jump = max(0.0, float(jump_penalty))
    max_dur = max(0.1, float(max_duration_seconds))
    dpen = max(0.0, float(duration_penalty))
    max_run = max(max_dur, float(max_run_seconds))
    J = max(1, int(max_forward_jump))
    bsp = float(boundary_stay_penalty)
    m = max(1e-6, max_dur)

    b = np.clip(np.asarray(boundary_prob, dtype=np.float32).reshape(-1), 1e-4, 1.0 - 1e-4)

    SP = np.zeros((n + 1, s), dtype=np.float64)
    SP[1:] = np.cumsum(scores.astype(np.float64), axis=0)

    BP = np.zeros(n + 1, dtype=np.float64)
    BP[1:] = np.cumsum(np.maximum(b - float(boundary_threshold), 0.0), dtype=np.float64)

    p64 = b.astype(np.float64)
    logit = np.log(p64 / (1.0 - p64))
    T = (
        float(boundary_reward) * np.tanh(0.5 * logit)[:, None]
        + float(transition_logits_weight) * next_logprob
        - float(premature_transition_penalty) * (1.0 - p64)[:, None]
    )

    NEG = -np.inf
    dp = np.full((n, s), NEG)
    back_start = np.full((n, s), -1, dtype=np.int32)
    back_prev = np.full((n, s), -1, dtype=np.int32)
    PB = np.full((n, s), NEG)                       # best previous dp over allowed prior slides
    PA = np.full((n, s), -1, dtype=np.int32)        # ...and which prior slide
    slide_idx = np.arange(s)

    for i in range(n):
        dur = (ends[i] - starts[: i + 1]).astype(np.float64)
        ok = dur <= max_run
        if not ok[i]:
            continue
        kmin = int(np.argmax(ok))
        K = i - kmin + 1
        d = dur[kmin:]

        excess = np.maximum(d - max_dur, 0.0)
        dcost = np.where(excess > 0.0, dpen * (excess / m) ** 2 * m, 0.0)
        stay = bsp * (BP[i + 1] - BP[kmin + 1: i + 2])
        run = SP[i + 1][None, :] - SP[kmin: i + 1]

        cand = np.full((K, s), NEG)
        r0 = 0
        if kmin == 0:
            cand[0, 0] = run[0, 0] - dcost[0] - stay[0]
            r0 = 1
        if r0 < K:
            ks = np.arange(kmin + r0, i + 1)
            val = PB[ks - 1] + run[r0:]
            val -= dcost[r0:, None]
            val -= stay[r0:, None]
            val -= jump
            val += T[ks]
            cand[r0:] = val

        ridx = np.argmax(cand[::-1], axis=0)         # ties -> largest k (matches the k-descending loop)
        kidx = K - 1 - ridx
        best = cand[kidx, slide_idx]
        finite = np.isfinite(best)
        kk = kmin + kidx
        dp[i] = best
        back_start[i] = np.where(finite, kk, -1)
        prev = np.where(kk > 0, PA[np.maximum(kk - 1, 0), slide_idx], -1)
        back_prev[i] = np.where(finite, prev, -1)

        for dd in range(J, 0, -1):                   # ascending prior-slide order, strict '>'
            if dd >= s:
                continue
            shifted = np.full(s, NEG)
            shifted[dd:] = dp[i, : s - dd]
            better = shifted > PB[i]
            PB[i][better] = shifted[better]
            PA[i][better] = (slide_idx - dd)[better]

    final_slide = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final_slide]):
        return None

    path = np.zeros(n, dtype=np.int32)
    end, slide = n - 1, final_slide
    while end >= 0:
        k = int(back_start[end, slide])
        if k < 0:
            return None
        path[k: end + 1] = slide
        if k == 0:
            break
        prev_slide = int(back_prev[end, slide])
        if prev_slide < 0 or prev_slide >= slide:
            return None
        end, slide = k - 1, prev_slide

    if path[0] != 0 or np.any(np.diff(path) < 0):
        return None
    return path.tolist()


def _decoder_kwargs(cfg):
    return dict(
        jump_penalty=float(cfg["jump_penalty"]),
        max_forward_jump=int(cfg["max_forward_jump"]),
        max_duration_seconds=float(cfg["max_duration_seconds"]),
        duration_penalty=float(cfg["duration_penalty"]),
        max_run_seconds=float(cfg["max_run_seconds"]),
        boundary_reward=float(cfg["boundary_reward"]),
        boundary_stay_penalty=float(cfg["boundary_stay_penalty"]),
        transition_logits_weight=float(cfg["transition_logits_weight"]),
        premature_transition_penalty=float(cfg["premature_transition_penalty"]),
        boundary_threshold=float(cfg["boundary_threshold"]),
    )


def real_dp(item, scores, cfg, rows=None):
    sl = slice(None) if rows is None else slice(0, rows)
    try:
        return boundary_aware_dp(
            scores[sl], item["boundary"][sl], item["next_logits"][sl],
            item["starts"][sl], item["ends"][sl], **_decoder_kwargs(cfg),
        )
    except (RuntimeError, ValueError):
        return None


def decode(item, scores, cfg):
    if USE_FAST_DP:
        path = fast_boundary_dp(
            scores, item["boundary"], item["next_logprob"], item["starts"], item["ends"],
            **_decoder_kwargs(cfg),
        )
    else:
        path = real_dp(item, scores, cfg)
    return None if path is None else np.asarray(path, dtype=np.int64)


def verify_fast_dp(dataset, baseline_cfg, rng, rows=140):
    """Compare the fast DP with the real boundary_aware_dp on slices of real data."""
    global USE_FAST_DP
    checks = bad = 0
    for item in dataset.values():
        m = min(rows, len(item["starts"]))
        if m < 20:
            continue
        scores = build_scores(item, baseline_cfg)
        sub = {
            "boundary": item["boundary"][:m], "next_logits": item["next_logits"][:m],
            "starts": item["starts"][:m], "ends": item["ends"][:m],
        }
        nlp = item["next_logprob"][:m]
        trials = [dict(baseline_cfg)]
        for _ in range(3):
            trials.append({k: rng.choice(SEARCH_SPACE[k]) for k in DECODER_KEYS} | {
                k: baseline_cfg[k] for k in FUSION_KEYS})
        for cfg in trials:
            ref = real_dp(sub, scores[:m], cfg)
            fast = fast_boundary_dp(scores[:m], sub["boundary"], nlp, sub["starts"], sub["ends"],
                                    **_decoder_kwargs(cfg))
            checks += 1
            if (ref is None) != (fast is None) or (ref is not None and list(ref) != list(fast)):
                bad += 1
    if checks == 0:
        print("DP verification: lectures too short, skipped (using fast DP).")
    elif bad == 0:
        print(f"DP verification: fast DP identical to boundary_aware_dp on {checks}/{checks} checks.")
    else:
        USE_FAST_DP = False
        print(f"WARNING: fast DP disagreed with boundary_aware_dp on {bad}/{checks} checks. "
              "Falling back to the real (slow) DP; the sweep will take much longer. "
              "Use --quick or a smaller --max-minutes budget.")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_lecture(item, cfg, baseline_fusion):
    scores = build_scores(item, cfg, baseline_fusion)
    path = decode(item, scores, cfg)
    total = len(item["gt_labels"])
    if path is None:
        return 0, total
    return int(np.sum(path[item["gt_indices"]] == item["gt_labels"])), total


def evaluate_config(dataset, cfg, baseline_fusion, workers=1):
    names = list(dataset)
    if workers > 1 and len(names) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(names))) as ex:
            outs = list(ex.map(lambda nm: evaluate_lecture(dataset[nm], cfg, baseline_fusion), names))
    else:
        outs = [evaluate_lecture(dataset[nm], cfg, baseline_fusion) for nm in names]

    accs = {nm: (c / t if t else 0.0) for nm, (c, t) in zip(names, outs)}
    correct = sum(c for c, _ in outs)
    total = sum(t for _, t in outs)
    return {
        "macro": float(np.mean(list(accs.values()))) if accs else 0.0,
        "weighted": correct / max(1, total),
        "accs": accs,
        "counts": {nm: (c, t) for nm, (c, t) in zip(names, outs)},
        "correct": correct,
        "total": total,
    }


def direct_accuracy(dataset, matrix_getter):
    """Plain per-segment argmax (no DP) - diagnostic only."""
    c = t = 0
    accs = {}
    for nm, item in dataset.items():
        pred = np.argmax(matrix_getter(item), axis=1)
        ok = int(np.sum(pred[item["gt_indices"]] == item["gt_labels"]))
        accs[nm] = ok / max(1, len(item["gt_labels"]))
        c += ok
        t += len(item["gt_labels"])
    return float(np.mean(list(accs.values()))), c / max(1, t)


# ============================================================
# BASELINE (what app.py / inference_temporal_v4.py produce today)
# ============================================================

def load_checkpoint_decoder(checkpoint: Path) -> dict:
    """Read the checkpoint's decoder_config via infer_v4.load_v4_checkpoint()."""
    try:
        _model, _base, _max_slides, _temperature, decoder_cfg = infer_v4.load_v4_checkpoint(checkpoint)
        cfg = dict(DEFAULT_DECODER)
        cfg.update({k: v for k, v in dict(decoder_cfg).items() if k in DECODER_KEYS})
        del _model, _base
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return cfg
    except Exception as exc:
        print(f"WARNING: could not load decoder_config from the checkpoint "
              f"({type(exc).__name__}: {exc}). Using inference defaults.")
        return dict(DEFAULT_DECODER)


def make_baseline_config(checkpoint_decoder: dict) -> dict:
    dec = dict(checkpoint_decoder)
    dec.update(APP_DECODER_OVERRIDES)               # exactly what app.py passes on the CLI
    cfg = {"residual_scale": 1.0, "video_weight": APP_VIDEO_WEIGHT, "score_temperature": 1.0}
    cfg.update({k: dec[k] for k in DECODER_KEYS})
    return cfg


def baseline_fusion_key(cfg):
    return (round(float(cfg["residual_scale"]), 6), round(float(cfg["video_weight"]), 6))


# ============================================================
# SEARCH
# ============================================================

class TimeBudget(Exception):
    pass


class Searcher:
    def __init__(self, dataset, baseline_cfg, space, objective, min_gain, workers, deadline, rng):
        self.dataset, self.base_cfg, self.space = dataset, baseline_cfg, space
        self.objective, self.min_gain, self.workers = objective, min_gain, workers
        self.deadline, self.rng = deadline, rng
        self.fusion_key = baseline_fusion_key(baseline_cfg)
        self.memo, self.rows = {}, []
        self.evals = 0

    def key(self, cfg):
        return tuple((k, round(float(cfg[k]), 8)) for k in ALL_KEYS)

    def obj(self, res):
        return res[self.objective]

    def score(self, cfg, stage):
        k = self.key(cfg)
        if k in self.memo:
            return self.memo[k]
        if time.time() > self.deadline:
            raise TimeBudget()
        t0 = time.time()
        res = evaluate_config(self.dataset, cfg, self.fusion_key, self.workers)
        res["seconds"] = time.time() - t0
        self.memo[k] = res
        self.evals += 1
        row = {"stage": stage, "macro": res["macro"], "weighted": res["weighted"],
               "seconds": res["seconds"]}
        row.update({f"acc_{nm}": a for nm, a in res["accs"].items()})
        row.update({k2: cfg[k2] for k2 in ALL_KEYS})
        self.rows.append(row)
        return res

    def n_changed(self, cfg):
        return sum(1 for k in ALL_KEYS if abs(float(cfg[k]) - float(self.base_cfg[k])) > 1e-12)

    def coordinate_descent(self, start, stage, max_rounds):
        cur = dict(start)
        cur_obj = self.obj(self.score(cur, stage))
        params = [p for p in ALL_KEYS if len(self.space[p]) > 1]
        for rnd in range(1, max_rounds + 1):
            improved = False
            self.rng.shuffle(params)
            for p in params:
                best_v, best_o = cur[p], cur_obj
                for v in self.space[p]:
                    if v == cur[p]:
                        continue
                    o = self.obj(self.score({**cur, p: v}, stage))
                    if o > best_o + 1e-12:
                        best_v, best_o = v, o
                if best_v != cur[p] and best_o >= cur_obj + self.min_gain:
                    cur[p], cur_obj, improved = best_v, best_o, True
            print(f"  [{stage}] round {rnd}: objective {pct(cur_obj)}  "
                  f"(evals so far: {self.evals})")
            if not improved:
                break
        return cur, cur_obj

    def random_cfg(self):
        cfg = {k: self.rng.choice(self.space[k]) for k in ALL_KEYS}
        # keep samples near the baseline so ~13 simultaneous changes don't dominate
        for k in ALL_KEYS:
            if self.rng.random() < 0.55:
                cfg[k] = self.base_cfg[k]
        return cfg

    def select_final(self):
        """Best objective; among configs within min_gain of it prefer the one
        closest to the baseline (simpler config = less validation overfitting)."""
        best = max(self.obj(r) for r in self.memo.values())
        pool = []
        for k, r in self.memo.items():
            if self.obj(r) >= best - self.min_gain:
                cfg = {kk: vv for kk, vv in k}
                pool.append((self.n_changed(cfg), -self.obj(r), cfg, r))
        pool.sort(key=lambda x: (x[0], x[1]))
        _, _, cfg, res = pool[0]
        return cfg, res


def build_space(dataset, baseline_cfg):
    space = {k: sorted(set(v) | {baseline_cfg[k]}) for k, v in SEARCH_SPACE.items()}
    if not any(it["can_residual"] for it in dataset.values()):
        space["residual_scale"] = [baseline_cfg["residual_scale"]]
        print("Note: base matrix missing/incompatible -> residual_scale is locked to baseline.")
    if not any(it["can_video"] for it in dataset.values()):
        space["video_weight"] = [baseline_cfg["video_weight"]]
        print("Note: compatible pre-video + video matrices unavailable -> video_weight is locked to baseline.")
    if FUSE_FAITHFUL is False:
        # The cached final matrix and reconstructed fusion do not match closely
        # enough. Searching these knobs would compare inconsistent score spaces.
        space["residual_scale"] = [baseline_cfg["residual_scale"]]
        space["video_weight"] = [baseline_cfg["video_weight"]]
        print("WARNING: fusion fidelity check failed; residual_scale/video_weight are locked. "
              "Continuing with score temperature and decoder tuning only.")
    return space


def save_sweep_csv(rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows).sort_values("macro", ascending=False)
        df.to_csv(CSV_PATH, index=False)


# ============================================================
# REPORTING
# ============================================================

def print_lecture_table(title, dataset, results: dict):
    """results: {column_label: eval result}"""
    labels = list(results)
    print(f"\n{title}")
    print(f"{'lecture':<34}" + "".join(f"{lb:>22}" for lb in labels))
    for nm in dataset:
        line = f"{nm:<34}"
        for lb in labels:
            c, t = results[lb]["counts"][nm]
            line += f"{f'{c}/{t}  {pct(c / max(1, t))}':>22}"
        print(line)
    line = f"{'MACRO (mean of lectures)':<34}" + "".join(f"{pct(results[lb]['macro']):>22}" for lb in labels)
    print(line)
    line = f"{'WEIGHTED (all GT points)':<34}" + "".join(
        f"{results[lb]['correct']}/{results[lb]['total']}  {pct(results[lb]['weighted'])}".rjust(22)
        for lb in labels
    )
    print(line)


def print_config_diff(base_cfg, cfg):
    print(f"\n{'parameter':<32}{'baseline':>12}{'selected':>12}")
    for k in ALL_KEYS:
        mark = "  <- changed" if abs(float(base_cfg[k]) - float(cfg[k])) > 1e-12 else ""
        print(f"{k:<32}{base_cfg[k]:>12g}{cfg[k]:>12g}{mark}")


# ============================================================
# VALIDATE MODE
# ============================================================

def validate_mode(args):
    t_start = time.time()
    rng = random.Random(args.seed)
    v4_root = Path(args.v4_root)

    print(f"\n{'=' * 78}\nLectureSense V4 decoder tuning  (VALIDATION lectures only)\n{'=' * 78}")
    print(f"Cached V4 outputs : {v4_root}")
    print(f"Checkpoint        : {args.checkpoint}  (read-only; never modified)")

    checkpoint_decoder = load_checkpoint_decoder(Path(args.checkpoint))
    baseline_cfg = make_baseline_config(checkpoint_decoder)

    dataset = load_dataset(lecture_specs("validation", v4_root))
    fusion_key = baseline_fusion_key(baseline_cfg)

    print()
    check_fusion_fidelity(dataset, baseline_cfg["video_weight"])
    verify_fast_dp(dataset, baseline_cfg, rng)

    # ---------------- baselines ----------------
    base_res = evaluate_config(dataset, baseline_cfg, fusion_key, args.workers)
    ckpt_cfg = make_baseline_config(checkpoint_decoder)
    ckpt_cfg.update({k: checkpoint_decoder[k] for k in DECODER_KEYS})
    ckpt_res = evaluate_config(dataset, ckpt_cfg, fusion_key, args.workers)

    print(f"\n{'=' * 78}\nCURRENT V4 BASELINE ON VALIDATION\n{'=' * 78}")
    print("Baseline = cached final matrix (video w=%.2f) + boundary DP with the decoder flags "
          "app.py passes to inference." % baseline_cfg["video_weight"])
    print_lecture_table("", dataset, {"app.py decoder": base_res, "checkpoint decoder": ckpt_res})
    dm, dw = direct_accuracy(
        dataset, lambda it: it["final"] if it["final"] is not None else it["adjusted"])
    print(f"\n(diagnostic) argmax without DP on the same matrices: macro {pct(dm)}, weighted {pct(dw)}")

    # ---------------- search ----------------
    space = build_space(dataset, baseline_cfg)
    deadline = time.time() + 60.0 * args.max_minutes
    searcher = Searcher(dataset, baseline_cfg, space, args.objective, args.min_gain,
                        args.workers, deadline, rng)
    searcher.memo[searcher.key(baseline_cfg)] = {**base_res, "seconds": 0.0}
    searcher.rows.append({"stage": "baseline", "macro": base_res["macro"], "weighted": base_res["weighted"],
                          "seconds": 0.0, **{f"acc_{n}": a for n, a in base_res["accs"].items()},
                          **{k: baseline_cfg[k] for k in ALL_KEYS}})

    try:
        print(f"\n{'=' * 78}\nSTAGE 1: coordinate descent from the baseline\n{'=' * 78}")
        s1_cfg, _ = searcher.coordinate_descent(baseline_cfg, "stage1_coord", args.max_rounds)
        save_sweep_csv(searcher.rows)

        print(f"\n{'=' * 78}\nSTAGE 2: random exploration ({args.random_samples} samples) + refinement\n{'=' * 78}")
        explored = []
        for i in range(args.random_samples):
            cfg = searcher.random_cfg()
            res = searcher.score(cfg, "stage2_random")
            explored.append((searcher.obj(res), cfg))
            if (i + 1) % 25 == 0:
                best_so_far = max(searcher.obj(r) for r in searcher.memo.values())
                print(f"  sampled {i + 1}/{args.random_samples}   best objective so far {pct(best_so_far)}")
        save_sweep_csv(searcher.rows)

        explored.sort(key=lambda x: -x[0])
        seeds, seen = [], set()
        for _, cfg in explored:
            k = searcher.key(cfg)
            if k not in seen:
                seen.add(k)
                seeds.append(cfg)
            if len(seeds) >= args.refine_top:
                break
        for j, cfg in enumerate(seeds, 1):
            print(f"\n  refining random candidate {j}/{len(seeds)}")
            searcher.coordinate_descent(cfg, f"stage3_refine{j}", max(1, args.max_rounds - 1))
        save_sweep_csv(searcher.rows)
    except TimeBudget:
        print(f"\nTime budget of {args.max_minutes} min reached; selecting the best configuration found so far.")
    except KeyboardInterrupt:
        print("\nInterrupted; selecting the best configuration found so far.")

    save_sweep_csv(searcher.rows)

    # ---------------- selection ----------------
    best_cfg, best_res = searcher.select_final()
    improvement = best_res["macro"] - base_res["macro"]
    improvement_w = best_res["weighted"] - base_res["weighted"]

    print(f"\n{'=' * 78}\nRESULT (validation)\n{'=' * 78}")
    print(f"Configurations evaluated : {searcher.evals}   ({time.time() - t_start:.0f}s total)")
    print_lecture_table("Per-lecture validation accuracy", dataset,
                        {"baseline": base_res, "tuned": best_res})
    print(f"\nCurrent baseline  macro : {pct(base_res['macro'])}   weighted : {pct(base_res['weighted'])}")
    print(f"Best validation   macro : {pct(best_res['macro'])}   weighted : {pct(best_res['weighted'])}")
    print(f"Improvement             : {100 * improvement:+.2f} pp macro, {100 * improvement_w:+.2f} pp weighted")
    print_config_diff(baseline_cfg, best_cfg)

    worst_drop = min(best_res["accs"][n] - base_res["accs"][n] for n in dataset)
    if worst_drop < -0.02:
        print(f"\nCAUTION: the tuned configuration lowers at least one validation lecture by "
              f"{100 * -worst_drop:.1f} pp; it may not generalise. Inspect {CSV_PATH.name}.")
    if improvement < 0.005:
        print("\nNOTE: the validation gain is below 0.5 pp - within noise for 4 lectures. "
              "Decoder tuning alone may not reach 70%; consider this a sign the remaining error is "
              "in the score matrices rather than the decoder.")

    # ---------------- save ----------------
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "LectureSense V4.2 (existing V4 checkpoint; no retraining)",
        "tuning_type": "validation_only_decoder_and_fusion",
        "validation_only": True,
        "test_used": False,
        "test_labels_used_in_this_tuning_run": False,
        "objective": args.objective,
        "checkpoint": str(Path(args.checkpoint)),
        "v4_root": str(v4_root),
        "validation_lectures": list(dataset),
        "baseline_config": {k: baseline_cfg[k] for k in ALL_KEYS},
        "baseline_validation": {"macro": base_res["macro"], "weighted": base_res["weighted"],
                                "per_lecture": base_res["accs"]},
        "validation_macro_accuracy": best_res["macro"],
        "validation_weighted_accuracy": best_res["weighted"],
        "validation_improvement_macro_pp": 100 * improvement,
        "validation_improvement_weighted_pp": 100 * improvement_w,
        "lecture_accuracy": best_res["accs"],
        "fusion": {k: best_cfg[k] for k in FUSION_KEYS},
        "decoder": {k: best_cfg[k] for k in DECODER_KEYS},
        "search": {"evaluations": searcher.evals, "seed": args.seed,
                   "seconds": round(time.time() - t_start, 1), "fast_dp": USE_FAST_DP,
                   "fusion_matches_cache": FUSE_FAITHFUL},
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    print(f"\nSaved config : {CONFIG_PATH}")
    print(f"Saved sweep  : {CSV_PATH}")
    print("\nNext (optional, once, on the project test split):")
    print(f"  python {Path(__file__).name} --mode test")
    print("TEST labels were not loaded in validation mode and were not used for tuning.")
    print("Note: these project test lectures were evaluated in earlier model experiments; this is not a fresh external benchmark.")


# ============================================================
# TEST MODE  (selected config only; no search, nothing written to the config)
# ============================================================

def test_mode(args):
    if not CONFIG_PATH.exists():
        sys.exit(f"Validation-tuned configuration not found: {CONFIG_PATH}\n"
                 f"Run validation first:  python {Path(__file__).name}")
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not payload.get("validation_only", False) or payload.get("test_used", True):
        sys.exit("Config is not marked validation-only / test_used=false; refusing to evaluate on test.")

    v4_root = Path(args.v4_root)
    tuned = {**payload["fusion"], **payload["decoder"]}
    baseline_cfg = payload.get("baseline_config")
    if baseline_cfg is None:                                    # older config files
        baseline_cfg = make_baseline_config(load_checkpoint_decoder(Path(args.checkpoint)))
    missing = [k for k in ALL_KEYS if k not in tuned or k not in baseline_cfg]
    if missing:
        sys.exit(f"Config is missing keys: {missing}. Re-run validation.")

    print(f"\n{'=' * 78}\nHELD-OUT TEST  (locked validation configuration, no tuning)\n{'=' * 78}")
    dataset = load_dataset(lecture_specs("test", v4_root))
    fusion_key = baseline_fusion_key(baseline_cfg)

    verify_fast_dp(dataset, baseline_cfg, random.Random(args.seed))

    base_res = evaluate_config(dataset, baseline_cfg, fusion_key, args.workers)
    tuned_res = evaluate_config(dataset, tuned, fusion_key, args.workers)

    print_lecture_table("Per-lecture test accuracy (path vs. ground truth)", dataset,
                        {"current V4": base_res, "tuned config": tuned_res})
    print(f"\nTest macro accuracy    : {pct(tuned_res['macro'])}   (current V4 {pct(base_res['macro'])},  "
          f"{100 * (tuned_res['macro'] - base_res['macro']):+.2f} pp)")
    print(f"Test weighted accuracy : {pct(tuned_res['weighted'])}   (current V4 {pct(base_res['weighted'])},  "
          f"{100 * (tuned_res['weighted'] - base_res['weighted']):+.2f} pp)")
    print(f"Test total             : {tuned_res['correct']}/{tuned_res['total']}")
    print("\nTest labels were not used for tuning or model selection.")
    print("Do not adjust the configuration based on these numbers; that would turn the test set "
          "into a validation set.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_REPORT_PATH.write_text(json.dumps({
        "config_file": str(CONFIG_PATH),
        "tuned": {"macro": tuned_res["macro"], "weighted": tuned_res["weighted"],
                  "per_lecture": tuned_res["accs"], "correct": tuned_res["correct"], "total": tuned_res["total"]},
        "current_v4": {"macro": base_res["macro"], "weighted": base_res["weighted"],
                       "per_lecture": base_res["accs"], "correct": base_res["correct"], "total": base_res["total"]},
    }, indent=2, default=float), encoding="utf-8")
    print(f"Saved report: {TEST_REPORT_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="LectureSense V4.2 validation-only decoder/fusion tuner (existing model; no retraining)")
    ap.add_argument("--mode", choices=["validate", "test"], default="validate")
    ap.add_argument("--v4-root", default=str(DEFAULT_V4_ROOT),
                    help="folder with one sub-folder of cached V4 matrices per lecture")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--objective", choices=["macro", "weighted"], default="weighted",
                    help="metric optimised on validation data (default: weighted; macro is also reported)")
    ap.add_argument("--random-samples", type=int, default=150)
    ap.add_argument("--refine-top", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--min-gain", type=float, default=0.001,
                    help="smallest objective gain (fraction) accepted as a real improvement")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small search for a fast smoke run")
    args = ap.parse_args()

    if args.quick:
        args.random_samples = min(args.random_samples, 30)
        args.max_rounds = min(args.max_rounds, 2)
        args.refine_top = 1
        args.max_minutes = min(args.max_minutes, 10.0)

    if args.mode == "validate":
        validate_mode(args)
    else:
        test_mode(args)


if __name__ == "__main__":
    main()
