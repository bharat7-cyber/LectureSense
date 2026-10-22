from __future__ import annotations

"""LectureSense V11: supervised candidate reranking + visual-semantic alignment.

Protocol
--------
train mode:
  * Fits only on official training lectures with usable labels/assets.
  * Uses the four official validation lectures for checkpoint/decoder selection.
  * Does not open held-out test ground truth.

test mode:
  * Loads the locked model and decoder chosen during validation.
  * Does no training, searching, or configuration changes.
  * Reports V11 beside the V4 baseline.

This script reuses the project's V7 asset/visual-cache helpers, but constructs a
new feature set from the V4 score matrices plus direct video-frame-to-slide
visual-semantic scores. It does not modify the V4 checkpoint.
"""

import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from boundary_dp_v4 import boundary_aware_dp
    from video_visual_alignment import fuse_model_and_video_scores
except Exception as exc:
    raise SystemExit(
        "Could not import V4 modules from src/. Run this script from the project root.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

V7_HELPERS_PATH = ROOT / "LectureSense_V7_FullDevelopment_GatedFusion.py"
if not V7_HELPERS_PATH.exists():
    raise SystemExit(
        f"Required helper not found: {V7_HELPERS_PATH}\n"
        "Keep LectureSense_V7_FullDevelopment_GatedFusion.py in the project root."
    )
_spec = importlib.util.spec_from_file_location("lecturesense_v7_helpers", V7_HELPERS_PATH)
v7 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(v7)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ROOT / "data" / "MaViLS" / "data"
DEFAULT_V4_ROOT = ROOT / "runs" / "v5_gate_train_v4"
OUT_ROOT = ROOT / "runs" / "v11_supervised_candidate_reranking"
CACHE_ROOT = OUT_ROOT / "feature_cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "v11_supervised_candidate_reranker.pt"
CONFIG_PATH = CHECKPOINT_DIR / "v11_supervised_candidate_reranker_config.json"
VALIDATION_REPORT = OUT_ROOT / "validation_report.json"
VALIDATION_CSV = OUT_ROOT / "validation_per_lecture.csv"
TEST_REPORT = OUT_ROOT / "test_report.json"
TEST_CSV = OUT_ROOT / "test_per_lecture.csv"

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
VAL_LECTURES = ["physics", "psychology", "reinforcement_learning", "sensory_systems"]
TEST_LECTURES = [
    "short_range",
    "solar_resource",
    "team_dynamics_game_design_MIT",
    "theory_of_computation",
]

# Feature names are created deterministically by build_candidate_features().
FEATURE_NAMES: list[str] = []
FINAL_RANK_IDX = -1
VISUAL_RANK_IDX = -1
FEATURE_VERSION = 1


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise_column_name(value) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def to_seconds(value) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
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
        parts = [float(x) for x in text.split(":")]
    except ValueError:
        return float("nan")
    total = 0.0
    for p in parts:
        total = total * 60.0 + p
    return total


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth workbook missing: {path}")
    df = pd.read_excel(path)
    df.columns = [normalise_column_name(c) for c in df.columns]
    time_col = next((c for c in ("time", "timestamp", "start", "key") if c in df.columns), None)
    slide_col = next(
        (c for c in ("slidenumber", "slide_number", "slide", "slideid", "slide_id") if c in df.columns),
        None,
    )
    if time_col is None or slide_col is None:
        raise RuntimeError(f"Cannot identify time/slide columns in {path}: {list(df.columns)}")
    out = df[[time_col, slide_col]].copy()
    out.columns = ["time", "slide"]
    out["time"] = out["time"].map(to_seconds)
    out["slide"] = pd.to_numeric(out["slide"], errors="coerce")
    out = out.dropna()
    out = out[(out["time"] >= 0) & (out["slide"] > 0)]
    out["slide"] = out["slide"].astype(int)
    out = out.sort_values("time").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No valid ground-truth rows in {path}")
    return out


def load_optional_matrix(folder: Path, filenames) -> np.ndarray | None:
    for filename in filenames:
        p = folder / filename
        if p.exists():
            try:
                return np.asarray(np.load(p), dtype=np.float32)
            except Exception as exc:
                print(f"  Warning: couldn't read {p.name}: {exc}")
    return None


def nearest_indices(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    sorted_values = np.asarray(sorted_values, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if len(sorted_values) == 0:
        raise ValueError("Cannot map times to an empty visual timestamp array")
    right = np.searchsorted(sorted_values, queries, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(queries - sorted_values[left]) <= np.abs(sorted_values[right] - queries)
    return np.where(choose_left, left, right).astype(np.int64)


def query_segment_indices(times: np.ndarray, segment_mids: np.ndarray) -> np.ndarray:
    # Matches the mapping convention used by the existing V7 evaluator.
    idx = np.searchsorted(segment_mids, np.asarray(times, dtype=np.float32), side="right") - 1
    return np.clip(idx, 0, len(segment_mids) - 1).astype(np.int64)


def segment_time(seg, field: str) -> float:
    if isinstance(seg, dict):
        if field == "start":
            value = seg.get("start")
            if value is None:
                value = seg.get("start_time", 0.0)
        else:
            value = seg.get("end")
            if value is None:
                value = seg.get("end_time")
            if value is None:
                value = seg.get("start")
            if value is None:
                value = seg.get("start_time", 0.0)
    else:
        value = getattr(seg, field, None)
        if value is None:
            value = getattr(seg, "start", 0.0)
    return float(value)


def load_checkpoint_decoder(checkpoint_path: Path) -> dict:
    defaults = {
        "jump_penalty": 0.05,
        "max_forward_jump": 1,
        "max_duration_seconds": 120.0,
        "duration_penalty": 0.001,
        "max_run_seconds": 1200.0,
        "boundary_reward": 1.0,
        "boundary_stay_penalty": 0.50,
        "transition_logits_weight": 0.35,
        "premature_transition_penalty": 0.50,
        "boundary_threshold": 0.20,
    }
    if checkpoint_path.exists():
        try:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(checkpoint_path, map_location="cpu")
        defaults.update({k: v for k, v in ck.get("decoder_config", {}).items() if k in defaults})
        del ck
    # Match the settings currently passed by the app/inference command.
    defaults.update({"jump_penalty": 0.05, "max_duration_seconds": 120.0, "duration_penalty": 0.003})
    return defaults


def resolve_v4_cache(lecture: str, split: str, v4_root: Path) -> Path:
    aliases = {
        "short_range": ["short_range_mit", "short_range"],
        "team_dynamics_game_design_MIT": ["team_dynamics_game_design_MIT", "team_dynamics"],
        "physics": ["physics", "physics_intro_02"],
        "reinforcement_learning": ["reinforcement_learning", "reinforcement_learning_silver"],
        "sensory_systems": ["sensory_systems", "visual_system_MIT"],
    }.get(lecture, [lecture])
    roots = [
        v4_root,
        ROOT / "runs" / "formal_test_eval_v4",
        ROOT / "runs" / "formal_test_eval_v4_final",
        ROOT / "runs" / "formal_test_eval",
        ROOT / "runs" / "unseen_eval",
    ]
    seen = set()
    for base in roots:
        for alias in aliases:
            candidate = base / alias
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "v4_final_score_matrix.npy").exists() or (candidate / "v4_adjusted_score_matrix.npy").exists():
                return candidate
    # If no completed cache exists, use the existing V7 helper to run V4 once.
    v7.V4_ROOT = v4_root
    srt, pdf, gt, video = v7.asset_paths(lecture)
    if srt is None or pdf is None or video is None:
        raise FileNotFoundError(f"No usable SRT/PDF/video assets for {lecture}; cache was not found either.")
    return Path(v7.ensure_v4(lecture, srt, pdf, video))


# ---------------------------------------------------------------------------
# Candidate feature construction
# ---------------------------------------------------------------------------

def row_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.clip((x - mu) / sd, -6.0, 6.0) / 3.0


def row_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def row_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(-x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[1], dtype=np.float32)[None, :]
    if x.shape[1] > 1:
        ranks /= float(x.shape[1] - 1)
    return 1.0 - ranks


def source_stats(x: np.ndarray) -> dict:
    z = row_zscore(x)
    p = row_softmax(z)
    ranks = row_rank(x)
    constant_rows = np.std(x, axis=1) < 1e-6
    if np.any(constant_rows):
        ranks[constant_rows] = 0.5
    sorted_z = np.sort(z, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2] if z.shape[1] > 1 else np.zeros(z.shape[0], np.float32)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
    entropy /= max(1e-6, math.log(max(2, z.shape[1])))
    top = np.argmax(x, axis=1).astype(np.int64)
    return {"z": z, "p": p, "rank": ranks, "margin": margin, "entropy": entropy, "top": top}


def build_candidate_features(
    query_times: np.ndarray,
    mids: np.ndarray,
    sources: dict[str, np.ndarray],
    boundary: np.ndarray,
    next_logits: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build (query-time, slide-candidate, feature) multimodal features."""
    query_times = np.asarray(query_times, dtype=np.float32)
    mids = np.asarray(mids, dtype=np.float32)
    seg_idx = query_segment_indices(query_times, mids)
    nseg, nslides = sources["final"].shape
    if any(x.shape != (nseg, nslides) for x in sources.values()):
        raise ValueError("All score sources must have identical (segments, slides) shapes")

    stats = {name: source_stats(matrix) for name, matrix in sources.items()}
    next_stats = source_stats(next_logits)
    prev_idx = np.clip(seg_idx - 1, 0, nseg - 1)
    next_idx = np.clip(seg_idx + 1, 0, nseg - 1)
    candidate_ids = np.arange(nslides, dtype=np.int64)[None, :]
    q = len(query_times)
    blocks: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, matrix: np.ndarray):
        a = np.asarray(matrix, dtype=np.float32)
        if a.shape != (q, nslides):
            raise ValueError(f"Feature {name} has shape {a.shape}, expected {(q, nslides)}")
        blocks.append(a)
        names.append(name)

    def add_row(name: str, row_values: np.ndarray):
        vals = np.asarray(row_values, dtype=np.float32).reshape(q)
        add(name, np.repeat(vals[:, None], nslides, axis=1))

    # Per-candidate score, probability and rank from each available modality.
    for src_name in ("base", "adjusted", "final", "v4video", "visual"):
        st = stats[src_name]
        add(f"{src_name}_z", st["z"][seg_idx])
        add(f"{src_name}_p", st["p"][seg_idx])
        add(f"{src_name}_rank", st["rank"][seg_idx])

    # Local temporal context: do nearby transcript moments support this candidate?
    for src_name in ("adjusted", "final", "v4video", "visual"):
        z = stats[src_name]["z"]
        add(f"{src_name}_prev_time_z", z[prev_idx])
        add(f"{src_name}_next_time_z", z[next_idx])

    # Adjacent slide candidates often look/textually resemble their neighbours.
    left_ids = np.clip(np.arange(nslides) - 1, 0, nslides - 1)
    right_ids = np.clip(np.arange(nslides) + 1, 0, nslides - 1)
    for src_name in ("adjusted", "final", "v4video", "visual"):
        z = stats[src_name]["z"][seg_idx]
        add(f"{src_name}_left_slide_z", z[:, left_ids])
        add(f"{src_name}_right_slide_z", z[:, right_ids])

    # Row confidence and ambiguity for all score sources.
    for src_name in ("base", "adjusted", "final", "v4video", "visual"):
        st = stats[src_name]
        add_row(f"{src_name}_margin", st["margin"][seg_idx])
        add_row(f"{src_name}_entropy", st["entropy"][seg_idx])

    # Explicit indicators: which modality picks this slide as its top candidate?
    for src_name in ("base", "adjusted", "final", "v4video", "visual"):
        tops = stats[src_name]["top"][seg_idx]
        add(f"is_top_{src_name}", (candidate_ids == tops[:, None]).astype(np.float32))

    # Candidate distance from the leading predictions.
    for src_name in ("adjusted", "final", "visual"):
        tops = stats[src_name]["top"][seg_idx]
        d = np.abs(candidate_ids - tops[:, None]).astype(np.float32)
        add(f"distance_to_{src_name}_top", d / max(1, nslides - 1))

    # Cross-modal agreement/difference between V4 semantic scores and direct video visuals.
    fz = stats["final"]["z"][seg_idx]
    vz = stats["visual"]["z"][seg_idx]
    add("final_visual_product", fz * vz)
    add("final_visual_difference", fz - vz)
    add("final_visual_abs_difference", np.abs(fz - vz))
    final_top = stats["final"]["top"][seg_idx]
    visual_top = stats["visual"]["top"][seg_idx]
    add_row("final_visual_top_agreement", (final_top == visual_top).astype(np.float32))
    add("is_top_both_final_visual", ((candidate_ids == final_top[:, None]) & (candidate_ids == visual_top[:, None])).astype(np.float32))

    # V4 learned next-slide head and boundary detector.
    add("next_slide_probability", next_stats["p"][seg_idx])
    add("next_slide_z", next_stats["z"][seg_idx])
    next_top = next_stats["top"][seg_idx]
    add("is_next_head_top", (candidate_ids == next_top[:, None]).astype(np.float32))
    add_row("boundary_probability", np.clip(boundary[seg_idx], 0.0, 1.0))

    # Time/slide position prior, kept deliberately weak and normalized.
    duration = max(1e-6, float(mids[-1] - mids[0]))
    tpos = np.clip((query_times - mids[0]) / duration, 0.0, 1.0)
    spos = np.broadcast_to(candidate_ids.astype(np.float32) / max(1, nslides - 1), (q, nslides))
    tpos_matrix = np.repeat(tpos[:, None], nslides, axis=1)
    add("candidate_slide_position", spos)
    add("query_time_position", tpos_matrix)
    add("slide_time_position_gap", spos - tpos_matrix)

    # Temporal transition context from neighbouring predicted top candidates.
    final_prev_top = stats["final"]["top"][prev_idx]
    final_next_top = stats["final"]["top"][next_idx]
    add("candidate_matches_prev_final_top", (candidate_ids == final_prev_top[:, None]).astype(np.float32))
    add("candidate_matches_next_final_top", (candidate_ids == final_next_top[:, None]).astype(np.float32))

    # Distribution changes between neighbouring rows identify likely slide transitions.
    for src_name in ("final", "visual"):
        p = stats[src_name]["p"]
        change = np.zeros(nseg, dtype=np.float32)
        if nseg > 1:
            change[1:] = np.mean(np.abs(p[1:] - p[:-1]), axis=1)
        add_row(f"{src_name}_probability_change", change[seg_idx])

    X = np.stack(blocks, axis=-1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


# ---------------------------------------------------------------------------
# Dataset/caches
# ---------------------------------------------------------------------------

def prepare_lecture(lecture: str, split: str, v4_root: Path) -> dict:
    srt, pdf, gt_path, video = v7.asset_paths(lecture)
    if srt is None or pdf is None or gt_path is None:
        raise FileNotFoundError(f"Missing SRT/PDF/ground-truth asset for {lecture}")
    if video is None:
        raise FileNotFoundError(f"No exact video is configured for {lecture}; skipping it")

    cache_dir = resolve_v4_cache(lecture, split, v4_root)
    print(f"\nPreparing {split}: {lecture}\n  SRT: {srt}\n  V4 cache: {cache_dir}")

    segments = v7.get_segments(srt)
    mids = np.asarray(v7.segment_mids(segments), dtype=np.float32)
    starts = np.asarray([segment_time(seg, "start") for seg in segments], dtype=np.float32)
    ends = np.asarray([segment_time(seg, "end") for seg in segments], dtype=np.float32)
    if not len(mids) or np.any(np.diff(mids) < 0):
        raise RuntimeError(f"Invalid transcript timestamps for {lecture}")

    base = load_optional_matrix(cache_dir, ["v4_base_score_matrix.npy", "base_score_matrix.npy"])
    adjusted = load_optional_matrix(cache_dir, ["v4_adjusted_score_matrix.npy", "adjusted_score_matrix.npy"])
    final = load_optional_matrix(cache_dir, ["v4_final_score_matrix.npy", "final_score_matrix.npy", "fused_score_matrix.npy"])
    v4video = load_optional_matrix(cache_dir, ["v4_video_score_matrix.npy", "video_score_matrix.npy"])
    boundary = load_optional_matrix(cache_dir, ["v4_boundary_probability.npy", "boundary_probability.npy"])
    next_logits = load_optional_matrix(cache_dir, ["v4_next_slide_logits.npy", "next_slide_logits.npy"])

    if final is None and adjusted is None and base is None:
        # If a cache was incomplete, regenerate once into the configured V4 root.
        v7.V4_ROOT = v4_root
        cache_dir = Path(v7.ensure_v4(lecture, srt, pdf, video))
        base = load_optional_matrix(cache_dir, ["v4_base_score_matrix.npy", "base_score_matrix.npy"])
        adjusted = load_optional_matrix(cache_dir, ["v4_adjusted_score_matrix.npy", "adjusted_score_matrix.npy"])
        final = load_optional_matrix(cache_dir, ["v4_final_score_matrix.npy", "final_score_matrix.npy", "fused_score_matrix.npy"])
        v4video = load_optional_matrix(cache_dir, ["v4_video_score_matrix.npy", "video_score_matrix.npy"])
        boundary = load_optional_matrix(cache_dir, ["v4_boundary_probability.npy", "boundary_probability.npy"])
        next_logits = load_optional_matrix(cache_dir, ["v4_next_slide_logits.npy", "next_slide_logits.npy"])

    if adjusted is None:
        adjusted = base.copy() if base is not None else final.copy()
    if base is None:
        # Safe fallback: residual-scaling features become neutral rather than aborting.
        base = adjusted.copy()
    if final is None:
        final = fuse_model_and_video_scores(adjusted, v4video, video_weight=0.40) if v4video is not None else adjusted.copy()

    nseg, nslides = final.shape
    if len(mids) != nseg:
        raise RuntimeError(f"{lecture}: transcript segments={len(mids)}, V4 matrix rows={nseg}")
    for label, matrix in (("base", base), ("adjusted", adjusted)):
        if matrix.shape != (nseg, nslides):
            print(f"  Warning: {label} matrix shape {matrix.shape} mismatches {final.shape}; using final matrix.")
            if label == "base":
                base = final.copy()
            else:
                adjusted = final.copy()
    if v4video is None or v4video.shape != (nseg, nslides):
        print("  Note: V4 video-score matrix missing/mismatched; zero-filled visual score source.")
        v4video = np.zeros((nseg, nslides), dtype=np.float32)
    if boundary is None or boundary.reshape(-1).shape[0] != nseg:
        boundary = np.zeros(nseg, dtype=np.float32)
    boundary = np.clip(boundary.reshape(-1).astype(np.float32), 1e-4, 1.0 - 1e-4)
    if next_logits is None or next_logits.shape != (nseg, nslides):
        next_logits = np.zeros((nseg, nslides), dtype=np.float32)

    visual_times, visual_scores = v7.get_visual_scores(lecture, pdf, video)
    visual_times = np.asarray(visual_times, dtype=np.float32)
    visual_scores = np.asarray(visual_scores, dtype=np.float32)
    if visual_scores.ndim != 2 or visual_scores.shape[1] != nslides or len(visual_times) != len(visual_scores):
        print("  Warning: visual-first cache shape mismatch; zero-filled visual-semantic source.")
        visual_by_seg = np.zeros((nseg, nslides), dtype=np.float32)
    else:
        vis_idx_for_mids = nearest_indices(visual_times, mids)
        visual_by_seg = visual_scores[vis_idx_for_mids]

    sources = {
        "base": base.astype(np.float32),
        "adjusted": adjusted.astype(np.float32),
        "final": final.astype(np.float32),
        "v4video": v4video.astype(np.float32),
        "visual": visual_by_seg.astype(np.float32),
    }

    gt = load_ground_truth(gt_path)
    keep = (gt["slide"].to_numpy(dtype=np.int64) >= 1) & (gt["slide"].to_numpy(dtype=np.int64) <= nslides)
    gt = gt.loc[keep].reset_index(drop=True)
    if gt.empty:
        raise RuntimeError(f"No ground-truth slides within the {nslides}-slide deck for {lecture}")
    gt_times = gt["time"].to_numpy(dtype=np.float32)
    gt_labels = gt["slide"].to_numpy(dtype=np.int64) - 1
    gt_idx = query_segment_indices(gt_times, mids)

    X_gt, feature_names = build_candidate_features(gt_times, mids, sources, boundary, next_logits)
    X_seq, seq_feature_names = build_candidate_features(mids, mids, sources, boundary, next_logits)
    if feature_names != seq_feature_names:
        raise RuntimeError("Feature builder emitted inconsistent feature order")

    next_prob = row_softmax(next_logits)
    next_logprob = np.log(np.maximum(next_prob, 1e-12)).astype(np.float64)

    print(f"  transcript segments={nseg}, slides={nslides}, GT={len(gt_labels)}, features={X_gt.shape[-1]}")
    print(f"  score matrices: base={'loaded' if base is not None else 'fallback'}, V4 video={'yes' if np.any(v4video) else 'empty'}, visual-first={'yes' if np.any(visual_by_seg) else 'empty'}")

    return {
        "lecture": lecture,
        "split": split,
        "srt": str(srt),
        "pdf": str(pdf),
        "video": str(video),
        "cache_dir": str(cache_dir),
        "mids": mids,
        "starts": starts,
        "ends": ends,
        "sources": sources,
        "boundary": boundary,
        "next_logits": next_logits.astype(np.float32),
        "next_logprob": next_logprob,
        "X_gt": X_gt,
        "Y": gt_labels,
        "X_seq": X_seq,
        "gt_idx": gt_idx,
        "gt_labels": gt_labels,
        "gt_times": gt_times,
        "slides": nslides,
        "final_scores": final.astype(np.float32),
        "feature_names": feature_names,
    }


def load_split(lectures: list[str], split: str, v4_root: Path, allow_skip: bool) -> list[dict]:
    items = []
    for lecture in lectures:
        try:
            items.append(prepare_lecture(lecture, split, v4_root))
        except Exception as exc:
            if not allow_skip:
                raise
            print(f"\n[{split.upper()} SKIP] {lecture}: {type(exc).__name__}: {exc}")
    if not items:
        raise RuntimeError(f"No usable {split} lectures loaded")
    return items


# ---------------------------------------------------------------------------
# Model + losses
# ---------------------------------------------------------------------------

class V11CandidateReranker(nn.Module):
    def __init__(self, feature_dim: int, final_rank_idx: int, visual_rank_idx: int):
        super().__init__()
        self.final_rank_idx = int(final_rank_idx)
        self.visual_rank_idx = int(visual_rank_idx)
        self.body = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.residual_head = nn.Linear(64, 1)
        self.visual_gate = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.anchor_raw = nn.Parameter(torch.tensor(1.20))
        self.visual_raw = nn.Parameter(torch.tensor(-1.20))
        nn.init.constant_(self.visual_gate[-1].bias, -1.0)

    def forward(self, x):
        h = self.body(x)
        residual = self.residual_head(h).squeeze(-1)
        gate = torch.sigmoid(self.visual_gate(h).squeeze(-1))
        anchor_scale = F.softplus(self.anchor_raw) + 0.20
        visual_scale = F.softplus(self.visual_raw)
        v4_anchor = x[:, :, self.final_rank_idx]
        visual_anchor = x[:, :, self.visual_rank_idx]
        return anchor_scale * v4_anchor + gate * visual_scale * visual_anchor + 0.70 * residual


def hard_negative_per_sample(logits, targets, margin=0.12, hard_k=6):
    batch, classes = logits.shape
    pos = logits.gather(1, targets.view(-1, 1))
    neg = logits.clone()
    neg.scatter_(1, targets.view(-1, 1), -torch.inf)
    k = min(hard_k, max(1, classes - 1))
    hardest = torch.topk(neg, k=k, dim=1).values
    return F.relu(margin - pos + hardest).mean(dim=1)


def make_v11_model(feature_dim: int, feature_names: list[str]) -> V11CandidateReranker:
    return V11CandidateReranker(
        feature_dim=feature_dim,
        final_rank_idx=feature_names.index("final_rank"),
        visual_rank_idx=feature_names.index("visual_rank"),
    ).to(DEVICE)


@torch.no_grad()
def predict_emissions(model_states: list[dict], feature_dim: int, feature_names: list[str], X_seq: np.ndarray, batch_size=96) -> np.ndarray:
    outputs = []
    models = []
    for state in model_states:
        m = make_v11_model(feature_dim, feature_names)
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
    for start in range(0, len(X_seq), batch_size):
        xb = torch.from_numpy(np.asarray(X_seq[start:start + batch_size], dtype=np.float32)).to(DEVICE)
        ensemble = [m(xb) for m in models]
        outputs.append(torch.stack(ensemble, dim=0).mean(dim=0).detach().cpu().numpy())
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def predict_single_model(model: nn.Module, X: np.ndarray, batch_size=96) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[start:start + batch_size], dtype=np.float32)).to(DEVICE)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary / transition-aware monotonic dynamic programming (fast vectorised)
# ---------------------------------------------------------------------------

def transition_viterbi(
    emissions: np.ndarray,
    boundary: np.ndarray,
    next_logprob: np.ndarray,
    max_forward_jump: int = 3,
    jump_penalty: float = 0.05,
    boundary_reward: float = 0.25,
    boundary_stay_penalty: float = 0.25,
    transition_logits_weight: float = 0.35,
    premature_transition_penalty: float = 0.25,
    boundary_threshold: float = 0.20,
) -> np.ndarray:
    """Fast monotonic DP. Starts on slide 1, only advances forward, uses V4 boundary/next-head evidence."""
    scores = np.asarray(emissions, dtype=np.float32)
    b = np.clip(np.asarray(boundary, dtype=np.float32).reshape(-1), 1e-4, 1.0 - 1e-4)
    nl = np.asarray(next_logprob, dtype=np.float64)
    n, s = scores.shape
    if n == 0 or s == 0:
        return np.zeros(0, dtype=np.int32)
    if b.shape[0] != n or nl.shape != (n, s):
        raise ValueError("V11 DP: boundary/next-logprob shape mismatch")
    max_jump = max(1, int(max_forward_jump))
    dp = np.full((n, s), -np.inf, dtype=np.float64)
    back_delta = np.full((n, s), -1, dtype=np.int16)
    dp[0, 0] = float(scores[0, 0])
    slide_ids = np.arange(s)
    for t in range(1, n):
        p = float(b[t])
        p_logit = math.log(p / (1.0 - p))
        jump_base = (
            float(boundary_reward) * math.tanh(0.5 * p_logit)
            - float(premature_transition_penalty) * (1.0 - p)
        )
        candidates = np.full((max_jump + 1, s), -np.inf, dtype=np.float64)
        # Stay at the same slide.
        stay_pen = float(boundary_stay_penalty) * max(0.0, p - float(boundary_threshold))
        candidates[0] = dp[t - 1] - stay_pen
        # Transition from p=j-delta to current slide j.
        for delta in range(1, max_jump + 1):
            if delta >= s:
                break
            trans = jump_base - float(jump_penalty) * delta
            target_slice = slice(delta, s)
            source_slice = slice(0, s - delta)
            candidates[delta, target_slice] = (
                dp[t - 1, source_slice]
                + trans
                + float(transition_logits_weight) * nl[t, target_slice]
            )
        best_delta = np.argmax(candidates, axis=0)
        best_value = candidates[best_delta, slide_ids]
        dp[t] = best_value + scores[t]
        back_delta[t] = best_delta.astype(np.int16)
    last_slide = int(np.argmax(dp[-1]))
    path = np.zeros(n, dtype=np.int32)
    path[-1] = last_slide
    for t in range(n - 1, 0, -1):
        delta = int(back_delta[t, path[t]])
        if delta < 0:
            delta = 0
        path[t - 1] = path[t] - delta
    if path[0] != 0 or np.any(np.diff(path) < 0):
        raise RuntimeError("V11 DP produced an invalid monotonic path")
    return path


def next_log_prob(next_logits: np.ndarray) -> np.ndarray:
    z = np.asarray(next_logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    return z - np.log(np.maximum(np.exp(z).sum(axis=1, keepdims=True), 1e-12))


def v11_decoder_defaults(checkpoint_decoder: dict) -> dict:
    return {
        "max_forward_jump": int(checkpoint_decoder.get("max_forward_jump", 1)),
        "jump_penalty": 0.05,
        "boundary_reward": float(checkpoint_decoder.get("boundary_reward", 1.0)),
        "boundary_stay_penalty": float(checkpoint_decoder.get("boundary_stay_penalty", 0.50)),
        "transition_logits_weight": float(checkpoint_decoder.get("transition_logits_weight", 0.35)),
        "premature_transition_penalty": float(checkpoint_decoder.get("premature_transition_penalty", 0.50)),
        "boundary_threshold": float(checkpoint_decoder.get("boundary_threshold", 0.20)),
    }


def evaluate_v11(items: list[dict], emissions_by_name: dict[str, np.ndarray], decoder: dict) -> dict:
    per = {}
    rows = []
    for item in items:
        path = transition_viterbi(
            emissions_by_name[item["lecture"]], item["boundary"], item["next_logprob"], **decoder
        )
        pred = path[item["gt_idx"]]
        correct = int(np.sum(pred == item["gt_labels"]))
        total = int(len(item["gt_labels"]))
        acc = correct / max(1, total)
        per[item["lecture"]] = {"correct": correct, "total": total, "accuracy": acc}
        rows.append({"lecture": item["lecture"], "correct": correct, "total": total, "accuracy": acc})
    correct_all = sum(x["correct"] for x in per.values())
    total_all = sum(x["total"] for x in per.values())
    return {
        "macro": float(np.mean([x["accuracy"] for x in per.values()])),
        "weighted": float(correct_all / max(1, total_all)),
        "correct": int(correct_all),
        "total": int(total_all),
        "per_lecture": per,
        "rows": rows,
    }


def evaluate_v4_baseline(item: dict, checkpoint_decoder: dict) -> tuple[int, int]:
    """Prefer the exact cached V4 timeline, avoiding expensive repeated DP runs.

    If no timeline JSON exists, recompute the V4 path using the same decoder
    overrides used by the existing inference command.
    """
    timeline_path = Path(item["cache_dir"]) / "timeline.json"
    if timeline_path.exists():
        try:
            payload = json.loads(timeline_path.read_text(encoding="utf-8"))
            entries = payload.get("timeline", [])
            parsed = []
            for row in entries:
                start = to_seconds(row.get("start_time", row.get("start", float("nan"))))
                end = to_seconds(row.get("end_time", row.get("end", float("nan"))))
                slide = row.get("slide_id")
                if slide is None and row.get("slide_index") is not None:
                    slide = int(row["slide_index"]) + 1
                if math.isfinite(start) and math.isfinite(end) and slide is not None and int(slide) > 0:
                    parsed.append((float(start), float(end), int(slide) - 1))
            parsed.sort(key=lambda x: x[0])
            if parsed:
                tstarts = np.asarray([x[0] for x in parsed], dtype=np.float32)
                preds = []
                for t in item["gt_times"]:
                    pos = int(np.searchsorted(tstarts, float(t), side="right") - 1)
                    pos = min(max(pos, 0), len(parsed) - 1)
                    start, end, slide = parsed[pos]
                    if float(t) > end and pos + 1 < len(parsed):
                        next_start = parsed[pos + 1][0]
                        if abs(next_start - float(t)) < abs(float(t) - end):
                            pos += 1
                            slide = parsed[pos][2]
                    preds.append(slide)
                pred = np.asarray(preds, dtype=np.int64)
                return int(np.sum(pred == item["gt_labels"])), int(len(item["gt_labels"]))
        except Exception as exc:
            print(f"  Warning: couldn't score cached V4 timeline for {item['lecture']}: {exc}; recomputing path.")

    cfg = dict(checkpoint_decoder)
    cfg.update({"jump_penalty": 0.05, "max_duration_seconds": 120.0, "duration_penalty": 0.003})
    path = boundary_aware_dp(
        item["final_scores"], item["boundary"], item["next_logits"], item["starts"], item["ends"],
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
    path = np.asarray(path, dtype=np.int64)
    return int(np.sum(path[item["gt_idx"]] == item["gt_labels"])), int(len(item["gt_labels"]))


def score_v4_baseline(items: list[dict], checkpoint_decoder: dict) -> dict:
    per = {}
    for item in items:
        correct, total = evaluate_v4_baseline(item, checkpoint_decoder)
        per[item["lecture"]] = {"correct": correct, "total": total, "accuracy": correct / max(1, total)}
    all_correct = sum(x["correct"] for x in per.values())
    all_total = sum(x["total"] for x in per.values())
    return {
        "macro": float(np.mean([x["accuracy"] for x in per.values()])),
        "weighted": float(all_correct / max(1, all_total)),
        "correct": int(all_correct),
        "total": int(all_total),
        "per_lecture": per,
    }


# ---------------------------------------------------------------------------
# Train, tune on validation, and locked test
# ---------------------------------------------------------------------------

def train_one_seed(train_items: list[dict], val_items: list[dict], feature_names: list[str], seed: int, epochs: int, patience: int, batch_size: int, checkpoint_decoder: dict) -> tuple[dict, dict]:
    set_seed(seed)
    feature_dim = len(feature_names)
    model = make_v11_model(feature_dim, feature_names)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)
    selection_decoder = v11_decoder_defaults(checkpoint_decoder)
    best_macro = -1.0
    best_state = None
    best_detail = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        losses = []
        for data_idx in order:
            item = train_items[data_idx]
            X = item["X_gt"]
            Y = item["Y"]
            n = len(Y)
            perm = np.random.permutation(n)
            # Accumulate one lecture-balanced gradient update per lecture.
            opt.zero_grad(set_to_none=True)
            for start in range(0, n, batch_size):
                ids = perm[start:start + batch_size]
                xb = torch.from_numpy(X[ids]).to(DEVICE)
                yb = torch.from_numpy(Y[ids]).long().to(DEVICE)
                logits = model(xb)
                ce = F.cross_entropy(logits, yb, reduction="none", label_smoothing=0.01)
                hn = hard_negative_per_sample(logits, yb, margin=0.12, hard_k=6)
                loss = (ce + 0.25 * hn).sum() / max(1, n)
                loss.backward()
                losses.append(float(loss.detach().cpu()) * max(1, n) / max(1, len(ids)))
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        val_emissions = {
            item["lecture"]: predict_single_model(model, item["X_seq"], batch_size=128)
            for item in val_items
        }
        val_res = evaluate_v11(val_items, val_emissions, selection_decoder)
        scheduler.step(val_res["macro"])
        lr = opt.param_groups[0]["lr"]
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"[V11 seed={seed}] epoch={epoch:02d}/{epochs} loss={avg_loss:.4f} "
            f"val macro={100 * val_res['macro']:.2f}% weighted={100 * val_res['weighted']:.2f}% lr={lr:.2e}"
        )
        if val_res["macro"] > best_macro + 1e-5:
            best_macro = val_res["macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_detail = val_res
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"  Early stopping seed={seed} after {epoch} epochs; best validation macro={100 * best_macro:.2f}%")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for seed {seed}")
    del model, opt, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_state, {"seed": seed, "best_validation_macro": best_macro, "best_validation": best_detail}


def tune_decoder_validation(val_items: list[dict], emissions: dict[str, np.ndarray], checkpoint_decoder: dict) -> tuple[dict, dict, list[dict]]:
    base = v11_decoder_defaults(checkpoint_decoder)
    candidates = []
    for max_jump, jump_penalty, boundary_reward in itertools.product(
        [1, 2, 3, 4], [0.0, 0.02, 0.05, 0.10], [0.0, 0.25, 0.50]
    ):
        cfg = dict(base)
        cfg.update({
            "max_forward_jump": int(max_jump),
            "jump_penalty": float(jump_penalty),
            "boundary_reward": float(boundary_reward),
        })
        res = evaluate_v11(val_items, emissions, cfg)
        candidates.append({
            "config": cfg,
            "macro": res["macro"],
            "weighted": res["weighted"],
            "correct": res["correct"],
            "total": res["total"],
            "per_lecture": res["per_lecture"],
        })
    # Prefer simpler/near-default configs when performance ties on this small validation set.
    best_macro = max(x["macro"] for x in candidates)
    near = [x for x in candidates if x["macro"] >= best_macro - 1e-5]
    def distance(row):
        c = row["config"]
        return (abs(c["max_forward_jump"] - 3) + abs(c["jump_penalty"] - 0.05) * 10 + abs(c["boundary_reward"] - 0.25))
    best = sorted(near, key=distance)[0]
    candidates.sort(key=lambda x: (x["macro"], x["weighted"]), reverse=True)
    return best["config"], best, candidates


def train_mode(args) -> None:
    v4_root = Path(args.v4_root).resolve()
    v7.V4_ROOT = v4_root
    print("=" * 88)
    print("LECTURESENSE V11 — SUPERVISED CANDIDATE RERANKING")
    print(f"Device: {DEVICE}" + (f" | GPU: {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else ""))
    print(f"V4 cache root: {v4_root}")
    print("Training lectures only for fitting; validation is for model/decoder selection; test labels are not loaded.")

    train_items = load_split(TRAIN_LECTURES, "train", v4_root, allow_skip=True)
    val_items = load_split(VAL_LECTURES, "validation", v4_root, allow_skip=False)
    feature_names = train_items[0]["feature_names"]
    for item in train_items + val_items:
        if item["feature_names"] != feature_names:
            raise RuntimeError("Feature ordering mismatch between lectures")
    if len(train_items) < 5:
        raise RuntimeError(f"Only {len(train_items)} usable training lectures were loaded; need at least 5")

    checkpoint_decoder = load_checkpoint_decoder(CHECKPOINT_DIR / "temporal_v4_best_model.pt")
    baseline_val = score_v4_baseline(val_items, checkpoint_decoder)
    print("\nCURRENT V4 VALIDATION BASELINE")
    for name, r in baseline_val["per_lecture"].items():
        print(f"  {name:<30} {r['correct']:>4}/{r['total']:<4} {100 * r['accuracy']:.2f}%")
    print(f"  macro={100 * baseline_val['macro']:.2f}% weighted={100 * baseline_val['weighted']:.2f}%")

    ensemble_states = []
    seed_reports = []
    for seed in args.seeds:
        print("\n" + "-" * 80)
        print(f"Training V11 member seed={seed}")
        state, report = train_one_seed(
            train_items, val_items, feature_names, seed,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            checkpoint_decoder=checkpoint_decoder,
        )
        ensemble_states.append(state)
        seed_reports.append(report)

    # Lock the ensemble first; only then search the V11 decoder on validation.
    val_emissions = {
        item["lecture"]: predict_emissions(
            ensemble_states, len(feature_names), feature_names, item["X_seq"], batch_size=128
        )
        for item in val_items
    }
    selected_decoder, selected_eval, sweep = tune_decoder_validation(val_items, val_emissions, checkpoint_decoder)
    tuned_val = evaluate_v11(val_items, val_emissions, selected_decoder)
    val_improvement = tuned_val["macro"] - baseline_val["macro"]
    use_v11 = val_improvement >= args.minimum_validation_gain

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "LectureSense-V11-Supervised-Candidate-Reranking",
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "final_rank_idx": feature_names.index("final_rank"),
        "visual_rank_idx": feature_names.index("visual_rank"),
        "models": ensemble_states,
        "seeds": args.seeds,
        "train_lectures_used": [x["lecture"] for x in train_items],
        "validation_lectures": [x["lecture"] for x in val_items],
        "test_lectures": TEST_LECTURES,
        "decoder": selected_decoder,
        "model_selected_over_v4_on_validation": bool(use_v11),
        "baseline_validation": {"macro": baseline_val["macro"], "weighted": baseline_val["weighted"], "per_lecture": baseline_val["per_lecture"]},
        "v11_validation": {"macro": tuned_val["macro"], "weighted": tuned_val["weighted"], "per_lecture": tuned_val["per_lecture"]},
        "validation_macro_gain": float(val_improvement),
        "validation_only_selection": True,
        "test_used_for_training_or_selection": False,
        "warning": "The project test split has been evaluated in earlier iterations; treat this as a fixed comparison rather than a pristine first-use test estimate.",
    }
    torch.save(payload, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({k: v for k, v in payload.items() if k not in {"models"}}, indent=2, default=float), encoding="utf-8")

    val_rows = []
    for item in val_items:
        v11r = tuned_val["per_lecture"][item["lecture"]]
        b = baseline_val["per_lecture"][item["lecture"]]
        val_rows.append({
            "lecture": item["lecture"],
            "v4_correct": b["correct"], "total": b["total"], "v4_accuracy": b["accuracy"],
            "v11_correct": v11r["correct"], "v11_accuracy": v11r["accuracy"],
            "change_pp": 100 * (v11r["accuracy"] - b["accuracy"]),
        })
    pd.DataFrame(val_rows).to_csv(VALIDATION_CSV, index=False)
    validation_report = {
        "baseline_v4": baseline_val,
        "v11": tuned_val,
        "selected_decoder": selected_decoder,
        "decoder_sweep_top_10": sweep[:10],
        "seed_reports": [{k: v for k, v in r.items() if k != "best_validation"} for r in seed_reports],
        "model_selected_over_v4": bool(use_v11),
        "minimum_validation_gain_required": args.minimum_validation_gain,
        "train_lectures_used": [x["lecture"] for x in train_items],
        "validation_lectures": [x["lecture"] for x in val_items],
        "test_used": False,
    }
    VALIDATION_REPORT.write_text(json.dumps(validation_report, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 88)
    print("V11 VALIDATION SUMMARY")
    print(f"V4 baseline macro : {100 * baseline_val['macro']:.2f}%")
    print(f"V11 tuned macro   : {100 * tuned_val['macro']:.2f}%")
    print(f"V4 baseline weighted: {100 * baseline_val['weighted']:.2f}%")
    print(f"V11 tuned weighted  : {100 * tuned_val['weighted']:.2f}%")
    print(f"Macro gain: {100 * val_improvement:+.2f} percentage points")
    print(f"V11 selected for testing based on validation: {'YES' if use_v11 else 'NO — preserve V4 as selected system'}")
    print(f"Saved model : {MODEL_PATH}")
    print(f"Saved config: {CONFIG_PATH}")
    print(f"Saved report: {VALIDATION_REPORT}")
    print("Test data/labels were not loaded in train mode.")


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"V11 model not found: {path}. Run --mode train first.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_or_validate_mode(args, split: str) -> None:
    bundle = load_bundle(MODEL_PATH)
    v4_root = Path(args.v4_root).resolve()
    v7.V4_ROOT = v4_root
    decoder = dict(bundle["decoder"])
    feature_names = list(bundle["feature_names"])
    model_states = bundle["models"]
    use_v11 = bool(bundle.get("model_selected_over_v4_on_validation", True))
    checkpoint_decoder = load_checkpoint_decoder(CHECKPOINT_DIR / "temporal_v4_best_model.pt")
    lectures = VAL_LECTURES if split == "validation" else TEST_LECTURES
    items = load_split(lectures, split, v4_root, allow_skip=False)
    for item in items:
        if item["feature_names"] != feature_names:
            raise RuntimeError(f"Feature-version mismatch in {item['lecture']}; retrain V11 with the current script")

    emissions = {
        item["lecture"]: predict_emissions(model_states, len(feature_names), feature_names, item["X_seq"], batch_size=128)
        for item in items
    }
    v11_res = evaluate_v11(items, emissions, decoder)
    baseline = score_v4_baseline(items, checkpoint_decoder)
    rows = []
    for item in items:
        name = item["lecture"]
        v4r = baseline["per_lecture"][name]
        v11r = v11_res["per_lecture"][name]
        rows.append({
            "lecture": name,
            "v4_correct": v4r["correct"], "total": v4r["total"], "v4_accuracy": v4r["accuracy"],
            "v11_correct": v11r["correct"], "v11_accuracy": v11r["accuracy"],
            "change_pp": 100 * (v11r["accuracy"] - v4r["accuracy"]),
        })
    out_csv = VALIDATION_CSV if split == "validation" else TEST_CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    report = {
        "split": split,
        "v4_baseline": baseline,
        "v11": v11_res,
        "decoder": decoder,
        "model_selected_over_v4_on_validation": use_v11,
        "selected_system": "V11" if use_v11 else "V4",
        "test_labels_used_for_training_or_selection": False,
        "project_test_split_caveat": "Test lectures were evaluated in earlier project iterations, so this is not a pristine first-use test estimate.",
    }
    report_path = VALIDATION_REPORT if split == "validation" else TEST_REPORT
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"LECTURESENSE V11 — {split.upper()} RESULTS")
    print(f"{'lecture':<36}{'V4 baseline':>15}{'V11':>15}{'change':>12}")
    for row in rows:
        print(f"{row['lecture']:<36}{100*row['v4_accuracy']:>14.2f}%{100*row['v11_accuracy']:>14.2f}%{row['change_pp']:>+11.2f}pp")
    print(f"Macro accuracy   V4={100*baseline['macro']:.2f}% | V11={100*v11_res['macro']:.2f}%")
    print(f"Weighted accuracy V4={100*baseline['weighted']:.2f}% | V11={100*v11_res['weighted']:.2f}%")
    if split == "test":
        print(f"V11 selected by validation? {'YES' if use_v11 else 'NO; selected system remains V4'}")
        print("No parameters were searched or changed on test data.")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="LectureSense V11 supervised reranker; train/validation/test separated.")
    parser.add_argument("--mode", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--v4-root", default=str(DEFAULT_V4_ROOT), help="Primary cache root with one V4 output folder per lecture")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--minimum-validation-gain", type=float, default=0.005,
                        help="Require this macro-accuracy gain over V4 on validation before selecting V11")
    args = parser.parse_args()
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "validate":
        test_or_validate_mode(args, "validation")
    else:
        test_or_validate_mode(args, "test")


if __name__ == "__main__":
    main()
