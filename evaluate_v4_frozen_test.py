from __future__ import annotations

"""Evaluate one fixed LectureSense V4 score-fusion/decoder configuration.

This script does not train, tune, or write to the V4 model checkpoint. It uses
cached test score matrices and reports a comparison against the current V4
cached final-score matrix. Do not use the result to tune the configuration.

Run from the project root:
    python evaluate_v4_frozen_test.py

If test caches are stored elsewhere:
    python evaluate_v4_frozen_test.py --v4-root "runs\\formal_test_eval_v4"
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import inference_temporal_v4 as infer_v4
    from boundary_dp_v4 import boundary_aware_dp
    from video_visual_alignment import fuse_model_and_video_scores
except Exception as exc:
    raise SystemExit(
        "Could not import LectureSense V4 modules from src/. Run this script "
        "from the project root containing src/.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    )

DATA_ROOT = ROOT / "data" / "MaViLS" / "data"
DEFAULT_V4_ROOT = ROOT / "runs" / "formal_test_eval_v4"
CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
OUTPUT_DIR = ROOT / "runs" / "v4_frozen_test"
REPORT_JSON = OUTPUT_DIR / "evaluation_report.json"
REPORT_CSV = OUTPUT_DIR / "per_lecture_results.csv"

# Frozen config supplied before this evaluation; this script performs no search.
FIXED_FUSION = {
    "residual_scale": 0.50,
    "video_weight": 0.50,
    "score_temperature": 3.00,
}
FIXED_DECODER_OVERRIDES = {
    "jump_penalty": 0.40,
    "max_duration_seconds": 90.0,
    "duration_penalty": 0.01,
    "max_run_seconds": 900.0,
}

# Use the official held-out test lecture SRTs and ground-truth workbooks.
TEST_LECTURES = {
    "short_range": {
        "srt": "short_range_mit.srt",
        "gt": "ground_truth_short_range.xlsx",
        "folder_candidates": ("short_range_mit", "short_range"),
    },
    "solar_resource": {
        "srt": "solar_resource.srt",
        "gt": "ground_truth_solar_resource.xlsx",
        "folder_candidates": ("solar_resource",),
    },
    "team_dynamics_game_design_MIT": {
        "srt": "team_dynamics_game_design_MIT.srt",
        "gt": "ground_truth_team_dynamics_game_design_MIT.xlsx",
        "folder_candidates": ("team_dynamics_game_design_MIT", "team_dynamics"),
    },
    "theory_of_computation": {
        "srt": "theory_of_computation_MIT.srt",
        "gt": "ground_truth_theory_of_computation.xlsx",
        "folder_candidates": ("theory_of_computation",),
    },
}


def norm_col(value) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def find_col(columns, candidates, contains=None):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if contains:
        for col in columns:
            if contains in col:
                return col
    return None


def to_seconds(value) -> float:
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
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return float("nan")
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth workbook not found: {path}")
    df = pd.read_excel(path)
    df.columns = [norm_col(col) for col in df.columns]
    time_col = find_col(df.columns, ["time", "timestamp", "start", "key"], contains="time")
    slide_col = find_col(
        df.columns,
        ["slidenumber", "slide_number", "slide", "slideid", "slide_id"],
        contains="slide",
    )
    if time_col is None or slide_col is None:
        raise RuntimeError(
            f"Could not identify timestamp/slide columns in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df[[time_col, slide_col]].copy()
    out.columns = ["time", "gt"]
    out["time"] = out["time"].map(to_seconds)
    out["gt"] = pd.to_numeric(out["gt"], errors="coerce")
    out = out.dropna()
    out = out[out["gt"] > 0]  # project convention: slide IDs <= 0 are invalid
    out["gt"] = out["gt"].astype(int)
    out = out.sort_values("time")
    if out.empty:
        raise RuntimeError(f"No valid ground-truth rows in {path}")
    return out.reset_index(drop=True)


def load_transcript(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Transcript SRT not found: {path}")
    segments = infer_v4.base.read_mavils_srt(str(path))
    if not segments:
        raise RuntimeError(f"No transcript segments read from {path}")
    starts = np.asarray(
        [float(seg.get("start", seg.get("start_time", 0.0))) for seg in segments],
        dtype=np.float32,
    )
    ends = np.asarray(
        [float(seg.get("end", seg.get("end_time", 0.0))) for seg in segments],
        dtype=np.float32,
    )
    if np.any(ends < starts) or np.any(np.diff(starts) < 0):
        raise RuntimeError(f"Transcript timestamps are invalid in {path}")
    return starts, ends


def map_gt_to_transcript(gt_df, starts, ends):
    indices, labels, times = [], [], []
    for timestamp, slide_id in zip(gt_df["time"].to_numpy(float), gt_df["gt"].to_numpy(int)):
        idx = int(np.searchsorted(starts, timestamp, side="right") - 1)
        idx = min(max(idx, 0), len(starts) - 1)
        if timestamp > float(ends[idx]) and idx + 1 < len(starts):
            if abs(float(starts[idx + 1]) - timestamp) < abs(timestamp - float(ends[idx])):
                idx += 1
        indices.append(idx)
        labels.append(int(slide_id) - 1)  # workbook IDs are one-based; model indices are zero-based
        times.append(float(timestamp))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times, dtype=np.float32),
    )


def load_matrix(folder: Path, names, required=False):
    for name in names:
        path = folder / name
        if path.exists():
            try:
                return np.load(path).astype(np.float32)
            except Exception as exc:
                raise RuntimeError(f"Could not load matrix {path}: {exc}") from exc
    if required:
        raise FileNotFoundError(
            f"Required matrix missing in {folder}. Tried: {', '.join(names)}"
        )
    return None


def resolve_cache_folder(v4_root: Path, candidates) -> Path:
    for name in candidates:
        candidate = v4_root / name
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(v4_root / name) for name in candidates)
    raise FileNotFoundError(f"Could not find a test cache folder. Tried: {tried}")


def load_checkpoint_decoder(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"V4 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
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
    defaults.update(checkpoint.get("decoder_config", {}))
    del checkpoint
    return defaults


def load_lecture(name: str, spec: dict, v4_root: Path) -> dict:
    cache_dir = resolve_cache_folder(v4_root, spec["folder_candidates"])
    srt_path = DATA_ROOT / "audioscripts" / spec["srt"]
    gt_path = DATA_ROOT / "ground_truth_files" / spec["gt"]

    print(f"\n{'=' * 78}\nLOADING {name}\n{'=' * 78}")
    print(f"Cache folder: {cache_dir}")
    starts, ends = load_transcript(srt_path)
    gt_df = load_ground_truth(gt_path)
    gt_idx, gt_labels, gt_times = map_gt_to_transcript(gt_df, starts, ends)

    base_scores = load_matrix(cache_dir, ["v4_base_score_matrix.npy", "base_score_matrix.npy"], required=True)
    adjusted = load_matrix(
        cache_dir, ["v4_adjusted_score_matrix.npy", "adjusted_score_matrix.npy"], required=True
    )
    video = load_matrix(
        cache_dir, ["v4_video_score_matrix.npy", "video_score_matrix.npy"], required=True
    )
    boundary = load_matrix(
        cache_dir, ["v4_boundary_probability.npy", "boundary_probability.npy"], required=True
    ).reshape(-1)
    next_logits = load_matrix(
        cache_dir, ["v4_next_slide_logits.npy", "next_slide_logits.npy"], required=True
    )
    final = load_matrix(
        cache_dir, ["v4_final_score_matrix.npy", "final_score_matrix.npy", "fused_score_matrix.npy"]
    )

    if base_scores.shape != adjusted.shape or base_scores.shape != video.shape:
        raise RuntimeError(
            f"Score matrix shape mismatch in {name}: base={base_scores.shape}, "
            f"adjusted={adjusted.shape}, video={video.shape}"
        )
    n, n_slides = adjusted.shape
    if len(starts) != n or len(ends) != n:
        raise RuntimeError(
            f"Transcript/matrix row mismatch in {name}: transcript={len(starts)}, scores={n}"
        )
    if boundary.shape != (n,):
        raise RuntimeError(f"Boundary matrix shape mismatch in {name}: {boundary.shape}, expected {(n,)}")
    if next_logits.shape != (n, n_slides):
        raise RuntimeError(
            f"Next-slide logits shape mismatch in {name}: {next_logits.shape}, expected {(n, n_slides)}"
        )
    if final is not None and final.shape != (n, n_slides):
        print(f"Warning: ignoring mismatched cached final matrix in {name}: {final.shape}")
        final = None

    keep = (gt_labels >= 0) & (gt_labels < n_slides) & (gt_idx < n)
    gt_idx, gt_labels, gt_times = gt_idx[keep], gt_labels[keep], gt_times[keep]
    if len(gt_labels) == 0:
        raise RuntimeError(f"No usable ground-truth rows for {name}")

    print(f"Transcript segments : {n}")
    print(f"Slide count         : {n_slides}")
    print(f"Ground-truth rows   : {len(gt_labels)}")
    print(f"Score matrix        : {adjusted.shape}")
    print("Cached video/boundary/next logits: present and shape-checked")

    return {
        "name": name,
        "starts": starts,
        "ends": ends,
        "base": base_scores,
        "adjusted": adjusted,
        "video": video,
        "final": final,
        "boundary": np.clip(boundary, 1e-4, 1.0 - 1e-4),
        "next_logits": next_logits,
        "gt_indices": gt_idx,
        "gt_labels": gt_labels,
        "gt_times": gt_times,
        "slides": n_slides,
        "cache_dir": str(cache_dir),
    }


def create_fixed_scores(item) -> np.ndarray:
    # Scale only the learned V4 residual, retaining the frozen base scores.
    v4_scores = item["base"] + FIXED_FUSION["residual_scale"] * (
        item["adjusted"] - item["base"]
    )
    fused = fuse_model_and_video_scores(
        v4_scores.astype(np.float32),
        item["video"].astype(np.float32),
        video_weight=FIXED_FUSION["video_weight"],
    )
    temperature = max(1e-3, FIXED_FUSION["score_temperature"])
    return (np.asarray(fused, dtype=np.float32) / np.float32(temperature)).astype(np.float32)


def decoder_kwargs(config: dict) -> dict:
    keys = (
        "jump_penalty", "max_forward_jump", "max_duration_seconds", "duration_penalty",
        "max_run_seconds", "boundary_reward", "boundary_stay_penalty",
        "transition_logits_weight", "premature_transition_penalty", "boundary_threshold",
    )
    return {key: config[key] for key in keys}


def evaluate_path(item, scores: np.ndarray, decoder: dict):
    path = boundary_aware_dp(
        scores,
        item["boundary"],
        item["next_logits"],
        item["starts"],
        item["ends"],
        **decoder_kwargs(decoder),
    )
    path = np.asarray(path, dtype=np.int64)
    total = len(item["gt_labels"])
    correct = int(np.sum(path[item["gt_indices"]] == item["gt_labels"]))
    return correct, total


def aggregate(results: dict) -> dict:
    correct = sum(row["correct"] for row in results.values())
    total = sum(row["total"] for row in results.values())
    macro = float(np.mean([row["accuracy"] for row in results.values()])) if results else 0.0
    return {
        "correct": int(correct),
        "total": int(total),
        "weighted_accuracy": float(correct / max(1, total)),
        "macro_accuracy": macro,
    }


def accuracy_label(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed V4 fusion/decoder configuration on cached held-out test outputs."
    )
    parser.add_argument(
        "--v4-root",
        default=str(DEFAULT_V4_ROOT),
        help="Directory containing per-lecture test cache folders (default: runs/formal_test_eval_v4).",
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT), help="Read-only V4 checkpoint path.")
    args = parser.parse_args()

    v4_root = Path(args.v4_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    print("LectureSense V4 — frozen test configuration")
    print(f"Test cache root : {v4_root}")
    print(f"Checkpoint      : {checkpoint_path}")
    print("No training, search, or parameter tuning will be performed.\n")

    decoder = load_checkpoint_decoder(checkpoint_path)
    decoder.update(FIXED_DECODER_OVERRIDES)
    print("Fixed score-fusion configuration:")
    print(json.dumps(FIXED_FUSION, indent=2))
    print("Decoder configuration (specified overrides applied; other values read from checkpoint):")
    print(json.dumps(decoder_kwargs(decoder), indent=2))

    dataset = {
        name: load_lecture(name, spec, v4_root)
        for name, spec in TEST_LECTURES.items()
    }

    baseline_results = {}
    fixed_results = {}
    csv_rows = []
    for name, item in dataset.items():
        # Production V4 baseline: cached final matrix produced at the original video weight.
        # If that optional matrix is absent, reconstruct the original 1.0 residual / 0.4 video fusion.
        if item["final"] is not None:
            baseline_scores = item["final"]
        else:
            base_production = item["base"] + (item["adjusted"] - item["base"])
            baseline_scores = fuse_model_and_video_scores(
                base_production.astype(np.float32), item["video"], video_weight=0.40
            )

        baseline_decoder = load_checkpoint_decoder(checkpoint_path)
        baseline_decoder.update({
            "jump_penalty": 0.05,
            "max_duration_seconds": 120.0,
            "duration_penalty": 0.003,
        })
        fixed_scores = create_fixed_scores(item)

        base_correct, total = evaluate_path(item, baseline_scores, baseline_decoder)
        fixed_correct, fixed_total = evaluate_path(item, fixed_scores, decoder)
        if total != fixed_total:
            raise RuntimeError(f"Internal metric mismatch on {name}")

        baseline_row = {"correct": base_correct, "total": total, "accuracy": base_correct / total}
        fixed_row = {"correct": fixed_correct, "total": total, "accuracy": fixed_correct / total}
        baseline_results[name] = baseline_row
        fixed_results[name] = fixed_row
        csv_rows.append({
            "lecture": name,
            "baseline_correct": base_correct,
            "total": total,
            "baseline_accuracy": baseline_row["accuracy"],
            "fixed_config_correct": fixed_correct,
            "fixed_config_accuracy": fixed_row["accuracy"],
            "change_percentage_points": 100.0 * (fixed_row["accuracy"] - baseline_row["accuracy"]),
            "cache_dir": item["cache_dir"],
        })
        print(
            f"\n{name:<34} baseline {base_correct:>4}/{total:<4} {accuracy_label(baseline_row['accuracy']):>8}"
            f" | fixed {fixed_correct:>4}/{total:<4} {accuracy_label(fixed_row['accuracy']):>8}"
            f" | change {100.0 * (fixed_row['accuracy'] - baseline_row['accuracy']):+.2f} pp"
        )

    base_agg = aggregate(baseline_results)
    fixed_agg = aggregate(fixed_results)
    print("\n" + "=" * 92)
    print(f"CURRENT V4 baseline weighted accuracy : {accuracy_label(base_agg['weighted_accuracy'])} "
          f"({base_agg['correct']}/{base_agg['total']})")
    print(f"CURRENT V4 baseline macro accuracy    : {accuracy_label(base_agg['macro_accuracy'])}")
    print(f"FIXED CONFIG weighted accuracy       : {accuracy_label(fixed_agg['weighted_accuracy'])} "
          f"({fixed_agg['correct']}/{fixed_agg['total']})")
    print(f"FIXED CONFIG macro accuracy          : {accuracy_label(fixed_agg['macro_accuracy'])}")
    print(f"Weighted change                      : "
          f"{100.0 * (fixed_agg['weighted_accuracy'] - base_agg['weighted_accuracy']):+.2f} pp")
    print(f"Macro change                         : "
          f"{100.0 * (fixed_agg['macro_accuracy'] - base_agg['macro_accuracy']):+.2f} pp")
    print("\nThis script did not search for parameters. Do not tune using these test results.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(REPORT_CSV, index=False)
    report = {
        "model": "LectureSense V4",
        "evaluation_type": "fixed_configuration_test_evaluation",
        "parameter_search_performed": False,
        "checkpoint": str(checkpoint_path),
        "test_cache_root": str(v4_root),
        "fixed_fusion": FIXED_FUSION,
        "fixed_decoder": decoder_kwargs(decoder),
        "baseline": {**base_agg, "per_lecture": baseline_results},
        "fixed_config": {**fixed_agg, "per_lecture": fixed_results},
        "warning": (
            "The test split has been evaluated in earlier project iterations. Treat this as a fixed-config "
            "comparison, not a pristine first-use test estimate. Do not change parameters based on this report."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"Saved CSV report : {REPORT_CSV}")
    print(f"Saved JSON report: {REPORT_JSON}")


if __name__ == "__main__":
    main()
