"""Evaluate different inference-time video fusion weights on saved matrices.

This script does not rerun video frame extraction. It expects:
  runs/session/score_matrix.npy
  runs/session/video_score_matrix.npy
  the MaViLS transcript and GT XLSX used for the lecture.

The reported accuracy is a diagnostic on the supplied lecture, not an
unbiased held-out test metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from dp_alignment import dp_align
from inference import read_mavils_srt
from video_visual_alignment import fuse_model_and_video_scores


SCORE_PATH = Path(r"runs\session\score_matrix.npy")
VIDEO_SCORE_PATH = Path(r"runs\session\video_score_matrix.npy")
TRANSCRIPT_PATH = Path(
    r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
)
GT_PATH = Path(
    r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"
)


model_scores = np.load(SCORE_PATH).astype(np.float32)
video_scores = np.load(VIDEO_SCORE_PATH).astype(np.float32)
segments = read_mavils_srt(TRANSCRIPT_PATH)
gt = pd.read_excel(GT_PATH)

if model_scores.shape != video_scores.shape:
    raise RuntimeError(
        f"Shape mismatch: model={model_scores.shape}, "
        f"video={video_scores.shape}"
    )

if model_scores.shape[0] != len(segments):
    raise RuntimeError(
        "Score matrices and transcript segment count do not match."
    )

starts = np.asarray(
    [item["start"] for item in segments],
    dtype=np.float32,
)
ends = np.asarray(
    [item["end"] for item in segments],
    dtype=np.float32,
)


def predict_slide(path: list[int], timestamp: float) -> int:
    index = np.searchsorted(
        starts,
        float(timestamp),
        side="right",
    ) - 1
    index = max(0, min(index, len(path) - 1))
    return int(path[index]) + 1


def evaluate(path: list[int]) -> float:
    correct = 0
    total = 0

    for _, row in gt.iterrows():
        true_slide = int(row["Slidenumber"])
        if true_slide < 1:
            continue

        predicted = predict_slide(path, float(row["Time"]))
        total += 1
        correct += int(predicted == true_slide)

    return correct / total if total else 0.0


weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
results = []

for weight in weights:
    fused = fuse_model_and_video_scores(
        model_scores,
        video_scores,
        video_weight=weight,
    )

    path = dp_align(
        fused,
        jump_penalty=0.05,
        max_forward_jump=1,
        segment_starts=starts,
        segment_ends=ends,
        max_duration_seconds=120.0,
        duration_penalty=0.01,
        max_run_seconds=480.0,
    )

    accuracy = evaluate(path)

    results.append(
        (accuracy, weight, path)
    )

results.sort(reverse=True, key=lambda x: x[0])

print("=" * 75)
print("VIDEO FUSION WEIGHT SWEEP")
print("=" * 75)

for accuracy, weight, _ in results:
    print(
        f"VideoWeight={weight:.2f} | "
        f"Accuracy={accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

best_accuracy, best_weight, best_path = results[0]

print()
print("=" * 75)
print("BEST DIAGNOSTIC CONFIGURATION")
print("=" * 75)
print(
    f"Video weight: {best_weight:.2f}"
)
print(
    f"Accuracy:     {best_accuracy:.4f} "
    f"({best_accuracy * 100:.2f}%)"
)

print()
print("Important timestamps:")
for timestamp in [474.26, 566.74, 744.93, 843.00, 929.23]:
    print(
        f"{timestamp:8.2f}s -> "
        f"Slide {predict_slide(best_path, timestamp)}"
    )

print()
print("Transitions 400s-650s:")
previous = best_path[0]
for i in range(1, len(best_path)):
    current = best_path[i]
    if current != previous:
        timestamp = float(starts[i])
        if 400.0 <= timestamp <= 650.0:
            print(
                f"{timestamp:8.2f}s : "
                f"Slide {previous + 1} -> Slide {current + 1}"
            )
        previous = current

print()
print("MONOTONIC:", bool(np.all(np.diff(best_path) >= 0)))
print()
print("This is diagnostic tuning on one lecture. Do not report the")
print("selected weight as an unbiased test result.")
