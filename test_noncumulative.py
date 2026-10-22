import numpy as np
import pandas as pd

SCORE_PATH = r"runs\session\score_matrix.npy"
TRANSCRIPT_PATH = r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
GT_PATH = r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"

import sys
sys.path.insert(0, "src")

from inference import read_mavils_srt

scores = np.load(SCORE_PATH).astype(np.float32)
segments = read_mavils_srt(TRANSCRIPT_PATH)
gt = pd.read_excel(GT_PATH)

starts = np.array([x["start"] for x in segments], dtype=float)

# Local best slide for each transcript segment.
local_path = np.argmax(scores, axis=1).astype(np.int32)

# Enforce monotonic order without accumulating scores over time.
monotonic_path = np.maximum.accumulate(local_path)

# Optional: prevent a single segment from jumping over multiple slides.
limited_path = np.zeros_like(local_path)
limited_path[0] = local_path[0]

for i in range(1, len(local_path)):
    previous = limited_path[i - 1]
    current = local_path[i]

    if current < previous:
        current = previous

    current = min(current, previous + 1)
    limited_path[i] = current


def predict(path, t):
    idx = np.searchsorted(starts, float(t), side="right") - 1
    idx = max(0, min(idx, len(path) - 1))
    return int(path[idx]) + 1


def evaluate(name, path):
    correct = 0
    total = 0

    for _, row in gt.iterrows():
        true_slide = int(row["Slidenumber"])

        if true_slide < 1:
            continue

        pred = predict(path, float(row["Time"]))
        total += 1

        if pred == true_slide:
            correct += 1

    acc = correct / total

    print(
        f"{name}: {acc:.4f} ({acc * 100:.2f}%)"
    )


def print_region(name, path):
    print()
    print(name)

    previous = path[0]

    for i in range(1, len(path)):
        if path[i] != previous:
            t = segments[i]["start"]

            if 100 <= t <= 2200:
                print(
                    f"{t:8.2f}s : "
                    f"Slide {previous + 1} -> "
                    f"Slide {path[i] + 1}"
                )

            previous = path[i]


print("=" * 70)
print("NON-CUMULATIVE MONOTONIC DECODER TEST")
print("=" * 70)

evaluate("Local Argmax", local_path)
evaluate("Monotonic Argmax", monotonic_path)
evaluate("Monotonic + Max 1 Slide Jump", limited_path)

print_region(
    "MONOTONIC ARGMAX TRANSITIONS",
    monotonic_path
)

print_region(
    "LIMITED-JUMP TRANSITIONS",
    limited_path
)

print()
print("Important timestamps:")

for t in [
    163.56,
    315.48,
    474.26,
    503.50,
    566.74,
    744.93,
    843.00,
    929.23,
    1152.29,
    1338.06,
    1496.04,
    2081.45,
]:
    print(
        f"{t:8.2f}s | "
        f"Local={predict(local_path, t):2d} | "
        f"Mono={predict(monotonic_path, t):2d} | "
        f"Limited={predict(limited_path, t):2d}"
    )
