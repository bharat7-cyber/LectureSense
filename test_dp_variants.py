import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from dp_alignment import dp_align
from inference import read_mavils_srt

SCORE_PATH = r"runs\session\score_matrix.npy"
TRANSCRIPT_PATH = r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
GT_PATH = r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"

scores = np.load(SCORE_PATH)
segments = read_mavils_srt(TRANSCRIPT_PATH)
gt = pd.read_excel(GT_PATH)

starts = np.array([x["start"] for x in segments], dtype=float)

def predict_slide_at_time(path, t):
    idx = np.searchsorted(starts, float(t), side="right") - 1
    idx = max(0, min(idx, len(path) - 1))
    return int(path[idx]) + 1

def smooth_scores(matrix, window):
    if window <= 1:
        return matrix.copy()

    half = window // 2
    padded = np.pad(
        matrix,
        ((half, half), (0, 0)),
        mode="edge"
    )

    out = np.zeros_like(matrix, dtype=np.float32)

    for i in range(len(matrix)):
        out[i] = padded[i:i + window].mean(axis=0)

    return out

results = []

penalties = [0.10, 0.20, 0.35, 0.50, 0.70]
max_jumps = [1, 2]
windows = [1, 3, 5]

for window in windows:

    test_scores = smooth_scores(scores, window)

    for max_jump in max_jumps:

        for penalty in penalties:

            path = dp_align(
                test_scores,
                jump_penalty=penalty,
                max_forward_jump=max_jump,
            )

            correct = 0
            total = 0

            for _, row in gt.iterrows():

                t = float(row["Time"])
                true_slide = int(row["Slidenumber"])

                if true_slide < 1:
                    continue

                pred_slide = predict_slide_at_time(path, t)

                total += 1

                if pred_slide == true_slide:
                    correct += 1

            accuracy = correct / total if total else 0.0

            timeline_switches = []

            previous = path[0]

            for i in range(1, len(path)):

                if path[i] != previous:

                    timeline_switches.append(
                        (
                            round(segments[i]["start"], 2),
                            previous + 1,
                            path[i] + 1
                        )
                    )

                    previous = path[i]

            results.append(
                {
                    "accuracy": accuracy,
                    "window": window,
                    "max_jump": max_jump,
                    "penalty": penalty,
                    "path": path,
                    "switches": timeline_switches,
                }
            )

results.sort(
    key=lambda x: x["accuracy"],
    reverse=True
)

print()
print("=" * 75)
print("DP VARIANT TEST")
print("=" * 75)

for r in results:

    print(
        f"Accuracy={r['accuracy']:.4f} | "
        f"Window={r['window']} | "
        f"MaxJump={r['max_jump']} | "
        f"Penalty={r['penalty']}"
    )

print()
print("=" * 75)
print("BEST VARIANT")
print("=" * 75)

best = results[0]

print(
    f"Accuracy : {best['accuracy']:.4f} "
    f"({best['accuracy'] * 100:.2f}%)"
)

print(f"Window   : {best['window']}")
print(f"MaxJump  : {best['max_jump']}")
print(f"Penalty  : {best['penalty']}")

print()
print("Switches around Slide 6-8 region:")

for t, old, new in best["switches"]:
    if 400 <= t <= 650:
        print(
            f"{t:8.2f}s : Slide {old} -> Slide {new}"
        )

print()
print("Predictions at important timestamps:")

for t in [474.26, 503.50, 520.31, 566.74, 574.0, 580.0]:

    slide = predict_slide_at_time(
        best["path"],
        t
    )

    print(
        f"{t:8.2f}s -> Slide {slide}"
    )

print()
print("NOTE: This test is for decoder diagnosis only.")
print("Do NOT use this lecture-specific tuning as the final reported test result.")
