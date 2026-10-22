import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from dp_alignment import dp_align
from inference import read_mavils_srt

SCORE_PATH = r"runs\session\score_matrix.npy"
TRANSCRIPT_PATH = r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
GT_PATH = r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"

scores = np.load(SCORE_PATH).astype(np.float32)
segments = read_mavils_srt(TRANSCRIPT_PATH)
gt = pd.read_excel(GT_PATH)

starts = np.array([x["start"] for x in segments], dtype=float)


def predict_slide_at_time(path, t):
    idx = np.searchsorted(starts, float(t), side="right") - 1
    idx = max(0, min(idx, len(path) - 1))
    return int(path[idx]) + 1


def row_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mean) / std


def row_softmax(x, temperature=1.0):
    z = x / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
    return np.log(p + 1e-12)


variants = {
    "RAW": scores,
    "ZSCORE": row_zscore(scores),
    "SOFTMAX_T0.5": row_softmax(scores, 0.5),
    "SOFTMAX_T1.0": row_softmax(scores, 1.0),
    "SOFTMAX_T2.0": row_softmax(scores, 2.0),
}

penalties = [0.05, 0.10, 0.20, 0.35]

results = []

for name, matrix in variants.items():

    for penalty in penalties:

        path = dp_align(
            matrix,
            jump_penalty=penalty,
            max_forward_jump=2,
        )

        correct = 0
        total = 0

        for _, row in gt.iterrows():

            true_slide = int(row["Slidenumber"])

            if true_slide < 1:
                continue

            pred = predict_slide_at_time(
                path,
                float(row["Time"])
            )

            total += 1

            if pred == true_slide:
                correct += 1

        accuracy = correct / total

        switches = []
        previous = path[0]

        for i in range(1, len(path)):
            if path[i] != previous:
                switches.append(
                    (
                        segments[i]["start"],
                        previous + 1,
                        path[i] + 1,
                    )
                )
                previous = path[i]

        results.append({
            "name": name,
            "penalty": penalty,
            "accuracy": accuracy,
            "path": path,
            "switches": switches,
        })


results.sort(
    key=lambda x: x["accuracy"],
    reverse=True
)

print()
print("=" * 80)
print("ROW-NORMALIZATION DP TEST")
print("=" * 80)

for r in results:
    print(
        f"{r['name']:16s} | "
        f"Penalty={r['penalty']:<4} | "
        f"Accuracy={r['accuracy']:.4f} "
        f"({r['accuracy']*100:.2f}%)"
    )


best = results[0]

print()
print("=" * 80)
print("BEST DIAGNOSTIC VARIANT")
print("=" * 80)

print("Method   :", best["name"])
print("Penalty  :", best["penalty"])
print(
    "Accuracy :",
    f"{best['accuracy']*100:.2f}%"
)

print()
print("Slide transitions 400s-650s:")

for t, old, new in best["switches"]:
    if 400 <= t <= 650:
        print(
            f"{t:8.2f}s : Slide {old} -> Slide {new}"
        )

print()
print("Important timestamps:")

for t in [474.26, 503.50, 520.31, 566.74, 574.0, 580.0]:

    pred = predict_slide_at_time(
        best["path"],
        t
    )

    print(
        f"{t:8.2f}s -> Slide {pred}"
    )

print()
print("This is a diagnostic experiment only.")
print("Do not replace the final decoder based only on this lecture.")
