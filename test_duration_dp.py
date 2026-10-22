import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from inference import read_mavils_srt

SCORE_PATH = r"runs\session\score_matrix.npy"
TRANSCRIPT_PATH = r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
GT_PATH = r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"

scores = np.load(SCORE_PATH).astype(np.float32)
segments = read_mavils_srt(TRANSCRIPT_PATH)
gt = pd.read_excel(GT_PATH)

starts = np.array([x["start"] for x in segments], dtype=float)
ends = np.array([x["end"] for x in segments], dtype=float)


def predict(path, t):
    idx = np.searchsorted(starts, float(t), side="right") - 1
    idx = max(0, min(idx, len(path) - 1))
    return int(path[idx]) + 1


def duration_dp(scores, starts, ends,
                 jump_penalty=0.05,
                 duration_penalty=0.003,
                 max_duration=240.0):

    n, m = scores.shape

    dp = np.full((n, m), -np.inf, dtype=np.float32)
    back = np.zeros((n, m), dtype=np.int32)

    dp[0] = scores[0]

    for i in range(1, n):

        current_time = ends[i]

        for slide in range(m):

            best = -np.inf
            best_prev = slide

            for prev in range(slide, min(m, slide + 2)):

                transition = 0.0

                if prev != slide:
                    transition = jump_penalty

                # Approximate how long this state has already persisted.
                # Search backwards until another slide state appears.
                run_start = i - 1

                while run_start > 0 and back[run_start, slide] == slide:
                    run_start -= 1

                duration = current_time - starts[run_start]

                penalty = 0.0

                if duration > max_duration:
                    excess = duration - max_duration
                    penalty = duration_penalty * (excess ** 2)

                candidate = (
                    dp[i - 1, prev]
                    - transition
                    - penalty
                )

                if candidate > best:
                    best = candidate
                    best_prev = prev

            dp[i, slide] = best + scores[i, slide]
            back[i, slide] = best_prev

    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))

    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    return path


def evaluate(path):
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

    return correct / total


configs = [
    (0.05, 0.001, 180),
    (0.05, 0.002, 180),
    (0.05, 0.003, 180),
    (0.05, 0.005, 180),
    (0.05, 0.001, 240),
    (0.05, 0.002, 240),
    (0.05, 0.003, 240),
    (0.05, 0.005, 240),
]

results = []

for jump, dur_pen, max_dur in configs:

    path = duration_dp(
        scores,
        starts,
        ends,
        jump_penalty=jump,
        duration_penalty=dur_pen,
        max_duration=max_dur,
    )

    acc = evaluate(path)

    results.append(
        (acc, jump, dur_pen, max_dur, path)
    )


results.sort(
    key=lambda x: x[0],
    reverse=True
)

print()
print("=" * 75)
print("DURATION-AWARE DP TEST")
print("=" * 75)

for acc, jump, dur_pen, max_dur, _ in results:

    print(
        f"Accuracy={acc:.4f} "
        f"({acc*100:.2f}%) | "
        f"Jump={jump} | "
        f"DurationPenalty={dur_pen} | "
        f"MaxDuration={max_dur}s"
    )


best = results[0]

print()
print("=" * 75)
print("BEST")
print("=" * 75)

print(
    f"Accuracy = {best[0]*100:.2f}%"
)

path = best[4]

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
        f"{t:8.2f}s -> Slide {predict(path, t)}"
    )

print()
print("Transitions around 400s-650s:")

previous = path[0]

for i in range(1, len(path)):

    if path[i] != previous:

        t = starts[i]

        if 400 <= t <= 650:

            print(
                f"{t:8.2f}s : "
                f"Slide {previous+1} -> "
                f"Slide {path[i]+1}"
            )

        previous = path[i]
