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

assert scores.shape[0] == len(segments), (
    f"Mismatch: scores={scores.shape[0]}, segments={len(segments)}"
)

n, m = scores.shape

starts = np.array([x["start"] for x in segments], dtype=np.float32)
ends = np.array([x["end"] for x in segments], dtype=np.float32)


def duration_dp(
    scores,
    starts,
    ends,
    max_duration=180.0,
    duration_penalty=0.02,
    jump_penalty=0.05,
):
    n, m = scores.shape

    # Maximum duration state is number of transcript segments.
    D = n

    NEG = -1e30

    dp = np.full(
        (n, m, D + 1),
        NEG,
        dtype=np.float32
    )

    back_slide = np.full(
        (n, m, D + 1),
        -1,
        dtype=np.int16
    )

    back_duration = np.full(
        (n, m, D + 1),
        -1,
        dtype=np.int32
    )

    # Lecture starts from slide 1.
    dp[0, 0, 1] = scores[0, 0]

    for i in range(1, n):

        for slide in range(m):

            # Stay on same slide.
            for d in range(2, i + 2):

                previous_value = dp[i - 1, slide, d - 1]

                if previous_value <= NEG / 2:
                    continue

                run_start = i - d + 1

                duration = (
                    ends[i] - starts[run_start]
                )

                excess = max(
                    0.0,
                    duration - max_duration
                )

                penalty = (
                    duration_penalty
                    * (excess / max_duration) ** 2
                    * max_duration
                )

                candidate = (
                    previous_value
                    + scores[i, slide]
                    - penalty
                )

                dp[i, slide, d] = candidate

                back_slide[i, slide, d] = slide
                back_duration[i, slide, d] = d - 1

            # Move forward exactly one slide.
            if slide > 0:

                best_prev_value = NEG
                best_prev_duration = -1

                for d in range(1, i + 1):

                    value = dp[i - 1, slide - 1, d]

                    if value > best_prev_value:
                        best_prev_value = value
                        best_prev_duration = d

                if best_prev_duration != -1:

                    candidate = (
                        best_prev_value
                        + scores[i, slide]
                        - jump_penalty
                    )

                    dp[i, slide, 1] = candidate

                    back_slide[i, slide, 1] = slide - 1
                    back_duration[i, slide, 1] = best_prev_duration

    # Best final state.
    last_slide, last_duration = np.unravel_index(
        np.argmax(dp[-1]),
        dp[-1].shape
    )

    path = np.zeros(
        n,
        dtype=np.int32
    )

    path[-1] = last_slide

    current_slide = last_slide
    current_duration = last_duration

    for i in range(n - 1, 0, -1):

        previous_slide = back_slide[
            i,
            current_slide,
            current_duration
        ]

        previous_duration = back_duration[
            i,
            current_slide,
            current_duration
        ]

        if previous_slide < 0:
            raise RuntimeError(
                f"Broken backpointer at i={i}"
            )

        path[i - 1] = previous_slide

        current_slide = previous_slide
        current_duration = previous_duration

    # Safety check.
    if np.any(np.diff(path) < 0):
        raise RuntimeError(
            "BACKWARD TRANSITION DETECTED"
        )

    return path


def predict(path, t):
    idx = np.searchsorted(
        starts,
        float(t),
        side="right"
    ) - 1

    idx = max(
        0,
        min(idx, len(path) - 1)
    )

    return int(path[idx]) + 1


def evaluate(path):
    correct = 0
    total = 0

    for _, row in gt.iterrows():

        true_slide = int(row["Slidenumber"])

        if true_slide < 1:
            continue

        predicted = predict(
            path,
            float(row["Time"])
        )

        total += 1

        if predicted == true_slide:
            correct += 1

    return correct / total


configs = [
    (120, 0.01),
    (150, 0.01),
    (180, 0.01),
    (210, 0.01),
    (240, 0.01),
    (300, 0.01),

    (150, 0.02),
    (180, 0.02),
    (210, 0.02),
    (240, 0.02),

    (180, 0.05),
    (210, 0.05),
    (240, 0.05),
]

results = []

for max_duration, duration_penalty in configs:

    path = duration_dp(
        scores,
        starts,
        ends,
        max_duration=max_duration,
        duration_penalty=duration_penalty,
        jump_penalty=0.05,
    )

    accuracy = evaluate(path)

    results.append(
        (
            accuracy,
            max_duration,
            duration_penalty,
            path
        )
    )

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print()
print("=" * 80)
print("CORRECT MONOTONIC DURATION DP")
print("=" * 80)

for accuracy, max_duration, duration_penalty, _ in results:

    print(
        f"Accuracy={accuracy:.4f} "
        f"({accuracy*100:.2f}%) | "
        f"MaxDuration={max_duration}s | "
        f"Penalty={duration_penalty}"
    )

best_accuracy, best_duration, best_penalty, best_path = results[0]

print()
print("=" * 80)
print("BEST")
print("=" * 80)

print(
    f"Accuracy = {best_accuracy*100:.2f}%"
)

print(
    f"MaxDuration = {best_duration}s"
)

print(
    f"DurationPenalty = {best_penalty}"
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
        f"{t:8.2f}s -> "
        f"Slide {predict(best_path, t)}"
    )

print()
print("Transitions around 400s-650s:")

previous = best_path[0]

for i in range(1, len(best_path)):

    current = best_path[i]

    if current != previous:

        t = starts[i]

        if 400 <= t <= 650:

            print(
                f"{t:8.2f}s : "
                f"Slide {previous + 1} -> "
                f"Slide {current + 1}"
            )

        previous = current

print()
print(
    "MONOTONIC CHECK:",
    bool(np.all(np.diff(best_path) >= 0))
)
