import json
import pandas as pd

timeline_path = r"runs\session\timeline.json"
gt_path = r"data\MaViLS\data\ground_truth_files\ground_truth_computer_vision_2_2.xlsx"

with open(timeline_path, encoding="utf-8") as f:
    timeline = json.load(f)["timeline"]

gt = pd.read_excel(gt_path)

def get_prediction(t):
    for item in timeline:
        if item["start_time"] <= t < item["end_time"]:
            return int(item["slide_id"])
    return -1

print("=" * 70)
print("GROUND TRUTH VS CURRENT TIMELINE")
print("=" * 70)

correct = 0
total = 0

for _, row in gt.iterrows():

    true_slide = int(row["Slidenumber"])

    if true_slide < 1:
        continue

    timestamp = float(row["Time"])
    predicted_slide = get_prediction(timestamp)

    status = "CORRECT" if predicted_slide == true_slide else "WRONG"

    if predicted_slide == true_slide:
        correct += 1

    total += 1

    print(
        f"{timestamp:8.2f}s | "
        f"GT Slide {true_slide:2d} | "
        f"Pred Slide {predicted_slide:2d} | "
        f"{status}"
    )

accuracy = correct / total if total else 0.0

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print()
print("=" * 70)
print("CURRENT TIMELINE")
print("=" * 70)

for item in timeline:
    print(
        f"Slide {int(item['slide_id']):2d}: "
        f"{float(item['start_time']):8.2f}s -> "
        f"{float(item['end_time']):8.2f}s"
    )
