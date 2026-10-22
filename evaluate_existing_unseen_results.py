from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
import re

ROOT = Path.cwd()
UNSEEN_DIR = ROOT / "runs" / "unseen_eval"
EVALUATOR = ROOT / "compare_timeline_fixed.py"
OUT_CSV = UNSEEN_DIR / "fixed_evaluation_results.csv"

GT_MAP = {
    "cities_and_decarbonization": "ground_truth_cities_and_decarbonization.xlsx",
    "climate_and_cities": "ground_truth_climate_and_cities.xlsx",
    "cognitive_robotics_MIT": "ground_truth_cognitive_robotics_MIT.xlsx",
    "creating_breakthrough_products_MIT": "ground_truth_creating_breakthrough_products_MIT.xlsx",
    "ML_for_health_MIT": "ground_truth_ML_for_health_MIT.xlsx",
}

GT_DIR = ROOT / "data" / "MaViLS" / "data" / "ground_truth_files"


def main():
    rows = []

    for lecture, gt_name in GT_MAP.items():
        timeline = UNSEEN_DIR / lecture / "timeline.json"
        gt = GT_DIR / gt_name

        print("\n" + "=" * 78)
        print(lecture)
        print("=" * 78)

        if not timeline.exists():
            print(f"SKIP: missing timeline: {timeline}")
            rows.append({
                "lecture": lecture,
                "status": "MISSING_TIMELINE",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        if not gt.exists():
            print(f"SKIP: missing GT: {gt}")
            rows.append({
                "lecture": lecture,
                "status": "MISSING_GT",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        cmd = [
            sys.executable,
            str(EVALUATOR),
            "--timeline",
            str(timeline),
            "--ground-truth",
            str(gt),
        ]

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )

        print(result.stdout)

        m = re.search(r"Correct\s*:\s*(\d+)\s*/\s*(\d+)", result.stdout)
        a = re.search(r"Accuracy:\s*([0-9.]+)", result.stdout)

        if not m:
            rows.append({
                "lecture": lecture,
                "status": "EVALUATOR_ERROR",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        correct = int(m.group(1))
        total = int(m.group(2))
        accuracy = float(a.group(1)) if a else (correct / total if total else 0.0)

        rows.append({
            "lecture": lecture,
            "status": "OK",
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        })

    ok = [r for r in rows if r["status"] == "OK"]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lecture", "status", "correct", "total", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "#" * 78)
    print("FIXED AGGREGATE RESULTS")
    print("#" * 78)

    if not ok:
        print("No successful evaluations.")
        print(f"CSV: {OUT_CSV}")
        return

    total_correct = sum(int(r["correct"]) for r in ok)
    total_points = sum(int(r["total"]) for r in ok)
    aggregate = total_correct / total_points if total_points else 0.0
    macro = sum(float(r["accuracy"]) for r in ok) / len(ok)

    print(f"Lectures evaluated: {len(ok)}")
    print(f"Aggregate correct:  {total_correct}/{total_points}")
    print(f"Aggregate accuracy: {aggregate:.4%}")
    print(f"Macro accuracy:     {macro:.4%}")
    print(f"CSV: {OUT_CSV}")

    print("\nPer lecture:")
    for r in ok:
        print(
            f"  {r['lecture']:<38} "
            f"{int(r['correct']):>4}/{int(r['total']):<4} "
            f"{float(r['accuracy']):.4%}"
        )


if __name__ == "__main__":
    main()
