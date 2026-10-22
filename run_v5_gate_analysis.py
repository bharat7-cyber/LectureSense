
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

LECTURES = [
    "physics_intro_02",
    "psychology_MIT",
    "reinforcement_learning_silver",
    "visual_system_MIT",
]

RESULT_RE = re.compile(r"Result:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)%")

def run_visual(root, lecture, sample_seconds, max_jump):
    cmd = [
        sys.executable,
        str(root / "run_visual_first_validation.py"),
        "--root", str(root),
        "--only", lecture,
        "--sample-seconds", str(sample_seconds),
        "--max-jump", str(max_jump),
    ]
    p = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
    raw = p.stdout + ("\n" + p.stderr if p.stderr else "")
    m = RESULT_RE.search(raw)
    if not m:
        raise RuntimeError(
            f"Visual evaluator failed for {lecture}:\n{raw[-3000:]}"
        )
    return int(m.group(1)), int(m.group(2)), raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--max-jump", type=int, default=3)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / "runs" / "v5_gate_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LECTURESENSE V5 — GATE ANALYSIS PREPARATION")
    print("=" * 78)
    print("Validation only.")
    print("No V4 checkpoint is modified.")
    print("No test lecture is loaded.")
    print("-" * 78)

    lectures = [args.only] if args.only else LECTURES
    rows = []

    for lecture in lectures:
        print(f"\n[{lecture}]")
        correct, total, raw = run_visual(
            root, lecture, args.sample_seconds, args.max_jump
        )
        (outdir / f"{lecture}_visual.log").write_text(raw, encoding="utf-8")

        rows.append({
            "lecture": lecture,
            "visual_correct": correct,
            "visual_total": total,
            "visual_accuracy_percent": 100.0 * correct / total,
            "v4_correct": "",
            "v4_total": "",
            "v4_accuracy_percent": "",
            "v4_correct_visual_wrong": "",
            "v4_wrong_visual_correct": "",
            "gate_possible_gain": "",
            "gate_possible_damage": "",
        })

        print(
            f"Visual: {correct}/{total} = "
            f"{100.0 * correct / total:.4f}%"
        )
        print(
            "V4 comparison is intentionally left blank here; "
            "timestamp-level V4 export must be aligned first."
        )

    csv_path = outdir / "gate_analysis.csv"
    fields = list(rows[0].keys()) if rows else []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "stage": "V5 gate analysis preparation",
        "validation_lectures": lectures,
        "visual_sample_seconds": args.sample_seconds,
        "visual_max_jump": args.max_jump,
        "test_used": False,
        "purpose": (
            "Verify the visual branch before timestamp-level V4/visual "
            "confidence gating."
        ),
    }

    (outdir / "gate_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("GATE ANALYSIS PREPARATION COMPLETE")
    print("=" * 78)
    print(f"Saved: {csv_path}")
    print("TEST SET WAS NOT USED.")
    print("=" * 78)

if __name__ == "__main__":
    main()
