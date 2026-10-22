from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
TRANSCRIPT_DIR = ROOT / "data" / "MaViLS" / "data" / "audioscripts"
SLIDE_DIR = ROOT / "data" / "MaViLS" / "data" / "lectures"
GT_DIR = ROOT / "data" / "MaViLS" / "data" / "ground_truth_files"
VIDEO_DIR = Path(r"C:\Users\Bharat\Downloads\archive\video")

INFERENCE = ROOT / "src" / "inference.py"
COMPARE = ROOT / "compare_timeline_fixed.py"
CHECKPOINT = ROOT / "checkpoints" / "ranking_4modal_hardneg_best_model.pt"

OUT_ROOT = ROOT / "runs" / "formal_test_eval"
RESULTS_CSV = OUT_ROOT / "results.csv"

# Official test split in this project's build_pairs.py:
# short_range, solar_resource, team_dynamics, theory_of_computation.
TEST_SET = {
    "short_range_mit": {
        "transcript": "short_range_mit.srt",
        "pdf": "short_range_mit.pdf",
        "gt": "ground_truth_short_range.xlsx",
        "video": "short_range_MIT.mp4",
    },
    "solar_resource": {
        "transcript": "solar_resource.srt",
        "pdf": "solar_resource.pdf",
        "gt": "ground_truth_solar_resource.xlsx",
        "video": "solar_resource_high_res.mp4",
    },
    "team_dynamics_game_design_MIT": {
        "transcript": "team_dynamics_game_design_MIT.srt",
        "pdf": "team_dynamics_game_design_mit.pdf",
        "gt": "ground_truth_team_dynamics_game_design_MIT.xlsx",
        "video": "team_dynamics_high_res.mp4",
    },
    "theory_of_computation": {
        "transcript": "theory_of_computation_MIT.srt",
        "pdf": "theory_of_computation.pdf",
        "gt": "ground_truth_theory_of_computation.xlsx",
        "video": "theory_of_computation_high_res.mp4",
    },
}

# LOCKED after tuning on computer_vision_2_2.
VIDEO_WEIGHT = 0.40
VIDEO_SAMPLE_SECONDS = 1.0
JUMP_PENALTY = 0.05
MAX_DURATION = 120
DURATION_PENALTY = 0.01


def run_and_capture(args: list[str], log_path: Path) -> tuple[int, str]:
    print("\n" + "=" * 78)
    print("RUNNING:")
    print(" ".join(f'"{a}"' if " " in a else a for a in args))
    print("=" * 78)

    p = subprocess.run(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(p.stdout, encoding="utf-8")
    print(p.stdout)
    return p.returncode, p.stdout


def parse_accuracy(text: str) -> tuple[int, int, float]:
    m = re.search(r"Correct\s*:\s*(\d+)\s*/\s*(\d+)", text)
    a = re.search(r"Accuracy:\s*([0-9.]+)", text)
    if not m:
        raise RuntimeError("Could not parse Correct X/Y from evaluator output.")
    correct = int(m.group(1))
    total = int(m.group(2))
    accuracy = float(a.group(1)) if a else (correct / total if total else 0.0)
    return correct, total, accuracy


def main():
    if not INFERENCE.exists():
        raise FileNotFoundError(INFERENCE)
    if not COMPARE.exists():
        raise FileNotFoundError(COMPARE)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    print("#" * 78)
    print("LECTURESENSE FORMAL TEST-SET EVALUATION")
    print("#" * 78)
    print("Test lectures: short_range, solar_resource, team_dynamics, theory_of_computation")
    print(f"Video weight: {VIDEO_WEIGHT}")
    print(f"Video sampling: {VIDEO_SAMPLE_SECONDS}s")
    print(f"Jump penalty: {JUMP_PENALTY}")
    print(f"Max duration: {MAX_DURATION}s")
    print(f"Duration penalty: {DURATION_PENALTY}")
    print("\nNo parameter tuning is performed in this script.")

    for idx, (lecture, info) in enumerate(TEST_SET.items(), start=1):
        transcript = TRANSCRIPT_DIR / info["transcript"]
        pdf = SLIDE_DIR / info["pdf"]
        gt = GT_DIR / info["gt"]
        video = VIDEO_DIR / info["video"]

        out_dir = OUT_ROOT / lecture
        out_dir.mkdir(parents=True, exist_ok=True)

        timeline = out_dir / "timeline.json"
        render_dir = out_dir / "rendered_slides"
        inference_log = out_dir / "inference.log"
        compare_log = out_dir / "compare.log"

        print("\n" + "-" * 78)
        print(f"[{idx}/4] {lecture}")
        print(f"Transcript: {transcript}")
        print(f"Slides:     {pdf}")
        print(f"Ground GT:  {gt}")
        print(f"Video:      {video}")

        missing = [str(p) for p in [transcript, pdf, gt, video] if not p.exists()]
        if missing:
            print("SKIPPED — missing:")
            for p in missing:
                print(" ", p)
            rows.append({
                "lecture": lecture,
                "status": "MISSING_INPUT",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        infer_args = [
            sys.executable,
            str(INFERENCE),
            "--transcript", str(transcript),
            "--slides", str(pdf),
            "--video", str(video),
            "--checkpoint", str(CHECKPOINT),
            "--out", str(timeline),
            "--render-dir", str(render_dir),
            "--jump-penalty", str(JUMP_PENALTY),
            "--max-duration", str(MAX_DURATION),
            "--duration-penalty", str(DURATION_PENALTY),
            "--video-weight", str(VIDEO_WEIGHT),
            "--video-sample-seconds", str(VIDEO_SAMPLE_SECONDS),
        ]

        code, _ = run_and_capture(infer_args, inference_log)

        if code != 0 or not timeline.exists():
            rows.append({
                "lecture": lecture,
                "status": "FAILED_INFERENCE",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        compare_args = [
            sys.executable,
            str(COMPARE),
            "--timeline", str(timeline),
            "--ground-truth", str(gt),
        ]

        code, compare_output = run_and_capture(compare_args, compare_log)

        if code != 0:
            rows.append({
                "lecture": lecture,
                "status": "FAILED_COMPARE",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            continue

        try:
            correct, total, accuracy = parse_accuracy(compare_output)
        except Exception as exc:
            rows.append({
                "lecture": lecture,
                "status": "FAILED_PARSE",
                "correct": "",
                "total": "",
                "accuracy": "",
            })
            print("Evaluator parsing failed:", exc)
            continue

        rows.append({
            "lecture": lecture,
            "status": "OK",
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        })

        print(f"TEST RESULT — {lecture}: {correct}/{total} = {accuracy:.4%}")

    fieldnames = ["lecture", "status", "correct", "total", "accuracy"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)

    ok = [r for r in rows if r["status"] == "OK"]

    print("\n" + "#" * 78)
    print("FORMAL TEST-SET RESULTS")
    print("#" * 78)

    if not ok:
        print("No successful test-set evaluations.")
        print(f"CSV: {RESULTS_CSV}")
        return

    total_correct = sum(int(r["correct"]) for r in ok)
    total_points = sum(int(r["total"]) for r in ok)
    aggregate = total_correct / total_points if total_points else 0.0
    macro = sum(float(r["accuracy"]) for r in ok) / len(ok)

    print(f"Lectures evaluated: {len(ok)}")
    print(f"Aggregate correct:  {total_correct}/{total_points}")
    print(f"Aggregate accuracy: {aggregate:.4%}")
    print(f"Macro accuracy:     {macro:.4%}")

    print("\nPer lecture:")
    for r in ok:
        print(
            f"  {r['lecture']:<38} "
            f"{int(r['correct']):>4}/{int(r['total']):<4} "
            f"{float(r['accuracy']):.4%}"
        )

    print(f"\nCSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
