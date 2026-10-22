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

# Your videos are outside the project directory.
VIDEO_DIR = Path(r"C:\Users\Bharat\Downloads\archive\video")

INFERENCE = ROOT / "src" / "inference.py"
COMPARE = ROOT / "compare_timeline.py"
CHECKPOINT = ROOT / "checkpoints" / "ranking_4modal_best_model.pt"

OUT_ROOT = ROOT / "runs" / "unseen_eval"
RESULTS_CSV = OUT_ROOT / "results.csv"

# This lecture was used for fusion/decoder tuning and is excluded.
EXCLUDE_TRANSCRIPT = "computer_vision_2_2_high_res.srt"

# Start with 5 usable lectures to keep runtime manageable.
# Set to None later to evaluate every usable lecture.
MAX_LECTURES = 5

# FIXED configuration. Do not tune these while measuring generalization.
VIDEO_WEIGHT = 0.40
VIDEO_SAMPLE_SECONDS = 1.0
JUMP_PENALTY = 0.05
MAX_DURATION = 120
DURATION_PENALTY = 0.01


# MaViLS filenames are not perfectly consistent across transcript,
# slide-PDF, and video directories, so use an explicit mapping.
LECTURE_MAP = {
    "cities_and_decarbonization": {
        "pdf": "cities&decarbonization.pdf",
        "video": "cities_and_decarbonization_standard_res.mp4",
    },
    "climate_and_cities": {
        "pdf": "cities&climate.pdf",
        "video": "cities_and_climate_high_res.mp4",
    },
    "climate_science_policy_MIT": {
        "pdf": "climate_science_policy_MIT.pdf",
        "video": None,
    },
    "cognitive_robotics_MIT": {
        "pdf": "cognitive_robotics_MIT.pdf",
        "video": "cognitive_robotics_high_res.mp4",
    },
    "creating_breakthrough_products_MIT": {
        "pdf": "creating_breakthrough_products.pdf",
        "video": "creating_breakthrough_products_MIT.mp4",
    },
    "cryptocurrency": {
        "pdf": "cryptocurrency.pdf",
        "video": "cryptocurrency_high_res.mp4",
    },
    "deeplearning_goodfellow": {
        "pdf": "deeplearning_goodfellow.pdf",
        "video": "deep_learning_high_res.mp4",
    },
    "image_processing_high_res": {
        "pdf": "image_processing.pdf",
        "video": "image_processing_high_res.mp4",
    },
    "ML_for_health_MIT": {
        "pdf": "ML_for_health_care_MIT.pdf",
        "video": "ML_for_health_high_res.mp4",
    },
    "numerics_hennig": {
        "pdf": "numerics.pdf",
        "video": "numerics_high_res.mp4",
    },
    "phonetics": {
        "pdf": "phonetics.pdf",
        "video": "phonetics_high_res.mp4",
    },
    "physics_intro_02": {
        "pdf": "physics_intro_02.pdf",
        "video": "physics_high_res.mp4",
    },
    "psychology_MIT": {
        "pdf": "psychology.pdf",
        "video": "psychology_high_res.mp4",
    },
    "reinforcement_learning_silver": {
        "pdf": "reinforcement_learning.pdf",
        "video": "reinforcement_learning_high_res.mp4",
    },
    "short_range_mit": {
        "pdf": "short_range_mit.pdf",
        "video": "short_range_MIT.mp4",
    },
    "solar_resource": {
        "pdf": "solar_resource.pdf",
        "video": "solar_resource_high_res.mp4",
    },
    "team_dynamics_game_design_MIT": {
        "pdf": "team_dynamics_game_design_mit.pdf",
        "video": "team_dynamics_high_res.mp4",
    },
    "theory_of_computation": {
        "pdf": "theory_of_computation.pdf",
        "video": "theory_of_computation_high_res.mp4",
    },
    # Transcript name differs from the PDF/video names.
    "visual_system_MIT": {
        "pdf": "sensory_systems.pdf",
        "video": "sensory_systems_high_res.mp4",
    },
    "computer_vision_2_2_high_res": {
        "pdf": "computer_vision_2_2_geiger.pdf",
        "video": "computer_vision_2_2_high_res.mp4",
    },
}


def run_command(args: list[str], log_path: Path) -> tuple[int, str]:
    print("\n" + "=" * 78)
    print("RUNNING:")
    print(" ".join(f'"{x}"' if " " in x else x for x in args))
    print("=" * 78)

    result = subprocess.run(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout)
    return result.returncode, result.stdout


def parse_accuracy(compare_output: str) -> tuple[int, int, float]:
    correct_match = re.search(r"Correct\s*:\s*(\d+)\s*/\s*(\d+)", compare_output)
    accuracy_match = re.search(r"Accuracy:\s*([0-9.]+)", compare_output)

    if not correct_match:
        raise RuntimeError(
            "Could not parse 'Correct : X/Y' from compare_timeline.py output."
        )

    correct = int(correct_match.group(1))
    total = int(correct_match.group(2))

    accuracy = (
        float(accuracy_match.group(1))
        if accuracy_match
        else (correct / total if total else 0.0)
    )

    return correct, total, accuracy


def main() -> None:
    if not INFERENCE.exists():
        raise FileNotFoundError(f"Missing inference script: {INFERENCE}")
    if not COMPARE.exists():
        raise FileNotFoundError(f"Missing evaluator: {COMPARE}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    if not TRANSCRIPT_DIR.exists():
        raise FileNotFoundError(f"Missing transcript directory: {TRANSCRIPT_DIR}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_transcripts = sorted(TRANSCRIPT_DIR.glob("*.srt"))

    usable = []
    skipped = []

    for transcript in all_transcripts:
        base = transcript.stem

        if transcript.name.lower() == EXCLUDE_TRANSCRIPT.lower():
            continue

        info = LECTURE_MAP.get(base)

        if info is None:
            skipped.append((base, "No explicit filename mapping"))
            continue

        video_name = info["video"]
        pdf_name = info["pdf"]

        if not video_name:
            skipped.append((base, "No video available"))
            continue

        pdf = SLIDE_DIR / pdf_name
        video = VIDEO_DIR / video_name
        gt = GT_DIR / f"ground_truth_{base}.xlsx"

        missing = []
        if not pdf.exists():
            missing.append(f"PDF: {pdf_name}")
        if not video.exists():
            missing.append(f"Video: {video_name}")
        if not gt.exists():
            missing.append(f"GT: ground_truth_{base}.xlsx")

        if missing:
            skipped.append((base, "; ".join(missing)))
            continue

        usable.append((transcript, pdf, gt, video))

    if MAX_LECTURES is not None:
        usable = usable[:MAX_LECTURES]

    print("\n" + "#" * 78)
    print("LECTURESENSE FIXED-CONFIGURATION GENERALIZATION CHECK")
    print("#" * 78)
    print(f"All SRT files found: {len(all_transcripts)}")
    print(f"Usable lectures selected: {len(usable)}")
    print(f"Excluded tuning lecture: {EXCLUDE_TRANSCRIPT}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Video weight: {VIDEO_WEIGHT}")
    print(f"Video sample interval: {VIDEO_SAMPLE_SECONDS}s")
    print(f"Jump penalty: {JUMP_PENALTY}")
    print(f"Max duration: {MAX_DURATION}s")
    print(f"Duration penalty: {DURATION_PENALTY}")

    if skipped:
        print("\nSkipped/unavailable:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    print("\nSelected lectures:")
    for i, (transcript, pdf, gt, video) in enumerate(usable, start=1):
        print(f"  {i}. {transcript.stem}")

    if not usable:
        print("\nNo usable lectures found.")
        return

    rows: list[dict[str, object]] = []

    for index, (transcript, pdf, gt, video) in enumerate(usable, start=1):
        lecture_id = transcript.stem
        out_dir = OUT_ROOT / lecture_id
        out_dir.mkdir(parents=True, exist_ok=True)

        timeline_path = out_dir / "timeline.json"
        render_dir = out_dir / "rendered_slides"
        inference_log = out_dir / "inference.log"
        compare_log = out_dir / "compare.log"

        print("\n" + "-" * 78)
        print(f"[{index}/{len(usable)}] {lecture_id}")
        print(f"Transcript: {transcript}")
        print(f"Slides:     {pdf}")
        print(f"Ground GT:  {gt}")
        print(f"Video:      {video}")
        print("-" * 78)

        inference_args = [
            sys.executable,
            str(INFERENCE),
            "--transcript",
            str(transcript),
            "--slides",
            str(pdf),
            "--video",
            str(video),
            "--checkpoint",
            str(CHECKPOINT),
            "--out",
            str(timeline_path),
            "--render-dir",
            str(render_dir),
            "--jump-penalty",
            str(JUMP_PENALTY),
            "--max-duration",
            str(MAX_DURATION),
            "--duration-penalty",
            str(DURATION_PENALTY),
            "--video-weight",
            str(VIDEO_WEIGHT),
            "--video-sample-seconds",
            str(VIDEO_SAMPLE_SECONDS),
        ]

        code, _ = run_command(inference_args, inference_log)

        if code != 0 or not timeline_path.exists():
            rows.append({
                "lecture": lecture_id,
                "status": "FAILED_INFERENCE",
                "correct": "",
                "total": "",
                "accuracy": "",
                "timeline": str(timeline_path),
                "log": str(inference_log),
                "reason": f"Return code {code}",
            })
            continue

        compare_args = [
            sys.executable,
            str(COMPARE),
            "--timeline",
            str(timeline_path),
            "--ground-truth",
            str(gt),
        ]

        code, output = run_command(compare_args, compare_log)

        if code != 0:
            rows.append({
                "lecture": lecture_id,
                "status": "FAILED_COMPARE",
                "correct": "",
                "total": "",
                "accuracy": "",
                "timeline": str(timeline_path),
                "log": str(compare_log),
                "reason": f"Return code {code}",
            })
            continue

        try:
            correct, total, accuracy = parse_accuracy(output)
        except Exception as exc:
            rows.append({
                "lecture": lecture_id,
                "status": "FAILED_PARSE",
                "correct": "",
                "total": "",
                "accuracy": "",
                "timeline": str(timeline_path),
                "log": str(compare_log),
                "reason": str(exc),
            })
            continue

        rows.append({
            "lecture": lecture_id,
            "status": "OK",
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "timeline": str(timeline_path),
            "log": str(compare_log),
            "reason": "",
        })

        print(f"\nRESULT — {lecture_id}: {correct}/{total} = {accuracy:.4%}")

    fieldnames = [
        "lecture", "status", "correct", "total", "accuracy",
        "timeline", "log", "reason"
    ]

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "OK"]

    print("\n" + "#" * 78)
    print("AGGREGATE RESULTS")
    print("#" * 78)

    if not ok_rows:
        print("No successful lecture evaluations.")
        print(f"CSV: {RESULTS_CSV}")
        return

    total_correct = sum(int(r["correct"]) for r in ok_rows)
    total_points = sum(int(r["total"]) for r in ok_rows)
    aggregate_accuracy = total_correct / total_points if total_points else 0.0
    macro_accuracy = sum(float(r["accuracy"]) for r in ok_rows) / len(ok_rows)

    print(f"Lectures evaluated: {len(ok_rows)}")
    print(f"Aggregate correct:  {total_correct}/{total_points}")
    print(f"Aggregate accuracy: {aggregate_accuracy:.4%}")
    print(f"Macro accuracy:     {macro_accuracy:.4%}")
    print(f"Results CSV: {RESULTS_CSV}")

    print("\nPer-lecture:")
    for r in ok_rows:
        print(
            f"  {r['lecture']:<38} "
            f"{int(r['correct']):>4}/{int(r['total']):<4} "
            f"{float(r['accuracy']):.4%}"
        )

    print("\nImportant:")
    print(
        "These measurements use a fixed configuration and exclude the "
        "computer_vision_2_2 tuning lecture. Before presenting them as a "
        "formal held-out test result, verify the MaViLS train/validation/test "
        "membership of each selected lecture."
    )


if __name__ == "__main__":
    main()
