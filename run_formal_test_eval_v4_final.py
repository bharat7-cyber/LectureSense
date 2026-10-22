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
INFERENCE = ROOT / "src" / "inference_temporal_v4_final.py"
COMPARE = ROOT / "compare_timeline_fixed.py"
CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
OUT_ROOT = ROOT / "runs" / "formal_test_eval_v4_final"
RESULTS_CSV = OUT_ROOT / "results.csv"
VIDEO_WEIGHT = 0.40
VIDEO_SAMPLE_SECONDS = 1.0

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


def run_and_capture(args, log_path):
    print("\n" + "=" * 78)
    print("RUNNING:")
    print(" ".join(f'"{a}"' if " " in a else a for a in args))
    print("=" * 78)
    p = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(p.stdout, encoding="utf-8")
    print(p.stdout)
    return p.returncode, p.stdout


def parse_accuracy(text):
    m = re.search(r"Correct\s*:\s*(\d+)\s*/\s*(\d+)", text)
    a = re.search(r"Accuracy:\s*([0-9.]+)", text)
    if not m:
        raise RuntimeError("Could not parse Correct X/Y")
    correct = int(m.group(1)); total = int(m.group(2))
    accuracy = float(a.group(1)) if a else correct / max(1, total)
    return correct, total, accuracy


def main():
    if not INFERENCE.exists(): raise FileNotFoundError(INFERENCE)
    if not COMPARE.exists(): raise FileNotFoundError(COMPARE)
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = CHECKPOINT.parent / "temporal_v4_final_decoder_config.json"
    print("#" * 78)
    print("LECTURESENSE V4 FINAL FORMAL TEST-SET EVALUATION")
    print("#" * 78)
    print("Test lectures: short_range, solar_resource, team_dynamics, theory_of_computation")
    print(f"Video weight: {VIDEO_WEIGHT}")
    print(f"Video sampling: {VIDEO_SAMPLE_SECONDS}")
    print(f"Decoder config: {cfg if cfg.exists() else 'checkpoint/default'}")
    print("No parameter tuning is performed on the test set.")

    rows = []
    for index, (lecture, item) in enumerate(TEST_SET.items(), start=1):
        print("\n" + "-" * 78)
        print(f"[{index}/4] {lecture}")
        transcript = TRANSCRIPT_DIR / item["transcript"]
        pdf = SLIDE_DIR / item["pdf"]
        gt = GT_DIR / item["gt"]
        video = VIDEO_DIR / item["video"]
        out_dir = OUT_ROOT / lecture
        timeline = out_dir / "timeline.json"
        render_dir = out_dir / "rendered_slides"
        infer_log = out_dir / "inference.log"
        compare_log = out_dir / "compare.log"
        args = [sys.executable, str(INFERENCE), "--transcript", str(transcript), "--slides", str(pdf), "--video", str(video), "--checkpoint", str(CHECKPOINT), "--out", str(timeline), "--render-dir", str(render_dir), "--video-weight", str(VIDEO_WEIGHT), "--video-sample-seconds", str(VIDEO_SAMPLE_SECONDS)]
        code, text = run_and_capture(args, infer_log)
        if code != 0 or not timeline.exists():
            print(f"FAILED: {lecture}"); continue
        compare_args = [sys.executable, str(COMPARE), "--timeline", str(timeline), "--ground-truth", str(gt)]
        code, text = run_and_capture(compare_args, compare_log)
        if code != 0:
            print(f"FAILED COMPARISON: {lecture}"); continue
        correct, total, acc = parse_accuracy(text)
        rows.append({"lecture": lecture, "correct": correct, "total": total, "accuracy": acc})
        print(f"\nTEST RESULT — {lecture}: {correct}/{total} = {acc*100:.4f}%")

    if not rows: raise RuntimeError("No test lecture completed successfully")
    aggregate_correct = sum(r["correct"] for r in rows); aggregate_total = sum(r["total"] for r in rows)
    aggregate_accuracy = aggregate_correct / max(1, aggregate_total)
    macro_accuracy = sum(r["accuracy"] for r in rows) / len(rows)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lecture", "correct", "total", "accuracy"])
        writer.writeheader(); writer.writerows(rows)
    print("\n" + "#" * 78)
    print("V4 FINAL FORMAL TEST-SET RESULTS")
    print("#" * 78)
    print(f"Lectures evaluated: {len(rows)}")
    print(f"Aggregate correct:  {aggregate_correct}/{aggregate_total}")
    print(f"Aggregate accuracy: {aggregate_accuracy*100:.4f}%")
    print(f"Macro accuracy:     {macro_accuracy*100:.4f}%")
    print("\nPer lecture:")
    for r in rows:
        print(f"  {r['lecture']:<40} {r['correct']}/{r['total']} {r['accuracy']*100:.4f}%")
    print(f"\nCSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
