from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.visual_first_alignment import find_asset, load_ground_truth, run_visual_first, evaluate_timeline, save_timeline

TEST = [
    "short_range_mit",
    "solar_resource",
    "team_dynamics_game_design_MIT",
    "theory_of_computation",
]


def asset_dirs(root: Path):
    base = root / "data" / "MaViLS" / "data"
    return base / "audioscripts", base / "lectures", root / "archive" / "video", base / "ground_truth_files"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--sample-seconds", type=float, default=1.0)
    p.add_argument("--max-jump", type=int, default=3)
    p.add_argument("--jump-penalty", type=float, default=0.015)
    p.add_argument("--out-dir", default="runs/visual_first_formal_test")
    args = p.parse_args()

    root = Path(args.root)
    audio_dir, lecture_dir, video_dir, gt_dir = asset_dirs(root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    results = []
    tc = tn = 0
    print("=" * 78)
    print("LECTURESENSE — VISUAL-FIRST FORMAL HELD-OUT TEST")
    print("=" * 78)
    print("No tuning is performed here. Parameters are command-line fixed.")

    for k, name in enumerate(TEST, 1):
        print("\n" + "-" * 78)
        print(f"[{k}/{len(TEST)}] {name}")
        pdf = find_asset(lecture_dir, name, [".pdf"])
        video = find_asset(video_dir, name, [".mp4", ".mkv", ".avi"])
        gt = find_asset(gt_dir, name, [".xlsx"])
        if not all([pdf, video, gt]):
            print(f"Missing asset: pdf={pdf}, video={video}, gt={gt}")
            continue
        start = time.time()
        timeline, _, _, _ = run_visual_first(video, pdf, args.sample_seconds, args.max_jump, args.jump_penalty, True)
        gt_rows = load_ground_truth(gt)
        correct, total, acc, _ = evaluate_timeline(timeline, gt_rows)
        save_timeline(out_dir / f"{name}_timeline.json", timeline, {"sample_seconds": args.sample_seconds, "max_jump": args.max_jump, "jump_penalty": args.jump_penalty})
        print(f"TEST RESULT — {name}: {correct}/{total} = {acc:.4%}")
        print(f"Runtime: {time.time()-start:.1f}s")
        results.append((name, correct, total, acc))
        tc += correct
        tn += total

    macro = sum(x[3] for x in results) / len(results) if results else 0.0
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "correct", "total", "accuracy"])
        w.writerows(results)
        w.writerow(["aggregate", tc, tn, tc / tn if tn else 0.0])
        w.writerow(["macro", "", "", macro])

    print("\n" + "=" * 78)
    print("VISUAL-FIRST FORMAL TEST RESULTS")
    print("=" * 78)
    print(f"Aggregate: {tc}/{tn} = {(tc/tn if tn else 0):.4%}")
    print(f"Macro:     {macro:.4%}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
