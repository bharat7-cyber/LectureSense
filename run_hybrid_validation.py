import argparse
import json
from pathlib import Path

from src.hybrid_alignment import (
    DEFAULT_DP, DEFAULT_WEIGHTS, prepare_lecture_cache,
    run_from_cache, load_ground_truth, score_predictions, build_timeline
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "MaViLS" / "data"
GT = DATA / "ground_truth_files"
SRT = DATA / "audioscripts"
PDF = DATA / "lectures"

VIDEO_DIRS = [
    Path(r"C:\Users\Bharat\Downloads\archive\video"),
    ROOT / "archive" / "video",
]

VAL = {
    "physics_intro_02": {
        "srt": SRT / "physics_intro_02.srt",
        "pdf": PDF / "physics_intro_02.pdf",
        "gt": GT / "ground_truth_physics.xlsx",
        "video": ["physics.mp4", "physics_high_res.mp4"],
    },
    "psychology_MIT": {
        "srt": SRT / "psychology_MIT.srt",
        "pdf": PDF / "psychology.pdf",
        "gt": GT / "ground_truth_psychology.xlsx",
        "video": ["psychology_high_res.mp4"],
    },
    "reinforcement_learning_silver": {
        "srt": SRT / "reinforcement_learning_silver.srt",
        "pdf": PDF / "reinforcement_learning.pdf",
        "gt": GT / "ground_truth_reinforcement_learning.xlsx",
        "video": ["reinforcement_learning_high_res.mp4"],
    },
    "visual_system_MIT": {
        "srt": SRT / "visual_system_MIT.srt",
        "pdf": PDF / "sensory_systems.pdf",
        "gt": GT / "ground_truth_sensory_systems.xlsx",
        "video": ["sensory_systems_high_res.mp4"],
    },
}


def find_video(names):
    for d in VIDEO_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--ocr-interval", type=float, default=6.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache_root = root / "runs" / "hybrid_validation_cache"
    run_root = root / "runs" / "hybrid_validation"
    cache_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    names = [args.only] if args.only else list(VAL)
    total_correct = total_n = 0

    print("=" * 78)
    print("LECTURESENSE — HYBRID FINAL VALIDATION (NO TEST DATA)")
    print("=" * 78)
    print("Primary evidence: video OCR + transcript semantics + visual structure")
    print("Test lectures are never loaded.")
    print()

    for i, name in enumerate(names, 1):
        cfg = VAL[name]
        video = find_video(cfg["video"])
        print("-" * 78)
        print(f"[{i}/{len(names)}] {name}")
        print(f"SRT:   {cfg['srt']}")
        print(f"PDF:   {cfg['pdf']}")
        print(f"GT:    {cfg['gt']}")
        print(f"VIDEO: {video}")

        if not video or not cfg["srt"].exists() or not cfg["pdf"].exists() or not cfg["gt"].exists():
            print("SKIPPED: missing asset")
            continue

        cache = cache_root / name
        prepare_lecture_cache(
            cfg["pdf"], cfg["srt"], video, cache,
            sample_seconds=args.sample_seconds,
            ocr_interval=args.ocr_interval
        )

        times, path, _ = run_from_cache(cache, DEFAULT_WEIGHTS, DEFAULT_DP)
        gt = load_ground_truth(cfg["gt"])
        correct, n = score_predictions(times, path, gt)
        timeline = build_timeline(times, path)

        out = run_root / name / "timeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"timeline": timeline}, indent=2), encoding="utf-8")

        print(f"Result: {correct}/{n} = {100*correct/max(n,1):.4f}%")
        print(f"Timeline entries: {len(timeline)}")
        total_correct += correct
        total_n += n

    print("=" * 78)
    print("HYBRID VALIDATION SUMMARY")
    print("=" * 78)
    print(f"Aggregate: {total_correct}/{total_n} = {100*total_correct/max(total_n,1):.4f}%")
    print("Test lectures were not loaded.")


if __name__ == "__main__":
    main()
