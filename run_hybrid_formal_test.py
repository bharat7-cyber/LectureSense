import argparse
import json
from pathlib import Path

from src.hybrid_alignment import (
    DEFAULT_DP, DEFAULT_WEIGHTS, prepare_lecture_cache,
    run_from_cache, load_ground_truth, score_predictions, build_timeline
)
from run_hybrid_validation import find_video

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "MaViLS" / "data"
GT = DATA / "ground_truth_files"
SRT = DATA / "audioscripts"
PDF = DATA / "lectures"

TEST = {
    "short_range_mit": {
        "srt": SRT / "short_range_mit.srt",
        "pdf": PDF / "short_range_mit.pdf",
        "gt": GT / "ground_truth_short_range.xlsx",
        "video": ["short_range_MIT.mp4"],
    },
    "solar_resource": {
        "srt": SRT / "solar_resource.srt",
        "pdf": PDF / "solar_resource.pdf",
        "gt": GT / "ground_truth_solar_resource.xlsx",
        "video": ["solar_resource_high_res.mp4"],
    },
    "team_dynamics_game_design_MIT": {
        "srt": SRT / "team_dynamics_game_design_MIT.srt",
        "pdf": PDF / "team_dynamics_game_design_mit.pdf",
        "gt": GT / "ground_truth_team_dynamics_game_design_MIT.xlsx",
        "video": ["team_dynamics_high_res.mp4"],
    },
    "theory_of_computation": {
        "srt": SRT / "theory_of_computation.srt",
        "pdf": PDF / "theory_of_computation.pdf",
        "gt": GT / "ground_truth_theory_of_computation.xlsx",
        "video": ["theory_of_computation_high_res.mp4"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--ocr-interval", type=float, default=6.0)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    config_path = root / "checkpoints" / "hybrid_final_validation_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Missing {config_path}. Run calibrate_hybrid_validation.py first."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = config["weights"]
    dp = config["dp"]

    cache_root = root / "runs" / "hybrid_test_cache"
    run_root = root / "runs" / "hybrid_formal_test"
    total_c = total_n = 0

    print("=" * 78)
    print("LECTURESENSE — HYBRID FINAL FORMAL TEST-SET EVALUATION")
    print("=" * 78)
    print("Validation-calibrated config loaded.")
    print("NO TEST-SET PARAMETER TUNING.")

    for i, (name, cfg) in enumerate(TEST.items(), 1):
        video = find_video(cfg["video"])
        print("-" * 78)
        print(f"[{i}/4] {name}")
        print(f"VIDEO: {video}")
        if video is None:
            print("SKIPPED: video missing")
            continue

        cache = cache_root / name
        prepare_lecture_cache(cfg["pdf"], cfg["srt"], video, cache,
                               sample_seconds=args.sample_seconds,
                               ocr_interval=args.ocr_interval)
        times, path, _ = run_from_cache(cache, weights, dp)
        gt = load_ground_truth(cfg["gt"])
        c, n = score_predictions(times, path, gt)

        timeline = build_timeline(times, path)
        out = run_root / name / "timeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"timeline": timeline}, indent=2), encoding="utf-8")

        print(f"TEST RESULT — {name}: {c}/{n} = {100*c/max(n,1):.4f}%")
        total_c += c
        total_n += n

    print("=" * 78)
    print("HYBRID FINAL TEST RESULTS")
    print("=" * 78)
    print(f"Aggregate: {total_c}/{total_n} = {100*total_c/max(total_n,1):.4f}%")
    print("Test lectures were evaluated only after validation calibration.")


if __name__ == "__main__":
    main()
