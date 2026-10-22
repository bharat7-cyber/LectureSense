import argparse
import json
from pathlib import Path

from src.hybrid_alignment import (
    DEFAULT_DP, prepare_lecture_cache, run_from_cache,
    load_ground_truth, score_predictions
)
from run_hybrid_validation import VAL, find_video

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--ocr-interval", type=float, default=6.0)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cache_root = root / "runs" / "hybrid_validation_cache"

    weight_candidates = [
        {"visual": .20, "ocr": .50, "semantic": .30, "boundary_to_score": .04},
        {"visual": .25, "ocr": .45, "semantic": .30, "boundary_to_score": .05},
        {"visual": .20, "ocr": .55, "semantic": .25, "boundary_to_score": .05},
        {"visual": .15, "ocr": .60, "semantic": .25, "boundary_to_score": .06},
        {"visual": .30, "ocr": .40, "semantic": .30, "boundary_to_score": .05},
        {"visual": .20, "ocr": .45, "semantic": .35, "boundary_to_score": .05},
        {"visual": .30, "ocr": .45, "semantic": .25, "boundary_to_score": .06},
        {"visual": .15, "ocr": .50, "semantic": .35, "boundary_to_score": .07},
    ]

    dp_candidates = [
        {"jump_penalty": .06, "boundary_reward": 1.6, "stay_boundary_penalty": .7, "max_forward_jump": 3},
        {"jump_penalty": .08, "boundary_reward": 1.8, "stay_boundary_penalty": .6, "max_forward_jump": 3},
        {"jump_penalty": .10, "boundary_reward": 2.0, "stay_boundary_penalty": .7, "max_forward_jump": 3},
        {"jump_penalty": .06, "boundary_reward": 2.0, "stay_boundary_penalty": .8, "max_forward_jump": 3},
    ]

    print("=" * 78)
    print("LECTURESENSE — HYBRID VALIDATION-ONLY CALIBRATION")
    print("=" * 78)
    print("Feature caches are built once. Test lectures are never loaded.")

    prepared = {}
    for name, cfg in VAL.items():
        video = find_video(cfg["video"])
        if video is None:
            print(f"Missing video: {name}")
            continue
        cache = cache_root / name
        prepare_lecture_cache(cfg["pdf"], cfg["srt"], video, cache,
                               sample_seconds=args.sample_seconds,
                               ocr_interval=args.ocr_interval)
        prepared[name] = (cache, cfg["gt"])

    best = None
    candidate_no = 0

    for wi, weights in enumerate(weight_candidates, 1):
        for di, dp in enumerate(dp_candidates, 1):
            candidate_no += 1
            total_correct = 0
            total_n = 0
            print("-" * 78)
            print(f"Candidate {candidate_no}/{len(weight_candidates)*len(dp_candidates)}")
            print(f"Weights: {weights}")
            print(f"DP: {dp}")
            for name, (cache, gt_path) in prepared.items():
                times, path, _ = run_from_cache(cache, weights, dp)
                gt = load_ground_truth(gt_path)
                c, n = score_predictions(times, path, gt)
                total_correct += c
                total_n += n
                print(f"  {name}: {c}/{n} = {100*c/max(n,1):.2f}%")
            acc = total_correct / max(total_n, 1)
            print(f"Candidate accuracy: {100*acc:.4f}%")
            if best is None or acc > best["accuracy"]:
                best = {"accuracy": acc, "weights": weights, "dp": dp}
                print("NEW BEST")

    out = root / "checkpoints" / "hybrid_final_validation_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"Best validation accuracy: {100*best['accuracy']:.4f}%")
    print(f"Saved: {out}")
    print("TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()
