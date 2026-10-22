import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from src.hybrid_alignment import (
    run_from_cache, load_ground_truth, nearest_indices
)

def softmax_confidence(row):
    x = np.asarray(row, dtype=np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    p /= np.sum(p) + 1e-12
    order = np.argsort(p)[::-1]
    top = p[order[0]]
    second = p[order[1]] if len(order) > 1 else 0.0
    return float(top), float(top - second)


def load_v4_scores(cache_dir: Path):
    """
    Adapter for a V4-compatible score matrix if one has already been
    exported by the user's existing inference/evaluation pipeline.

    Expected:
      runs/.../v4_scores/<lecture>.npy

    Shape:
      [video/sample timestamps, number_of_slides]
    """
    candidates = [
        cache_dir / "v4_scores.npy",
        cache_dir.parent / "v4_scores.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.load(p)
    return None


def load_visual_predictions(cache_dir: Path):
    times, path, parts = run_from_cache(
        cache_dir,
        {
            "visual": 1.0,
            "ocr": 0.0,
            "semantic": 0.0,
            "boundary_to_score": 0.0,
        },
        {
            "jump_penalty": 0.08,
            "boundary_reward": 1.8,
            "stay_boundary_penalty": 0.6,
            "max_forward_jump": 3,
        },
    )
    return times, path + 1, parts


def evaluate_gate(cache_dir, gt_path, threshold, margin_threshold):
    gt = load_ground_truth(gt_path)

    times, visual_pred, parts = load_visual_predictions(cache_dir)
    visual_scores = parts["visual"]

    v_idx = nearest_indices(gt["timestamp"].to_numpy(), times)
    visual_at_gt = visual_pred[v_idx]

    rows = []
    for i, ts in enumerate(gt["timestamp"].to_numpy()):
        row = visual_scores[v_idx[i]]
        vc, vm = softmax_confidence(row)
        rows.append({
            "timestamp": float(ts),
            "gt": int(gt["slide"].iloc[i]),
            "visual_pred": int(visual_at_gt[i]),
            "visual_confidence": vc,
            "visual_margin": vm,
        })

    return pd.DataFrame(rows)


def summarize(df):
    visual_correct = (df["visual_pred"] == df["gt"])
    return {
        "n": int(len(df)),
        "visual_correct": int(visual_correct.sum()),
        "visual_accuracy": float(visual_correct.mean()) if len(df) else 0.0,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = evaluate_gate(Path(args.cache), Path(args.gt), 0.0, 0.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(json.dumps(summarize(df), indent=2))
    print(f"Saved diagnostics: {args.out}")


if __name__ == "__main__":
    main()
