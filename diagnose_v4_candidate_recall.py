from __future__ import annotations

"""Validation-only candidate-recall and decoder diagnostic for LectureSense V4.

Run from the project root after placing this file beside tune_v4_decoder_only.py:
    python diagnose_v4_candidate_recall.py

Reads only the four validation lecture caches and labels. It does not train a model,
modify the V4 checkpoint, or read official test labels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import tune_v4_decoder_only as tuner
except Exception as exc:
    raise SystemExit(
        "Could not import tune_v4_decoder_only.py. Keep this diagnostic beside that file.\n"
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


def _candidate_metrics(scores: np.ndarray, item: dict) -> dict:
    gt_idx = np.asarray(item["gt_indices"], dtype=np.int64)
    gt_labels = np.asarray(item["gt_labels"], dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape={scores.shape}")
    if scores.shape[0] != len(item["starts"]):
        raise ValueError(
            f"Score rows ({scores.shape[0]}) do not match transcript segments "
            f"({len(item['starts'])}) for {item['name']}"
        )
    keep = (gt_idx >= 0) & (gt_idx < scores.shape[0]) & (gt_labels >= 0) & (gt_labels < scores.shape[1])
    gt_idx, gt_labels = gt_idx[keep], gt_labels[keep]
    if len(gt_labels) == 0:
        raise ValueError(f"No valid mapped ground-truth labels for {item['name']}")

    point_scores = scores[gt_idx]
    # Descending argsort provides exact candidate rank for each GT point.
    ranking = np.argsort(-point_scores, axis=1)
    ranks = np.argmax(ranking == gt_labels[:, None], axis=1) + 1

    out = {
        "lecture": item["name"],
        "gt_points": int(len(gt_labels)),
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= min(3, scores.shape[1]))),
        "top5": float(np.mean(ranks <= min(5, scores.shape[1]))),
        "top10": float(np.mean(ranks <= min(10, scores.shape[1]))),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_gt_rank": float(np.median(ranks)),
        "mean_gt_rank": float(np.mean(ranks)),
    }
    return out


def _aggregate(rows: list[dict], name: str) -> dict:
    total = sum(int(r["gt_points"]) for r in rows)
    out = {"matrix": name, "gt_points": total}
    for key in ("top1", "top3", "top5", "top10", "mrr"):
        out[f"macro_{key}"] = float(np.mean([r[key] for r in rows])) if rows else 0.0
        out[f"weighted_{key}"] = (
            sum(float(r[key]) * int(r["gt_points"]) for r in rows) / max(1, total)
        )
    out["median_gt_rank_macro"] = float(np.mean([r["median_gt_rank"] for r in rows])) if rows else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how often the correct slide is in V4's top-K candidates.")
    parser.add_argument("--v4-root", default=str(tuner.DEFAULT_V4_ROOT))
    parser.add_argument("--checkpoint", default=str(tuner.DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "v4_candidate_diagnostics"))
    parser.add_argument("--skip-dp", action="store_true", help="Skip full-path DP baseline/tuned metrics; top-K runs only.")
    args = parser.parse_args()

    v4_root = Path(args.v4_root)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("LectureSense V4 candidate recall diagnostic (VALIDATION ONLY)")
    print(f"V4 cache root: {v4_root}")
    print(f"Checkpoint: {checkpoint} (read-only)")
    print("Official test labels will not be loaded.\n")

    checkpoint_decoder = tuner.load_checkpoint_decoder(checkpoint)
    baseline_cfg = tuner.make_baseline_config(checkpoint_decoder)
    baseline_fusion = tuner.baseline_fusion_key(baseline_cfg)

    dataset = tuner.load_dataset(tuner.lecture_specs("validation", v4_root))
    config_payload = None
    tuned_cfg = None
    config_path = tuner.CONFIG_PATH
    if config_path.exists():
        try:
            candidate = json.loads(config_path.read_text(encoding="utf-8"))
            if candidate.get("validation_only") and not candidate.get("test_used", True):
                tuned_cfg = {**candidate.get("fusion", {}), **candidate.get("decoder", {})}
                required = set(tuner.ALL_KEYS)
                if not required.issubset(tuned_cfg):
                    tuned_cfg = None
                else:
                    config_payload = candidate
                    print(f"Loaded validation-selected config: {config_path}")
        except Exception as exc:
            print(f"Note: ignored config file ({type(exc).__name__}: {exc})")
    if tuned_cfg is None:
        print("No valid validation-only selected config found; only baseline / cached matrices will be analysed.")

    per_rows: list[dict] = []
    matrix_rows: dict[str, list[dict]] = {}
    path_rows: list[dict] = []

    for name, item in dataset.items():
        matrices: dict[str, np.ndarray] = {}
        if item.get("base") is not None:
            matrices["base_no_temporal"] = item["base"]
        if item.get("adjusted") is not None:
            matrices["v4_temporal_adjusted"] = item["adjusted"]
        if item.get("final") is not None:
            matrices["cached_final_video_fusion"] = item["final"]
        matrices["production_baseline_scores"] = tuner.build_scores(item, baseline_cfg, baseline_fusion)
        if tuned_cfg is not None:
            matrices["validation_tuned_scores"] = tuner.build_scores(item, tuned_cfg, baseline_fusion)

        for matrix_name, matrix in matrices.items():
            metric = _candidate_metrics(matrix, item)
            metric["matrix"] = matrix_name
            per_rows.append(metric)
            matrix_rows.setdefault(matrix_name, []).append(metric)

        if not args.skip_dp:
            c0, t0 = tuner.evaluate_lecture(item, baseline_cfg, baseline_fusion)
            record = {
                "lecture": name,
                "gt_points": int(t0),
                "baseline_dp_correct": int(c0),
                "baseline_dp_accuracy": float(c0 / max(1, t0)),
            }
            if tuned_cfg is not None:
                c1, t1 = tuner.evaluate_lecture(item, tuned_cfg, baseline_fusion)
                record["tuned_dp_correct"] = int(c1)
                record["tuned_dp_accuracy"] = float(c1 / max(1, t1))
            path_rows.append(record)

    per_df = pd.DataFrame(per_rows)
    per_path = out_dir / "candidate_recall_by_lecture.csv"
    per_df.to_csv(per_path, index=False)

    aggregate = [_aggregate(rows, matrix_name) for matrix_name, rows in matrix_rows.items()]
    aggregate_df = pd.DataFrame(aggregate).sort_values("weighted_top1", ascending=False)
    aggregate_path = out_dir / "candidate_recall_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    if path_rows:
        path_df = pd.DataFrame(path_rows)
        path_path = out_dir / "decoder_path_accuracy.csv"
        path_df.to_csv(path_path, index=False)
    else:
        path_path = None

    print("\nTOP-K CANDIDATE RECALL (higher means the correct slide is available to a reranker)")
    for _, row in aggregate_df.iterrows():
        print(
            f"{row['matrix']:<30} "
            f"Top1 macro/weighted={100*row['macro_top1']:.2f}%/{100*row['weighted_top1']:.2f}%  "
            f"Top3={100*row['weighted_top3']:.2f}%  Top5={100*row['weighted_top5']:.2f}%  "
            f"Top10={100*row['weighted_top10']:.2f}%  weighted MRR={row['weighted_mrr']:.4f}"
        )

    if path_rows:
        print("\nFULL-PATH DECODER ACCURACY")
        for r in path_rows:
            s = f"{r['lecture']:<30} baseline={100*r['baseline_dp_accuracy']:.2f}%"
            if "tuned_dp_accuracy" in r:
                s += f"  tuned={100*r['tuned_dp_accuracy']:.2f}%"
            print(s)

    summary = {
        "validation_only": True,
        "test_labels_loaded": False,
        "checkpoint": str(checkpoint),
        "v4_root": str(v4_root),
        "tuned_config_path": str(config_path) if config_payload else None,
        "lectures": list(dataset),
        "candidate_summary": aggregate,
        "decoder_path_results": path_rows,
        "outputs": {
            "per_lecture_csv": str(per_path),
            "summary_csv": str(aggregate_path),
            "decoder_csv": str(path_path) if path_path else None,
        },
    }
    summary_path = out_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved per-lecture candidate metrics: {per_path}")
    print(f"Saved aggregate candidate summary : {aggregate_path}")
    if path_path:
        print(f"Saved decoder path comparison    : {path_path}")
    print(f"Saved diagnostic JSON             : {summary_path}")
    print("\nInterpretation: if Top-5 recall is well below 90%, a decoder-only change cannot plausibly deliver 90% point accuracy.")


if __name__ == "__main__":
    main()
