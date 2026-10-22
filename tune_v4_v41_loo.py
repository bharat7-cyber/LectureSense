from __future__ import annotations

"""LectureSense V4.1: leave-one-validation-lecture-out fusion/decoder experiment.

This script imports the user's existing tune_v4_decoder_only.py as a utility
module. It uses cached score matrices only; it does not retrain or modify the
V4 checkpoint, and it does not load held-out test lecture labels.

Run from project root:
    python tune_v4_v41_loo.py

The four official validation lectures are used in four folds. In each fold,
three lectures select the candidate configuration and the fourth is evaluated
only after selection. Results are saved to runs/v4_v41_loo/.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tune_v4_decoder_only as tuner
except Exception as exc:
    raise SystemExit(
        "Could not import tune_v4_decoder_only.py. Keep this script beside that file "
        "and run it from your LectureSense project root.\n"
        f"{type(exc).__name__}: {exc}"
    )

OUT_DIR = ROOT / "runs" / "v4_v41_loo"
REPORT_PATH = OUT_DIR / "loo_report.json"
SWEEP_PATH = OUT_DIR / "fold_search.csv"

# Restricted, interpretable search ranges to reduce overfitting on only four
# validation lectures. Baseline values are always considered explicitly.
SPACE = {
    "residual_scale": [0.50, 0.75, 1.00, 1.25],
    "video_weight": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
    "score_temperature": [0.60, 0.80, 1.00, 1.25, 1.50, 2.00],
    "jump_penalty": [0.00, 0.02, 0.05, 0.10, 0.20],
    "max_forward_jump": [1, 2, 3],
    "max_duration_seconds": [60.0, 90.0, 120.0, 180.0, 240.0],
    "duration_penalty": [0.0005, 0.001, 0.003, 0.01],
    "max_run_seconds": [600.0, 900.0, 1200.0, 1800.0],
    "boundary_reward": [0.00, 0.50, 1.00, 1.50],
    "boundary_stay_penalty": [0.00, 0.25, 0.50, 1.00, 2.00],
    "transition_logits_weight": [0.00, 0.10, 0.20, 0.35, 0.50],
    "premature_transition_penalty": [0.00, 0.25, 0.50, 1.00],
    "boundary_threshold": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    # Parameters used only by confidence-adaptive fusion.
    "adaptive_gate_slope": [0.0, 2.0, 4.0, 8.0],
    "adaptive_gate_bias": [-1.0, 0.0, 1.0],
    "disagree_factor": [0.25, 0.50, 0.75, 1.00],
}
DECODER_KEYS = tuple(tuner.DECODER_KEYS)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def row_zscore(matrix):
    """Normalize each candidate-score row, keeping modality scales comparable."""
    x = np.asarray(matrix, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return np.clip((x - mean) / np.maximum(std, 1e-5), -8.0, 8.0)


def row_confidence(z_scores, temperature):
    """Normalized certainty (1 - entropy/log(slides)) for each row."""
    x = np.asarray(z_scores, dtype=np.float32) / max(1e-3, float(temperature))
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(np.clip(x, -40.0, 0.0))
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-8, 1.0)), axis=1)
    return np.clip(1.0 - entropy / max(1e-8, math.log(max(2, p.shape[1]))), 0.0, 1.0), np.argmax(p, axis=1)


def v4_pre_video_scores(item, residual_scale):
    """Rebuild V4 temporal scores from the cached base/residual matrices."""
    adjusted = np.asarray(item["adjusted"], dtype=np.float32)
    if item.get("can_residual") and item.get("base") is not None:
        base = np.asarray(item["base"], dtype=np.float32)
        return (base + float(residual_scale) * (adjusted - base)).astype(np.float32)
    return adjusted


def build_v41_scores(item, cfg, baseline_fusion_key):
    """Build fixed-fusion or confidence-adaptive score matrix for a candidate."""
    mode = cfg["fusion_mode"]
    temp = max(1e-3, float(cfg["score_temperature"]))

    if mode == "fixed":
        scores = tuner.build_scores(item, cfg, baseline_fusion_key)
        return np.asarray(scores, dtype=np.float32)

    # In adaptive mode, normalize V4 and visual scores per transcript segment,
    # then let visual evidence contribute more when it is confident and agrees
    # with the V4 candidate; disagreement reduces its contribution.
    v4_scores = v4_pre_video_scores(item, cfg["residual_scale"])
    video_scores = item.get("video")
    if not item.get("can_video") or video_scores is None:
        return (row_zscore(v4_scores) / temp).astype(np.float32)

    v4_z = row_zscore(v4_scores)
    video_z = row_zscore(video_scores)
    v4_conf, v4_top = row_confidence(v4_z, temp)
    video_conf, video_top = row_confidence(video_z, temp)

    gate = sigmoid(
        float(cfg["adaptive_gate_bias"])
        + float(cfg["adaptive_gate_slope"]) * (video_conf - v4_conf)
    )
    agree = (v4_top == video_top)
    agree_multiplier = np.where(agree, 1.0, float(cfg["disagree_factor"]))
    gate = gate * agree_multiplier

    weight = float(cfg["video_weight"])
    mixed = v4_z + weight * gate[:, None] * (video_z - v4_z)
    return (mixed / temp).astype(np.float32)


def full_baseline_candidate(baseline):
    cfg = dict(baseline)
    cfg.update({
        "fusion_mode": "fixed",
        "adaptive_gate_slope": 4.0,
        "adaptive_gate_bias": 0.0,
        "disagree_factor": 0.50,
    })
    return cfg


def candidate_key(cfg):
    keys = tuple(tuner.ALL_KEYS) + ("fusion_mode", "adaptive_gate_slope", "adaptive_gate_bias", "disagree_factor")
    return tuple((k, str(cfg[k]) if isinstance(cfg[k], str) else round(float(cfg[k]), 8)) for k in keys)


def random_candidate(rng, baseline, fold_dataset):
    cfg = dict(baseline)
    can_residual = any(item.get("can_residual") for item in fold_dataset.values())
    can_video = any(item.get("can_video") for item in fold_dataset.values())

    cfg["fusion_mode"] = rng.choice(["fixed", "adaptive"] if can_video else ["fixed"])
    for key, values in SPACE.items():
        if key in DECODER_KEYS or key in ("residual_scale", "video_weight", "score_temperature"):
            if key == "residual_scale" and not can_residual:
                cfg[key] = baseline[key]
            elif key == "video_weight" and not can_video:
                cfg[key] = baseline[key]
            else:
                # Keep some values anchored to the baseline, especially with
                # only three training lectures in a fold.
                if key in baseline and rng.random() < 0.35:
                    cfg[key] = baseline[key]
                else:
                    cfg[key] = rng.choice(values)
        else:
            cfg[key] = rng.choice(values)
    if cfg["fusion_mode"] == "adaptive" and not can_video:
        cfg["fusion_mode"] = "fixed"
    return cfg


def changed_count(cfg, baseline):
    keys = list(tuner.ALL_KEYS) + ["fusion_mode", "adaptive_gate_slope", "adaptive_gate_bias", "disagree_factor"]
    return sum(1 for k in keys if cfg.get(k) != (baseline.get(k) if k in baseline else None) and
               not (k == "fusion_mode" and cfg.get(k) == "fixed"))


def evaluate_one(item, cfg, baseline_fusion_key):
    scores = build_v41_scores(item, cfg, baseline_fusion_key)
    path = tuner.decode(item, scores, cfg)
    total = len(item["gt_labels"])
    if path is None:
        return 0, total
    return int(np.sum(path[item["gt_indices"]] == item["gt_labels"])), total


def evaluate_dataset(dataset, cfg, baseline_fusion_key):
    counts = {}
    for name, item in dataset.items():
        counts[name] = evaluate_one(item, cfg, baseline_fusion_key)
    per_lecture = {name: c / max(1, t) for name, (c, t) in counts.items()}
    correct = sum(c for c, _ in counts.values())
    total = sum(t for _, t in counts.values())
    return {
        "macro": float(np.mean(list(per_lecture.values()))) if per_lecture else 0.0,
        "weighted": correct / max(1, total),
        "per_lecture": per_lecture,
        "counts": counts,
        "correct": correct,
        "total": total,
    }


def config_to_json(cfg):
    keys = list(tuner.ALL_KEYS) + ["fusion_mode", "adaptive_gate_slope", "adaptive_gate_bias", "disagree_factor"]
    return {k: cfg[k] for k in keys}


def main():
    ap = argparse.ArgumentParser(description="LectureSense V4.1 leave-one-validation-lecture-out experiment")
    ap.add_argument("--v4-root", default=str(tuner.DEFAULT_V4_ROOT), help="Cached V4 validation matrices root")
    ap.add_argument("--checkpoint", default=str(tuner.DEFAULT_CHECKPOINT), help="V4 checkpoint; read-only")
    ap.add_argument("--trials-per-fold", type=int, default=40, help="Random candidate configurations per fold")
    ap.add_argument("--max-minutes-per-fold", type=float, default=7.0, help="Hard search budget per fold")
    ap.add_argument("--seed", type=int, default=1041)
    ap.add_argument("--skip-dp-verify", action="store_true", help="Skip startup fast-DP equivalence checks")
    args = ap.parse_args()

    if args.trials_per_fold < 1:
        ap.error("--trials-per-fold must be at least 1")
    if args.max_minutes_per_fold <= 0:
        ap.error("--max-minutes-per-fold must be positive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    v4_root = Path(args.v4_root)
    checkpoint = Path(args.checkpoint)

    print("\n" + "=" * 78)
    print("LECTURESENSE V4.1 — LEAVE-ONE-VALIDATION-LECTURE-OUT")
    print("=" * 78)
    print(f"Validation cache : {v4_root}")
    print(f"V4 checkpoint    : {checkpoint} (read-only; will not be modified)")
    print(f"Trials per fold  : {args.trials_per_fold}")
    print("Test lectures    : NOT LOADED")

    checkpoint_decoder = tuner.load_checkpoint_decoder(checkpoint)
    baseline = tuner.make_baseline_config(checkpoint_decoder)
    baseline_candidate = full_baseline_candidate(baseline)
    base_fusion_key = tuner.baseline_fusion_key(baseline)
    dataset = tuner.load_dataset(tuner.lecture_specs("validation", v4_root))

    if not args.skip_dp_verify:
        tuner.verify_fast_dp(dataset, baseline, rng)

    print("\nLoading complete. Evaluating the unchanged production V4 baseline...")
    baseline_all = evaluate_dataset(dataset, baseline_candidate, base_fusion_key)
    for name, acc in baseline_all["per_lecture"].items():
        c, t = baseline_all["counts"][name]
        print(f"  baseline {name:<28} {c:4d}/{t:<4d} {100*acc:6.2f}%")
    print(f"  baseline validation macro={100*baseline_all['macro']:.2f}% weighted={100*baseline_all['weighted']:.2f}%")

    fold_results = []
    sweep_rows = []
    names = list(dataset)

    for fold_idx, held_name in enumerate(names, start=1):
        train_names = [n for n in names if n != held_name]
        train_data = {n: dataset[n] for n in train_names}
        held_item = dataset[held_name]
        fold_rng = random.Random(args.seed + fold_idx * 1009)
        baseline_train = evaluate_dataset(train_data, baseline_candidate, base_fusion_key)
        deadline = time.monotonic() + args.max_minutes_per_fold * 60.0

        print("\n" + "=" * 78)
        print(f"FOLD {fold_idx}/{len(names)} — held out: {held_name}")
        print(f"Tune on: {', '.join(train_names)}")
        print(f"Held-out lecture labels are used only once, after candidate selection.")
        print("=" * 78)

        candidates = [baseline_candidate]
        seen = {candidate_key(baseline_candidate)}
        tries = 0
        while len(candidates) < args.trials_per_fold + 1 and time.monotonic() < deadline:
            cand = random_candidate(fold_rng, baseline, train_data)
            key = candidate_key(cand)
            if key in seen:
                tries += 1
                if tries > args.trials_per_fold * 20:
                    break
                continue
            seen.add(key)
            candidates.append(cand)

        best_cfg = baseline_candidate
        best_train = baseline_train
        best_tie = changed_count(best_cfg, baseline)
        completed = 0
        for cand_i, cand in enumerate(candidates, start=1):
            if cand_i > 1 and time.monotonic() > deadline:
                print("  Fold time budget reached; selecting best candidate evaluated so far.")
                break
            train_res = evaluate_dataset(train_data, cand, base_fusion_key)
            completed += 1
            sweep_rows.append({
                "fold": fold_idx,
                "held_out_lecture": held_name,
                "candidate": cand_i,
                "train_macro": train_res["macro"],
                "train_weighted": train_res["weighted"],
                **{f"train_acc_{k}": v for k, v in train_res["per_lecture"].items()},
                **config_to_json(cand),
            })
            tie = changed_count(cand, baseline)
            if (train_res["macro"] > best_train["macro"] + 1e-10 or
                (abs(train_res["macro"] - best_train["macro"]) <= 1e-10 and tie < best_tie)):
                best_cfg, best_train, best_tie = cand, train_res, tie
            if cand_i == 1 or cand_i % 10 == 0 or cand_i == len(candidates):
                print(f"  candidate {cand_i:3d}/{len(candidates):3d} | best train macro {100*best_train['macro']:.2f}%")

        # Held-out lecture is evaluated only for the selected fold config.
        held_base_c, held_total = evaluate_one(held_item, baseline_candidate, base_fusion_key)
        held_c, held_t = evaluate_one(held_item, best_cfg, base_fusion_key)
        assert held_total == held_t
        fold = {
            "fold": fold_idx,
            "held_out_lecture": held_name,
            "training_lectures": train_names,
            "candidates_evaluated": completed,
            "baseline_train_macro": baseline_train["macro"],
            "selected_train_macro": best_train["macro"],
            "held_out_baseline": {"correct": held_base_c, "total": held_total, "accuracy": held_base_c / max(1, held_total)},
            "held_out_selected": {"correct": held_c, "total": held_t, "accuracy": held_c / max(1, held_t)},
            "selected_config": config_to_json(best_cfg),
        }
        fold_results.append(fold)
        print(f"\n  Fold result on held-out {held_name}:")
        print(f"    baseline: {held_base_c}/{held_total} ({100*held_base_c/max(1,held_total):.2f}%)")
        print(f"    V4.1    : {held_c}/{held_t} ({100*held_c/max(1,held_t):.2f}%)")
        print(f"    selected config (trained on other 3 lectures): {config_to_json(best_cfg)}")

    # Aggregate LOO score is calculated from the four held-out fold predictions.
    loo_base_correct = sum(x["held_out_baseline"]["correct"] for x in fold_results)
    loo_tuned_correct = sum(x["held_out_selected"]["correct"] for x in fold_results)
    loo_total = sum(x["held_out_selected"]["total"] for x in fold_results)
    loo_base_macro = float(np.mean([x["held_out_baseline"]["accuracy"] for x in fold_results]))
    loo_tuned_macro = float(np.mean([x["held_out_selected"]["accuracy"] for x in fold_results]))
    loo_base_weighted = loo_base_correct / max(1, loo_total)
    loo_tuned_weighted = loo_tuned_correct / max(1, loo_total)

    # Choose no universal deployed config here: each fold config was selected
    # without its held-out lecture. The report is an honest CV estimate, not a
    # production configuration tuned on all four validation lectures.
    report = {
        "project": "LectureSense V4.1",
        "experiment": "leave_one_validation_lecture_out",
        "created_unix_time": time.time(),
        "checkpoint": str(checkpoint),
        "checkpoint_modified": False,
        "test_lectures_loaded": False,
        "validation_root": str(v4_root),
        "trials_per_fold_requested": args.trials_per_fold,
        "max_minutes_per_fold": args.max_minutes_per_fold,
        "seed": args.seed,
        "validation_lectures": names,
        "all_validation_baseline": {
            "macro": baseline_all["macro"], "weighted": baseline_all["weighted"],
            "per_lecture": baseline_all["per_lecture"],
            "counts": {k: {"correct": v[0], "total": v[1]} for k, v in baseline_all["counts"].items()},
        },
        "loo_baseline": {"macro": loo_base_macro, "weighted": loo_base_weighted,
                         "correct": loo_base_correct, "total": loo_total},
        "loo_v41": {"macro": loo_tuned_macro, "weighted": loo_tuned_weighted,
                    "correct": loo_tuned_correct, "total": loo_total},
        "loo_improvement_percentage_points": {
            "macro": 100.0 * (loo_tuned_macro - loo_base_macro),
            "weighted": 100.0 * (loo_tuned_weighted - loo_base_weighted),
        },
        "folds": fold_results,
        "interpretation": (
            "Each held-out validation lecture was excluded from its fold's configuration search. "
            "This estimates the configuration-selection procedure across the four development lectures; "
            "it does not guarantee held-out test improvement. No single deployment config was fitted to "
            "all validation lectures by this script. The formal test set was not loaded."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if sweep_rows:
        keys = []
        for row in sweep_rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with SWEEP_PATH.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(sweep_rows)

    print("\n" + "=" * 78)
    print("V4.1 LEAVE-ONE-OUT SUMMARY")
    print("=" * 78)
    print(f"Baseline LOO macro    : {100*loo_base_macro:.2f}%")
    print(f"V4.1 LOO macro        : {100*loo_tuned_macro:.2f}% ({100*(loo_tuned_macro-loo_base_macro):+.2f} pp)")
    print(f"Baseline LOO weighted : {100*loo_base_weighted:.2f}%")
    print(f"V4.1 LOO weighted     : {100*loo_tuned_weighted:.2f}% ({100*(loo_tuned_weighted-loo_base_weighted):+.2f} pp)")
    print(f"Total held-out points : {loo_tuned_correct}/{loo_total}")
    print(f"Report                : {REPORT_PATH}")
    print(f"Search rows           : {SWEEP_PATH}")
    print("\nDo not use the held-out test set to select fusion/decoder parameters.")


if __name__ == "__main__":
    main()
