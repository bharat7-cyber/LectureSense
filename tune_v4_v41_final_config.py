from __future__ import annotations

"""Select one global LectureSense V4.1 fusion/decoder configuration.

This script tunes one shared configuration on all four official validation
lectures using cached V4 matrices. It does not retrain or modify the V4 model
checkpoint and does not load held-out test lecture labels.

Place this file in the LectureSense project root beside:
  - tune_v4_decoder_only.py
  - tune_v4_v41_loo.py

Run:
  python tune_v4_v41_final_config.py

Optional:
  python tune_v4_v41_final_config.py --trials 160 --max-minutes 12

Outputs:
  checkpoints/v4_decoder_tuned_config.json
  runs/v4_v41_final_config/final_report.json
  runs/v4_v41_final_config/validation_sweep.csv

Note: because all four validation lectures are used to select the final global
configuration, its displayed validation score is a configuration-selection
score, not an unbiased estimate of generalization.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tune_v4_v41_loo as v41
except Exception as exc:
    raise SystemExit(
        "Could not import tune_v4_v41_loo.py. Keep this script beside that file "
        "and tune_v4_decoder_only.py in your LectureSense project root.\n"
        f"{type(exc).__name__}: {exc}"
    )

tuner = v41.tuner
OUT_DIR = ROOT / "runs" / "v4_v41_final_config"
REPORT_PATH = OUT_DIR / "final_report.json"
SWEEP_PATH = OUT_DIR / "validation_sweep.csv"
CONFIG_PATH = ROOT / "checkpoints" / "v4_decoder_tuned_config.json"
LOO_REPORT_PATH = ROOT / "runs" / "v4_v41_loo" / "loo_report.json"
EXTRA_KEYS = ("fusion_mode", "adaptive_gate_slope", "adaptive_gate_bias", "disagree_factor")
ALL_CONFIG_KEYS = tuple(tuner.ALL_KEYS) + EXTRA_KEYS


def as_config(raw: dict, baseline: dict) -> dict:
    """Normalize a candidate so every config key is populated."""
    cfg = v41.full_baseline_candidate(baseline)
    cfg.update(raw or {})
    for key in ALL_CONFIG_KEYS:
        if key not in cfg:
            if key == "fusion_mode":
                cfg[key] = "fixed"
            elif key == "adaptive_gate_slope":
                cfg[key] = 4.0
            elif key == "adaptive_gate_bias":
                cfg[key] = 0.0
            elif key == "disagree_factor":
                cfg[key] = 0.5
            else:
                raise KeyError(f"Configuration is missing required key: {key}")
    if cfg["fusion_mode"] not in ("fixed", "adaptive"):
        cfg["fusion_mode"] = "fixed"
    return cfg


def load_loo_seed_candidates(baseline: dict) -> list[tuple[str, dict]]:
    """Reuse fold-selected configurations as starting candidates, not as test data."""
    seeds: list[tuple[str, dict]] = []
    if not LOO_REPORT_PATH.exists():
        print(f"LOO report not found; continuing without LOO seeds: {LOO_REPORT_PATH}")
        return seeds

    try:
        report = json.loads(LOO_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read LOO report; continuing without LOO seeds ({exc})")
        return seeds

    folds = report.get("folds", [])
    fold_cfgs: list[dict] = []
    for fold in folds:
        raw = fold.get("selected_config")
        if isinstance(raw, dict):
            cfg = as_config(raw, baseline)
            seeds.append((f"loo_fold_{fold.get('fold', len(seeds) + 1)}", cfg))
            fold_cfgs.append(cfg)

    # A consensus seed uses the median for numeric parameters and the most
    # common category for categorical parameters. It is only a search seed;
    # the final config is selected by evaluating the global validation set.
    if fold_cfgs:
        consensus = as_config({}, baseline)
        for key in ALL_CONFIG_KEYS:
            values = [cfg[key] for cfg in fold_cfgs]
            if key == "fusion_mode":
                counts = Counter(values)
                best = max(counts.values())
                tied = [value for value, count in counts.items() if count == best]
                consensus[key] = baseline.get(key, "fixed") if baseline.get(key) in tied else sorted(tied)[0]
            elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                med = median(float(value) for value in values)
                if key == "max_forward_jump":
                    consensus[key] = max(1, int(round(med)))
                else:
                    consensus[key] = float(med)
            else:
                counts = Counter(values)
                best = max(counts.values())
                tied = [value for value, count in counts.items() if count == best]
                consensus[key] = baseline.get(key, tied[0]) if baseline.get(key) in tied else sorted(tied, key=str)[0]
        seeds.append(("loo_consensus", consensus))
    return seeds


def objective_value(metrics: dict, objective: str) -> float:
    return float(metrics[objective])


def candidate_tiebreak(cfg: dict, metrics: dict, baseline: dict) -> tuple[float, float, int]:
    """Prefer higher weighted accuracy, then the simpler/less changed config."""
    return (
        float(metrics["weighted"]),
        -int(v41.changed_count(cfg, v41.full_baseline_candidate(baseline))),
        0,
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a single global V4.1 configuration using validation lectures only"
    )
    parser.add_argument(
        "--v4-root",
        default=str(tuner.DEFAULT_V4_ROOT),
        help="Directory containing cached validation score matrices",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(tuner.DEFAULT_CHECKPOINT),
        help="Existing V4 checkpoint (read-only)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=160,
        help="Maximum total configurations evaluated, including baseline and LOO seeds",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=12.0,
        help="Wall-clock budget for global configuration search",
    )
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--objective", choices=("macro", "weighted"), default="macro")
    parser.add_argument(
        "--skip-dp-verify",
        action="store_true",
        help="Skip the fast-DP equivalence check",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    v4_root = Path(args.v4_root).expanduser().resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("\n" + "=" * 78)
    print("LECTURESENSE V4.1 — GLOBAL VALIDATION CONFIGURATION SELECTION")
    print("=" * 78)
    print(f"Validation cache : {v4_root}")
    print(f"V4 checkpoint    : {checkpoint} (read-only)")
    print(f"Trial limit      : {args.trials}")
    print(f"Time budget      : {args.max_minutes:.1f} minutes")
    print(f"Selection metric : validation {args.objective} accuracy")
    print("Test labels      : NOT LOADED")

    checkpoint_decoder = tuner.load_checkpoint_decoder(checkpoint)
    baseline = tuner.make_baseline_config(checkpoint_decoder)
    baseline_candidate = v41.full_baseline_candidate(baseline)
    baseline_fusion_key = tuner.baseline_fusion_key(baseline)
    dataset = tuner.load_dataset(tuner.lecture_specs("validation", v4_root))
    if len(dataset) < 2:
        raise SystemExit(f"Expected multiple validation lectures, found {len(dataset)}")

    print("\nValidation lectures loaded:")
    for name, item in dataset.items():
        print(
            f"  {name:<28} GT={len(item['gt_labels']):4d} "
            f"V4={item['adjusted'].shape} video={'YES' if item.get('can_video') else 'NO'}"
        )

    if not args.skip_dp_verify:
        print("\nVerifying fast DP implementation...")
        tuner.verify_fast_dp(dataset, baseline, rng)

    print("\nEvaluating baseline across all validation lectures...")
    baseline_metrics = v41.evaluate_dataset(dataset, baseline_candidate, baseline_fusion_key)
    for name, accuracy in baseline_metrics["per_lecture"].items():
        correct, total = baseline_metrics["counts"][name]
        print(f"  baseline {name:<28} {correct:4d}/{total:<4d} {100*accuracy:6.2f}%")
    print(
        f"Baseline macro={100*baseline_metrics['macro']:.2f}% "
        f"weighted={100*baseline_metrics['weighted']:.2f}%"
    )

    # Include the baseline and configurations that generalized to one held-out
    # validation lecture in the previous LOO experiment. Then search globally.
    seed_candidates: list[tuple[str, dict]] = [("baseline", baseline_candidate)]
    seed_candidates.extend(load_loo_seed_candidates(baseline))
    candidates: list[tuple[str, dict]] = []
    seen: set[tuple] = set()
    for source, cfg in seed_candidates:
        key = v41.candidate_key(cfg)
        if key not in seen:
            seen.add(key)
            candidates.append((source, cfg))

    # Reserve room for random candidates while respecting --trials.
    candidates = candidates[: args.trials]
    attempts = 0
    max_attempts = max(100, args.trials * 50)
    while len(candidates) < args.trials and attempts < max_attempts:
        attempts += 1
        cfg = v41.random_candidate(rng, baseline, dataset)
        key = v41.candidate_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(("random", cfg))

    print(f"\nUnique configurations queued: {len(candidates)}")
    deadline = time.monotonic() + args.max_minutes * 60.0
    best_cfg = baseline_candidate
    best_metrics = baseline_metrics
    best_source = "baseline"
    best_value = objective_value(baseline_metrics, args.objective)
    best_tie = candidate_tiebreak(best_cfg, best_metrics, baseline)
    rows: list[dict] = []
    completed = 0

    for index, (source, cfg) in enumerate(candidates, start=1):
        if index > 1 and time.monotonic() >= deadline:
            print("Time budget reached; keeping the best candidate evaluated so far.")
            break
        try:
            metrics = v41.evaluate_dataset(dataset, cfg, baseline_fusion_key)
            value = objective_value(metrics, args.objective)
            tie = candidate_tiebreak(cfg, metrics, baseline)
        except Exception as exc:
            print(f"  candidate {index:3d}/{len(candidates):3d} failed: {type(exc).__name__}: {exc}")
            continue

        completed += 1
        row = {
            "candidate": index,
            "source": source,
            "objective": args.objective,
            "selection_score": value,
            "validation_macro": metrics["macro"],
            "validation_weighted": metrics["weighted"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            **{f"acc_{name}": acc for name, acc in metrics["per_lecture"].items()},
            **v41.config_to_json(cfg),
        }
        rows.append(row)

        better = value > best_value + 1e-10
        tied = abs(value - best_value) <= 1e-10
        if better or (tied and tie > best_tie):
            best_cfg = dict(cfg)
            best_metrics = metrics
            best_source = source
            best_value = value
            best_tie = tie

        if index == 1 or index % 10 == 0 or index == len(candidates):
            print(
                f"  candidate {index:3d}/{len(candidates):3d} [{source:<14}] "
                f"best {args.objective}={100*best_value:.2f}% "
                f"(current={100*value:.2f}%)"
            )

    if completed == 0:
        raise SystemExit("No candidate could be evaluated. Review validation cache paths and logs.")

    write_csv(rows, SWEEP_PATH)
    final_cfg = v41.config_to_json(best_cfg)
    selected_summary = {
        "project": "LectureSense V4.1",
        "method": "global validation-only fusion and decoder configuration selection",
        "selected_config": final_cfg,
        "selection_metric": args.objective,
        "selected_candidate_source": best_source,
        "validation_selection_metrics": {
            "macro": best_metrics["macro"],
            "weighted": best_metrics["weighted"],
            "correct": best_metrics["correct"],
            "total": best_metrics["total"],
            "per_lecture": best_metrics["per_lecture"],
            "counts": {
                name: {"correct": int(pair[0]), "total": int(pair[1])}
                for name, pair in best_metrics["counts"].items()
            },
        },
        "baseline_validation_metrics": {
            "macro": baseline_metrics["macro"],
            "weighted": baseline_metrics["weighted"],
            "correct": baseline_metrics["correct"],
            "total": baseline_metrics["total"],
            "per_lecture": baseline_metrics["per_lecture"],
            "counts": {
                name: {"correct": int(pair[0]), "total": int(pair[1])}
                for name, pair in baseline_metrics["counts"].items()
            },
        },
        "validation_improvement_percentage_points": {
            "macro": 100.0 * (best_metrics["macro"] - baseline_metrics["macro"]),
            "weighted": 100.0 * (best_metrics["weighted"] - baseline_metrics["weighted"]),
        },
        "validation_lectures": list(dataset.keys()),
        "checkpoint": str(checkpoint),
        "checkpoint_modified": False,
        "test_labels_loaded": False,
        "trial_limit": args.trials,
        "candidates_evaluated": completed,
        "time_budget_minutes": args.max_minutes,
        "seed": args.seed,
        "objective": args.objective,
        "warning": (
            "The selected metrics are measured on the same validation lectures used to choose this global config. "
            "They are selection metrics, not an unbiased generalization estimate. Freeze this config before any "
            "formal test evaluation. The existing V4 checkpoint remains unchanged."
        ),
    }

    # Human-readable deployment config: keep its parameters at top-level while
    # retaining selection provenance. Inference will not read this file unless
    # explicitly wired to do so or its values are passed through the CLI.
    config_file = {
        **final_cfg,
        "project": "LectureSense V4.1",
        "checkpoint": str(checkpoint),
        "selected_on": "validation_only",
        "selection_metric": args.objective,
        "validation_selection_macro": best_metrics["macro"],
        "validation_selection_weighted": best_metrics["weighted"],
        "checkpoint_modified": False,
        "test_labels_loaded": False,
        "warning": selected_summary["warning"],
    }
    CONFIG_PATH.write_text(json.dumps(config_file, indent=2, default=float), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(selected_summary, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 78)
    print("V4.1 GLOBAL CONFIGURATION SELECTION COMPLETE")
    print("=" * 78)
    print(f"Baseline validation macro    : {100*baseline_metrics['macro']:.2f}%")
    print(f"Selected validation macro    : {100*best_metrics['macro']:.2f}% "
          f"({100*(best_metrics['macro']-baseline_metrics['macro']):+.2f} pp)")
    print(f"Baseline validation weighted : {100*baseline_metrics['weighted']:.2f}%")
    print(f"Selected validation weighted : {100*best_metrics['weighted']:.2f}% "
          f"({100*(best_metrics['weighted']-baseline_metrics['weighted']):+.2f} pp)")
    print(f"Selected candidate source    : {best_source}")
    print("\nPer-lecture selected validation results:")
    for name, accuracy in best_metrics["per_lecture"].items():
        correct, total = best_metrics["counts"][name]
        print(f"  {name:<28} {correct:4d}/{total:<4d} {100*accuracy:6.2f}%")
    print(f"\nConfig file : {CONFIG_PATH}")
    print(f"Full report : {REPORT_PATH}")
    print(f"Sweep CSV   : {SWEEP_PATH}")
    print("V4 .pt checkpoint: unchanged")
    print("Test labels: not loaded or used")
    print("IMPORTANT: validation score is used for config selection; it is not a final test accuracy.")
    print("Next: freeze this config, then run the formal evaluation once and report its result separately.")


if __name__ == "__main__":
    main()
