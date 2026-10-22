from __future__ import annotations

"""
Evaluate the frozen V4.1 configuration once on the held-out test lectures.

This script does no search, writes no configuration, and does not change the
V4 model checkpoint. The configuration must first be selected on validation
lectures using tune_v4_v41_final_config.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tune_v4_v41_loo as v41
except Exception as exc:
    raise SystemExit(
        "Could not import tune_v4_v41_loo.py. Keep all V4.1 scripts in the "
        f"LectureSense project root.\n{type(exc).__name__}: {exc}"
    )

tuner = v41.tuner

DEFAULT_CONFIG = ROOT / "checkpoints" / "v4_decoder_tuned_config.json"
DEFAULT_TEST_ROOT = ROOT / "runs" / "formal_test_eval_v4"
REPORT_PATH = ROOT / "runs" / "v4_v41_final_config" / "test_report.json"

TEST_FOLDER_MAP = {
    "short_range": "short_range_mit",
    "solar_resource": "solar_resource",
    "team_dynamics_game_design_MIT": "team_dynamics_game_design_MIT",
    "theory_of_computation": "theory_of_computation",
}


def load_test_specs(test_root: Path) -> dict:
    specs = tuner.lecture_specs("test", test_root)
    for name, folder in TEST_FOLDER_MAP.items():
        if name not in specs:
            raise RuntimeError(f"Test specification missing lecture: {name}")
        specs[name]["v4_dir"] = test_root / folder
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one locked V4.1 configuration on held-out test lectures"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    test_root = Path(args.test_root).expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(
            f"Selected config not found: {config_path}\n"
            "Run tune_v4_v41_final_config.py on validation data first."
        )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("selected_on") != "validation_only":
        raise SystemExit(
            "Refusing to evaluate: config is not marked as selected_on='validation_only'."
        )
    if payload.get("test_labels_loaded", True) is not False:
        raise SystemExit(
            "Refusing to evaluate: config does not confirm that test labels were unused."
        )

    required = [
    "residual_scale",
    "video_weight",
    "score_temperature",
    "jump_penalty",
    "max_forward_jump",
    "max_duration_seconds",
    "duration_penalty",
    "max_run_seconds",
    "boundary_reward",
    "boundary_stay_penalty",
    "transition_logits_weight",
    "premature_transition_penalty",
    "boundary_threshold",
    "fusion_mode",
    "adaptive_gate_slope",
    "adaptive_gate_bias",
    "disagree_factor",
]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(
            f"Selected config is missing keys {missing}. "
            "Regenerate it with tune_v4_v41_final_config.py."
        )

    cfg = {key: payload[key] for key in required}
    checkpoint = Path(
        args.checkpoint or payload.get("checkpoint", tuner.DEFAULT_CHECKPOINT)
    ).expanduser().resolve()
    test_root_arg = test_root

    print("=" * 78)
    print("LECTURESENSE V4.1 — LOCKED HELD-OUT TEST EVALUATION")
    print("=" * 78)
    print(f"Configuration : {config_path}")
    print(f"Test cache    : {test_root_arg}")
    print(f"Checkpoint    : {checkpoint} (read-only)")
    print("Search/tuning : DISABLED")
    print("Configuration : will not be modified")
    print()

    checkpoint_decoder = tuner.load_checkpoint_decoder(checkpoint)
    baseline = tuner.make_baseline_config(checkpoint_decoder)
    baseline_candidate = v41.full_baseline_candidate(baseline)
    baseline_fusion_key = tuner.baseline_fusion_key(baseline)

    specs = load_test_specs(test_root_arg)
    dataset = tuner.load_dataset(specs)
    if len(dataset) != 4:
        raise SystemExit(f"Expected 4 held-out test lectures, loaded {len(dataset)}.")

    baseline_metrics = v41.evaluate_dataset(
        dataset, baseline_candidate, baseline_fusion_key
    )
    tuned_metrics = v41.evaluate_dataset(dataset, cfg, baseline_fusion_key)

    print(f"{'Lecture':<36}{'Baseline':>16}{'V4.1 selected':>18}")
    print("-" * 70)
    for name in dataset:
        bc, bt = baseline_metrics["counts"][name]
        tc, tt = tuned_metrics["counts"][name]
        if bt != tt:
            raise RuntimeError(f"Ground-truth row count changed for {name}: {bt} != {tt}")
        print(
            f"{name:<36}{bc:>5}/{bt:<10} {100*bc/max(1,bt):>6.2f}%"
            f"{tc:>5}/{tt:<10} {100*tc/max(1,tt):>6.2f}%"
        )

    print("-" * 70)
    print(
        f"Macro accuracy       : baseline {100*baseline_metrics['macro']:.2f}%"
        f" | V4.1 {100*tuned_metrics['macro']:.2f}%"
    )
    print(
        f"Weighted accuracy    : baseline {100*baseline_metrics['weighted']:.2f}%"
        f" | V4.1 {100*tuned_metrics['weighted']:.2f}%"
    )
    print(
        f"Correct / total      : {tuned_metrics['correct']}/{tuned_metrics['total']}"
    )
    print(
        "\nInterpretation: this is one locked test evaluation. Do not tune the "
        "configuration based on these results; changes after inspecting test "
        "accuracy would turn the test set into validation data."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "locked_v4_1_test_evaluation",
        "configuration_path": str(config_path),
        "test_root": str(test_root_arg),
        "checkpoint": str(checkpoint),
        "search_performed": False,
        "configuration_modified": False,
        "test_labels_used_for_selection": False,
        "baseline": {
            "macro": baseline_metrics["macro"],
            "weighted": baseline_metrics["weighted"],
            "per_lecture": baseline_metrics["per_lecture"],
            "counts": {
                name: {"correct": int(c), "total": int(t)}
                for name, (c, t) in baseline_metrics["counts"].items()
            },
        },
        "selected_v4_1": {
            "macro": tuned_metrics["macro"],
            "weighted": tuned_metrics["weighted"],
            "correct": int(tuned_metrics["correct"]),
            "total": int(tuned_metrics["total"]),
            "per_lecture": tuned_metrics["per_lecture"],
            "counts": {
                name: {"correct": int(c), "total": int(t)}
                for name, (c, t) in tuned_metrics["counts"].items()
            },
        },
        "delta_percentage_points": {
            "macro": 100.0 * (tuned_metrics["macro"] - baseline_metrics["macro"]),
            "weighted": 100.0 * (tuned_metrics["weighted"] - baseline_metrics["weighted"]),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print(f"Saved test report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
