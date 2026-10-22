from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the existing validation-artifact loader only.
# No new OCR, SBERT, ResNet or video extraction is performed.
import tune_v4_decoder_only as source_tuner
from boundary_dp_v4 import boundary_aware_dp

OUTPUT_DIR = ROOT / "runs" / "v4_fast_accuracy_tuning"
CONFIG_PATH = ROOT / "checkpoints" / "v4_fast_tuned_config.json"
CSV_PATH = OUTPUT_DIR / "validation_results.csv"

DEFAULT_DECODER = {
    "jump_penalty": 0.05,
    "max_forward_jump": 1,
    "max_duration_seconds": 120.0,
    "duration_penalty": 0.001,
    "max_run_seconds": 1200.0,
    "boundary_reward": 1.0,
    "boundary_stay_penalty": 0.50,
    "transition_logits_weight": 0.35,
    "premature_transition_penalty": 0.50,
    "boundary_threshold": 0.20,
}


def row_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.maximum(np.std(x, axis=1, keepdims=True), 1e-4)
    return ((x - mean) / std).astype(np.float32)


def row_softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    t = max(1e-6, float(temperature))
    z = x / t
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return (e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def confidence_features(scores):
    probs = row_softmax(scores)
    order = np.argsort(probs, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2] if scores.shape[1] >= 2 else top1_idx
    p1 = probs[np.arange(len(probs)), top1_idx]
    p2 = probs[np.arange(len(probs)), top2_idx]
    margin = np.clip(p1 - p2, 0.0, 1.0)
    entropy_den = max(1e-8, math.log(max(2, scores.shape[1])))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1) / entropy_den
    entropy = np.clip(entropy, 0.0, 1.0)
    return top1_idx, margin.astype(np.float32), entropy.astype(np.float32)


def adaptive_visual_fusion(
    v4_scores,
    video_scores,
    visual_weight=0.0,
    v4_temperature=1.0,
    visual_temperature=1.0,
    uncertainty_power=1.0,
    visual_confidence_power=1.0,
    disagreement_scale=0.55,
):
    v4_scores = np.asarray(v4_scores, dtype=np.float32)
    if video_scores is None:
        return v4_scores.copy()

    video_scores = np.asarray(video_scores, dtype=np.float32)
    if video_scores.shape != v4_scores.shape:
        return v4_scores.copy()

    if float(visual_weight) <= 0.0:
        return v4_scores.copy()

    v4_z = row_zscore(v4_scores / max(1e-6, float(v4_temperature)))
    vis_z = row_zscore(video_scores / max(1e-6, float(visual_temperature)))

    v4_top, v4_margin, v4_entropy = confidence_features(v4_scores)
    vis_top, vis_margin, vis_entropy = confidence_features(video_scores)

    # Trust video mainly when V4 is uncertain and video is itself confident.
    v4_uncertainty = np.clip((1.0 - v4_entropy) * (1.0 - v4_margin), 0.0, 1.0)
    v4_uncertainty = np.power(np.maximum(v4_uncertainty, 1e-6), float(uncertainty_power))

    vis_confidence = np.clip(
        0.65 * vis_margin + 0.35 * (1.0 - vis_entropy),
        0.0,
        1.0,
    )
    vis_confidence = np.power(np.maximum(vis_confidence, 1e-6), float(visual_confidence_power))

    agreement = np.where(v4_top == vis_top, 1.0, float(disagreement_scale))
    gate = np.clip(v4_uncertainty * vis_confidence * agreement, 0.0, 1.0)

    return (
        v4_z + float(visual_weight) * gate[:, None] * vis_z
    ).astype(np.float32)


def fast_monotonic_viterbi(
    scores,
    boundary,
    next_logits,
    starts,
    ends,
    jump_penalty,
    max_forward_jump,
    boundary_reward,
    boundary_stay_penalty,
    transition_logits_weight,
    premature_transition_penalty,
    boundary_threshold,
    duration_penalty,
    max_duration_seconds,
):
    """Fast O(N*S*J) monotonic decoder for parameter search."""
    scores = np.asarray(scores, dtype=np.float32)
    boundary = np.clip(np.asarray(boundary, dtype=np.float32).reshape(-1), 1e-4, 1.0 - 1e-4)
    next_logits = np.asarray(next_logits, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)

    n, s = scores.shape
    if n == 0 or s == 0:
        return np.zeros(n, dtype=np.int64)

    max_jump = max(1, int(max_forward_jump))

    next_logprob = next_logits.astype(np.float64)
    next_logprob -= np.max(next_logprob, axis=1, keepdims=True)
    next_logprob -= np.log(
        np.maximum(np.sum(np.exp(next_logprob), axis=1, keepdims=True), 1e-12)
    )

    dp_prev = np.full(s, -np.inf, dtype=np.float64)
    dp_prev[0] = float(scores[0, 0])
    back = np.full((n, s), -1, dtype=np.int16)
    back[0, 0] = 0

    for i in range(1, n):
        dp_cur = np.full(s, -np.inf, dtype=np.float64)
        stay_pen = boundary_stay_penalty * max(0.0, float(boundary[i] - boundary_threshold))

        p = float(boundary[i - 1])
        boundary_logit = math.log(p / (1.0 - p))
        trans_strength = (
            boundary_reward * math.tanh(0.5 * boundary_logit)
            - premature_transition_penalty * (1.0 - p)
        )

        dt = max(0.0, float(ends[i] - starts[i - 1])) if i > 0 else 0.0
        duration_cost = 0.0
        if dt > max_duration_seconds:
            duration_cost = duration_penalty * (
                (dt - max_duration_seconds) / max(1.0, max_duration_seconds)
            )

        for slide in range(s):
            # Stay on same slide.
            best_value = dp_prev[slide] - stay_pen
            best_prev = slide

            # Move forward by 1..J slides.
            first_prev = max(0, slide - max_jump)
            for prev_slide in range(first_prev, slide):
                if not np.isfinite(dp_prev[prev_slide]):
                    continue
                jump = slide - prev_slide
                candidate = (
                    dp_prev[prev_slide]
                    - float(jump_penalty) * jump
                    + trans_strength
                    + float(transition_logits_weight) * float(next_logprob[i - 1, slide])
                    - duration_cost
                )
                if candidate > best_value:
                    best_value = candidate
                    best_prev = prev_slide

            if np.isfinite(best_value):
                dp_cur[slide] = best_value + float(scores[i, slide])
                back[i, slide] = best_prev

        dp_prev = dp_cur

    final_slide = int(np.argmax(dp_prev))
    path = np.zeros(n, dtype=np.int64)
    path[-1] = final_slide

    for i in range(n - 1, 0, -1):
        prev_slide = int(back[i, path[i]])
        if prev_slide < 0:
            prev_slide = int(path[i])
        path[i - 1] = prev_slide

    return np.maximum.accumulate(np.clip(path, 0, s - 1))


def score_item(item, path):
    idx = np.asarray(item["gt_indices"], dtype=np.int64)
    labels = np.asarray(item["gt_labels"], dtype=np.int64)
    if len(labels) == 0:
        return 0, 0
    pred = np.asarray(path, dtype=np.int64)
    return int(np.sum(pred[idx] == labels)), int(len(labels))


def exact_v4_decode(item, scores, decoder):
    path = boundary_aware_dp(
        scores,
        item["boundary"],
        item["next_logits"],
        np.asarray(item["starts"], dtype=np.float32),
        np.asarray(item["ends"], dtype=np.float32),
        jump_penalty=float(decoder["jump_penalty"]),
        max_forward_jump=int(decoder["max_forward_jump"]),
        max_duration_seconds=float(decoder["max_duration_seconds"]),
        duration_penalty=float(decoder["duration_penalty"]),
        max_run_seconds=float(decoder["max_run_seconds"]),
        boundary_reward=float(decoder["boundary_reward"]),
        boundary_stay_penalty=float(decoder["boundary_stay_penalty"]),
        transition_logits_weight=float(decoder["transition_logits_weight"]),
        premature_transition_penalty=float(decoder["premature_transition_penalty"]),
        boundary_threshold=float(decoder["boundary_threshold"]),
    )
    return np.asarray(path, dtype=np.int64)


def evaluate_fast(dataset, fusion, decoder):
    rows = []
    total_correct = 0
    total = 0

    for name, item in dataset.items():
        scores = adaptive_visual_fusion(
            item["adjusted"],
            item.get("video"),
            **fusion,
        )
        path = fast_monotonic_viterbi(
            scores,
            item["boundary"],
            item["next_logits"],
            item["starts"],
            item["ends"],
            **{k: decoder[k] for k in (
                "jump_penalty",
                "max_forward_jump",
                "boundary_reward",
                "boundary_stay_penalty",
                "transition_logits_weight",
                "premature_transition_penalty",
                "boundary_threshold",
                "duration_penalty",
                "max_duration_seconds",
            )},
        )
        correct, count = score_item(item, path)
        rows.append((name, correct, count))
        total_correct += correct
        total += count

    macro = float(np.mean([c / max(1, t) for _, c, t in rows]))
    weighted = total_correct / max(1, total)
    return macro, weighted, rows


def evaluate_exact(dataset, fusion, decoder):
    rows = []
    total_correct = 0
    total = 0

    for name, item in dataset.items():
        scores = adaptive_visual_fusion(
            item["adjusted"],
            item.get("video"),
            **fusion,
        )
        path = exact_v4_decode(item, scores, decoder)
        correct, count = score_item(item, path)
        rows.append((name, correct, count))
        total_correct += correct
        total += count

    macro = float(np.mean([c / max(1, t) for _, c, t in rows]))
    weighted = total_correct / max(1, total)
    return macro, weighted, rows


def fusion_grid():
    out = []
    for visual_weight in (0.0, 0.08, 0.16, 0.24):
        for uncertainty_power in (0.7, 1.2):
            out.append(
                {
                    "visual_weight": visual_weight,
                    "v4_temperature": 1.0,
                    "visual_temperature": 1.0,
                    "uncertainty_power": uncertainty_power,
                    "visual_confidence_power": 1.0,
                    "disagreement_scale": 0.55,
                }
            )
    return out


def decoder_grid():
    presets = [
        (0.00, 1, 0.0, 0.00, 0.00, 0.00),
        (0.02, 1, 0.5, 0.20, 0.10, 0.25),
        (0.05, 1, 1.0, 0.50, 0.35, 0.50),
        (0.02, 2, 0.5, 0.20, 0.10, 0.25),
        (0.05, 2, 1.0, 0.50, 0.35, 0.50),
        (0.08, 2, 1.0, 0.25, 0.20, 0.25),
        (0.02, 3, 0.5, 0.20, 0.10, 0.25),
        (0.05, 3, 1.0, 0.50, 0.35, 0.50),
        (0.00, 2, 0.5, 0.00, 0.10, 0.00),
        (0.03, 2, 1.5, 0.30, 0.20, 0.40),
        (0.03, 3, 1.5, 0.30, 0.20, 0.40),
        (0.06, 3, 1.5, 0.50, 0.35, 0.40),
    ]
    out = []
    for jp, mj, br, bsp, tlw, pp in presets:
        d = dict(DEFAULT_DECODER)
        d.update(
            {
                "jump_penalty": jp,
                "max_forward_jump": mj,
                "boundary_reward": br,
                "boundary_stay_penalty": bsp,
                "transition_logits_weight": tlw,
                "premature_transition_penalty": pp,
            }
        )
        out.append(d)
    return out


def save_results(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "macro", "weighted", "fusion", "decoder"])
        for r in records:
            w.writerow(
                [
                    r["stage"],
                    r["macro"],
                    r["weighted"],
                    json.dumps(r["fusion"], sort_keys=True),
                    json.dumps(r["decoder"], sort_keys=True),
                ]
            )


def main():
    print("=" * 78)
    print("LectureSense V4 FAST ACCURACY TUNER")
    print("=" * 78)
    print("Uses saved V4 validation artifacts only.")
    print("No new OCR/SBERT/ResNet/video extraction.")
    print("Held-out test labels are not used.")

    dataset = source_tuner.load_dataset(source_tuner.VALIDATION_LECTURES)
    records = []

    print()
    print("=" * 78)
    print("FAST SEARCH: 8 fusion x 12 decoder = 96 lightweight trials")
    print("=" * 78)

    candidates = []
    fgrid = fusion_grid()
    dgrid = decoder_grid()
    total_trials = len(fgrid) * len(dgrid)
    trial = 0

    for fusion in fgrid:
        for decoder in dgrid:
            trial += 1
            macro, weighted, rows = evaluate_fast(dataset, fusion, decoder)
            cand = {
                "macro": macro,
                "weighted": weighted,
                "fusion": fusion,
                "decoder": decoder,
                "rows": rows,
            }
            candidates.append(cand)
            records.append({
                "stage": "fast",
                "macro": macro,
                "weighted": weighted,
                "fusion": fusion,
                "decoder": decoder,
            })
            print(
                f"[{trial:03d}/{total_trials}] "
                f"macro={macro * 100:6.2f}% "
                f"weighted={weighted * 100:6.2f}% "
                f"visual={fusion['visual_weight']:.2f} "
                f"jump={decoder['jump_penalty']:.2f} "
                f"max_jump={decoder['max_forward_jump']}"
            )

    candidates.sort(key=lambda x: (x["macro"], x["weighted"]), reverse=True)

    # Exact-check the default V4 configuration and the single best fast candidate.
    exact_candidates = []
    baseline_fusion = {
        "visual_weight": 0.0,
        "v4_temperature": 1.0,
        "visual_temperature": 1.0,
        "uncertainty_power": 1.0,
        "visual_confidence_power": 1.0,
        "disagreement_scale": 0.55,
    }

    print()
    print("=" * 78)
    print("EXACT V4 CHECK")
    print("=" * 78)

    exact_macro, exact_weighted, exact_rows = evaluate_exact(
        dataset, baseline_fusion, DEFAULT_DECODER
    )
    exact_candidates.append({
        "macro": exact_macro,
        "weighted": exact_weighted,
        "fusion": baseline_fusion,
        "decoder": dict(DEFAULT_DECODER),
        "rows": exact_rows,
    })
    records.append({
        "stage": "exact_baseline",
        "macro": exact_macro,
        "weighted": exact_weighted,
        "fusion": baseline_fusion,
        "decoder": DEFAULT_DECODER,
    })
    print(f"Default V4 exact macro    : {exact_macro * 100:.2f}%")
    print(f"Default V4 exact weighted : {exact_weighted * 100:.2f}%")

    best_fast = candidates[0]
    fm, fw, fr = evaluate_exact(dataset, best_fast["fusion"], best_fast["decoder"])
    exact_candidates.append({
        "macro": fm,
        "weighted": fw,
        "fusion": best_fast["fusion"],
        "decoder": best_fast["decoder"],
        "rows": fr,
    })
    records.append({
        "stage": "exact_best_fast",
        "macro": fm,
        "weighted": fw,
        "fusion": best_fast["fusion"],
        "decoder": best_fast["decoder"],
    })
    print(f"Best fast candidate exact: {fm * 100:.2f}%")
    print(f"Best fast candidate weighted: {fw * 100:.2f}%")

    exact_candidates.sort(key=lambda x: (x["macro"], x["weighted"]), reverse=True)
    best = exact_candidates[0]

    print()
    print("=" * 78)
    print("LOCKED VALIDATION CONFIG")
    print("=" * 78)
    print(f"Validation macro    : {best['macro'] * 100:.2f}%")
    print(f"Validation weighted : {best['weighted'] * 100:.2f}%")
    for name, correct, total in best["rows"]:
        print(f"{name:<28} {correct:4d}/{total:<4d} {correct / max(1, total) * 100:6.2f}%")

    payload = {
        "project": "LectureSense",
        "model": "LectureSense V4 + Fast Adaptive Decoder/Fusion",
        "validation_only": True,
        "test_used": False,
        "validation_macro_accuracy": float(best["macro"]),
        "validation_weighted_accuracy": float(best["weighted"]),
        "fusion": best["fusion"],
        "decoder": best["decoder"],
        "lecture_accuracy": {
            name: float(correct / max(1, total))
            for name, correct, total in best["rows"]
        },
        "search": {
            "fast_fusion_trials": len(fgrid),
            "fast_decoder_trials": len(dgrid),
            "fast_trials_total": total_trials,
            "exact_checks": 2,
        },
        "source": "saved V4 validation artifacts",
        "note": "Held-out test labels were not used during tuning.",
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_results(records)

    print()
    print(f"Saved config : {CONFIG_PATH}")
    print(f"Saved results: {CSV_PATH}")
    print("Next step: use this locked validation config on the untouched test set.")


if __name__ == "__main__":
    main()
