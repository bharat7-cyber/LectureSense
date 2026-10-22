from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_temporal_v4 import BoundaryAwareTemporalGRU, make_v4_features, MAX_SLIDES_DEFAULT, EXTRA_FEATURES
from boundary_dp_v4_final import boundary_aware_dp_final

CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
CACHE_DIR = ROOT / "runs" / "temporal_v3_cache"
OUT_CONFIG = CHECKPOINT.parent / "temporal_v4_final_decoder_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DEFAULTS = {
    "jump_penalty": 0.03,
    "max_forward_jump": 2,
    "max_duration_seconds": 120.0,
    "duration_penalty": 0.001,
    "max_run_seconds": 1200.0,
    "boundary_reward": 1.50,
    "boundary_stay_penalty": 0.25,
    "transition_logits_weight": 0.25,
    "premature_transition_penalty": 0.25,
    "boundary_threshold": 0.20,
    "adjacent_margin_weight": 1.00,
    "lookahead_weight": 0.60,
    "visual_change_weight": 1.50,
    "stay_pressure_weight": 1.20,
    "stay_margin": 0.05,
    "stay_temperature": 0.25,
    "lookahead_segments": 4,
}

# Small targeted search around the anti-sticking decoder.
# The original calibration tested 36 combinations. This tests 10,
# prints progress after every validation lecture, and can prune weak
# candidates before running every lecture.
CANDIDATES = [
    (2, 1.50, 1.00, 1.20),
    (2, 2.00, 1.00, 1.20),
    (2, 2.50, 1.00, 1.20),
    (2, 1.50, 1.50, 1.20),
    (2, 2.00, 1.50, 1.20),
    (2, 2.50, 1.50, 1.20),
    (3, 1.50, 1.00, 1.50),
    (3, 2.00, 1.00, 1.50),
    (2, 2.00, 0.50, 0.80),
    (1, 1.50, 1.00, 0.80),
]


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    max_slides = int(ckpt.get("max_slides", MAX_SLIDES_DEFAULT))
    hidden = int(ckpt.get("hidden", 160))
    layers = int(ckpt.get("layers", 2))
    heads = int(ckpt.get("heads", 4))
    temperature = float(ckpt.get("temperature", 1.0))
    model = BoundaryAwareTemporalGRU(
        max_slides + int(ckpt.get("extra_features", EXTRA_FEATURES)),
        max_slides,
        hidden,
        layers,
        heads,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, max_slides, temperature


def prepare_lectures(model, max_slides, temperature):
    items = []
    for cache in sorted(CACHE_DIR.glob("val_*.npz")):
        data = np.load(cache)
        base_scores = data["scores"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        timestamps = data["timestamps"].astype(np.float32)
        starts = timestamps
        ends = np.r_[timestamps[1:], timestamps[-1] + 8.0]
        time_norm = data["time_norm"].astype(np.float32)
        features = make_v4_features(base_scores, time_norm, max_slides, temperature)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
            delta, boundary_logit, next_logits = model(x)
        adjusted = base_scores + delta[0, :, :base_scores.shape[1]].cpu().numpy()
        b = torch.sigmoid(boundary_logit[0]).cpu().numpy()
        nxt = next_logits[0, :, :base_scores.shape[1]].cpu().numpy()
        items.append((cache.stem.replace("val_", ""), adjusted, b, nxt, starts, ends, labels))
    if not items:
        raise RuntimeError(f"No validation caches found in {CACHE_DIR}")
    return items


def make_cfg(candidate):
    max_jump, boundary_reward, adjacent_weight, stay_weight = candidate
    cfg = dict(BASE_DEFAULTS)
    cfg.update({
        "max_forward_jump": max_jump,
        "boundary_reward": boundary_reward,
        "adjacent_margin_weight": adjacent_weight,
        "stay_pressure_weight": stay_weight,
    })
    return cfg


def score_candidate(items, cfg, best_acc):
    correct = 0
    total = 0
    per = []
    t0 = time.time()
    for idx, (name, scores, b, nxt, starts, ends, labels) in enumerate(items, 1):
        path = boundary_aware_dp_final(
            scores, b, nxt, starts, ends,
            visual_change=None,
            **cfg,
        )
        path = np.asarray(path, dtype=np.int64)
        c = int(np.sum(path == labels))
        t = len(labels)
        correct += c
        total += t
        per.append((name, c, t))
        partial = correct / max(1, total)
        # Even perfect remaining lectures cannot beat the current best.
        remaining = sum(len(x[-1]) for x in items[idx:])
        upper = (correct + remaining) / max(1, total + remaining)
        print(
            f"      {name}: {c}/{t} = {partial:.2%} | upper bound {upper:.2%}",
            flush=True,
        )
        if upper <= best_acc:
            print("      -> pruned (cannot beat current best)", flush=True)
            return partial, per, time.time() - t0, True
    return correct / max(1, total), per, time.time() - t0, False


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print("=" * 78, flush=True)
    print("LECTURESENSE V4 FINAL — FAST VALIDATION-ONLY CALIBRATION", flush=True)
    print("=" * 78, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Candidates: {len(CANDIDATES)} (original search: 36)", flush=True)
    print("Test lectures are never loaded.", flush=True)

    t0 = time.time()
    model, max_slides, temperature = load_model()
    items = prepare_lectures(model, max_slides, temperature)
    print(f"Validation lectures loaded: {len(items)}", flush=True)
    print(f"Feature preparation: {time.time() - t0:.1f}s", flush=True)

    best_acc = -1.0
    best_cfg = None
    best_per = None
    results = []

    for i, candidate in enumerate(CANDIDATES, 1):
        cfg = make_cfg(candidate)
        print("\n" + "-" * 78, flush=True)
        print(f"Candidate {i}/{len(CANDIDATES)}", flush=True)
        print(
            f"max_jump={candidate[0]} | boundary_reward={candidate[1]:.2f} | "
            f"adjacent={candidate[2]:.2f} | stay_pressure={candidate[3]:.2f}",
            flush=True,
        )
        acc, per, elapsed, pruned = score_candidate(items, cfg, best_acc)
        results.append((acc, cfg, per))
        print(f"Candidate result: {acc:.4%} | {elapsed:.1f}s", flush=True)
        if acc > best_acc and not pruned:
            best_acc = acc
            best_cfg = cfg
            best_per = per
            print(f"NEW BEST: {best_acc:.4%}", flush=True)

    if best_cfg is None:
        # Keep the first configuration if all candidates are pruned.
        results.sort(key=lambda x: x[0], reverse=True)
        best_acc, best_cfg, best_per = results[0]

    OUT_CONFIG.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print("FAST CALIBRATION COMPLETE", flush=True)
    print("=" * 78, flush=True)
    print(f"Best validation accuracy: {best_acc:.4%}", flush=True)
    print("Best decoder config:", flush=True)
    print(json.dumps(best_cfg, indent=2), flush=True)
    print("Per lecture:", flush=True)
    for name, c, t in best_per:
        print(f"  {name}: {c}/{t} = {c / max(1, t):.4%}", flush=True)
    print(f"Saved: {OUT_CONFIG}", flush=True)
    print(f"Total calibration time: {time.time() - t0:.1f}s", flush=True)
    print("Test lectures were not used.", flush=True)
    print("Next command: python run_formal_test_eval_v4_final.py", flush=True)


if __name__ == "__main__":
    main()
