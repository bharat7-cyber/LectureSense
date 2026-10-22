from __future__ import annotations

import json
import sys
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


def score_cfg(items, cfg):
    correct = 0
    total = 0
    per = []
    for name, scores, b, nxt, starts, ends, labels in items:
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
    return correct / max(1, total), per


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    model, max_slides, temperature = load_model()
    items = prepare_lectures(model, max_slides, temperature)

    candidates = []
    # Small, deterministic validation-only search over the parameters that
    # directly affect the observed long-slide sticking failure mode.
    for max_jump in [1, 2, 3]:
        for boundary_reward in [1.0, 1.5, 2.0]:
            for adjacent_weight in [0.5, 1.0]:
                for stay_weight in [0.8, 1.2]:
                    cfg = dict(BASE_DEFAULTS)
                    cfg.update({
                        "max_forward_jump": max_jump,
                        "boundary_reward": boundary_reward,
                        "adjacent_margin_weight": adjacent_weight,
                        "stay_pressure_weight": stay_weight,
                    })
                    acc, per = score_cfg(items, cfg)
                    candidates.append((acc, cfg, per))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_cfg, per = candidates[0]
    OUT_CONFIG.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    print("=" * 78)
    print("LECTURESENSE V4 FINAL — VALIDATION-ONLY DECODER CALIBRATION")
    print("=" * 78)
    print(f"Device: {DEVICE}")
    print(f"Validation lectures: {len(items)}")
    print(f"Candidates tested: {len(candidates)}")
    print(f"Best validation accuracy: {best_acc:.4%}")
    print("Best decoder config:")
    print(json.dumps(best_cfg, indent=2))
    print("Per lecture:")
    for name, c, t in per:
        print(f"  {name}: {c}/{t} = {c/max(1,t):.4%}")
    print(f"Saved: {OUT_CONFIG}")
    print("Test lectures were not used.")


if __name__ == "__main__":
    main()
