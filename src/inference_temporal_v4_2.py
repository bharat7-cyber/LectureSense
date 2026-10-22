from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import inference as base
from boundary_dp_v4 import boundary_aware_dp
from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRA_FEATURES = 6
DEFAULT_CONFIG = ROOT / "checkpoints" / "v4_decoder_tuned_config.json"

V4_DEFAULT_FUSION = {
    "residual_scale": 1.0,
    "video_weight": 0.40,
    "score_temperature": 1.0,
}

V4_APP_DECODER_OVERRIDES = {
    "jump_penalty": 0.05,
    "max_duration_seconds": 120.0,
    "duration_penalty": 0.003,
}


class BoundaryAwareTemporalGRU(nn.Module):
    def __init__(self, input_dim, max_slides, hidden=160, layers=2, heads=4):
        super().__init__()
        if hidden * 2 % heads != 0:
            raise ValueError("hidden*2 must be divisible by heads")

        self.max_slides = max_slides
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.gru = nn.GRU(
            128,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.20 if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(
            hidden * 2,
            num_heads=heads,
            dropout=0.10,
            batch_first=True,
        )
        self.attn_gate = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(hidden * 2)

        self.delta = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max_slides),
        )
        self.boundary = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )
        self.next_slide = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max_slides),
        )


    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        gate = torch.tanh(self.attn_gate)
        h = self.norm(h + gate * attn_out)
        return (
            self.delta(h),
            self.boundary(h).squeeze(-1),
            self.next_slide(h),
        )


def make_v4_features(scores, time_norm, max_slides, temperature):
    n, s = scores.shape
    logits = torch.from_numpy(scores.astype(np.float32))
    probs = F.softmax(logits / temperature, dim=-1).numpy()

    p_sorted = np.sort(probs, axis=1)
    top1 = p_sorted[:, -1]
    top2 = p_sorted[:, -2] if s >= 2 else np.zeros(n, dtype=np.float32)
    margin = top1 - top2

    entropy_denom = max(1e-8, float(np.log(max(2, s))))
    entropy = -np.sum(
        probs * np.log(np.clip(probs, 1e-8, 1.0)),
        axis=1,
    ) / entropy_denom

    change = np.zeros(n, dtype=np.float32)
    if n > 1:
        a = probs[:-1]
        c = probs[1:]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        change[1:] = np.clip(
            1.0 - np.sum(a * c, axis=1),
            0.0,
            1.0,
        )

    durations = np.zeros(n, dtype=np.float32)
    if n > 1:
        durations[:-1] = np.clip(np.diff(time_norm), 0.0, 1.0)
        durations[-1] = durations[-2]

    x = np.zeros((n, max_slides + EXTRA_FEATURES), dtype=np.float32)
    x[:, :s] = probs
    x[:, max_slides] = time_norm
    x[:, max_slides + 1] = durations
    x[:, max_slides + 2] = top1
    x[:, max_slides + 3] = margin
    x[:, max_slides + 4] = entropy
    x[:, max_slides + 5] = change
    return x


def load_v4_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    max_slides = int(ckpt["max_slides"])
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

    base_model = base.LectureSenseCrossModal().to(DEVICE)
    base_model.load_state_dict(ckpt["base_model_state_dict"], strict=True)
    base_model.eval()

    decoder_cfg = ckpt.get(
        "decoder_config",
        {
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
        },
    )

    print(f"V4 checkpoint: {path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'n/a')}")
    print(f"Checkpoint Val Top-1: {ckpt.get('val_top1', 0.0):.4f}")
    print(f"Checkpoint Val MRR: {ckpt.get('val_mrr', 0.0):.4f}")
    print(f"Validation DP accuracy: {ckpt.get('val_decoder_accuracy', 'n/a')}")

    return model, base_model, max_slides, temperature, decoder_cfg


def temporal_adjust_scores(
    base_scores,
    transcript_segments,
    model,
    max_slides,
    temperature,
):
    n, s = base_scores.shape
    if s > max_slides:
        raise RuntimeError(
            f"Lecture has {s} slides, checkpoint supports {max_slides}"
        )

    starts = np.array(
        [float(x["start"]) for x in transcript_segments],
        dtype=np.float32,
    )

    total = max(1.0, float(starts[-1] - starts[0]))
    time_norm = (starts - starts[0]) / total

    features = make_v4_features(
        base_scores,
        time_norm,
        max_slides,
        temperature,
    )

    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        delta, boundary_logit, next_logits = model(x)

    adjusted = (
        base_scores
        + delta[0, :, :s].cpu().numpy()
    )
    boundary_prob = torch.sigmoid(
        boundary_logit[0]
    ).cpu().numpy()
    next_logits = next_logits[
        0, :, :s
    ].cpu().numpy()

    return adjusted, boundary_prob, next_logits



def resolve_project_path(value):
    """Resolve absolute paths as-is and relative paths from the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_v4_2_config(config_path):
    """Load the validation-only tuning artifact produced by tune_v4_decoder_only.py.

    Returns None when the file is absent. A present config must explicitly be
    marked as validation-only and test-unused to avoid applying a test-tuned file.
    """
    if not config_path.exists():
        print(f"Tuned config not found: {config_path}")
        print("Using the current V4 application defaults instead.")
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("validation_only") is not True or payload.get("test_used") is not False:
        raise ValueError(
            "Refusing to load this tuning config: it must declare "
            "validation_only=true and test_used=false."
        )

    fusion = payload.get("fusion")
    decoder = payload.get("decoder")
    if not isinstance(fusion, dict) or not isinstance(decoder, dict):
        raise ValueError("Tuned config must contain 'fusion' and 'decoder' objects.")

    required_fusion = ("residual_scale", "video_weight", "score_temperature")
    required_decoder = (
        "jump_penalty", "max_forward_jump", "max_duration_seconds",
        "duration_penalty", "max_run_seconds", "boundary_reward",
        "boundary_stay_penalty", "transition_logits_weight",
        "premature_transition_penalty", "boundary_threshold",
    )
    missing = [f"fusion.{key}" for key in required_fusion if key not in fusion]
    missing += [f"decoder.{key}" for key in required_decoder if key not in decoder]
    if missing:
        raise ValueError(f"Tuned config is missing required settings: {', '.join(missing)}")

    clean_fusion = {key: float(fusion[key]) for key in required_fusion}
    clean_decoder = dict(decoder)
    for key in required_decoder:
        if key == "max_forward_jump":
            clean_decoder[key] = int(decoder[key])
        else:
            clean_decoder[key] = float(decoder[key])

    if clean_fusion["score_temperature"] <= 0:
        raise ValueError("fusion.score_temperature must be greater than zero")
    if not 0.0 <= clean_fusion["video_weight"] <= 1.0:
        raise ValueError("fusion.video_weight must be in [0, 1]")
    if clean_fusion["residual_scale"] < 0:
        raise ValueError("fusion.residual_scale must be non-negative")

    print(f"Loaded validation-tuned V4.2 config: {config_path}")
    print(f"Validation macro accuracy in config: {100.0 * float(payload.get('validation_macro_accuracy', 0.0)):.2f}%")
    print("Config declares test_used=false; test labels were not used for tuning.")
    return {"fusion": clean_fusion, "decoder": clean_decoder, "payload": payload}


def apply_residual_scaling(base_scores, adjusted_scores, residual_scale):
    """Faithful tuner formula: base + scale * (adjusted - base)."""
    base_scores = np.asarray(base_scores, dtype=np.float32)
    adjusted_scores = np.asarray(adjusted_scores, dtype=np.float32)
    if base_scores.shape != adjusted_scores.shape:
        raise ValueError(
            f"Base/adjusted score shape mismatch: {base_scores.shape} vs {adjusted_scores.shape}"
        )
    return (base_scores + float(residual_scale) * (adjusted_scores - base_scores)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="LectureSense V4.2 tuned fusion + boundary-aware temporal inference"
    )
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/temporal_v4_best_model.pt")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="validation-only config saved by tune_v4_decoder_only.py")
    parser.add_argument("--ignore-config", action="store_true",
                        help="ignore the V4.2 tuning config and use V4 application defaults")
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)

    # Optional CLI overrides. When omitted, a loaded V4.2 config takes precedence.
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--video-weight", type=float, default=None)
    parser.add_argument("--score-temperature", type=float, default=None)
    parser.add_argument("--jump-penalty", type=float, default=None)
    parser.add_argument("--max-forward-jump", type=int, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--duration-penalty", type=float, default=None)
    parser.add_argument("--max-run-seconds", type=float, default=None)
    parser.add_argument("--boundary-reward", type=float, default=None)
    parser.add_argument("--boundary-stay-penalty", type=float, default=None)
    parser.add_argument("--transition-logits-weight", type=float, default=None)
    parser.add_argument("--premature-transition-penalty", type=float, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=None)

    args = parser.parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    out_path = resolve_project_path(args.out)
    render_dir = resolve_project_path(args.render_dir)
    transcript_path = resolve_project_path(args.transcript)
    slides_path = resolve_project_path(args.slides)
    video_path = resolve_project_path(args.video) if args.video else None

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, base_model, max_slides, model_temperature, checkpoint_decoder = load_v4_checkpoint(
        checkpoint_path
    )
    decoder_cfg = dict(checkpoint_decoder)
    # These were the exact V4 defaults used by the production app during tuning.
    decoder_cfg.update(V4_APP_DECODER_OVERRIDES)
    fusion_cfg = dict(V4_DEFAULT_FUSION)

    config_path = resolve_project_path(args.config)
    tuned_config = None
    if not args.ignore_config:
        tuned_config = load_v4_2_config(config_path)
    if tuned_config is not None:
        fusion_cfg.update(tuned_config["fusion"])
        decoder_cfg.update(tuned_config["decoder"])

    # Explicit CLI flags are deliberate overrides of the stored config.
    cli_fusion = {
        "residual_scale": args.residual_scale,
        "video_weight": args.video_weight,
        "score_temperature": args.score_temperature,
    }
    for key, value in cli_fusion.items():
        if value is not None:
            fusion_cfg[key] = float(value)

    cli_decoder = {
        "jump_penalty": args.jump_penalty,
        "max_forward_jump": args.max_forward_jump,
        "max_duration_seconds": args.max_duration,
        "duration_penalty": args.duration_penalty,
        "max_run_seconds": args.max_run_seconds,
        "boundary_reward": args.boundary_reward,
        "boundary_stay_penalty": args.boundary_stay_penalty,
        "transition_logits_weight": args.transition_logits_weight,
        "premature_transition_penalty": args.premature_transition_penalty,
        "boundary_threshold": args.boundary_threshold,
    }
    for key, value in cli_decoder.items():
        if value is not None:
            decoder_cfg[key] = int(value) if key == "max_forward_jump" else float(value)

    if fusion_cfg["score_temperature"] <= 0:
        raise ValueError("score_temperature must be greater than zero")
    if not 0.0 <= fusion_cfg["video_weight"] <= 1.0:
        raise ValueError("video_weight must be between 0 and 1")
    if fusion_cfg["residual_scale"] < 0:
        raise ValueError("residual_scale must be non-negative")
    if args.video_sample_seconds <= 0:
        raise ValueError("video-sample-seconds must be greater than zero")

    print("\nActive V4.2 fusion configuration:")
    for key, value in fusion_cfg.items():
        print(f"  {key}: {value}")
    print("Active V4.2 decoder configuration:")
    for key in (
        "jump_penalty", "max_forward_jump", "max_duration_seconds", "duration_penalty",
        "max_run_seconds", "boundary_reward", "boundary_stay_penalty",
        "transition_logits_weight", "premature_transition_penalty", "boundary_threshold",
    ):
        print(f"  {key}: {decoder_cfg[key]}")

    print("\n1/6 READING TRANSCRIPT")
    transcript_segments = base.read_mavils_srt(str(transcript_path))
    if not transcript_segments:
        raise RuntimeError(f"No transcript segments found: {transcript_path}")

    print("\n2/6 RENDERING SLIDES")
    render_dir.mkdir(parents=True, exist_ok=True)
    slide_paths = base.render_pdf(str(slides_path), str(render_dir))
    if not slide_paths:
        raise RuntimeError(f"No slides rendered from PDF: {slides_path}")

    print("\n3/6 EXTRACTING OCR")
    ocr_texts = base.extract_ocr(slide_paths)
    if len(ocr_texts) != len(slide_paths):
        raise RuntimeError("OCR count does not match slide count")

    print("\n4/6 CREATING MULTIMODAL FEATURES")
    sentence_model = base.load_sbert()
    transcript_embeddings = base.create_text_embeddings(
        sentence_model, [x["text"] for x in transcript_segments]
    )
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n5/6 V4.2 SCORE FUSION")
    base_scores = base.compute_score_matrix(
        base_model,
        transcript_embeddings,
        ocr_embeddings,
        image_embeddings,
        batch_size=max(1, args.score_batch),
    )
    adjusted_scores, boundary_prob, next_logits = temporal_adjust_scores(
        base_scores, transcript_segments, model, max_slides, model_temperature
    )

    # Match tune_v4_decoder_only.py exactly: base + residual_scale*(adjusted-base).
    residual_scaled_scores = apply_residual_scaling(
        base_scores, adjusted_scores, fusion_cfg["residual_scale"]
    )
    video_score_matrix = None
    fused_score_matrix = residual_scaled_scores

    if video_path is not None:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        print("Extracting actual lecture-video visual evidence...")
        video_embeddings, video_timestamps, _ = create_video_frame_embeddings(
            str(video_path),
            device=DEVICE,
            sample_every_seconds=max(0.25, args.video_sample_seconds),
            batch_size=8,
        )
        video_score_matrix = build_video_slide_score_matrix(
            video_embeddings,
            video_timestamps,
            image_embeddings,
            segment_starts=[x["start"] for x in transcript_segments],
            segment_ends=[x["end"] for x in transcript_segments],
        )
        if video_score_matrix.shape != residual_scaled_scores.shape:
            raise RuntimeError(
                f"Video/V4 score shape mismatch: {video_score_matrix.shape} vs "
                f"{residual_scaled_scores.shape}"
            )
        fused_score_matrix = fuse_model_and_video_scores(
            residual_scaled_scores,
            video_score_matrix,
            video_weight=fusion_cfg["video_weight"],
        ).astype(np.float32)

    # Tuner applies score_temperature to the fused matrix immediately before DP.
    final_score_matrix = (
        fused_score_matrix / np.float32(fusion_cfg["score_temperature"])
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.parent / "v4_base_score_matrix.npy", base_scores)
    np.save(out_path.parent / "v4_adjusted_score_matrix.npy", adjusted_scores)
    np.save(out_path.parent / "v4_residual_scaled_score_matrix.npy", residual_scaled_scores)
    np.save(out_path.parent / "v4_fused_pre_temperature_score_matrix.npy", fused_score_matrix)
    np.save(out_path.parent / "v4_final_score_matrix.npy", final_score_matrix)
    np.save(out_path.parent / "v4_boundary_probability.npy", boundary_prob)
    np.save(out_path.parent / "v4_next_slide_logits.npy", next_logits)
    if video_score_matrix is not None:
        np.save(out_path.parent / "v4_video_score_matrix.npy", video_score_matrix)

    print("\n6/6 RUNNING BOUNDARY-AWARE DYNAMIC PROGRAMMING")
    starts = [x["start"] for x in transcript_segments]
    ends = [x["end"] for x in transcript_segments]
    alignment = boundary_aware_dp(
        final_score_matrix,
        boundary_prob,
        next_logits,
        starts,
        ends,
        jump_penalty=decoder_cfg["jump_penalty"],
        max_forward_jump=decoder_cfg["max_forward_jump"],
        max_duration_seconds=decoder_cfg["max_duration_seconds"],
        duration_penalty=decoder_cfg["duration_penalty"],
        max_run_seconds=decoder_cfg["max_run_seconds"],
        boundary_reward=decoder_cfg["boundary_reward"],
        boundary_stay_penalty=decoder_cfg["boundary_stay_penalty"],
        transition_logits_weight=decoder_cfg["transition_logits_weight"],
        premature_transition_penalty=decoder_cfg["premature_transition_penalty"],
        boundary_threshold=decoder_cfg["boundary_threshold"],
    )
    if not alignment:
        raise RuntimeError("Boundary-aware DP returned no result")

    timeline = base.make_timeline(transcript_segments, slide_paths, alignment)
    metadata = {
        "model": "LectureSenseBoundaryAwareTemporalGRU_V4.2",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path) if tuned_config is not None else None,
        "config_validation_only": bool(tuned_config is not None),
        "config_test_used": False if tuned_config is not None else None,
        "temporal_max_slides": max_slides,
        "temporal_feature_temperature": model_temperature,
        "fusion": {
            "residual_scale": fusion_cfg["residual_scale"],
            "video_weight": fusion_cfg["video_weight"] if video_score_matrix is not None else 0.0,
            "score_temperature": fusion_cfg["score_temperature"],
            "video_enabled": video_score_matrix is not None,
        },
        "video_sample_seconds": args.video_sample_seconds if video_score_matrix is not None else None,
        "decoder": decoder_cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }
    base.save_result(str(out_path), timeline, metadata)
    print(f"\nV4.2 INFERENCE COMPLETE\nTimeline: {out_path.resolve()}")


if __name__ == "__main__":
    main()
