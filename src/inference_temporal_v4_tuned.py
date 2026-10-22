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
DEFAULT_TUNED_CONFIG = ROOT / "checkpoints" / "v4_2_decoder_tuned_config.json"
FUSION_KEYS = ("residual_scale", "video_weight", "score_temperature")
DECODER_KEYS = (
    "jump_penalty", "max_forward_jump", "max_duration_seconds", "duration_penalty",
    "max_run_seconds", "boundary_reward", "boundary_stay_penalty",
    "transition_logits_weight", "premature_transition_penalty", "boundary_threshold",
)


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


def load_tuned_config(path: Path):
    """Load the validation-tuned V4.2 config; refuse unvalidated/test-tuned files."""
    if not path.exists():
        raise FileNotFoundError(
            f"V4.2 config not found: {path}. Run the validation-only tuner first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read V4.2 config {path}: {exc}") from exc

    if payload.get("validation_only") is not True or payload.get("test_used") is not False:
        raise RuntimeError(
            "Refusing to use this config: it must be marked validation_only=true and test_used=false."
        )
    if payload.get("test_labels_used_in_this_tuning_run", False) is not False:
        raise RuntimeError("Refusing to use a config whose tuning run used test labels.")

    fusion = payload.get("fusion")
    decoder = payload.get("decoder")
    if not isinstance(fusion, dict) or not isinstance(decoder, dict):
        raise RuntimeError("V4.2 config must contain 'fusion' and 'decoder' objects.")

    missing_fusion = [key for key in FUSION_KEYS if key not in fusion]
    missing_decoder = [key for key in DECODER_KEYS if key not in decoder]
    if missing_fusion or missing_decoder:
        raise RuntimeError(
            f"V4.2 config is incomplete. Missing fusion keys={missing_fusion}; "
            f"decoder keys={missing_decoder}. Re-run validation-only tuning."
        )

    fusion = {key: float(fusion[key]) for key in FUSION_KEYS}
    decoder = {key: decoder[key] for key in DECODER_KEYS}
    decoder = {key: (int(value) if key == "max_forward_jump" else float(value))
               for key, value in decoder.items()}

    if fusion["score_temperature"] <= 0:
        raise RuntimeError("V4.2 score_temperature must be greater than zero.")
    if not 0.0 <= fusion["video_weight"] <= 1.0:
        raise RuntimeError("V4.2 video_weight must be between 0 and 1.")
    if fusion["residual_scale"] < 0:
        raise RuntimeError("V4.2 residual_scale cannot be negative.")
    if decoder["max_forward_jump"] < 1:
        raise RuntimeError("V4.2 max_forward_jump must be at least 1.")

    return payload, fusion, decoder


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


def main():
    parser = argparse.ArgumentParser(
        description="LectureSense V4.2 tuned temporal inference"
    )
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_TUNED_CONFIG),
                        help="Validation-tuned V4.2 config JSON")
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--video-weight", type=float, default=None,
                        help="Optional manual override; leave unset to use V4.2 config")

    parser.add_argument("--jump-penalty", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--duration-penalty", type=float, default=None)
    parser.add_argument("--max-run-seconds", type=float, default=None)

    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config_path = Path(args.config).resolve()
    config_payload, fusion_cfg, tuned_decoder_cfg = load_tuned_config(config_path)
    model, base_model, max_slides, temperature, decoder_cfg = load_v4_checkpoint(
        Path(args.checkpoint)
    )

    # V4.2 config takes precedence over checkpoint defaults. CLI overrides are
    # retained only for explicit diagnostics; the Streamlit app passes none.
    decoder_cfg.update(tuned_decoder_cfg)
    if args.video_weight is not None:
        fusion_cfg["video_weight"] = float(args.video_weight)

    if args.jump_penalty is not None:
        decoder_cfg["jump_penalty"] = args.jump_penalty
    if args.max_duration is not None:
        decoder_cfg["max_duration_seconds"] = args.max_duration
    if args.duration_penalty is not None:
        decoder_cfg["duration_penalty"] = args.duration_penalty
    if args.max_run_seconds is not None:
        decoder_cfg["max_run_seconds"] = args.max_run_seconds

    print("\n1/6 READING TRANSCRIPT")
    transcript_segments = base.read_mavils_srt(args.transcript)
    if not transcript_segments:
        raise RuntimeError("No transcript segments found")

    print("\n2/6 RENDERING SLIDES")
    slide_paths = base.render_pdf(args.slides, args.render_dir)
    if not slide_paths:
        raise RuntimeError("No slides rendered")

    print("\n3/6 EXTRACTING OCR")
    ocr_texts = base.extract_ocr(slide_paths)
    if len(ocr_texts) != len(slide_paths):
        raise RuntimeError("OCR count does not match slide count")

    print("\n4/6 CREATING MULTIMODAL FEATURES")
    sentence_model = base.load_sbert()
    transcript_embeddings = base.create_text_embeddings(
        sentence_model,
        [x["text"] for x in transcript_segments],
    )
    ocr_embeddings = base.create_text_embeddings(
        sentence_model,
        ocr_texts,
    )
    del sentence_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n5/6 V4 BOUNDARY-AWARE TEMPORAL ALIGNMENT")
    base_scores = base.compute_score_matrix(
        base_model,
        transcript_embeddings,
        ocr_embeddings,
        image_embeddings,
        batch_size=max(1, args.score_batch),
    )

    adjusted_scores, boundary_prob, next_logits = temporal_adjust_scores(
        base_scores,
        transcript_segments,
        model,
        max_slides,
        temperature,
    )

    # Match the tuned workflow exactly: scale the temporal residual relative to
    # the frozen base score matrix, fuse video evidence, then apply score temperature.
    residual_scale = float(fusion_cfg["residual_scale"])
    residual_scaled_scores = (
        base_scores + residual_scale * (adjusted_scores - base_scores)
    ).astype(np.float32)
    fused_score_matrix = residual_scaled_scores
    video_score_matrix = None
    effective_video_weight = float(fusion_cfg["video_weight"])

    if args.video and effective_video_weight > 0.0:
        print("\nExtracting actual lecture-video visual evidence...")
        video_embeddings, video_timestamps, _ = create_video_frame_embeddings(
            args.video,
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
        fused_score_matrix = fuse_model_and_video_scores(
            residual_scaled_scores,
            video_score_matrix,
            video_weight=effective_video_weight,
        ).astype(np.float32)
        np.save(Path(args.out).parent / "v4_video_score_matrix.npy", video_score_matrix)
    else:
        stale_video_matrix = Path(args.out).parent / "v4_video_score_matrix.npy"
        if stale_video_matrix.exists():
            stale_video_matrix.unlink()
        if args.video and effective_video_weight <= 0.0:
            print("\nV4.2 config sets video_weight=0; skipping video-frame embedding extraction.")
        elif not args.video and effective_video_weight > 0.0:
            print("WARNING: no --video supplied; tuned visual fusion cannot be applied.")

    score_temperature = float(fusion_cfg["score_temperature"])
    final_score_matrix = (fused_score_matrix / max(1e-6, score_temperature)).astype(np.float32)

    np.save(
        Path(args.out).parent / "v4_base_score_matrix.npy",
        base_scores,
    )
    np.save(
        Path(args.out).parent / "v4_adjusted_score_matrix.npy",
        adjusted_scores,
    )
    np.save(
        Path(args.out).parent / "v4_boundary_probability.npy",
        boundary_prob,
    )
    np.save(
        Path(args.out).parent / "v4_next_slide_logits.npy",
        next_logits,
    )
    np.save(
        Path(args.out).parent / "v4_residual_scaled_score_matrix.npy",
        residual_scaled_scores,
    )
    np.save(
        Path(args.out).parent / "v4_final_score_matrix.npy",
        fused_score_matrix,
    )
    np.save(
        Path(args.out).parent / "v4_decoder_input_score_matrix.npy",
        final_score_matrix,
    )

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

    timeline = base.make_timeline(
        transcript_segments,
        slide_paths,
        alignment,
    )

    metadata = {
        "model": "LectureSenseBoundaryAwareTemporalGRU_V4.2_Tuned",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tuning_config": str(config_path),
        "tuning_type": config_payload.get("tuning_type", "validation_only_decoder_and_fusion"),
        "temporal_max_slides": max_slides,
        "temporal_temperature": temperature,
        "score_temperature": score_temperature,
        "residual_scale": residual_scale,
        "video_enabled": video_score_matrix is not None,
        "video_weight": effective_video_weight if video_score_matrix is not None else 0.0,
        "video_sample_seconds": args.video_sample_seconds if video_score_matrix is not None else None,
        "fusion": fusion_cfg,
        "decoder": decoder_cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }

    base.save_result(args.out, timeline, metadata)

    print(
        f"\nV4.2 TUNED INFERENCE COMPLETE\n"
        f"Timeline: {Path(args.out).resolve()}"
    )


if __name__ == "__main__":
    main()
