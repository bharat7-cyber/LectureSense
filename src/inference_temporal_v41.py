from __future__ import annotations

"""LectureSense V4.1 inference using a validation-selected fusion/decoder config.

This script reuses the original, frozen V4 checkpoint and temporal model. It
applies the selected post-model configuration in the same order as
`tune_v4_decoder_only.py`:
  1. residual scaling between base and temporal-adjusted V4 scores
  2. video-score fusion
  3. score-temperature scaling
  4. boundary-aware dynamic programming with the selected decoder parameters

It never trains or overwrites the V4 checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import inference as base
import inference_temporal_v4 as v4
from boundary_dp_v4 import boundary_aware_dp
from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)

DEVICE = v4.DEVICE
CONFIG_DEFAULT = ROOT / "checkpoints" / "v4_decoder_tuned_config.json"
CHECKPOINT_DEFAULT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"

FUSION_KEYS = ("residual_scale", "video_weight", "score_temperature")
DECODER_KEYS = (
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
)


def load_selected_config(path: Path) -> tuple[dict, dict]:
    """Load a config written by validation-only tune_v4_decoder_only.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Selected V4.1 config not found: {path}\n"
            "Run the validation-only tuner first: python tune_v4_decoder_only.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("validation_only", False) or payload.get("test_used", True):
        raise RuntimeError(
            "Refusing to load this config: it is not marked as validation-only "
            "with test_used=false. Re-run the validation-only tuner."
        )
    fusion = payload.get("fusion") or {}
    decoder = payload.get("decoder") or {}
    missing_fusion = [key for key in FUSION_KEYS if key not in fusion]
    missing_decoder = [key for key in DECODER_KEYS if key not in decoder]
    if missing_fusion or missing_decoder:
        raise RuntimeError(
            "Selected config is incomplete. Missing fusion keys="
            f"{missing_fusion}; missing decoder keys={missing_decoder}."
        )

    fusion_cfg = {
        "residual_scale": float(fusion["residual_scale"]),
        "video_weight": float(fusion["video_weight"]),
        "score_temperature": max(1e-3, float(fusion["score_temperature"])),
    }
    decoder_cfg = {
        "jump_penalty": float(decoder["jump_penalty"]),
        "max_forward_jump": int(decoder["max_forward_jump"]),
        "max_duration_seconds": float(decoder["max_duration_seconds"]),
        "duration_penalty": float(decoder["duration_penalty"]),
        "max_run_seconds": float(decoder["max_run_seconds"]),
        "boundary_reward": float(decoder["boundary_reward"]),
        "boundary_stay_penalty": float(decoder["boundary_stay_penalty"]),
        "transition_logits_weight": float(decoder["transition_logits_weight"]),
        "premature_transition_penalty": float(decoder["premature_transition_penalty"]),
        "boundary_threshold": float(decoder["boundary_threshold"]),
    }
    return {"payload": payload, "path": str(path.resolve())}, {**fusion_cfg, **decoder_cfg}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LectureSense V4.1 inference with validation-selected fusion and decoder"
    )
    parser.add_argument("--transcript", required=True, help="MaViLS-compatible SRT transcript")
    parser.add_argument("--slides", required=True, help="Lecture slide PDF")
    parser.add_argument("--video", required=True, help="Lecture MP4 used for visual score fusion")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT))
    parser.add_argument("--tuned-config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--out", required=True, help="Output timeline JSON path")
    parser.add_argument("--render-dir", required=True, help="Directory for rendered slide images")
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    args = parser.parse_args()

    for label, raw_path in (
        ("transcript", args.transcript),
        ("slides PDF", args.slides),
        ("lecture video", args.video),
        ("V4 checkpoint", args.checkpoint),
    ):
        if not Path(raw_path).exists():
            raise FileNotFoundError(f"{label} not found: {raw_path}")

    config_info, cfg = load_selected_config(Path(args.tuned_config))
    fusion_cfg = {key: cfg[key] for key in FUSION_KEYS}
    decoder_cfg = {key: cfg[key] for key in DECODER_KEYS}

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"V4 checkpoint (read-only): {Path(args.checkpoint).resolve()}")
    print(f"V4.1 selected config: {config_info['path']}")
    print(f"Fusion config: {fusion_cfg}")
    print(f"Decoder config: {decoder_cfg}")

    model, base_model, max_slides, model_temperature, _checkpoint_decoder = (
        v4.load_v4_checkpoint(Path(args.checkpoint))
    )

    out_path = Path(args.out).resolve()
    render_dir = Path(args.render_dir).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    print("\n1/6 READING TRANSCRIPT")
    transcript_segments = base.read_mavils_srt(args.transcript)
    if not transcript_segments:
        raise RuntimeError("No transcript segments found")

    print("\n2/6 RENDERING SLIDES")
    slide_paths = base.render_pdf(args.slides, str(render_dir))
    if not slide_paths:
        raise RuntimeError("No slides rendered")

    print("\n3/6 EXTRACTING OCR")
    ocr_texts = base.extract_ocr(slide_paths)
    if len(ocr_texts) != len(slide_paths):
        raise RuntimeError("OCR count does not match slide count")

    print("\n4/6 CREATING MULTIMODAL FEATURES")
    sentence_model = base.load_sbert()
    transcript_embeddings = base.create_text_embeddings(
        sentence_model, [item["text"] for item in transcript_segments]
    )
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n5/6 V4.1 SCORE FUSION")
    base_scores = base.compute_score_matrix(
        base_model,
        transcript_embeddings,
        ocr_embeddings,
        image_embeddings,
        batch_size=max(1, args.score_batch),
    )
    adjusted_scores, boundary_prob, next_logits = v4.temporal_adjust_scores(
        base_scores,
        transcript_segments,
        model,
        max_slides,
        model_temperature,
    )

    # Match tune_v4_decoder_only.fused_scores() exactly:
    # base + residual_scale * (adjusted - base), then the production video fuse.
    v4_scores = (
        base_scores
        + fusion_cfg["residual_scale"] * (adjusted_scores - base_scores)
    ).astype(np.float32)

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
        segment_starts=[float(item["start"]) for item in transcript_segments],
        segment_ends=[float(item["end"]) for item in transcript_segments],
    )

    if video_score_matrix.shape != v4_scores.shape:
        raise RuntimeError(
            f"Video score matrix shape {video_score_matrix.shape} does not match V4 scores {v4_scores.shape}"
        )

    fused_scores = fuse_model_and_video_scores(
        v4_scores,
        video_score_matrix,
        video_weight=fusion_cfg["video_weight"],
    ).astype(np.float32)

    score_temperature = fusion_cfg["score_temperature"]
    final_score_matrix = (fused_scores / np.float32(score_temperature)).astype(np.float32)

    # Persist diagnostic matrices without modifying the original checkpoint.
    np.save(out_path.parent / "v4_base_score_matrix.npy", base_scores)
    np.save(out_path.parent / "v4_adjusted_score_matrix.npy", adjusted_scores)
    np.save(out_path.parent / "v4_residual_scaled_score_matrix.npy", v4_scores)
    np.save(out_path.parent / "v4_video_score_matrix.npy", video_score_matrix)
    np.save(out_path.parent / "v4_boundary_probability.npy", boundary_prob)
    np.save(out_path.parent / "v4_next_slide_logits.npy", next_logits)
    np.save(out_path.parent / "v4_final_score_matrix.npy", final_score_matrix)

    print("\n6/6 RUNNING V4.1 BOUNDARY-AWARE DYNAMIC PROGRAMMING")
    starts = [float(item["start"]) for item in transcript_segments]
    ends = [float(item["end"]) for item in transcript_segments]
    alignment = boundary_aware_dp(
        final_score_matrix,
        boundary_prob,
        next_logits,
        starts,
        ends,
        **decoder_cfg,
    )
    if not alignment:
        raise RuntimeError("Boundary-aware DP returned no alignment")

    timeline = base.make_timeline(transcript_segments, slide_paths, alignment)
    validation_summary = {
        "macro_accuracy": config_info["payload"].get("validation_macro_accuracy"),
        "weighted_accuracy": config_info["payload"].get("validation_weighted_accuracy"),
        "validation_lectures": config_info["payload"].get("validation_lectures", []),
        "test_used_for_selection": False,
    }
    metadata = {
        "model": "LectureSenseBoundaryAwareTemporalGRU_V4.1",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tuned_config": config_info["path"],
        "tuning_type": "validation_selected_fusion_and_decoder",
        "validation_summary": validation_summary,
        "temporal_max_slides": max_slides,
        "temporal_temperature": model_temperature,
        "video_enabled": True,
        "video_weight": fusion_cfg["video_weight"],
        "video_sample_seconds": max(0.25, args.video_sample_seconds),
        "residual_scale": fusion_cfg["residual_scale"],
        "score_temperature": score_temperature,
        "decoder": decoder_cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }
    base.save_result(str(out_path), timeline, metadata)
    print(f"\nV4.1 INFERENCE COMPLETE\nTimeline: {out_path}")
    print("Original V4 checkpoint was not modified.")


if __name__ == "__main__":
    main()
