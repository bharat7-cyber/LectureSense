from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference as base
from boundary_dp_v4 import boundary_aware_dp
from video_visual_alignment import fuse_model_and_video_scores

V4_INFERENCE = SRC / "inference_temporal_v4.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
DEFAULT_CONFIG = ROOT / "checkpoints" / "v4_1_decoder_tuned_config.json"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"V4.1 configuration not found: {path}\n"
            f"Run validation tuning first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("validation_only") is not True:
        raise RuntimeError("Config is not marked validation_only=true.")

    if payload.get("test_used") is True:
        raise RuntimeError("Refusing to use a config marked test_used=true.")

    fusion = payload.get("fusion", {})
    decoder = payload.get("decoder", {})

    required_fusion = [
        "residual_scale",
        "video_weight",
        "score_temperature",
    ]

    required_decoder = [
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
    ]

    missing = [
        k for k in required_fusion
        if k not in fusion
    ] + [
        k for k in required_decoder
        if k not in decoder
    ]

    if missing:
        raise RuntimeError(
            "V4.1 config is missing keys: "
            + ", ".join(missing)
        )

    return {
        "fusion": fusion,
        "decoder": decoder,
        "payload": payload,
    }


def run_v4_stage(
    transcript: Path,
    slides: Path,
    video: Path,
    checkpoint: Path,
    raw_out: Path,
    render_dir: Path,
    video_weight: float,
) -> None:
    if not V4_INFERENCE.exists():
        raise FileNotFoundError(f"Missing V4 inference script: {V4_INFERENCE}")

    cmd = [
        sys.executable,
        str(V4_INFERENCE),
        "--transcript",
        str(transcript),
        "--slides",
        str(slides),
        "--video",
        str(video),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(raw_out),
        "--render-dir",
        str(render_dir),
        "--jump-penalty",
        "0.05",
        "--max-duration",
        "120",
        "--duration-penalty",
        "0.003",
        "--video-weight",
        str(float(video_weight)),
        "--video-sample-seconds",
        "1.0",
    ]

    print("\n" + "=" * 78)
    print("V4.1 INFERENCE - V4 FEATURE EXTRACTION")
    print("=" * 78)
    print("Running existing V4 model. Checkpoint is read-only.")
    print(f"Video weight: {video_weight:.4f}")

    process = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        timeout=1800,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"V4 inference failed with exit code {process.returncode}"
        )

    if not raw_out.exists():
        raise RuntimeError(
            f"V4 inference completed but did not create {raw_out}"
        )


def load_matrix(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Required V4 matrix missing: {path} ({name})"
        )
    return np.load(path).astype(np.float32)


def build_tuned_scores(
    base_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    video_scores: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    residual_scale = float(cfg["residual_scale"])
    video_weight = float(cfg["video_weight"])
    temperature = max(1e-4, float(cfg["score_temperature"]))

    if base_scores.shape == adjusted_scores.shape:
        v4_scores = (
            base_scores
            + residual_scale * (adjusted_scores - base_scores)
        ).astype(np.float32)
    else:
        print(
            "WARNING: base/adjusted shapes differ; "
            "residual_scale disabled for this lecture."
        )
        v4_scores = adjusted_scores.copy()

    if video_scores.shape == v4_scores.shape and video_weight > 0:
        fused = fuse_model_and_video_scores(
            v4_scores,
            video_scores,
            video_weight=video_weight,
        )
    else:
        fused = v4_scores

    return (
        fused * np.float32(1.0 / temperature)
    ).astype(np.float32)


def run_decoder(
    scores: np.ndarray,
    boundary: np.ndarray,
    next_logits: np.ndarray,
    starts: list[float],
    ends: list[float],
    decoder: dict,
) -> np.ndarray:
    path = boundary_aware_dp(
        scores,
        boundary,
        next_logits,
        starts,
        ends,
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

    if not path:
        raise RuntimeError("V4.1 decoder returned no alignment.")

    return np.asarray(path, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LectureSense V4.1 production inference with validation-locked tuning."
    )

    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)

    args = parser.parse_args()

    transcript = Path(args.transcript).resolve()
    slides = Path(args.slides).resolve()
    video = Path(args.video).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    out_path = Path(args.out).resolve()
    render_dir = Path(args.render_dir).resolve()

    cfg_payload = load_config(config_path)
    fusion = cfg_payload["fusion"]
    decoder = cfg_payload["decoder"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_path.with_name(
        out_path.stem + "_raw_v4.json"
    )

    run_v4_stage(
        transcript=transcript,
        slides=slides,
        video=video,
        checkpoint=checkpoint,
        raw_out=raw_out,
        render_dir=render_dir,
        video_weight=float(fusion["video_weight"]),
    )

    base_scores = load_matrix(
        raw_out.parent / "v4_base_score_matrix.npy",
        "base scores",
    )

    adjusted_scores = load_matrix(
        raw_out.parent / "v4_adjusted_score_matrix.npy",
        "adjusted scores",
    )

    video_scores = load_matrix(
        raw_out.parent / "v4_video_score_matrix.npy",
        "video scores",
    )

    boundary = load_matrix(
        raw_out.parent / "v4_boundary_probability.npy",
        "boundary probabilities",
    ).reshape(-1)

    next_logits = load_matrix(
        raw_out.parent / "v4_next_slide_logits.npy",
        "next-slide logits",
    )

    transcript_segments = base.read_mavils_srt(
        str(transcript)
    )

    if not transcript_segments:
        raise RuntimeError("No transcript segments found.")

    starts = [
        float(x["start"])
        for x in transcript_segments
    ]

    ends = [
        float(x["end"])
        for x in transcript_segments
    ]

    if len(transcript_segments) != base_scores.shape[0]:
        raise RuntimeError(
            "Transcript segment count does not match "
            "V4 score rows."
        )

    scores = build_tuned_scores(
        base_scores,
        adjusted_scores,
        video_scores,
        fusion,
    )

    alignment = run_decoder(
        scores,
        boundary,
        next_logits,
        starts,
        ends,
        decoder,
    )

    slide_paths = sorted(
        render_dir.glob("slide_*.png")
    )

    if len(slide_paths) != scores.shape[1]:
        raise RuntimeError(
            f"Rendered slides ({len(slide_paths)}) "
            f"do not match score columns ({scores.shape[1]})."
        )

    timeline = base.make_timeline(
        transcript_segments,
        slide_paths,
        alignment,
    )

    metadata = {
        "model": "LectureSenseBoundaryAwareTemporalGRU_V4.1",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "validation_locked_config": True,
        "video_enabled": True,
        "video_weight": float(fusion["video_weight"]),
        "score_temperature": float(fusion["score_temperature"]),
        "residual_scale": float(fusion["residual_scale"]),
        "decoder": decoder,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }

    base.save_result(
        str(out_path),
        timeline,
        metadata,
    )

    try:
        raw_out.unlink()
    except OSError:
        pass

    print("\n" + "=" * 78)
    print("V4.1 INFERENCE COMPLETE")
    print("=" * 78)
    print(f"Timeline : {out_path}")
    print(f"Slides   : {render_dir}")
    print(
        "Notes/UI compatibility: YES - output uses the same "
        "LectureSense timeline structure."
    )


if __name__ == "__main__":
    main()
