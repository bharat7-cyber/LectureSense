from __future__ import annotations

"""Run LectureSense V11 on an arbitrary lecture SRT + PDF + MP4.

Requires, in the project root:
  - LectureSense_V11_Supervised_Candidate_Reranking.py
  - LectureSense_V7_FullDevelopment_GatedFusion.py
  - src/inference_temporal_v4.py and its dependencies
  - src/visual_first_alignment.py
  - checkpoints/temporal_v4_best_model.pt
  - checkpoints/v11_supervised_candidate_reranker.pt

The existing V4 checkpoint is used only to generate score/boundary evidence. The
V11 reranker and decoder from the locked V11 bundle produce the final timeline.
No training, validation tuning, or test-label access occurs in this script.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import inference as base
    import LectureSense_V11_Supervised_Candidate_Reranking as v11
    from visual_first_alignment import run_visual_first
except Exception as exc:
    raise SystemExit(
        "LectureSense V11 inference imports failed. Keep this script in the project root "
        "beside LectureSense_V11_Supervised_Candidate_Reranking.py and the src/ directory.\n"
        f"{type(exc).__name__}: {exc}"
    )


def load_matrix(folder: Path, names: list[str], required: bool = False) -> np.ndarray | None:
    for name in names:
        path = folder / name
        if path.exists():
            value = np.asarray(np.load(path), dtype=np.float32)
            if not np.all(np.isfinite(value)):
                value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value
    if required:
        raise FileNotFoundError(f"Required score artifact is missing in {folder}: {names}")
    return None


def run_v4_stage(
    transcript: Path,
    slides: Path,
    video: Path,
    v4_checkpoint: Path,
    work_dir: Path,
    render_dir: Path,
    video_weight: float,
    video_sample_seconds: float,
    timeout_seconds: int,
) -> Path:
    if not v4_checkpoint.exists():
        raise FileNotFoundError(f"V4 checkpoint not found: {v4_checkpoint}")
    script = SRC / "inference_temporal_v4.py"
    if not script.exists():
        raise FileNotFoundError(f"V4 inference script not found: {script}")

    work_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    v4_timeline = work_dir / "v4_timeline.json"
    cmd = [
        sys.executable,
        str(script),
        "--transcript", str(transcript),
        "--slides", str(slides),
        "--video", str(video),
        "--checkpoint", str(v4_checkpoint),
        "--out", str(v4_timeline),
        "--render-dir", str(render_dir),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", str(video_weight),
        "--video-sample-seconds", str(video_sample_seconds),
    ]
    print("\n[1/3] Generating V4 multimodal score evidence...")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-35:] + (proc.stderr or "").splitlines()[-35:])
        raise RuntimeError("V4 inference failed. Recent subprocess output:\n" + tail)
    if not v4_timeline.exists():
        raise RuntimeError("V4 stage exited successfully but did not produce its timeline JSON.")
    print(proc.stdout[-5000:] if proc.stdout else "V4 stage completed.")
    return work_dir


def align_visual_scores(video: Path, slides_pdf: Path, segment_mids: np.ndarray) -> np.ndarray:
    print("\n[2/3] Computing visual-first scores (same 2-second sampling convention as V11 training)...")
    _visual_path, visual_scores, visual_times, _extra = run_visual_first(
        Path(video),
        Path(slides_pdf),
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )
    visual_scores = np.asarray(visual_scores, dtype=np.float32)
    visual_times = np.asarray(visual_times, dtype=np.float32).reshape(-1)
    if visual_scores.ndim != 2 or len(visual_times) != len(visual_scores):
        raise RuntimeError(
            f"Visual-first output shape mismatch: scores={visual_scores.shape}, times={visual_times.shape}"
        )
    if len(visual_times) == 0:
        raise RuntimeError("Visual-first branch produced no timestamped scores.")
    idx = v11.nearest_indices(visual_times, np.asarray(segment_mids, dtype=np.float32))
    return visual_scores[idx].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LectureSense V11 on uploaded lecture files")
    parser.add_argument("--transcript", required=True, help="SRT file, e.g. faster-whisper output")
    parser.add_argument("--slides", required=True, help="Lecture slide PDF")
    parser.add_argument("--video", required=True, help="Lecture MP4 video")
    parser.add_argument("--out", required=True, help="Output timeline JSON path")
    parser.add_argument("--render-dir", required=True, help="Directory for rendered slide PNGs")
    parser.add_argument(
        "--v11-checkpoint",
        default=str(ROOT / "checkpoints" / "v11_supervised_candidate_reranker.pt"),
        help="Locked V11 model bundle (.pt), not the JSON config",
    )
    parser.add_argument(
        "--v4-checkpoint",
        default=str(ROOT / "checkpoints" / "temporal_v4_best_model.pt"),
        help="Existing V4 checkpoint used to create base evidence",
    )
    parser.add_argument("--video-weight", type=float, default=0.40)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    transcript = Path(args.transcript).resolve()
    slides_pdf = Path(args.slides).resolve()
    video = Path(args.video).resolve()
    out_path = Path(args.out).resolve()
    render_dir = Path(args.render_dir).resolve()
    v11_checkpoint = Path(args.v11_checkpoint).resolve()
    v4_checkpoint = Path(args.v4_checkpoint).resolve()

    for label, path in (("Transcript", transcript), ("Slide PDF", slides_pdf), ("Video", video)):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if not v11_checkpoint.exists():
        raise FileNotFoundError(
            f"V11 model bundle not found: {v11_checkpoint}. It must be the .pt file, not the JSON config."
        )

    try:
        bundle = torch.load(v11_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(v11_checkpoint, map_location="cpu")

    if not isinstance(bundle, dict) or "models" not in bundle or "feature_names" not in bundle:
        raise RuntimeError("The supplied V11 .pt file does not contain the expected trained model bundle.")
    feature_names = list(bundle["feature_names"])
    expected_dim = int(bundle.get("feature_dim", len(feature_names)))
    if len(feature_names) != expected_dim:
        raise RuntimeError(f"V11 feature mismatch: feature_names={len(feature_names)}, feature_dim={expected_dim}")

    selected_v11 = bool(bundle.get("model_selected_over_v4_on_validation", True))
    if not selected_v11:
        raise RuntimeError(
            "This V11 bundle says V11 was not selected over V4 on validation. "
            "Use the selected V4 inference path rather than presenting V11 as the selected model."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / "_v11_v4_evidence"
    work_dir.mkdir(parents=True, exist_ok=True)

    run_v4_stage(
        transcript=transcript,
        slides=slides_pdf,
        video=video,
        v4_checkpoint=v4_checkpoint,
        work_dir=work_dir,
        render_dir=render_dir,
        video_weight=args.video_weight,
        video_sample_seconds=args.video_sample_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    segments = v11.v7.get_segments(transcript)
    mids = np.asarray(v11.v7.segment_mids(segments), dtype=np.float32)
    if len(mids) == 0:
        raise RuntimeError("Transcript parser returned no usable segments.")

    base_scores = load_matrix(work_dir, ["v4_base_score_matrix.npy"], required=True)
    adjusted_scores = load_matrix(work_dir, ["v4_adjusted_score_matrix.npy"], required=True)
    final_scores = load_matrix(work_dir, ["v4_final_score_matrix.npy"], required=True)
    v4_video_scores = load_matrix(work_dir, ["v4_video_score_matrix.npy"])
    boundary = load_matrix(work_dir, ["v4_boundary_probability.npy"], required=True)
    next_logits = load_matrix(work_dir, ["v4_next_slide_logits.npy"], required=True)

    n_segments, n_slides = final_scores.shape
    if len(mids) != n_segments:
        raise RuntimeError(
            f"SRT/V4 score mismatch: transcript has {len(mids)} segments but V4 matrix has {n_segments} rows."
        )
    if base_scores.shape != final_scores.shape or adjusted_scores.shape != final_scores.shape:
        raise RuntimeError("V4 base/adjusted/final score matrix shapes do not match.")
    if boundary.reshape(-1).shape[0] != n_segments or next_logits.shape != final_scores.shape:
        raise RuntimeError("V4 boundary/next-slide artifacts do not match score matrix dimensions.")
    boundary = np.clip(boundary.reshape(-1).astype(np.float32), 1e-4, 1.0 - 1e-4)
    if v4_video_scores is None or v4_video_scores.shape != final_scores.shape:
        v4_video_scores = np.zeros_like(final_scores, dtype=np.float32)

    visual_scores_by_segment = align_visual_scores(video, slides_pdf, mids)
    if visual_scores_by_segment.shape != final_scores.shape:
        raise RuntimeError(
            f"Visual score shape {visual_scores_by_segment.shape} does not match V4 matrix {final_scores.shape}."
        )

    sources = {
        "base": base_scores.astype(np.float32),
        "adjusted": adjusted_scores.astype(np.float32),
        "final": final_scores.astype(np.float32),
        "v4video": v4_video_scores.astype(np.float32),
        "visual": visual_scores_by_segment.astype(np.float32),
    }
    X_seq, actual_feature_names = v11.build_candidate_features(
        mids, mids, sources, boundary, next_logits.astype(np.float32)
    )
    if actual_feature_names != feature_names:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(actual_feature_names, feature_names)) if a != b),
            min(len(actual_feature_names), len(feature_names)),
        )
        raise RuntimeError(
            "V11 feature-version/order mismatch. Reuse the inference script matching the trained model. "
            f"Expected {len(feature_names)} features, built {len(actual_feature_names)}; first difference at {first_diff}."
        )

    print("\n[3/3] Applying the locked V11 ensemble and decoder...")
    emissions = v11.predict_emissions(
        model_states=bundle["models"],
        feature_dim=expected_dim,
        feature_names=feature_names,
        X_seq=X_seq,
        batch_size=128,
    )
    next_logprob = v11.next_log_prob(next_logits.astype(np.float32))
    decoder = dict(bundle["decoder"])
    required_decoder = {
        "max_forward_jump", "jump_penalty", "boundary_reward", "boundary_stay_penalty",
        "transition_logits_weight", "premature_transition_penalty", "boundary_threshold",
    }
    missing = required_decoder - set(decoder)
    if missing:
        raise RuntimeError(f"V11 checkpoint decoder is missing required fields: {sorted(missing)}")
    alignment = v11.transition_viterbi(emissions, boundary, next_logprob, **decoder)

    slide_paths = sorted(render_dir.glob("slide_*.png"))
    if len(slide_paths) != n_slides:
        raise RuntimeError(
            f"Rendered slide count mismatch: found {len(slide_paths)} slide_*.png files but score matrix has {n_slides} slides."
        )
    timeline = base.make_timeline(segments, slide_paths, alignment.tolist())
    metadata = {
        "project": "LectureSense",
        "model": "LectureSenseV11SupervisedCandidateReranker",
        "selected_system": "V11",
        "checkpoint": str(v11_checkpoint),
        "base_model": "LectureSenseBoundaryAwareTemporalGRU_V4",
        "v4_checkpoint": str(v4_checkpoint),
        "v11_feature_version": int(bundle.get("feature_version", 1)),
        "v11_feature_dim": expected_dim,
        "v11_seeds": bundle.get("seeds", []),
        "decoder": decoder,
        "video_enabled": True,
        "video_weight_for_v4_evidence": float(args.video_weight),
        "visual_first_sample_seconds": 2.0,
        "num_transcript_segments": int(n_segments),
        "num_slides": int(n_slides),
        "timeline_entries": int(len(timeline)),
    }
    base.save_result(str(out_path), timeline, metadata)
    np.save(out_path.parent / "v11_emission_matrix.npy", emissions)
    np.save(out_path.parent / "v11_alignment.npy", np.asarray(alignment, dtype=np.int32))

    print("\nV11 INFERENCE COMPLETE")
    print(f"Timeline: {out_path}")
    print(f"Slides: {len(slide_paths)}")
    print(f"Transcript segments: {n_segments}")
    print(f"Selected system: V11")


if __name__ == "__main__":
    main()
