from __future__ import annotations

import argparse
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
from dp_alignment import dp_align
from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalResidualGRU(nn.Module):
    def __init__(self, input_dim, max_slides, hidden=128, layers=2):
        super().__init__()
        self.max_slides = max_slides
        self.input_proj = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.10))
        self.gru = nn.GRU(128, hidden, num_layers=layers, batch_first=True,
                           dropout=0.20 if layers > 1 else 0.0, bidirectional=True)
        self.delta = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden, max_slides))

    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)
        return self.delta(h)


def load_v3_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    max_slides = int(ckpt["max_slides"])
    hidden = int(ckpt.get("hidden", 128))
    layers = int(ckpt.get("layers", 2))
    temperature = float(ckpt.get("temperature", 1.0))
    temporal = TemporalResidualGRU(max_slides + 2, max_slides, hidden, layers).to(DEVICE)
    temporal.load_state_dict(ckpt["model_state_dict"], strict=True)
    temporal.eval()
    base_model = base.LectureSenseCrossModal().to(DEVICE)
    base_model.load_state_dict(ckpt["base_model_state_dict"], strict=True)
    base_model.eval()
    print(f"V3 checkpoint: {path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'n/a')}")
    print(f"Checkpoint Val Top-1: {ckpt.get('val_top1', 0.0):.4f}")
    print(f"Checkpoint Val MRR: {ckpt.get('val_mrr', 0.0):.4f}")
    return temporal, base_model, max_slides, temperature


def temporal_adjust_scores(base_scores, transcript_segments, temporal_model, max_slides, temperature):
    n, s = base_scores.shape
    if s > max_slides:
        raise RuntimeError(f"Lecture has {s} slides, checkpoint supports {max_slides}")
    starts = np.array([float(x["start"]) for x in transcript_segments], dtype=np.float32)
    ends = np.array([float(x["end"]) for x in transcript_segments], dtype=np.float32)
    total = max(1.0, float(ends[-1] - starts[0]))
    time_norm = (starts - starts[0]) / total
    durations = np.clip((ends - starts) / total, 0.0, 1.0)
    probs = np.zeros((n, max_slides), dtype=np.float32)
    probs[:, :s] = F.softmax(torch.from_numpy(base_scores.astype(np.float32)) / temperature, dim=-1).numpy()
    features = np.zeros((n, max_slides + 2), dtype=np.float32)
    features[:, :max_slides] = probs
    features[:, max_slides] = time_norm
    features[:, max_slides + 1] = durations
    with torch.no_grad():
        delta = temporal_model(torch.from_numpy(features).unsqueeze(0).to(DEVICE))[0, :, :s].cpu().numpy()
    return base_scores + delta, delta


def main():
    parser = argparse.ArgumentParser(description="LectureSense V3 temporal inference")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--jump-penalty", type=float, default=0.05)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--duration-penalty", type=float, default=0.01)
    parser.add_argument("--max-run-seconds", type=float, default=480.0)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--video-weight", type=float, default=0.40)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    temporal_model, base_model, max_slides, temperature = load_v3_checkpoint(Path(args.checkpoint))

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
    transcript_embeddings = base.create_text_embeddings(sentence_model, [x["text"] for x in transcript_segments])
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n5/6 TEMPORAL MULTIMODAL ALIGNMENT")
    base_scores = base.compute_score_matrix(base_model, transcript_embeddings, ocr_embeddings, image_embeddings, batch_size=max(1, args.score_batch))
    adjusted_scores, delta = temporal_adjust_scores(base_scores, transcript_segments, temporal_model, max_slides, temperature)
    np.save(Path(args.out).parent / "v3_base_score_matrix.npy", base_scores)
    np.save(Path(args.out).parent / "v3_temporal_residual.npy", delta)
    final_score_matrix = adjusted_scores

    if args.video:
        print("\nExtracting actual lecture-video visual evidence...")
        video_embeddings, video_timestamps, _ = create_video_frame_embeddings(
            args.video, device=DEVICE,
            sample_every_seconds=max(0.25, args.video_sample_seconds), batch_size=8,
        )
        video_score_matrix = build_video_slide_score_matrix(
            video_embeddings, video_timestamps, image_embeddings,
            segment_starts=[x["start"] for x in transcript_segments],
            segment_ends=[x["end"] for x in transcript_segments],
        )
        final_score_matrix = fuse_model_and_video_scores(
            adjusted_scores, video_score_matrix, video_weight=args.video_weight,
        )
        np.save(Path(args.out).parent / "v3_video_score_matrix.npy", video_score_matrix)
        np.save(Path(args.out).parent / "v3_fused_score_matrix.npy", final_score_matrix)

    print("\n6/6 RUNNING DURATION-AWARE DYNAMIC PROGRAMMING")
    alignment = dp_align(
        final_score_matrix,
        jump_penalty=args.jump_penalty,
        max_forward_jump=1,
        segment_starts=[x["start"] for x in transcript_segments],
        segment_ends=[x["end"] for x in transcript_segments],
        max_duration_seconds=args.max_duration,
        duration_penalty=args.duration_penalty,
        max_run_seconds=args.max_run_seconds,
    )
    if not alignment:
        raise RuntimeError("Dynamic-programming alignment returned no result")

    timeline = base.make_timeline(transcript_segments, slide_paths, alignment)
    metadata = {
        "model": "LectureSenseTemporalResidualGRU",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(Path(args.checkpoint)),
        "temporal_max_slides": max_slides,
        "temporal_temperature": temperature,
        "video_enabled": bool(args.video),
        "video_weight": args.video_weight if args.video else 0.0,
        "video_sample_seconds": args.video_sample_seconds if args.video else None,
        "jump_penalty": args.jump_penalty,
        "max_duration_seconds": args.max_duration,
        "duration_penalty": args.duration_penalty,
        "max_run_seconds": args.max_run_seconds,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }
    base.save_result(args.out, timeline, metadata)
    print(f"\nV3 INFERENCE COMPLETE\nTimeline: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
