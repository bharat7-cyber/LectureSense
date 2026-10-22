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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import inference as base
from boundary_dp_v4_1 import boundary_aware_dp_v41
from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRA_FEATURES = 12


class BoundaryAwareTemporalGRUV41(nn.Module):
    def __init__(self, input_dim, max_slides, hidden=192, layers=2, heads=4):
        super().__init__()
        self.max_slides = max_slides
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.gru = nn.GRU(
            160, hidden, num_layers=layers, batch_first=True,
            dropout=0.20 if layers > 1 else 0.0, bidirectional=True
        )
        d_model = hidden * 2
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=heads, dropout=0.10, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )
        self.delta = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max_slides),
        )
        self.boundary = nn.Sequential(
            nn.Linear(d_model, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )
        self.next_slide = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max_slides),
        )


    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        gate = torch.sigmoid(self.gate(h)).squeeze(-1)
        return (
            self.delta(h),
            self.boundary(h).squeeze(-1),
            self.next_slide(h),
            gate,
        )


def make_v41_features(scores, time_norm, max_slides, temperature):
    n, s = scores.shape
    if s > max_slides:
        raise ValueError(f"Lecture has {s} slides but checkpoint supports {max_slides}")

    logits = torch.from_numpy(scores.astype(np.float32))
    probs = F.softmax(logits / temperature, dim=-1).numpy()

    p_sorted = np.sort(probs, axis=1)
    top1 = p_sorted[:, -1]
    top2 = p_sorted[:, -2] if s >= 2 else np.zeros(n, dtype=np.float32)
    top3 = p_sorted[:, -3] if s >= 3 else top2
    margin = top1 - top2
    top3_mean = (top1 + top2 + top3) / 3.0
    peak_gap = top1 - top3_mean

    denom = max(1e-8, float(np.log(max(2, s))))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1) / denom

    argmax_idx = np.argmax(probs, axis=1).astype(np.float32)
    argmax_norm = argmax_idx / max(1.0, float(max(1, s - 1)))

    change = np.zeros(n, dtype=np.float32)
    top1_change = np.zeros(n, dtype=np.float32)
    margin_change = np.zeros(n, dtype=np.float32)
    entropy_change = np.zeros(n, dtype=np.float32)
    argmax_change = np.zeros(n, dtype=np.float32)
    if n > 1:
        prev = probs[:-1]
        cur = probs[1:]
        prev_norm = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-8)
        cur_norm = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8)
        change[1:] = np.clip(
            1.0 - np.sum(prev_norm * cur_norm, axis=1), 0.0, 1.0
        )
        top1_change[1:] = np.abs(top1[1:] - top1[:-1])
        margin_change[1:] = np.abs(margin[1:] - margin[:-1])
        entropy_change[1:] = np.abs(entropy[1:] - entropy[:-1])
        argmax_change[1:] = (argmax_idx[1:] != argmax_idx[:-1]).astype(np.float32)

    durations = np.zeros(n, dtype=np.float32)
    if n > 1:
        durations[:-1] = np.clip(np.diff(time_norm), 0.0, 1.0)
        durations[-1] = durations[-2]

    x = np.zeros((n, max_slides + EXTRA_FEATURES), dtype=np.float32)
    x[:, :s] = probs
    extras = [
        time_norm, durations, top1, margin, entropy, change,
        top1_change, margin_change, entropy_change, argmax_change,
        argmax_norm, peak_gap,
    ]
    for j, feat in enumerate(extras):
        x[:, max_slides + j] = feat
    return x


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    max_slides = int(ckpt["max_slides"])
    hidden = int(ckpt.get("hidden", 192))
    layers = int(ckpt.get("layers", 2))
    heads = int(ckpt.get("heads", 4))
    temperature = float(ckpt.get("temperature", 0.90))

    model = BoundaryAwareTemporalGRUV41(
        max_slides + int(ckpt.get("extra_features", EXTRA_FEATURES)),
        max_slides, hidden, layers, heads
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    decoder_cfg = ckpt.get("decoder_config")
    if not decoder_cfg:
        decoder_path = ROOT / "checkpoints" / "temporal_v4_1_decoder_config.json"
        if decoder_path.exists():
            decoder_cfg = json.loads(decoder_path.read_text(encoding="utf-8"))

    if not decoder_cfg:
        decoder_cfg = {
            "jump_penalty": 0.03,
            "max_forward_jump": 2,
            "max_duration_seconds": 150.0,
            "duration_penalty": 0.001,
            "max_run_seconds": 1500.0,
            "boundary_reward": 0.75,
            "boundary_stay_penalty": 0.35,
            "transition_logits_weight": 0.30,
            "premature_transition_penalty": 0.35,
            "boundary_threshold": 0.20,
            "jump_penalty_power": 1.35,
        }

    print(f"V4.1 checkpoint: {path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'n/a')}")
    print(f"Validation Top-1: {ckpt.get('val_top1', 'n/a')}")
    print(f"Validation MRR: {ckpt.get('val_mrr', 'n/a')}")
    print(f"Validation DP accuracy: {ckpt.get('val_decoder_accuracy', 'n/a')}")
    return model, max_slides, temperature, decoder_cfg, ckpt


def temporal_adjust_scores(base_scores, transcript_segments, model, max_slides, temperature):
    n, s = base_scores.shape
    if s > max_slides:
        raise RuntimeError(f"Lecture has {s} slides but checkpoint supports {max_slides}")

    starts = np.array([float(x["start"]) for x in transcript_segments], dtype=np.float32)
    total = max(1.0, float(starts[-1] - starts[0]))
    time_norm = (starts - starts[0]) / total

    features = make_v41_features(base_scores, time_norm, max_slides, temperature)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        delta, boundary_logit, next_logits, gate = model(x)

    # Same probabilistic base as V4, but gated by learned confidence.
    probs = torch.from_numpy(base_scores.astype(np.float32)).to(DEVICE)
    probs = F.softmax(probs / temperature, dim=-1).clamp_min(1e-6)
    base_logits = torch.log(probs)

    adjusted = (
        base_logits
        + gate[0].unsqueeze(-1) * delta[0, :, :s]
    ).cpu().numpy()

    boundary_prob = torch.sigmoid(boundary_logit[0]).cpu().numpy()
    next_logits = next_logits[0, :, :s].cpu().numpy()
    return adjusted, boundary_prob, next_logits


def main():
    parser = argparse.ArgumentParser(description="LectureSense V4.1 inference")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--video-weight", type=float, default=0.40)
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
    parser.add_argument("--jump-penalty-power", type=float, default=None)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, max_slides, temperature, decoder_cfg, ckpt = load_checkpoint(Path(args.checkpoint))

    overrides = {
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
        "jump_penalty_power": args.jump_penalty_power,
    }
    for k, v in overrides.items():
        if v is not None:
            decoder_cfg[k] = v

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
        sentence_model, [x["text"] for x in transcript_segments]
    )
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n5/6 V4.1 GATED TEMPORAL ALIGNMENT")
    base_model = base.LectureSenseCrossModal().to(DEVICE)
    base_model.load_state_dict(ckpt["base_model_state_dict"], strict=True)
    base_model.eval()

    base_scores = base.compute_score_matrix(
        base_model,
        transcript_embeddings,
        ocr_embeddings,
        image_embeddings,
        batch_size=max(1, args.score_batch),
    )
    adjusted_scores, boundary_prob, next_logits = temporal_adjust_scores(
        base_scores, transcript_segments, model, max_slides, temperature
    )

    final_score_matrix = adjusted_scores

    if args.video:
        print("\nExtracting actual lecture-video visual evidence...")
        video_embeddings, video_timestamps, _ = create_video_frame_embeddings(
            args.video, device=DEVICE,
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
        final_score_matrix = fuse_model_and_video_scores(
            adjusted_scores, video_score_matrix, video_weight=args.video_weight
        )
        np.save(Path(args.out).parent / "v4_1_video_score_matrix.npy", video_score_matrix)

    np.save(Path(args.out).parent / "v4_1_base_score_matrix.npy", base_scores)
    np.save(Path(args.out).parent / "v4_1_adjusted_score_matrix.npy", adjusted_scores)
    np.save(Path(args.out).parent / "v4_1_boundary_probability.npy", boundary_prob)
    np.save(Path(args.out).parent / "v4_1_next_slide_logits.npy", next_logits)
    np.save(Path(args.out).parent / "v4_1_final_score_matrix.npy", final_score_matrix)

    print("\n6/6 RUNNING V4.1 DYNAMIC PROGRAMMING")
    starts = [x["start"] for x in transcript_segments]
    ends = [x["end"] for x in transcript_segments]
    alignment = boundary_aware_dp_v41(
        final_score_matrix,
        boundary_prob,
        next_logits,
        starts,
        ends,
        **decoder_cfg,
    )
    if not alignment:
        raise RuntimeError("V4.1 DP returned no result")

    timeline = base.make_timeline(transcript_segments, slide_paths, alignment)
    metadata = {
        "model": "LectureSenseBoundaryAwareTemporalGRU_V4_1",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(Path(args.checkpoint)),
        "temporal_max_slides": max_slides,
        "temporal_temperature": temperature,
        "video_enabled": bool(args.video),
        "video_weight": args.video_weight if args.video else 0.0,
        "video_sample_seconds": args.video_sample_seconds if args.video else None,
        "decoder": decoder_cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }
    base.save_result(args.out, timeline, metadata)

    print(f"\nV4.1 INFERENCE COMPLETE\nTimeline: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
