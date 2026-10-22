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
from boundary_dp_v4_final import boundary_aware_dp_final
from video_visual_alignment import create_video_frame_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRA_FEATURES = 6


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
        h = self.norm(h + torch.tanh(self.attn_gate) * attn_out)
        return self.delta(h), self.boundary(h).squeeze(-1), self.next_slide(h)


def make_v4_features(scores, time_norm, max_slides, temperature):
    n, s = scores.shape
    if s > max_slides:
        raise ValueError(f"Lecture has {s} slides but checkpoint supports {max_slides}")
    logits = torch.from_numpy(scores.astype(np.float32))
    probs = F.softmax(logits / temperature, dim=-1).numpy()
    p_sorted = np.sort(probs, axis=1)
    top1 = p_sorted[:, -1]
    top2 = p_sorted[:, -2] if s >= 2 else np.zeros(n, dtype=np.float32)
    margin = top1 - top2
    denom = max(1e-8, float(np.log(max(2, s))))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1) / denom
    change = np.zeros(n, dtype=np.float32)
    if n > 1:
        prev = probs[:-1]
        cur = probs[1:]
        prev = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-8)
        cur = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8)
        change[1:] = np.clip(1.0 - np.sum(prev * cur, axis=1), 0.0, 1.0)
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

    decoder_cfg = ckpt.get("decoder_config", {})
    external_cfg = path.parent / "temporal_v4_final_decoder_config.json"
    if external_cfg.exists():
        import json
        decoder_cfg = json.loads(external_cfg.read_text(encoding="utf-8"))
        print(f"Loaded validation-calibrated decoder config: {external_cfg}")
    defaults = {
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
    defaults.update(decoder_cfg)
    print(f"V4 FINAL checkpoint: {path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'n/a')}")
    print(f"Checkpoint Val Top-1: {ckpt.get('val_top1', 0.0):.4f}")
    print(f"Checkpoint Val MRR: {ckpt.get('val_mrr', 0.0):.4f}")
    print("Final decoder: visual-change + adjacent-evidence + anti-sticking semi-Markov")
    return model, base_model, max_slides, temperature, defaults


def _row_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def build_precise_video_scores(
    video_embeddings,
    video_timestamps,
    slide_embeddings,
    segment_starts,
    segment_ends,
    radius_seconds=0.75,
):
    """Use local time around each transcript midpoint instead of averaging
    the whole subtitle segment. This reduces boundary smearing at slide changes.
    """
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    video_timestamps = np.asarray(video_timestamps, dtype=np.float32)
    slide_embeddings = np.asarray(slide_embeddings, dtype=np.float32)
    frame_norm = video_embeddings / (np.linalg.norm(video_embeddings, axis=1, keepdims=True) + 1e-8)
    slide_norm = slide_embeddings / (np.linalg.norm(slide_embeddings, axis=1, keepdims=True) + 1e-8)
    frame_slide = frame_norm @ slide_norm.T

    frame_change = np.zeros(len(video_timestamps), dtype=np.float32)
    if len(video_embeddings) > 1:
        a = frame_norm[:-1]
        c = frame_norm[1:]
        frame_change[1:] = np.clip(1.0 - np.sum(a * c, axis=1), 0.0, 1.0)
        if len(frame_change) >= 3:
            p = np.pad(frame_change, (1, 1), mode="edge")
            frame_change = (
                p[:-2] * 0.25 + p[1:-1] * 0.50 + p[2:] * 0.25
            ).astype(np.float32)

    starts = np.asarray(segment_starts, dtype=np.float32)
    ends = np.asarray(segment_ends, dtype=np.float32)
    out = np.zeros((len(starts), slide_embeddings.shape[0]), dtype=np.float32)
    changes = np.zeros(len(starts), dtype=np.float32)

    for i, (a, z) in enumerate(zip(starts, ends)):
        midpoint = 0.5 * (float(a) + float(z))
        left = int(np.searchsorted(video_timestamps, midpoint - radius_seconds, side="left"))
        right = int(np.searchsorted(video_timestamps, midpoint + radius_seconds, side="right"))
        if right <= left:
            idx = int(np.argmin(np.abs(video_timestamps - midpoint)))
            out[i] = frame_slide[idx]
            changes[i] = frame_change[idx]
            continue
        ts = video_timestamps[left:right].astype(np.float64)
        dist = np.abs(ts - midpoint)
        sigma = max(0.25, radius_seconds * 0.70)
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        weights /= np.maximum(weights.sum(), 1e-8)
        out[i] = np.sum(frame_slide[left:right] * weights[:, None], axis=0)
        changes[i] = float(np.sum(frame_change[left:right] * weights))

    return out, changes


def temporal_adjust_scores(base_scores, transcript_segments, model, max_slides, temperature):
    starts = np.asarray([float(x["start"]) for x in transcript_segments], dtype=np.float32)
    total = max(1.0, float(starts[-1] - starts[0]))
    time_norm = (starts - starts[0]) / total
    features = make_v4_features(base_scores, time_norm, max_slides, temperature)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        delta, boundary_logit, next_logits = model(x)
    adjusted = base_scores + delta[0, :, :base_scores.shape[1]].cpu().numpy()
    boundary_prob = torch.sigmoid(boundary_logit[0]).cpu().numpy()
    next_logits = next_logits[0, :, :base_scores.shape[1]].cpu().numpy()
    return adjusted, boundary_prob, next_logits


def main():
    parser = argparse.ArgumentParser(description="LectureSense V4 final accuracy decoder")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--video-weight", type=float, default=0.40)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, base_model, max_slides, temperature, decoder_cfg = load_v4_checkpoint(Path(args.checkpoint))

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
    sentence_model = base.load_sbert()
    transcript_embeddings = base.create_text_embeddings(sentence_model, [x["text"] for x in transcript_segments])
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_embeddings = base.create_image_embeddings(slide_paths)

    print("\n4/6 CREATING MULTIMODAL FEATURES")
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

    final_score_matrix = adjusted_scores.copy()
    video_change = np.zeros(len(transcript_segments), dtype=np.float32)

    if args.video:
        print("\n5/6 EXTRACTING PRECISE VIDEO VISUAL EVIDENCE")
        video_embeddings, video_timestamps, duration = create_video_frame_embeddings(
            args.video,
            device=DEVICE,
            sample_every_seconds=max(0.25, args.video_sample_seconds),
            batch_size=8,
        )
        video_scores, video_change = build_precise_video_scores(
            video_embeddings,
            video_timestamps,
            image_embeddings,
            [x["start"] for x in transcript_segments],
            [x["end"] for x in transcript_segments],
            radius_seconds=max(0.5, args.video_sample_seconds * 0.75),
        )
        model_z = _row_zscore(adjusted_scores)
        video_z = _row_zscore(video_scores)
        w = float(np.clip(args.video_weight, 0.0, 1.0))
        final_score_matrix = ((1.0 - w) * model_z + w * video_z).astype(np.float32)
        np.save(Path(args.out).parent / "v4_final_video_score_matrix.npy", video_scores)
        np.save(Path(args.out).parent / "v4_final_video_change.npy", video_change)
        print(f"Video frames: {len(video_embeddings)}")
        print(f"Video duration: {duration:.1f}s")
    else:
        print("\n5/6 NO VIDEO PROVIDED — score-distribution change signal used")

    np.save(Path(args.out).parent / "v4_final_base_score_matrix.npy", base_scores)
    np.save(Path(args.out).parent / "v4_final_adjusted_score_matrix.npy", adjusted_scores)
    np.save(Path(args.out).parent / "v4_final_boundary_probability.npy", boundary_prob)
    np.save(Path(args.out).parent / "v4_final_next_slide_logits.npy", next_logits)
    np.save(Path(args.out).parent / "v4_final_score_matrix.npy", final_score_matrix)

    print("\n6/6 RUNNING FINAL HYBRID SEMI-MARKOV DECODER")
    starts = [x["start"] for x in transcript_segments]
    ends = [x["end"] for x in transcript_segments]
    alignment = boundary_aware_dp_final(
        final_score_matrix,
        boundary_prob,
        next_logits,
        starts,
        ends,
        visual_change=video_change,
        **decoder_cfg,
    )
    timeline = base.make_timeline(transcript_segments, slide_paths, alignment)

    metadata = {
        "model": "LectureSenseV4_Final_HybridVisualTemporal",
        "base_model": "LectureSenseCrossModal",
        "checkpoint": str(Path(args.checkpoint)),
        "video_enabled": bool(args.video),
        "video_weight": float(args.video_weight) if args.video else 0.0,
        "video_sample_seconds": float(args.video_sample_seconds) if args.video else None,
        "decoder": decoder_cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
    }
    base.save_result(args.out, timeline, metadata)
    print(f"\nV4 FINAL INFERENCE COMPLETE\nTimeline: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
