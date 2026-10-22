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

import inference_temporal_v4 as v4
import inference as base
from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)
from candidate_semi_markov import semi_markov_decode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOVE_VALUES = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)


class CandidateTemporalModel(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int, hidden: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.context_proj = nn.Sequential(nn.Linear(context_dim, 96), nn.GELU(), nn.Dropout(0.10))
        self.gru = nn.GRU(
            96,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        dim = hidden * 2
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.10, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.10), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(dim + candidate_dim, 160), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.boundary = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.movement = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 5))

    def forward(self, context: torch.Tensor, candidates: torch.Tensor):
        z = self.context_proj(context)
        h, _ = self.gru(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))
        b, t, k, _ = candidates.shape
        h_exp = h.unsqueeze(2).expand(-1, -1, k, -1)
        logits = self.candidate_mlp(torch.cat([h_exp, candidates], dim=-1)).squeeze(-1)
        boundary = self.boundary(h).squeeze(-1)
        movement = self.movement(h)
        return logits, boundary, movement


def load_b_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Candidate B checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    model = CandidateTemporalModel(
        int(ckpt.get("context_dim", 15)),
        int(ckpt.get("candidate_dim", 11)),
        int(ckpt.get("hidden", 96)),
        int(ckpt.get("layers", 2)),
        int(ckpt.get("heads", 4)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    cfg = ckpt.get("decoder_config", {})
    print(f"Candidate B checkpoint: {path}")
    print(f"Validation DP accuracy: {ckpt.get('validation_dp_accuracy', 'n/a')}")
    return model, ckpt, cfg


def zscore(a: np.ndarray) -> np.ndarray:
    mu = np.mean(a, axis=-1, keepdims=True)
    sd = np.std(a, axis=-1, keepdims=True)
    return (a - mu) / np.maximum(sd, 1e-5)


def row_softmax(a: np.ndarray) -> np.ndarray:
    x = a - np.max(a, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -30, 30))
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-8)


def candidate_features(base_scores, adjusted, boundary, next_logits, topk=8, radius=2, max_candidates=48):
    base_scores = np.asarray(base_scores, dtype=np.float32)
    adjusted = np.asarray(adjusted, dtype=np.float32)
    n, s = base_scores.shape
    bp = row_softmax(base_scores)
    ap = row_softmax(adjusted)
    np_ = row_softmax(np.asarray(next_logits, dtype=np.float32))
    bz = zscore(base_scores)
    az = zscore(adjusted)
    base_order = np.argsort(base_scores, axis=1)[:, ::-1]
    adj_order = np.argsort(adjusted, axis=1)[:, ::-1]
    bt = base_order[:, 0]
    at = adj_order[:, 0]

    duration_like = np.zeros(n, dtype=np.float32)
    change = np.zeros(n, dtype=np.float32)
    if n > 1:
        duration_like[1:] = 1.0
        p = ap[:-1] / np.maximum(np.linalg.norm(ap[:-1], axis=1, keepdims=True), 1e-8)
        c = ap[1:] / np.maximum(np.linalg.norm(ap[1:], axis=1, keepdims=True), 1e-8)
        change[1:] = np.clip(1.0 - np.sum(p * c, axis=1), 0, 1)

    base_entropy = -np.sum(bp * np.log(np.clip(bp, 1e-8, 1)), axis=1) / max(1e-8, np.log(max(2, s)))
    adj_entropy = -np.sum(ap * np.log(np.clip(ap, 1e-8, 1)), axis=1) / max(1e-8, np.log(max(2, s)))
    b = np.asarray(boundary, dtype=np.float32)
    context = np.stack([
        np.linspace(0, 1, n, dtype=np.float32),
        duration_like,
        bp[np.arange(n), bt],
        bp[np.arange(n), base_order[:, 1] if s > 1 else bt] * 0 + bp[np.arange(n), bt] - bp[np.arange(n), base_order[:, 1] if s > 1 else bt],
        base_entropy,
        ap[np.arange(n), at],
        ap[np.arange(n), adj_order[:, 1] if s > 1 else at] * 0 + ap[np.arange(n), at] - ap[np.arange(n), adj_order[:, 1] if s > 1 else at],
        adj_entropy,
        b,
        change,
        (bt == at).astype(np.float32),
        np.clip((at - bt) / max(1, s - 1), -1, 1),
        bt / max(1, s - 1),
        at / max(1, s - 1),
        np_[np.arange(n), at],
    ], axis=1).astype(np.float32)

    ranks_b = np.empty_like(base_order)
    ranks_b[np.arange(n)[:, None], base_order] = np.arange(s)[None, :]
    ranks_a = np.empty_like(adj_order)
    ranks_a[np.arange(n)[:, None], adj_order] = np.arange(s)[None, :]

    ids = np.full((n, max_candidates), -1, dtype=np.int64)
    feats = np.zeros((n, max_candidates, 11), dtype=np.float32)
    mask = np.zeros((n, max_candidates), dtype=np.float32)

    for i in range(n):
        pool = {0}
        seeds = list(base_order[i, :min(topk, s)]) + list(adj_order[i, :min(topk, s)])
        for c0 in seeds:
            lo = max(0, int(c0) - radius)
            hi = min(s - 1, int(c0) + radius)
            pool.update(range(lo, hi + 1))
        ordered = sorted(pool, key=lambda sid: (-float(az[i, sid]), sid))[:max_candidates]
        anchor = int(at[i])
        for j, sid in enumerate(ordered):
            ids[i, j] = sid
            feats[i, j] = np.asarray([
                bz[i, sid], az[i, sid], bp[i, sid], ap[i, sid], np_[i, sid],
                ranks_b[i, sid] / max(1, s - 1), ranks_a[i, sid] / max(1, s - 1),
                (sid - anchor) / max(1, s - 1), abs(sid - anchor) / max(1, s - 1),
                1.0 if sid == anchor else 0.0, sid / max(1, s - 1)
            ], dtype=np.float32)
            mask[i, j] = 1.0
    return context, ids, feats, mask


def run_b_alignment(model, base_scores, adjusted, boundary, next_logits, timestamps, cfg):
    context, ids, cand, mask = candidate_features(
        base_scores, adjusted, boundary, next_logits,
        topk=int(cfg.get("candidate_topk", 8)),
        radius=int(cfg.get("candidate_radius", 2)),
        max_candidates=int(cfg.get("max_candidates", 48)),
    )
    with torch.no_grad():
        logits, b_logit, movement = model(
            torch.from_numpy(context).unsqueeze(0).to(DEVICE),
            torch.from_numpy(cand).unsqueeze(0).to(DEVICE),
        )
    logits = logits[0].cpu().numpy()
    bprob = torch.sigmoid(b_logit[0]).cpu().numpy()
    movement = movement[0].cpu().numpy()

    n, s = adjusted.shape
    emission = np.full((n, s), -1e9, dtype=np.float64)
    az = zscore(adjusted)
    bz = zscore(base_scores)
    for i in range(n):
        valid = mask[i] > 0
        cids = ids[i, valid]
        cl = logits[i, valid]
        if len(cl) > 1:
            cl = (cl - cl.mean()) / max(1e-5, cl.std())
        for j, sid in enumerate(cids):
            emission[i, int(sid)] = (
                float(cfg.get("candidate_weight", 1.0)) * float(cl[j])
                + float(cfg.get("v4_weight", 0.8)) * float(az[i, sid])
                + float(cfg.get("base_weight", 0.0)) * float(bz[i, sid])
            )

    m = movement - movement.max(axis=1, keepdims=True)
    m = m - np.log(np.maximum(np.exp(m).sum(axis=1, keepdims=True), 1e-12))
    starts = np.asarray(timestamps, dtype=np.float32)
    ends = np.r_[starts[1:], starts[-1] + 8.0]
    path = semi_markov_decode(
        emission,
        bprob,
        m,
        starts,
        ends,
        max_forward_jump=int(cfg.get("max_forward_jump", 1)),
        jump_penalty=float(cfg.get("jump_penalty", 0.0)),
        boundary_reward=float(cfg.get("boundary_reward", 0.25)),
        boundary_stay_penalty=float(cfg.get("boundary_stay_penalty", 0.15)),
        movement_weight=float(cfg.get("movement_weight", 0.30)),
        duration_penalty=float(cfg.get("duration_penalty", 0.001)),
        max_duration_seconds=float(cfg.get("max_duration_seconds", 150.0)),
        max_run_seconds=float(cfg.get("max_run_seconds", 1200.0)),
    )
    return np.asarray(path, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="LectureSense Experiment B candidate-level inference")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--slides", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--v4-checkpoint", default=str(ROOT / "checkpoints" / "temporal_v4_best_model.pt"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "temporal_b_candidate_best.pt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--score-batch", type=int, default=4)
    parser.add_argument("--video-sample-seconds", type=float, default=1.0)
    parser.add_argument("--video-weight", type=float, default=0.4)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    b_model, b_ckpt, cfg = load_b_checkpoint(Path(args.checkpoint))
    v4_model, v4_base_model, max_slides, temperature, _ = v4.load_v4_checkpoint(Path(args.v4_checkpoint))
    v4_model.eval(); v4_base_model.eval()

    print("\n1/7 READING TRANSCRIPT")
    transcript_segments = base.read_mavils_srt(args.transcript)
    if not transcript_segments:
        raise RuntimeError("No transcript segments found")

    print("\n2/7 RENDERING SLIDES")
    slide_paths = base.render_pdf(args.slides, args.render_dir)
    if not slide_paths:
        raise RuntimeError("No slides rendered")

    print("\n3/7 EXTRACTING OCR")
    ocr_texts = base.extract_ocr(slide_paths)
    if len(ocr_texts) != len(slide_paths):
        raise RuntimeError("OCR count does not match slide count")

    print("\n4/7 BUILDING V4 MULTIMODAL SCORES")
    sentence_model = base.load_sbert()
    transcript_embeddings = base.create_text_embeddings(sentence_model, [x["text"] for x in transcript_segments])
    ocr_embeddings = base.create_text_embeddings(sentence_model, ocr_texts)
    del sentence_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image_embeddings = base.create_image_embeddings(slide_paths)
    base_scores = base.compute_score_matrix(
        v4_base_model,
        transcript_embeddings,
        ocr_embeddings,
        image_embeddings,
        batch_size=max(1, args.score_batch),
    )

    print("\n5/7 RUNNING FROZEN V4 TEMPORAL MODEL")
    adjusted, boundary, next_logits = v4.temporal_adjust_scores(
        base_scores,
        transcript_segments,
        v4_model,
        max_slides,
        temperature,
    )
    final_scores = adjusted

    video_used = False
    if args.video:
        print("\nExtracting actual lecture video evidence...")
        video_embeddings, video_timestamps, _ = create_video_frame_embeddings(
            args.video,
            device=DEVICE,
            sample_every_seconds=max(0.25, args.video_sample_seconds),
            batch_size=8,
        )
        video_scores = build_video_slide_score_matrix(
            video_embeddings,
            video_timestamps,
            image_embeddings,
            segment_starts=[x["start"] for x in transcript_segments],
            segment_ends=[x["end"] for x in transcript_segments],
        )
        final_scores = fuse_model_and_video_scores(
            adjusted,
            video_scores,
            video_weight=args.video_weight,
        )
        np.save(Path(args.out).parent / "v4_video_score_matrix.npy", video_scores)
        video_used = True

    print("\n6/7 CANDIDATE-LEVEL TEMPORAL FUSION + SEMI-MARKOV DP")
    path = run_b_alignment(
        b_model,
        base_scores,
        final_scores,
        boundary,
        next_logits,
        np.asarray([x["start"] for x in transcript_segments], dtype=np.float32),
        cfg,
    )

    if len(path) != len(transcript_segments):
        raise RuntimeError("Decoded path length does not match transcript")

    print("\n7/7 SAVING TIMELINE")
    timeline = base.make_timeline(transcript_segments, slide_paths, path.tolist())
    metadata = {
        "project": "LectureSense",
        "model": "LectureSenseCandidateTemporalB",
        "architecture": {
            "frozen_v4_backbone": True,
            "candidate_level_reranker": True,
            "temporal_backbone": "BiGRU + self-attention",
            "heads": ["current_slide", "boundary", "relative_slide_movement"],
            "movement_classes": [-2, -1, 0, 1, 2],
            "decoder": "monotonic semi-Markov DP",
        },
        "checkpoint": str(Path(args.checkpoint)),
        "v4_checkpoint": str(Path(args.v4_checkpoint)),
        "video_enabled": video_used,
        "video_weight": args.video_weight if video_used else 0.0,
        "decoder_config": cfg,
        "num_transcript_segments": len(transcript_segments),
        "num_slides": len(slide_paths),
        "timeline_entries": len(timeline),
        "test_labels_used": False,
    }
    base.save_result(args.out, timeline, metadata)
    np.save(Path(args.out).parent / "b_base_score_matrix.npy", base_scores)
    np.save(Path(args.out).parent / "b_v4_adjusted_score_matrix.npy", adjusted)
    np.save(Path(args.out).parent / "b_final_score_matrix.npy", final_scores)
    np.save(Path(args.out).parent / "b_v4_boundary_probability.npy", boundary)
    np.save(Path(args.out).parent / "b_v4_next_slide_logits.npy", next_logits)

    print("\nEXPERIMENT B INFERENCE COMPLETE")
    print(f"Timeline: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
