from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

V3_PATH = SRC / "train_temporal_v3.py"

spec = importlib.util.spec_from_file_location("lecturesense_src_v3", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load V3 helper module: {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


import inference_temporal_v4 as v4
from candidate_semi_markov_fixed import semi_markov_decode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SLIDES = 96
MOVE_VALUES = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)

# Fixed V4 anchor checkpoint. Never overwritten by this experiment.
V4_CHECKPOINT = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
OUT_DIR = ROOT / "checkpoints"
OUT_CHECKPOINT = OUT_DIR / "temporal_b_candidate_fixed_best.pt"
CACHE_DIR = ROOT / "runs" / "temporal_b_cache_fixed"
RESULT_DIR = ROOT / "runs" / "temporal_b_fixed"


class CandidateTemporalModel(nn.Module):
    """Candidate-level temporal reranker.

    The temporal backbone sees lecture context, not a global 96-way class
    vector. A separate candidate scorer compares only locally plausible
    slides. Auxiliary heads predict boundary probability and relative slide
    movement {-2,-1,0,+1,+2}.
    """

    def __init__(self, context_dim: int, candidate_dim: int, hidden: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.GELU(),
            nn.Dropout(0.10),
        )
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
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

        self.candidate_mlp = nn.Sequential(
            nn.Linear(dim + candidate_dim, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.boundary = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 1)
        )
        self.movement = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, 5)
        )

    def forward(self, context: torch.Tensor, candidates: torch.Tensor):
        # context: [B,T,C]
        # candidates: [B,T,K,F]
        z = self.context_proj(context)
        h, _ = self.gru(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))

        b, t, k, f = candidates.shape
        h_exp = h.unsqueeze(2).expand(-1, -1, k, -1)
        cand_in = torch.cat([h_exp, candidates], dim=-1)
        cand_logits = self.candidate_mlp(cand_in).squeeze(-1)
        boundary_logit = self.boundary(h).squeeze(-1)
        movement_logits = self.movement(h)
        return cand_logits, boundary_logit, movement_logits


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    mu = np.mean(a, axis=-1, keepdims=True)
    sd = np.std(a, axis=-1, keepdims=True)
    return (a - mu) / np.maximum(sd, 1e-5)


def row_softmax(a: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = a / max(1e-5, float(temp))
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -30, 30))
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-8)


def make_context_and_candidate_features(item: dict, candidate_topk: int = 8, radius: int = 2, max_candidates: int = 48):
    base = np.asarray(item["scores"], dtype=np.float32)
    adj = np.asarray(item["v4_adjusted"], dtype=np.float32)
    b = np.asarray(item["boundary"], dtype=np.float32)
    nxt = np.asarray(item["next_logits"], dtype=np.float32)
    times = np.asarray(item["timestamps"], dtype=np.float32)
    n, s = base.shape

    bp = row_softmax(base)
    ap = row_softmax(adj)
    np_ = row_softmax(nxt)
    bz = zscore(base)
    az = zscore(adj)
    nz = zscore(nxt)

    base_order = np.argsort(base, axis=1)[:, ::-1]
    adj_order = np.argsort(adj, axis=1)[:, ::-1]
    base_top1 = base_order[:, 0]
    adj_top1 = adj_order[:, 0]
    base_top2 = base_order[:, 1] if s > 1 else base_order[:, 0]
    adj_top2 = adj_order[:, 1] if s > 1 else adj_order[:, 0]

    base_p1 = bp[np.arange(n), base_top1]
    base_p2 = bp[np.arange(n), base_top2]
    adj_p1 = ap[np.arange(n), adj_top1]
    adj_p2 = ap[np.arange(n), adj_top2]

    base_entropy = -np.sum(bp * np.log(np.clip(bp, 1e-8, 1.0)), axis=1) / max(1e-8, math.log(max(2, s)))
    adj_entropy = -np.sum(ap * np.log(np.clip(ap, 1e-8, 1.0)), axis=1) / max(1e-8, math.log(max(2, s)))

    time_norm = (times - times[0]) / max(1.0, float(times[-1] - times[0]))
    duration = np.zeros(n, dtype=np.float32)
    if n > 1:
        duration[:-1] = np.clip(np.diff(times), 0, None)
        duration[-1] = duration[-2]
    duration_norm = np.clip(duration / 30.0, 0, 5)

    score_change = np.zeros(n, dtype=np.float32)
    if n > 1:
        prev = ap[:-1] / np.maximum(np.linalg.norm(ap[:-1], axis=1, keepdims=True), 1e-8)
        cur = ap[1:] / np.maximum(np.linalg.norm(ap[1:], axis=1, keepdims=True), 1e-8)
        score_change[1:] = np.clip(1.0 - np.sum(prev * cur, axis=1), 0, 1)

    agreement = (base_top1 == adj_top1).astype(np.float32)
    top_delta = np.clip((adj_top1 - base_top1) / max(1, s - 1), -1, 1).astype(np.float32)

    context = np.stack(
        [
            time_norm,
            duration_norm,
            base_p1,
            base_p1 - base_p2,
            base_entropy,
            adj_p1,
            adj_p1 - adj_p2,
            adj_entropy,
            b,
            score_change,
            agreement,
            top_delta,
            base_top1 / max(1, s - 1),
            adj_top1 / max(1, s - 1),
            np_[np.arange(n), adj_top1],
        ],
        axis=1,
    ).astype(np.float32)

    cand_ids = np.full((n, max_candidates), -1, dtype=np.int64)
    cand_feats = np.zeros((n, max_candidates, 11), dtype=np.float32)
    cand_mask = np.zeros((n, max_candidates), dtype=np.float32)

    ranks_base = np.empty_like(base_order)
    ranks_base[np.arange(n)[:, None], base_order] = np.arange(s)[None, :]
    ranks_adj = np.empty_like(adj_order)
    ranks_adj[np.arange(n)[:, None], adj_order] = np.arange(s)[None, :]

    for i in range(n):
        pool = set()
        seeds = list(base_order[i, :min(candidate_topk, s)]) + list(adj_order[i, :min(candidate_topk, s)])
        for center in seeds:
            lo = max(0, int(center) - radius)
            hi = min(s - 1, int(center) + radius)
            for sid in range(lo, hi + 1):
                pool.add(sid)
        # Always include slide 0 because the monotonic decoder must start there.
        pool.add(0)
        # Keep candidates ordered by fused V4 confidence, then slide id.
        ordered = sorted(
            pool,
            key=lambda sid: (-float(az[i, sid]) - 0.25 * float(bz[i, sid]), sid),
        )[:max_candidates]

        for j, sid in enumerate(ordered):
            cand_ids[i, j] = sid
            anchor = int(adj_top1[i])
            cand_feats[i, j] = np.asarray(
                [
                    bz[i, sid],
                    az[i, sid],
                    bp[i, sid],
                    ap[i, sid],
                    np_[i, sid],
                    ranks_base[i, sid] / max(1, s - 1),
                    ranks_adj[i, sid] / max(1, s - 1),
                    (sid - anchor) / max(1, s - 1),
                    abs(sid - anchor) / max(1, s - 1),
                    1.0 if sid == anchor else 0.0,
                    sid / max(1, s - 1),
                ],
                dtype=np.float32,
            )
            cand_mask[i, j] = 1.0

    labels = np.asarray(item["labels"], dtype=np.int64)
    target_pos = np.full(n, -1, dtype=np.int64)
    candidate_recall = 0
    for i in range(n):
        where = np.where(cand_ids[i] == labels[i])[0]
        if len(where):
            target_pos[i] = int(where[0])
            candidate_recall += 1

    movement_target = np.zeros(n, dtype=np.int64)
    if n > 1:
        movement_target[1:] = np.clip(labels[1:] - labels[:-1], -2, 2) + 2
    movement_target[0] = 2

    return {
        "context": context,
        "cand_ids": cand_ids,
        "cand_feats": cand_feats,
        "cand_mask": cand_mask,
        "target_pos": target_pos,
        "movement_target": movement_target,
        "candidate_recall": candidate_recall / max(1, n),
    }


def attach_v4_outputs(item: dict, temporal_model, max_slides: int, temperature: float):
    segs = [{"start": float(t)} for t in item["timestamps"]]
    adjusted, boundary, next_logits = v4.temporal_adjust_scores(
        np.asarray(item["scores"], dtype=np.float32),
        segs,
        temporal_model,
        max_slides,
        temperature,
    )
    out = dict(item)
    out["v4_adjusted"] = adjusted.astype(np.float32)
    out["boundary"] = boundary.astype(np.float32)
    out["next_logits"] = next_logits.astype(np.float32)
    return out


def _canonical_lecture_name(name: str) -> str:
    value = "".join(ch.lower() for ch in str(name) if ch.isalnum())
    aliases = {
        "physicsintro02": "physics",
        "psychologymit": "psychology",
        "reinforcementlearningsilver": "reinforcementlearning",
        "visualsystemmit": "sensorysystems",
        "shortrangemit": "shortrange",
    }
    return aliases.get(value, value)


EXPECTED_SPLIT_LECTURES = {
    "val": {"physics", "psychology", "reinforcementlearning", "sensorysystems"},
    "validation": {"physics", "psychology", "reinforcementlearning", "sensorysystems"},
    "test": {"shortrange", "solarresource", "teamdynamicsgamedesignmit", "theoryofcomputation"},
}


def require_complete_split(lectures: dict, split_name: str) -> None:
    """Fail loudly instead of reporting a partial-split score as the benchmark."""
    expected = EXPECTED_SPLIT_LECTURES.get(str(split_name).lower())
    if not expected:
        return
    loaded = {_canonical_lecture_name(name) for name in lectures}
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(
            f"Incomplete {split_name.upper()} split. Missing expected lectures: {missing}. "
            f"Loaded: {sorted(loaded)}. Do not treat a partial-split score as the full benchmark. "
            "Repair the corresponding MaViLS slide inventory/embeddings before evaluating."
        )


def load_lecture_items(split_name: str, cache_prefix: str, score_batch: int, temporal_model, max_slides: int, temperature: float):
    pairs = v3.normalize_pairs(v3.load_csv(v3.PAIRS))
    split_values = [split_name] if split_name != "val" else ["val", "validation"]
    df = pairs[pairs["split"].isin(split_values)].copy()

    text_embeddings = np.load(v3.TEXT_EMB).astype(np.float32)
    ocr_embeddings = np.load(v3.OCR_EMB).astype(np.float32)
    image_embeddings = np.load(v3.IMAGE_EMB).astype(np.float32)
    text_index = v3.load_csv(v3.TEXT_INDEX)
    ocr_index = v3.load_csv(v3.OCR_INDEX)
    image_index = v3.load_csv(v3.IMAGE_INDEX)

    text_lookup = v3.make_text_lookup(text_index, len(text_embeddings))
    ocr_lookup = v3.make_slide_lookup(ocr_index, "embedding_row", len(ocr_embeddings))
    image_lookup = v3.make_slide_lookup(image_index, "embedding_idx", len(image_embeddings))
    slides_by_lecture = v3.build_slide_inventory(image_index, ocr_lookup, image_lookup)
    sequences = v3.build_positive_sequences(df)

    base_model = v3.load_base_model(DEVICE)
    base_model.eval()

    out = {}
    for lecture, rows in sequences.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            print(f"[SKIP] {lecture}: no slide inventory")
            continue
        cache = CACHE_DIR / f"{cache_prefix}_{lecture}.npz"
        item = v3.load_cached_or_score(
            cache,
            lecture,
            rows,
            slides,
            text_lookup,
            ocr_lookup,
            image_lookup,
            text_embeddings,
            ocr_embeddings,
            image_embeddings,
            base_model,
            DEVICE,
            score_batch,
        )
        item = attach_v4_outputs(item, temporal_model, max_slides, temperature)
        out[lecture] = item
        print(f"[READY] {lecture}: segments={len(item['labels'])} slides={item['scores'].shape[1]}")
    return out


class WindowDataset(Dataset):
    def __init__(self, lectures: dict, window: int = 64, stride: int = 32):
        self.examples = []
        self.window = window
        self.stride = stride
        self.lectures = lectures

        for lecture, item in lectures.items():
            feats = make_context_and_candidate_features(item)
            n = len(item["labels"])

            starts = list(range(0, max(1, n - window + 1), stride))
            last = max(0, n - window)

            if not starts or starts[-1] != last:
                starts.append(last)

            for st in starts:
                en = min(st + window, n)
                length = en - st

                self.examples.append(
                    (
                        lecture,
                        st,
                        en,
                        feats,
                        length,
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        lecture, st, en, feats, length = self.examples[idx]

        cdim = feats["context"].shape[1]
        k = feats["cand_feats"].shape[1]
        f = feats["cand_feats"].shape[2]

        context = np.zeros(
            (self.window, cdim),
            dtype=np.float32,
        )

        cand_feats = np.zeros(
            (self.window, k, f),
            dtype=np.float32,
        )

        cand_mask = np.zeros(
            (self.window, k),
            dtype=np.float32,
        )

        target = np.zeros(
            self.window,
            dtype=np.int64,
        )

        movement = np.zeros(
            self.window,
            dtype=np.int64,
        )

        boundary = np.zeros(
            self.window,
            dtype=np.float32,
        )

        mask = np.zeros(
            self.window,
            dtype=np.float32,
        )

        context[:length] = feats["context"][st:en]
        cand_feats[:length] = feats["cand_feats"][st:en]
        cand_mask[:length] = feats["cand_mask"][st:en]

        # Target is candidate position, not raw slide ID.
        target[:length] = feats["target_pos"][st:en]

        # Movement target is already encoded as classes 0..4
        # inside make_context_and_candidate_features().
        movement[:length] = feats["movement_target"][st:en]

        # Boundary target = 1 when the ground-truth slide changes.
        labels = np.asarray(
            self.lectures[lecture]["labels"],
            dtype=np.int64,
        )

        boundary_values = np.zeros(
            len(labels),
            dtype=np.float32,
        )

        if len(labels) > 1:
            boundary_values[1:] = (
                labels[1:] != labels[:-1]
            ).astype(np.float32)

        boundary[:length] = boundary_values[st:en]

        mask[:length] = 1.0

        return (
            torch.from_numpy(context),
            torch.from_numpy(cand_feats),
            torch.from_numpy(cand_mask),
            torch.from_numpy(target),
            torch.from_numpy(movement),
            torch.from_numpy(boundary),
            torch.from_numpy(mask),
        )


def batch_metrics(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for context, cand, mask, target_pos, movement, boundary, valid in loader:
            context = context.to(DEVICE)
            cand = cand.to(DEVICE)
            logits, _, _ = model(context, cand)
            logits = logits.masked_fill(mask.to(DEVICE) <= 0, -1e9)
            pred = logits.argmax(dim=-1).cpu()
            usable = (valid > 0) & (target_pos >= 0)
            correct += int(((pred == target_pos) & usable).sum())
            total += int(usable.sum())
    return correct / max(1, total)


def lecture_candidates(model, item, *, candidate_topk=8, radius=2, max_candidates=48):
    feats = make_context_and_candidate_features(item, candidate_topk, radius, max_candidates)
    context = torch.from_numpy(feats["context"]).unsqueeze(0).to(DEVICE)
    cand = torch.from_numpy(feats["cand_feats"]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, boundary_logit, movement_logits = model(context, cand)
    return feats, logits[0].cpu().numpy(), torch.sigmoid(boundary_logit[0]).cpu().numpy(), movement_logits[0].cpu().numpy()


def decode_lecture(model, item, cfg):
    """Run the exact same score construction used during validation tuning.

    The old evaluator gave every non-shortlisted slide an emission of -1e9,
    whereas the training sweep kept a finite V4 fallback score for those
    slides. That changed the decoder between model selection and evaluation.
    This adapter routes both paths through decode_evidence to keep them equal.
    """
    feats, cand_logits, boundary_prob, movement_logits = lecture_candidates(
        model,
        item,
        candidate_topk=int(cfg.get("candidate_topk", 8)),
        radius=int(cfg.get("candidate_radius", 2)),
        max_candidates=int(cfg.get("max_candidates", 48)),
    )
    evidence = {
        "item": item,
        "cand_ids": feats["cand_ids"],
        "cand_mask": feats["cand_mask"],
        "candidate_recall": feats["candidate_recall"],
        "cand_logits": cand_logits,
        "boundary_prob": boundary_prob,
        "movement_logits": movement_logits,
    }
    path = decode_evidence(evidence, cfg)
    return np.asarray(path, dtype=np.int64), float(feats["candidate_recall"])


def evaluate_decode(model, lectures, cfg, verbose=True):
    totals = []
    recalls = []
    per = {}
    for lecture, item in lectures.items():
        # Validation labels are used only for evaluation / config selection.
        path, recall = decode_lecture(model, item, cfg)
        labels = np.asarray(item["labels"], dtype=np.int64)
        correct = int(np.sum(path == labels))
        total = len(labels)
        acc = correct / max(1, total)
        per[lecture] = {"correct": correct, "total": total, "accuracy": acc, "candidate_recall": recall}
        totals.append((correct, total))
        recalls.append(recall)
        if verbose:
            print(f"  {lecture:30s} {correct:4d}/{total:4d}  {acc*100:6.2f}%  cand_recall={recall*100:6.2f}%")
    macro = float(np.mean([x[1]["accuracy"] for x in per.items()])) if per else 0.0
    weighted = sum(c for c, _ in totals) / max(1, sum(t for _, t in totals))
    return macro, weighted, per, float(np.mean(recalls) if recalls else 0.0)


def build_decoder_candidates():
    candidates = []
    for candidate_weight in [0.8, 1.0, 1.3]:
        for v4_weight in [0.6, 0.9]:
            for base_weight in [0.0, 0.10]:
                for movement_weight in [0.20, 0.35]:
                    candidates.append(
                        {
                            "candidate_weight": candidate_weight,
                            "v4_weight": v4_weight,
                            "base_weight": base_weight,
                            "fallback_weight": 0.20,
                            "movement_weight": movement_weight,
                            "max_forward_jump": 1,
                            "jump_penalty": 0.0,
                            "boundary_reward": 0.25,
                            "boundary_stay_penalty": 0.15,
                            "duration_penalty": 0.001,
                            "max_duration_seconds": 150.0,
                            "max_run_seconds": 1200.0,
                            "candidate_topk": 8,
                            "candidate_radius": 2,
                            "max_candidates": 48,
                        }
                    )
    return candidates


def precompute_decode_evidence(model, lectures, cfg):
    evidence = {}
    for lecture, item in lectures.items():
        feats = make_context_and_candidate_features(
            item,
            candidate_topk=int(cfg.get("candidate_topk", 8)),
            radius=int(cfg.get("candidate_radius", 2)),
            max_candidates=int(cfg.get("max_candidates", 48)),
        )
        context = torch.from_numpy(feats["context"]).unsqueeze(0).to(DEVICE)
        cand = torch.from_numpy(feats["cand_feats"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, b_logit, movement = model(context, cand)
        evidence[lecture] = {
            "item": item,
            "cand_ids": feats["cand_ids"],
            "cand_mask": feats["cand_mask"],
            "candidate_recall": feats["candidate_recall"],
            "cand_logits": logits[0].cpu().numpy(),
            "boundary_prob": torch.sigmoid(b_logit[0]).cpu().numpy(),
            "movement_logits": movement[0].cpu().numpy(),
        }
    return evidence


def decode_evidence(ev, cfg):
    item = ev["item"]

    base = np.asarray(
        item["scores"],
        dtype=np.float32,
    )

    adj = np.asarray(
        item["v4_adjusted"],
        dtype=np.float32,
    )

    n, s = adj.shape

    az = zscore(adj)
    bz = zscore(base)

    # --------------------------------------------------------------
    # Dense finite fallback.
    #
    # Every slide remains legal for the monotonic decoder.
    # Candidate-level model gets a strong advantage only for
    # locally selected candidates.
    # --------------------------------------------------------------

    fallback_weight = float(
        cfg.get("fallback_weight", 0.20)
    )

    emission = (
        fallback_weight * az
        + float(cfg.get("base_weight", 0.0)) * bz
    ).astype(np.float64)

    cand_mask = ev["cand_mask"] > 0
    cand_ids_all = ev["cand_ids"]
    cand_logits_all = ev["cand_logits"]

    candidate_weight = float(
        cfg.get("candidate_weight", 1.0)
    )

    v4_weight = float(
        cfg.get("v4_weight", 0.8)
    )

    for i in range(n):

        valid = cand_mask[i]

        ids = cand_ids_all[
            i,
            valid,
        ]

        cl = cand_logits_all[
            i,
            valid,
        ].astype(np.float64)

        if len(cl) > 1:
            cl = (
                cl - cl.mean()
            ) / max(
                1e-5,
                cl.std(),
            )

        for j, sid in enumerate(ids):

            sid = int(sid)

            emission[i, sid] = (
                candidate_weight
                * float(cl[j])
                + v4_weight
                * float(az[i, sid])
                + float(
                    cfg.get(
                        "base_weight",
                        0.0,
                    )
                )
                * float(bz[i, sid])
            )

    movement_logits = np.asarray(
        ev["movement_logits"],
        dtype=np.float64,
    )

    movement_logits = (
        movement_logits
        - movement_logits.max(
            axis=1,
            keepdims=True,
        )
    )

    movement_logits = (
        movement_logits
        - np.log(
            np.maximum(
                np.exp(movement_logits).sum(
                    axis=1,
                    keepdims=True,
                ),
                1e-12,
            )
        )
    )

    starts = np.asarray(
        item["timestamps"],
        dtype=np.float32,
    )

    ends = np.r_[
        starts[1:],
        starts[-1] + 8.0,
    ]

    path = semi_markov_decode(
        emission,
        ev["boundary_prob"],
        movement_logits,
        starts,
        ends,

        max_forward_jump=int(
            cfg.get(
                "max_forward_jump",
                1,
            )
        ),

        jump_penalty=float(
            cfg.get(
                "jump_penalty",
                0.0,
            )
        ),

        boundary_reward=float(
            cfg.get(
                "boundary_reward",
                0.25,
            )
        ),

        boundary_stay_penalty=float(
            cfg.get(
                "boundary_stay_penalty",
                0.15,
            )
        ),

        movement_weight=float(
            cfg.get(
                "movement_weight",
                0.30,
            )
        ),

        duration_penalty=float(
            cfg.get(
                "duration_penalty",
                0.001,
            )
        ),

        max_duration_seconds=float(
            cfg.get(
                "max_duration_seconds",
                150.0,
            )
        ),

        max_run_seconds=float(
            cfg.get(
                "max_run_seconds",
                1200.0,
            )
        ),
    )

    return (
        np.asarray(
            path,
            dtype=np.int64,
        )
    )


def evaluate_precomputed(evidence, cfg, verbose=False):
    per = {}
    total_correct = 0
    total_rows = 0
    recalls = []
    for lecture, ev in evidence.items():
        path = decode_evidence(ev, cfg)
        labels = np.asarray(ev["item"]["labels"], dtype=np.int64)
        correct = int(np.sum(path == labels))
        total = len(labels)
        acc = correct / max(1, total)
        per[lecture] = {"correct": correct, "total": total, "accuracy": acc, "candidate_recall": ev["candidate_recall"]}
        total_correct += correct
        total_rows += total
        recalls.append(ev["candidate_recall"])
        if verbose:
            print(f"  {lecture:30s} {correct:4d}/{total:4d}  {acc*100:6.2f}%  cand_recall={ev['candidate_recall']*100:6.2f}%")
    macro = float(np.mean([v["accuracy"] for v in per.values()])) if per else 0.0
    weighted = total_correct / max(1, total_rows)
    return macro, weighted, per, float(np.mean(recalls) if recalls else 0.0)

def train_one_seed(train_lectures, val_lectures, args, seed: int):
    seed_everything(seed)
    model = CandidateTemporalModel(
        context_dim=15,
        candidate_dim=11,
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
    ).to(DEVICE)

    ds = WindowDataset(train_lectures, window=args.window, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.boundary_pos_weight, device=DEVICE))

    best_path = None
    best_val = -1.0
    best_state = None
    stale = 0

    train_candidate_recall = np.mean([
        make_context_and_candidate_features(v)["candidate_recall"] for v in train_lectures.values()
    ]) if train_lectures else 0.0
    print(f"Seed {seed}: training windows={len(ds)}, candidate recall={train_candidate_recall*100:.2f}%")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        rank_sum = 0.0
        boundary_sum = 0.0
        movement_sum = 0.0
        steps = 0
        for context, cand, mask, target_pos, movement, boundary, valid in loader:
            context = context.to(DEVICE)
            cand = cand.to(DEVICE)
            mask = mask.to(DEVICE)
            target_pos = target_pos.to(DEVICE)
            movement = movement.to(DEVICE)
            boundary = boundary.to(DEVICE)
            valid = valid.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            cand_logits, b_logit, move_logits = model(context, cand)
            # Work in FP32 and use a finite mask value. More importantly, a row
            # contributes to CE/ranking only if its target is a real candidate.
            cand_logits = cand_logits.float()
            candidate_valid = mask > 0.5
            num_candidates = cand_logits.shape[-1]
            safe_target = target_pos.clamp(min=0, max=num_candidates - 1)
            target_is_valid = candidate_valid.gather(
                2, safe_target.unsqueeze(-1)
            ).squeeze(-1)
            usable = (
                (valid > 0)
                & (target_pos >= 0)
                & (target_pos < num_candidates)
                & target_is_valid
            )
            masked_logits = cand_logits.masked_fill(~candidate_valid, -1e4)

            if usable.any():
                ce = F.cross_entropy(masked_logits[usable], target_pos[usable])
                target_logits = cand_logits.gather(
                    2, safe_target.unsqueeze(-1)
                ).squeeze(-1)
                target_one_hot = F.one_hot(
                    safe_target, num_classes=num_candidates
                ).bool()
                negative_valid = candidate_valid & ~target_one_hot
                has_negative = usable & negative_valid.any(dim=-1)
                if has_negative.any():
                    negative_logits = cand_logits.masked_fill(~negative_valid, -1e4)
                    hard_negative = negative_logits.max(dim=-1).values
                    rank_rows = F.relu(
                        args.rank_margin - target_logits + hard_negative
                    )
                    rank = rank_rows[has_negative].mean()
                else:
                    rank = ce * 0.0
            else:
                ce = cand_logits.sum() * 0.0
                rank = ce

            valid_flat = valid > 0
            b_loss = (
                bce(b_logit.float()[valid_flat], boundary.float()[valid_flat]).mean()
                if valid_flat.any() else ce * 0.0
            )
            move_loss = (
                F.cross_entropy(move_logits.float()[valid_flat], movement[valid_flat])
                if valid_flat.any() else ce * 0.0
            )

            loss = (
                ce
                + args.rank_loss_weight * rank
                + args.boundary_loss_weight * b_loss
                + args.movement_loss_weight * move_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite Candidate B loss. "
                    f"CE={ce.detach().item():.6g}, rank={rank.detach().item():.6g}, "
                    f"boundary={b_loss.detach().item():.6g}, movement={move_loss.detach().item():.6g}."
                )
            if float(loss.detach().item()) > 1e4:
                raise RuntimeError(
                    "Candidate B loss exceeded 10000; refusing unstable optimizer step. "
                    f"CE={ce.detach().item():.6g}, rank={rank.detach().item():.6g}, "
                    f"boundary={b_loss.detach().item():.6g}, movement={move_loss.detach().item():.6g}, "
                    f"usable_targets={int(usable.sum().item())}, "
                    f"valid_frames={int(valid_flat.sum().item())}."
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                raise RuntimeError("Non-finite gradient norm; optimizer step cancelled.")
            opt.step()
            loss_sum += float(loss.item())
            ce_sum += float(ce.item())
            rank_sum += float(rank.item())
            boundary_sum += float(b_loss.item())
            movement_sum += float(move_loss.item())
            steps += 1

        # Model training metric on held-out validation is candidate Top-1.
        val_candidate = evaluate_candidate_top1(model, val_lectures)
        sched.step(val_candidate)
        print(
            f"Seed {seed} Epoch {epoch:02d}/{args.epochs} | "
            f"loss={loss_sum/max(1,steps):.4f} "
            f"(ce={ce_sum/max(1,steps):.3f}, rank={rank_sum/max(1,steps):.3f}, "
            f"boundary={boundary_sum/max(1,steps):.3f}, movement={movement_sum/max(1,steps):.3f}) | "
            f"val candidate Top1={val_candidate:.4f} | lr={opt.param_groups[0]['lr']:.2e}"
        )

        if val_candidate > best_val + 1e-6:
            best_val = val_candidate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Seed {seed}: early stop")
                break

    if best_state is None:
        raise RuntimeError("No model state saved")
    model.load_state_dict(best_state)
    return model, best_val


def evaluate_candidate_top1(model, lectures):
    model.eval()
    all_acc = []
    for item in lectures.values():
        feats = make_context_and_candidate_features(item)
        context = torch.from_numpy(feats["context"]).unsqueeze(0).to(DEVICE)
        cand = torch.from_numpy(feats["cand_feats"]).unsqueeze(0).to(DEVICE)
        mask = torch.from_numpy(feats["cand_mask"]).unsqueeze(0).to(DEVICE)
        target = torch.from_numpy(feats["target_pos"]).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(context, cand)
        logits = logits[0].masked_fill(mask[0] <= 0, -1e9)
        pred = logits.argmax(dim=-1).cpu().numpy()
        usable = target.detach().cpu().numpy() >= 0
        if usable.any():
            target_np = target.detach().cpu().numpy()
            all_acc.extend(
                (pred[usable] == target_np[usable])
                .astype(np.float32)
                .tolist()
            )
    return float(np.mean(all_acc)) if all_acc else 0.0


def save_checkpoint(model, cfg, args, seed, val_candidate, val_macro, val_weighted):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "model_type": "LectureSenseCandidateTemporalB",
        "model_state_dict": model.state_dict(),
        "context_dim": 15,
        "candidate_dim": 11,
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "candidate_topk": cfg["candidate_topk"],
        "candidate_radius": cfg["candidate_radius"],
        "max_candidates": cfg["max_candidates"],
        "v4_checkpoint": str(V4_CHECKPOINT),
        "validation_candidate_top1": val_candidate,
        "validation_dp_macro_accuracy": val_macro,
        "validation_dp_weighted_accuracy": val_weighted,
        "validation_dp_accuracy": val_weighted,
        "decoder_config": cfg,
        "seed": seed,
        "test_used_for_selection": False,
    }
    torch.save(state, OUT_CHECKPOINT)
    config_only = {k: v for k, v in state.items() if k != "model_state_dict"}
    with open(OUT_DIR / "temporal_b_candidate_fixed_config.json", "w", encoding="utf-8") as f:
        json.dump(config_only, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="LectureSense Experiment B: Candidate-Level Temporal Fusion")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="17,41,73")
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.35)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.40)
    parser.add_argument("--movement-loss-weight", type=float, default=0.35)
    parser.add_argument("--boundary-pos-weight", type=float, default=2.5)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not V4_CHECKPOINT.exists():
        raise FileNotFoundError(f"Required V4 checkpoint not found: {V4_CHECKPOINT}")

    print("\nLoading frozen V4 backbone...")
    v4_model, _, max_slides, temperature, _ = v4.load_v4_checkpoint(V4_CHECKPOINT)
    v4_model.eval()

    print("\nLoading TRAIN lectures...")
    train_lectures = load_lecture_items("train", "train", args.score_batch, v4_model, max_slides, temperature)
    print(f"TRAIN lectures loaded: {len(train_lectures)}")

    print("\nLoading VALIDATION lectures...")
    val_lectures = load_lecture_items("val", "val", args.score_batch, v4_model, max_slides, temperature)
    print(f"VALIDATION lectures loaded: {len(val_lectures)}")
    require_complete_split(val_lectures, "val")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required via --seeds")

    # Use a fixed, modest decoder config to choose among seeds by actual decoded
    # validation accuracy (weighted first, macro second), not conditional candidate
    # Top-1 alone. The held-out test split is never loaded here.
    seed_selection_cfg = {
        "candidate_weight": 1.0,
        "v4_weight": 0.8,
        "base_weight": 0.0,
        "fallback_weight": 0.20,
        "movement_weight": 0.30,
        "max_forward_jump": 1,
        "jump_penalty": 0.0,
        "boundary_reward": 0.25,
        "boundary_stay_penalty": 0.15,
        "duration_penalty": 0.001,
        "max_duration_seconds": 150.0,
        "max_run_seconds": 1200.0,
        "candidate_topk": 8,
        "candidate_radius": 2,
        "max_candidates": 48,
    }
    seed_records = []
    for seed in seeds:
        model, val_candidate = train_one_seed(
            train_lectures, val_lectures, args, seed
        )
        val_macro_seed, val_weighted_seed, _, val_recall_seed = evaluate_decode(
            model, val_lectures, seed_selection_cfg, verbose=False
        )
        seed_records.append({
            "seed": seed,
            "model": model,
            "candidate_top1": val_candidate,
            "validation_macro": val_macro_seed,
            "validation_weighted": val_weighted_seed,
            "candidate_recall": val_recall_seed,
        })
        print(
            f"Seed {seed} decoded validation: "
            f"macro={val_macro_seed:.4f}, weighted={val_weighted_seed:.4f}, "
            f"candidate Top1={val_candidate:.4f}"
        )

    seed_records.sort(
        key=lambda r: (r["validation_weighted"], r["validation_macro"], r["candidate_top1"]),
        reverse=True,
    )
    winner = seed_records[0]
    best_model = winner["model"]
    best_seed = int(winner["seed"])
    best_candidate = float(winner["candidate_top1"])
    print(
        f"\nSelected seed by decoded VALIDATION accuracy: seed={best_seed}, "
        f"weighted={winner['validation_weighted']:.4f}, "
        f"macro={winner['validation_macro']:.4f}"
    )

    print("\nVALIDATION DECODER SWEEP")
    configs = build_decoder_candidates()
    sweep_rows = []
    print("Precomputing candidate model outputs once for the sweep...")
    evidence = precompute_decode_evidence(best_model, val_lectures, configs[0])
    for idx, cfg in enumerate(configs, 1):
        macro, weighted, per, recall = evaluate_precomputed(evidence, cfg, verbose=False)
        row = {"idx": idx, "macro": macro, "weighted": weighted, "candidate_recall": recall, **cfg}
        sweep_rows.append(row)
        if idx % 8 == 0 or idx == len(configs):
            print(f"  sweep {idx:03d}/{len(configs)}: macro={macro:.4f} weighted={weighted:.4f}")

    # A small second-stage search varies the temporal constraints around the
    # best score. This is still validation-only and avoids a huge grid.
    sweep_rows.sort(key=lambda r: (r["weighted"], r["macro"]), reverse=True)
    finalists = []
    for seed_cfg in sweep_rows[:4]:
        for max_jump in [1, 2]:
            for stay_penalty in [0.0, 0.15, 0.30]:
                cfg2 = dict(seed_cfg)
                cfg2["max_forward_jump"] = max_jump
                cfg2["boundary_stay_penalty"] = stay_penalty
                macro2, weighted2, per2, recall2 = evaluate_precomputed(evidence, cfg2, verbose=False)
                cfg2.update({"macro": macro2, "weighted": weighted2, "candidate_recall": recall2, "stage": 2})
                finalists.append(cfg2)
    sweep_rows.extend(finalists)
    sweep_rows.sort(key=lambda r: (r["weighted"], r["macro"]), reverse=True)
    best_cfg = sweep_rows[0]
    print("\nBEST VALIDATION CONFIG")
    print(json.dumps(best_cfg, indent=2))
    macro, weighted, per, recall = evaluate_precomputed(evidence, best_cfg, verbose=True)
    print(f"Validation macro={macro:.4f} weighted={weighted:.4f} candidate_recall={recall:.4f}")

    save_checkpoint(best_model, best_cfg, args, best_seed, best_candidate, macro, weighted)

    # Save sweep as JSONL/CSV without requiring pandas.
    sweep_path = RESULT_DIR / "validation_decoder_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_rows, f, indent=2)
    with open(RESULT_DIR / "validation_best.json", "w", encoding="utf-8") as f:
        json.dump({"macro": macro, "weighted": weighted, "candidate_recall": recall, "per_lecture": per, "config": best_cfg}, f, indent=2)

    print("\n============================================")
    print("EXPERIMENT B TRAINING COMPLETE")
    print("============================================")
    print(f"Checkpoint: {OUT_CHECKPOINT}")
    print(f"Validation macro: {macro*100:.2f}%")
    print(f"Validation weighted: {weighted*100:.2f}%")
    print(f"Validation candidate recall: {recall*100:.2f}%")
    print("TEST SET WAS NOT USED FOR TRAINING OR MODEL/DECODER SELECTION.")


if __name__ == "__main__":
    main()
