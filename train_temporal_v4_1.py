from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import train_temporal_v3 as v3
from boundary_dp_v4_1 import boundary_aware_dp_v41

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "runs" / "temporal_v3_cache"
V3_CHECKPOINT = CHECKPOINT_DIR / "temporal_v3_best_model.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "temporal_v4_1_best_model.pt"
OUT_CONFIG = CHECKPOINT_DIR / "temporal_v4_1_decoder_config.json"

MAX_SLIDES_DEFAULT = 96
EXTRA_FEATURES = 12


class BoundaryAwareTemporalGRUV41(nn.Module):
    def __init__(self, input_dim, max_slides, hidden=192, layers=2, heads=4):
        super().__init__()
        if hidden * 2 % heads != 0:
            raise ValueError("hidden*2 must be divisible by heads")

        self.max_slides = max_slides
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.gru = nn.GRU(
            160,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.20 if layers > 1 else 0.0,
            bidirectional=True,
        )
        d_model = hidden * 2
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads=heads,
            dropout=0.10,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Confidence gate: prevents the temporal model from overwriting
        # confident base-model predictions unnecessarily.
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

        nn.init.constant_(self.boundary[-1].bias, -2.2)
        nn.init.constant_(self.gate[-1].bias, -0.2)
        nn.init.zeros_(self.next_slide[-1].weight)
        nn.init.zeros_(self.next_slide[-1].bias)

    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)

        h = self.norm2(h + self.ffn(h))

        gate = torch.sigmoid(self.gate(h)).squeeze(-1)
        delta = self.delta(h)
        boundary_logit = self.boundary(h).squeeze(-1)
        next_logits = self.next_slide(h)
        return delta, boundary_logit, next_logits, gate


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
    top3_mean = (top1 + top2 + top3) / 3.0
    margin = top1 - top2
    peak_gap = top1 - top3_mean

    denom = max(1e-8, float(np.log(max(2, s))))
    entropy = -np.sum(
        probs * np.log(np.clip(probs, 1e-8, 1.0)),
        axis=1,
    ) / denom

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
        time_norm,
        durations,
        top1,
        margin,
        entropy,
        change,
        top1_change,
        margin_change,
        entropy_change,
        argmax_change,
        argmax_norm,
        peak_gap,
    ]
    for j, feat in enumerate(extras):
        x[:, max_slides + j] = feat
    return x


def make_boundary_targets(labels):
    labels = np.asarray(labels, dtype=np.int64)
    b = np.zeros(len(labels), dtype=np.float32)
    if len(labels) > 1:
        b[1:] = (labels[1:] != labels[:-1]).astype(np.float32)
    return b


class TransitionWindowDataset(Dataset):
    def __init__(
        self,
        lectures,
        window,
        stride,
        max_slides,
        temperature,
        boundary_radius=16,
    ):
        self.samples = []
        self.weights = []

        for lecture, item in lectures.items():
            features = make_v41_features(
                item["scores"], item["time_norm"], max_slides, temperature
            )
            labels = item["labels"]
            boundaries = make_boundary_targets(labels)
            n = len(labels)

            starts = list(range(0, max(1, n - window + 1), stride))
            last = max(0, n - window)
            if not starts or starts[-1] != last:
                starts.append(last)

            boundary_starts = []
            for idx in np.where(boundaries > 0.5)[0].tolist():
                start = max(0, idx - boundary_radius)
                start = min(start, last)
                boundary_starts.append(start)

            starts.extend(boundary_starts)

            unique = sorted(set(starts))
            boundary_start_set = set(boundary_starts)

            for start in unique:
                end = min(start + window, n)
                length = end - start

                x = np.zeros((window, features.shape[1]), dtype=np.float32)
                y = np.zeros(window, dtype=np.int64)
                b = np.zeros(window, dtype=np.float32)
                mask = np.zeros(window, dtype=np.float32)

                x[:length] = features[start:end]
                y[:length] = labels[start:end]
                b[:length] = boundaries[start:end]
                mask[:length] = 1.0

                self.samples.append((x, y, b, mask))

                # Explicitly oversample transition-centered windows.
                self.weights.append(3.0 if start in boundary_start_set else 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, b, mask = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(b),
            torch.from_numpy(mask),
        )


def warm_start_from_v3(model, device):
    # Copy only compatible recurrent/input layers from V3/V4-style
    # checkpoints; new V4.1 heads remain freshly initialized.
    source_path = V3_CHECKPOINT
    if not source_path.exists():
        raise FileNotFoundError(f"Required warm-start checkpoint not found: {source_path}")

    ckpt = torch.load(source_path, map_location=device)
    source = ckpt["model_state_dict"]
    target = model.state_dict()

    copied = []
    for key in ["gru"]:
        for name, value in source.items():
            if name.startswith(key + ".") and name in target and target[name].shape == value.shape:
                target[name].copy_(value)
                copied.append(name)

    src_weight = source.get("input_proj.0.weight")
    if src_weight is not None and target["input_proj.0.weight"].shape[0] == src_weight.shape[0]:
        # Only copy the overlapping feature columns because V4.1 adds scalars.
        dst = target["input_proj.0.weight"]
        cols = min(src_weight.shape[1], dst.shape[1])
        with torch.no_grad():
            dst[:, :cols].copy_(src_weight[:, :cols])
        copied.append("input_proj.0.weight")

    for name in ["input_proj.0.bias"]:
        if name in source and name in target and source[name].shape == target[name].shape:
            target[name].copy_(source[name])
            copied.append(name)

    model.load_state_dict(target, strict=True)
    print(f"Warm-started V4.1 from V3: {len(copied)} tensors")


def focal_bce_with_logits(logits, targets, mask, pos_weight, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal = ((1.0 - p_t).clamp_min(1e-6) ** gamma) * ce
    return (focal * mask).sum() / mask.sum().clamp_min(1.0)


def hard_negative_pairwise_loss(logits, labels, mask, margin_value=0.20, top_k=5):
    losses = []
    bsz, t, s = logits.shape
    for bi in range(bsz):
        valid_t = torch.nonzero(mask[bi] > 0.5, as_tuple=False).flatten()
        for ti in valid_t.tolist():
            y = int(labels[bi, ti].item())
            if y < 0 or y >= s:
                continue
            row = logits[bi, ti]
            neg = row.clone()
            neg[y] = -1e9
            k = min(top_k, s - 1)
            hard = torch.topk(neg, k=k).values
            diffs = row[y] - hard
            losses.append(F.softplus(margin_value - diffs).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def run_model_outputs(model, lectures, device, max_slides, temperature):
    model.eval()
    outputs = {}
    with torch.no_grad():
        for lecture, item in lectures.items():
            features = make_v41_features(
                item["scores"], item["time_norm"], max_slides, temperature
            )
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            delta, boundary_logit, next_logits, gate = model(x)

            probs = x[:, :, :max_slides].clamp_min(1e-6)
            base_logits = torch.log(probs)
            adjusted = (
                base_logits
                + gate[0].unsqueeze(-1) * delta[0]
            )
            s = item["scores"].shape[1]

            outputs[lecture] = {
                "scores": adjusted[:, :s].cpu().numpy(),
                "boundary": torch.sigmoid(boundary_logit[0]).cpu().numpy(),
                "next": next_logits[0, :, :s].cpu().numpy(),
                "labels": np.asarray(item["labels"], dtype=np.int64),
                "starts": np.asarray(item["timestamps"], dtype=np.float32),
                "ends": np.r_[
                    np.asarray(item["timestamps"][1:], dtype=np.float32),
                    float(item["timestamps"][-1]) + 8.0,
                ],
            }
    return outputs


def evaluate_model(model, lectures, device, max_slides, temperature):
    outputs = run_model_outputs(model, lectures, device, max_slides, temperature)
    top1, top3, mrr = [], [], []
    boundary_true, boundary_pred = [], []

    for lecture, out in outputs.items():
        fused = out["scores"]
        target = out["labels"]
        order = np.argsort(fused, axis=1)[:, ::-1]

        t1 = (order[:, 0] == target).astype(np.float32)
        t3 = np.any(order[:, :min(3, order.shape[1])] == target[:, None], axis=1).astype(np.float32)

        rr = []
        for i in range(len(target)):
            pos = np.where(order[i] == target[i])[0]
            rr.append(1.0 / float(pos[0] + 1) if len(pos) else 0.0)

        b_true = make_boundary_targets(target)
        b_pred = (out["boundary"] >= 0.20).astype(np.float32)

        top1.extend(t1.tolist())
        top3.extend(t3.tolist())
        mrr.extend(rr)
        boundary_true.extend(b_true.tolist())
        boundary_pred.extend(b_pred.tolist())

        print(
            f"  {lecture}: Top1={np.mean(t1):.4f} "
            f"Top3={np.mean(t3):.4f} MRR={np.mean(rr):.4f}"
        )

    tp = sum(t == 1 and p == 1 for t, p in zip(boundary_true, boundary_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(boundary_true, boundary_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(boundary_true, boundary_pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return float(np.mean(top1)), float(np.mean(top3)), float(np.mean(mrr)), float(f1)


def decode_outputs(outputs, cfg):
    correct = 0
    total = 0
    per_lecture = {}

    for lecture, out in outputs.items():
        path = boundary_aware_dp_v41(
            out["scores"],
            out["boundary"],
            out["next"],
            out["starts"],
            out["ends"],
            **cfg,
        )
        path = np.asarray(path, dtype=np.int64)
        labels = out["labels"]
        c = int(np.sum(path == labels))
        per_lecture[lecture] = (c, len(labels))
        correct += c
        total += len(labels)

    return correct / max(1, total), per_lecture


def coordinate_decoder_tune(outputs, initial=None, rounds=2):
    cfg = {
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
    if initial:
        cfg.update(initial)

    grid = {
        "max_forward_jump": [1, 2, 3, 4],
        "jump_penalty": [0.0, 0.015, 0.03, 0.05, 0.08],
        "max_duration_seconds": [90.0, 120.0, 150.0, 180.0],
        "duration_penalty": [0.0, 0.0005, 0.001, 0.002, 0.003],
        "max_run_seconds": [900.0, 1200.0, 1500.0, 1800.0],
        "boundary_reward": [0.0, 0.5, 0.75, 1.0, 1.5],
        "boundary_stay_penalty": [0.0, 0.25, 0.35, 0.5, 0.75],
        "transition_logits_weight": [0.0, 0.15, 0.30, 0.45, 0.60],
        "premature_transition_penalty": [0.0, 0.2, 0.35, 0.5, 0.75],
        "boundary_threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
        "jump_penalty_power": [1.0, 1.25, 1.5, 1.75],
    }

    best_acc, _ = decode_outputs(outputs, cfg)
    print(f"Decoder start accuracy: {best_acc:.4f}")

    for rnd in range(1, rounds + 1):
        improved = False
        for key, values in grid.items():
            local_best = best_acc
            local_cfg = dict(cfg)
            for value in values:
                trial = dict(cfg)
                trial[key] = value
                acc, _ = decode_outputs(outputs, trial)
                print(f"  round={rnd} param={key} value={value} acc={acc:.4f}")
                if acc > local_best + 1e-8:
                    local_best = acc
                    local_cfg = trial
            if local_best > best_acc + 1e-8:
                best_acc = local_best
                cfg = local_cfg
                improved = True
                print(f"  improved -> {best_acc:.4f}")
        if not improved:
            break

    return best_acc, cfg


def main():
    parser = argparse.ArgumentParser(description="LectureSense V4.1 training")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--max-slides", type=int, default=MAX_SLIDES_DEFAULT)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--boundary-radius", type=int, default=16)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.80)
    parser.add_argument("--next-loss-weight", type=float, default=0.30)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.35)
    parser.add_argument("--pairwise-margin", type=float, default=0.20)
    parser.add_argument("--residual-reg", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    v3.seed_everything(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_pairs = v3.normalize_pairs(v3.load_csv(v3.PAIRS))
    train_df = all_pairs[all_pairs["split"] == "train"].copy()
    val_df = all_pairs[all_pairs["split"].isin(["val", "validation"])].copy()

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

    train_seq = v3.build_positive_sequences(train_df)
    val_seq = v3.build_positive_sequences(val_df)

    base_model = v3.load_base_model(device)
    base_model.eval()

    train_lectures = {}
    val_lectures = {}

    print("\nLoading/generating TRAIN caches...")
    for lecture, rows in train_seq.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            continue
        cache = CACHE_DIR / f"train_{lecture}.npz"
        train_lectures[lecture] = v3.load_cached_or_score(
            cache, lecture, rows, slides, text_lookup, ocr_lookup, image_lookup,
            text_embeddings, ocr_embeddings, image_embeddings, base_model, device, args.score_batch
        )

    print("\nLoading/generating VALIDATION caches...")
    for lecture, rows in val_seq.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            continue
        cache = CACHE_DIR / f"val_{lecture}.npz"
        val_lectures[lecture] = v3.load_cached_or_score(
            cache, lecture, rows, slides, text_lookup, ocr_lookup, image_lookup,
            text_embeddings, ocr_embeddings, image_embeddings, base_model, device, args.score_batch
        )

    model = BoundaryAwareTemporalGRUV41(
        args.max_slides + EXTRA_FEATURES,
        args.max_slides,
        args.hidden,
        args.layers,
        args.heads,
    ).to(device)
    warm_start_from_v3(model, device)

    dataset = TransitionWindowDataset(
        train_lectures,
        args.window,
        args.stride,
        args.max_slides,
        args.temperature,
        boundary_radius=args.boundary_radius,
    )
    weights = torch.DoubleTensor(dataset.weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_pos, total_neg = 0, 0
    for item in train_lectures.values():
        b = make_boundary_targets(item["labels"])
        total_pos += int(np.sum(b))
        total_neg += int(len(b) - np.sum(b))

    pos_weight = min(
        12.0,
        max(2.5, float(np.sqrt(max(1.0, total_neg / max(1, total_pos)))) * 1.25),
    )
    print(f"Transition windows: {len(dataset)}")
    print(f"Boundary positives: {total_pos}")
    print(f"Boundary negatives: {total_neg}")
    print(f"Boundary BCE pos_weight: {pos_weight:.3f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-6
    )
    b_pos = torch.tensor(pos_weight, device=device)

    best_score = -1.0
    best_top1 = -1.0
    best_mrr = -1.0
    stale = 0

    print("\nStarting V4.1 training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for x, y, b, mask in loader:
            x = x.to(device)
            y = y.to(device)
            b = b.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            delta, boundary_logit, next_logits, gate = model(x)

            probs = x[:, :, :args.max_slides].clamp_min(1e-6)
            base_logits = torch.log(probs)
            fused = base_logits + gate.unsqueeze(-1) * delta

            ce_raw = F.cross_entropy(
                fused.reshape(-1, args.max_slides),
                y.reshape(-1),
                reduction="none",
                label_smoothing=0.02,
            )
            ce = (ce_raw * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)

            b_loss = focal_bce_with_logits(
                boundary_logit.reshape(-1),
                b.reshape(-1),
                mask.reshape(-1),
                b_pos,
                gamma=2.0,
            )

            boundary_mask = b * mask
            next_ce_raw = F.cross_entropy(
                next_logits.reshape(-1, args.max_slides),
                y.reshape(-1),
                reduction="none",
            )
            next_loss = (
                (next_ce_raw * boundary_mask.reshape(-1)).sum()
                / boundary_mask.sum().clamp_min(1.0)
            )

            pair_loss = hard_negative_pairwise_loss(
                fused, y, mask, margin_value=args.pairwise_margin, top_k=5
            )

            reg = (
                (gate.unsqueeze(-1) * delta).pow(2) * mask.unsqueeze(-1)
            ).mean()

            loss = (
                ce
                + args.boundary_loss_weight * b_loss
                + args.next_loss_weight * next_loss
                + args.pairwise_loss_weight * pair_loss
                + args.residual_reg * reg
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        val_top1, val_top3, val_mrr, boundary_f1 = evaluate_model(
            model, val_lectures, device, args.max_slides, args.temperature
        )
        scheduler.step(val_mrr)

        # Save by a validation objective that reflects ranking quality,
        # while the final decoder is separately optimized on validation only.
        selection = 0.60 * val_mrr + 0.40 * val_top1

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {total_loss / max(1, steps):.4f} | "
            f"Val Top1 {val_top1:.4f} | Val Top3 {val_top3:.4f} | "
            f"Val MRR {val_mrr:.4f} | Boundary F1 {boundary_f1:.4f} | "
            f"Select {selection:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if selection > best_score + 1e-6:
            best_score = selection
            best_top1 = val_top1
            best_mrr = val_mrr
            stale = 0
            torch.save(
                {
                    "model_type": "BoundaryAwareTemporalGRU_V4_1",
                    "model_state_dict": model.state_dict(),
                    "base_model_state_dict": base_model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "val_mrr": val_mrr,
                    "val_boundary_f1": boundary_f1,
                    "max_slides": args.max_slides,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "heads": args.heads,
                    "temperature": args.temperature,
                    "window": args.window,
                    "stride": args.stride,
                    "extra_features": EXTRA_FEATURES,
                    "selection_score": selection,
                    "seed": args.seed,
                },
                OUT_CHECKPOINT,
            )
            print(f"  Saved best V4.1 checkpoint -> {OUT_CHECKPOINT}")
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("\nRunning VALIDATION-ONLY decoder coordinate search...")
    best_ckpt = torch.load(OUT_CHECKPOINT, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    outputs = run_model_outputs(
        model, val_lectures, device, args.max_slides, args.temperature
    )
    decoder_acc, decoder_cfg = coordinate_decoder_tune(outputs, rounds=2)

    import json
    best_ckpt["decoder_config"] = decoder_cfg
    best_ckpt["val_decoder_accuracy"] = decoder_acc
    torch.save(best_ckpt, OUT_CHECKPOINT)
    OUT_CONFIG.write_text(json.dumps(decoder_cfg, indent=2), encoding="utf-8")

    print("\n================================")
    print("V4.1 TRAINING COMPLETE")
    print("================================")
    print(f"Best validation Top-1 : {best_top1:.4f}")
    print(f"Best validation MRR   : {best_mrr:.4f}")
    print(f"Best validation DP    : {decoder_acc:.4f}")
    print(f"Checkpoint            : {OUT_CHECKPOINT}")
    print(f"Decoder config        : {OUT_CONFIG}")
    print("TEST SET: NOT USED for training, model selection, or decoder tuning.")


if __name__ == "__main__":
    main()
