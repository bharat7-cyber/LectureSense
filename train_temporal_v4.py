from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_temporal_v3 as v3
from boundary_dp_v4 import boundary_aware_dp

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "runs" / "temporal_v3_cache"
V3_CHECKPOINT = CHECKPOINT_DIR / "temporal_v3_best_model.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "temporal_v4_best_model.pt"

MAX_SLIDES_DEFAULT = 96
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

        nn.init.constant_(self.boundary[-1].bias, -2.0)
        nn.init.zeros_(self.next_slide[-1].weight)
        nn.init.zeros_(self.next_slide[-1].bias)

    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        gate = torch.tanh(self.attn_gate)
        h = self.norm(h + gate * attn_out)
        delta = self.delta(h)
        boundary_logit = self.boundary(h).squeeze(-1)
        next_logits = self.next_slide(h)
        return delta, boundary_logit, next_logits


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
        prev_norm = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-8)
        cur_norm = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-8)
        change[1:] = np.clip(
            1.0 - np.sum(prev_norm * cur_norm, axis=1),
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
        add_boundary_windows=True,
        boundary_radius=16,
    ):
        self.samples = []

        for lecture, item in lectures.items():
            features = make_v4_features(
                item["scores"],
                item["time_norm"],
                max_slides,
                temperature,
            )
            labels = item["labels"]
            boundaries = make_boundary_targets(labels)
            n = len(labels)

            starts = list(range(0, max(1, n - window + 1), stride))
            last = max(0, n - window)
            if not starts or starts[-1] != last:
                starts.append(last)

            extra = []
            if add_boundary_windows:
                for idx in np.where(boundaries > 0.5)[0].tolist():
                    start = max(0, idx - boundary_radius)
                    start = min(start, last)
                    extra.append(start)
            starts.extend(extra)

            for start in sorted(set(starts)):
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

                self.samples.append(
                    (
                        x,
                        y,
                        b,
                        mask,
                    )
                )

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


def evaluate_model(model, lectures, device, max_slides, temperature):
    model.eval()
    top1, top3, mrr = [], [], []
    boundary_true, boundary_pred = [], []

    with torch.no_grad():
        for lecture, item in lectures.items():
            x = torch.from_numpy(
                make_v4_features(
                    item["scores"],
                    item["time_norm"],
                    max_slides,
                    temperature,
                )
            ).unsqueeze(0).to(device)

            delta, boundary_logit, next_logits = model(x)
            base_scores = torch.from_numpy(item["scores"]).to(device)
            fused = base_scores + delta[0, :, :base_scores.shape[1]]

            order = torch.argsort(fused, dim=1, descending=True)
            target = torch.from_numpy(item["labels"]).to(device)

            t1 = (order[:, 0] == target).float().cpu().numpy()
            t3 = (
                order[:, :min(3, order.shape[1])]
                == target[:, None]
            ).any(dim=1).float().cpu().numpy()

            ranks = []
            for i in range(len(target)):
                pos = (order[i] == target[i]).nonzero(as_tuple=False)
                ranks.append(1.0 / (int(pos[0].item()) + 1))

            b_true = make_boundary_targets(item["labels"])
            b_pred = (
                torch.sigmoid(boundary_logit[0]).cpu().numpy() >= 0.5
            ).astype(np.float32)

            top1.extend(t1.tolist())
            top3.extend(t3.tolist())
            mrr.extend(ranks)
            boundary_true.extend(b_true.tolist())
            boundary_pred.extend(b_pred.tolist())

            print(
                f"  {lecture}: "
                f"Top1={np.mean(t1):.4f} "
                f"Top3={np.mean(t3):.4f} "
                f"MRR={np.mean(ranks):.4f}"
            )

    tp = sum(1 for t, p in zip(boundary_true, boundary_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(boundary_true, boundary_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(boundary_true, boundary_pred) if t == 1 and p == 0)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    boundary_f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return (
        float(np.mean(top1)),
        float(np.mean(top3)),
        float(np.mean(mrr)),
        float(boundary_f1),
    )


def decode_validation(model, lectures, device, max_slides, temperature, decoder_cfg):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in lectures.values():
            features = make_v4_features(
                item["scores"],
                item["time_norm"],
                max_slides,
                temperature,
            )
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            delta, boundary_logit, next_logits = model(x)

            base_scores = item["scores"]
            adjusted = base_scores + delta[0, :, :base_scores.shape[1]].cpu().numpy()
            b = torch.sigmoid(boundary_logit[0]).cpu().numpy()
            nxt = next_logits[0, :, :base_scores.shape[1]].cpu().numpy()

            path = boundary_aware_dp(
                adjusted,
                b,
                nxt,
                item["timestamps"],
                np.r_[item["timestamps"][1:], item["timestamps"][-1] + 8.0],
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

            path = np.asarray(path, dtype=np.int64)
            labels = np.asarray(item["labels"], dtype=np.int64)
            correct += int(np.sum(path == labels))
            total += len(labels)

    return correct / max(1, total)


def warm_start_from_v3(model, device):
    if not V3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Required warm-start checkpoint not found: {V3_CHECKPOINT}"
        )

    ckpt = torch.load(V3_CHECKPOINT, map_location=device)
    source = ckpt["model_state_dict"]
    target = model.state_dict()

    copied = []
    for key in ["gru", "delta"]:
        for name, value in source.items():
            if not name.startswith(key + "."):
                continue
            if name in target and target[name].shape == value.shape:
                target[name].copy_(value)
                copied.append(name)

    src_weight = source.get("input_proj.0.weight")
    if src_weight is not None:
        dst_weight = target["input_proj.0.weight"]
        cols = min(src_weight.shape[1], dst_weight.shape[1])
        with torch.no_grad():
            dst_weight[:, :cols].copy_(src_weight[:, :cols])
            if dst_weight.shape[1] > cols:
                dst_weight[:, cols:].zero_()
        copied.append("input_proj.0.weight")

    for name in ["input_proj.0.bias"]:
        if name in source and name in target and source[name].shape == target[name].shape:
            target[name].copy_(source[name])
            copied.append(name)

    model.load_state_dict(target, strict=True)
    print(f"Warm-started V4 from V3: {len(copied)} tensors")


def run_decoder_sweep(model, val_lectures, device, max_slides, temperature):
    candidates = []
    for max_run in [900.0, 1200.0, 1500.0]:
        for duration_penalty in [0.0, 0.001, 0.003]:
            cfg = {
                "jump_penalty": 0.05,
                "max_forward_jump": 1,
                "max_duration_seconds": 120.0,
                "duration_penalty": duration_penalty,
                "max_run_seconds": max_run,
                "boundary_reward": 1.0,
                "boundary_stay_penalty": 0.50,
                "transition_logits_weight": 0.35,
                "premature_transition_penalty": 0.50,
                "boundary_threshold": 0.20,
            }
            acc = decode_validation(
                model,
                val_lectures,
                device,
                max_slides,
                temperature,
                cfg,
            )
            candidates.append((acc, cfg))
            print(
                f"Decoder sweep | max_run={max_run:.0f} "
                f"duration_penalty={duration_penalty:.3f} "
                f"Val DP accuracy={acc:.4f}"
            )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_acc, best_cfg = candidates[0]
    print(f"Best validation decoder accuracy: {best_acc:.4f}")
    print(f"Best decoder config: {best_cfg}")
    return best_acc, best_cfg


def main():
    parser = argparse.ArgumentParser(
        description="LectureSense V4 boundary-aware temporal training"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-slides", type=int, default=MAX_SLIDES_DEFAULT)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--boundary-radius", type=int, default=16)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.65)
    parser.add_argument("--next-loss-weight", type=float, default=0.25)
    parser.add_argument("--residual-reg", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v3.seed_everything(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading frozen V3 score caches...")
    train_df = v3.normalize_pairs(v3.load_csv(v3.PAIRS))
    train_df = train_df[train_df["split"] == "train"].copy()
    val_df = train_df.iloc[0:0].copy()
    all_pairs = v3.normalize_pairs(v3.load_csv(v3.PAIRS))
    val_df = all_pairs[all_pairs["split"].isin(["val", "validation"])].copy()

    text_embeddings = np.load(v3.TEXT_EMB).astype(np.float32)
    ocr_embeddings = np.load(v3.OCR_EMB).astype(np.float32)
    image_embeddings = np.load(v3.IMAGE_EMB).astype(np.float32)
    text_index = v3.load_csv(v3.TEXT_INDEX)
    ocr_index = v3.load_csv(v3.OCR_INDEX)
    image_index = v3.load_csv(v3.IMAGE_INDEX)

    text_lookup = v3.make_text_lookup(text_index, len(text_embeddings))
    ocr_lookup = v3.make_slide_lookup(
        ocr_index, "embedding_row", len(ocr_embeddings)
    )
    image_lookup = v3.make_slide_lookup(
        image_index, "embedding_idx", len(image_embeddings)
    )
    slides_by_lecture = v3.build_slide_inventory(
        image_index,
        ocr_lookup,
        image_lookup,
    )

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
            device,
            args.score_batch,
        )
        train_lectures[lecture] = item

    print("\nLoading/generating VALIDATION caches...")
    for lecture, rows in val_seq.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            continue
        cache = CACHE_DIR / f"val_{lecture}.npz"
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
            device,
            args.score_batch,
        )
        val_lectures[lecture] = item

    model = BoundaryAwareTemporalGRU(
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
        add_boundary_windows=True,
        boundary_radius=args.boundary_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    total_boundary_pos = 0
    total_boundary_neg = 0
    for item in train_lectures.values():
        b = make_boundary_targets(item["labels"])
        total_boundary_pos += int(np.sum(b))
        total_boundary_neg += int(len(b) - np.sum(b))

    pos_weight = min(
        10.0,
        max(
            2.0,
            float(np.sqrt(max(1.0, total_boundary_neg / max(1, total_boundary_pos))))
        )
    )
    print(f"Transition windows: {len(dataset)}")
    print(f"Boundary positives: {total_boundary_pos}")
    print(f"Boundary negatives: {total_boundary_neg}")
    print(f"Boundary BCE pos_weight: {pos_weight:.3f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    boundary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device),
        reduction="none",
    )

    best_mrr = -1.0
    best_top1 = -1.0
    best_boundary_f1 = -1.0
    stale = 0

    print("\nStarting V4 training...")
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

            delta, boundary_logit, next_logits = model(x)

            probs = x[:, :, :args.max_slides].clamp_min(1e-6)
            base_logits = torch.log(probs)
            fused = base_logits + delta

            ce_raw = F.cross_entropy(
                fused.reshape(-1, args.max_slides),
                y.reshape(-1),
                reduction="none",
                label_smoothing=0.03,
            )
            ce = (ce_raw * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)

            b_raw = boundary_loss_fn(boundary_logit.reshape(-1), b.reshape(-1))
            b_loss = (b_raw * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)

            boundary_mask = b * mask
            next_ce_raw = F.cross_entropy(
                next_logits.reshape(-1, args.max_slides),
                y.reshape(-1),
                reduction="none",
            )
            next_denom = boundary_mask.sum().clamp_min(1.0)
            next_loss = (
                (next_ce_raw * boundary_mask.reshape(-1)).sum()
                / next_denom
            )

            valid_delta = delta.pow(2) * mask.unsqueeze(-1)
            reg = valid_delta.mean()

            loss = (
                ce
                + args.boundary_loss_weight * b_loss
                + args.next_loss_weight * next_loss
                + args.residual_reg * reg
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        val_top1, val_top3, val_mrr, boundary_f1 = evaluate_model(
            model,
            val_lectures,
            device,
            args.max_slides,
            args.temperature,
        )
        scheduler.step(val_mrr)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {total_loss / max(1, steps):.4f} | "
            f"Val Top1 {val_top1:.4f} | "
            f"Val Top3 {val_top3:.4f} | "
            f"Val MRR {val_mrr:.4f} | "
            f"Boundary F1 {boundary_f1:.4f} | "
            f"LR {lr:.2e}"
        )

        improved = (
            val_mrr > best_mrr + 1e-6
            or (
                abs(val_mrr - best_mrr) <= 1e-6
                and val_top1 > best_top1 + 1e-6
            )
        )

        if improved:
            best_mrr = val_mrr
            best_top1 = val_top1
            best_boundary_f1 = boundary_f1
            stale = 0

            torch.save(
                {
                    "model_type": "BoundaryAwareTemporalGRU_V4",
                    "model_state_dict": model.state_dict(),
                    "base_model_state_dict": base_model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "val_mrr": val_mrr,
                    "val_boundary_f1": boundary_f1,
                    "base_val_top1": v3.evaluate_base(val_lectures)[0],
                    "base_val_top3": v3.evaluate_base(val_lectures)[1],
                    "base_val_mrr": v3.evaluate_base(val_lectures)[2],
                    "max_slides": args.max_slides,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "heads": args.heads,
                    "temperature": args.temperature,
                    "window": args.window,
                    "stride": args.stride,
                    "extra_features": EXTRA_FEATURES,
                    "decoder_config": {
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
                    "seed": args.seed,
                },
                OUT_CHECKPOINT,
            )
            print(f"  Saved best V4 checkpoint -> {OUT_CHECKPOINT}")
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("\nRunning VALIDATION-ONLY decoder sweep...")
    best_ckpt = torch.load(OUT_CHECKPOINT, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    decoder_acc, decoder_cfg = run_decoder_sweep(
        model,
        val_lectures,
        device,
        args.max_slides,
        args.temperature,
    )

    best_ckpt["decoder_config"] = decoder_cfg
    best_ckpt["val_decoder_accuracy"] = decoder_acc
    torch.save(best_ckpt, OUT_CHECKPOINT)

    print("\n================================")
    print("V4 TRAINING COMPLETE")
    print("================================")
    print(f"Best validation Top-1 : {best_top1:.4f}")
    print(f"Best validation MRR   : {best_mrr:.4f}")
    print(f"Best boundary F1      : {best_boundary_f1:.4f}")
    print(f"Best decoder accuracy : {decoder_acc:.4f}")
    print(f"Checkpoint            : {OUT_CHECKPOINT}")
    print("TEST SET: NOT USED for training, model selection, or decoder tuning.")


if __name__ == "__main__":
    main()
