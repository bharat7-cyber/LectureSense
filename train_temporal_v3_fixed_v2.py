from __future__ import annotations

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "runs" / "temporal_v3_cache"

TEXT_EMB = DATA_DIR / "text_embeddings.npy"
OCR_EMB = DATA_DIR / "ocr_embeddings.npy"
IMAGE_EMB = DATA_DIR / "image_embeddings.npy"
TEXT_INDEX = DATA_DIR / "text_embedding_index.csv"
OCR_INDEX = DATA_DIR / "ocr_embedding_index.csv"
IMAGE_INDEX = DATA_DIR / "image_embedding_index.csv"
PAIRS = DATA_DIR / "pairs.csv"
BASE_CHECKPOINT = CHECKPOINT_DIR / "ranking_4modal_best_model.pt"
OUT_CHECKPOINT = CHECKPOINT_DIR / "temporal_v3_best_model.pt"

TEXT_DIM = 384
OCR_DIM = 384
IMAGE_DIM = 512
HIDDEN_DIM = 256
MAX_SLIDES_DEFAULT = 96

LECTURE_ID_MAP = {
    "ML_for_health_MIT": "ML_for_health_MIT",
    "cities_and_decarbonization": "cities_and_decarbonization",
    "climate_and_cities": "climate_and_cities",
    "climate_science_policy_MIT": "climate_science_policy_MIT2",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT",
    "computer_vision_2_2": "computer_vision_2_2",
    "creating_breakthrough_products": "creating_breakthrough_products_MIT",
    "cryptocurrency": "cryptocurrency_MIT",
    "deeplearning_goodfellow": "deeplearning",
    "image_processing_high_res": "image_processing",
    "numerics_hennig": "numerics",
    "phonetics": "phonetics",
    "physics_intro_02": "physics",
    "psychology_MIT": "psychology",
    "reinforcement_learning_silver": "reinforcement_learning",
    "short_range_mit": "short_range",
    "solar_resource": "solar_resource",
    "team_dynamics_game_design_mit": "team_dynamics",
    "theory_of_computation": "theory_of_computation",
    "visual_system_MIT": "sensory_systems",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sentence(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_reverse_lecture_map() -> Dict[str, str]:
    return {v: k for k, v in LECTURE_ID_MAP.items()}


def make_text_lookup(text_index: pd.DataFrame, n_text: int):
    df = text_index.copy()
    df["lecture_id"] = df["lecture_id"].astype(str).str.strip()
    sentence_col = "sentence_norm" if "sentence_norm" in df.columns else "sentence_text"
    if sentence_col not in df.columns or "embedding_idx" not in df.columns:
        raise ValueError("text_embedding_index.csv must contain sentence_text/sentence_norm and embedding_idx")
    df["sentence_norm"] = df[sentence_col].map(normalize_sentence)
    df["embedding_idx"] = pd.to_numeric(df["embedding_idx"], errors="coerce")
    df = df.dropna(subset=["embedding_idx"]).copy()
    df["embedding_idx"] = df["embedding_idx"].astype(int)
    df = df[(df["embedding_idx"] >= 0) & (df["embedding_idx"] < n_text)]
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[(str(row.lecture_id), str(row.sentence_norm))] = int(row.embedding_idx)
    return lookup


def make_slide_lookup(index: pd.DataFrame, emb_key: str, n_emb: int):
    df = index.copy()
    if "lecture_id" not in df.columns or "slide_id" not in df.columns:
        raise ValueError("Slide embedding index must contain lecture_id and slide_id")
    if emb_key not in df.columns:
        if emb_key == "embedding_idx":
            df["_derived_embedding_idx"] = np.arange(len(df), dtype=np.int64)
            actual = "_derived_embedding_idx"
        else:
            raise ValueError(f"Missing embedding column: {emb_key}")
    else:
        actual = emb_key
    reverse = build_reverse_lecture_map()
    df["lecture_id"] = df["lecture_id"].astype(str).str.strip().map(lambda x: reverse.get(x, x))
    df["slide_id"] = pd.to_numeric(df["slide_id"], errors="coerce")
    df[actual] = pd.to_numeric(df[actual], errors="coerce")
    df = df.dropna(subset=["slide_id", actual]).copy()
    df["slide_id"] = df["slide_id"].astype(int)
    df[actual] = df[actual].astype(int)
    df = df[(df[actual] >= 0) & (df[actual] < n_emb)]
    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["lecture_id"]), int(row["slide_id"]))] = int(row[actual])
    return lookup


def normalize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs.copy()
    df["lecture_id"] = df["lecture_id"].astype(str).str.strip()
    df["sentence_text"] = df["sentence_text"].fillna("").astype(str)
    df["sentence_norm"] = df["sentence_text"].map(normalize_sentence)
    df["slide_id"] = pd.to_numeric(df["slide_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") if "timestamp" in df.columns else np.nan
    df["sentence_idx"] = pd.to_numeric(df["sentence_idx"], errors="coerce") if "sentence_idx" in df.columns else np.arange(len(df))
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["slide_id", "label"]).copy()
    df["slide_id"] = df["slide_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df["feature_lecture_id"] = df["lecture_id"].map(LECTURE_ID_MAP).fillna(df["lecture_id"])
    return df


def build_slide_inventory(image_index, ocr_lookup, image_lookup):
    reverse = build_reverse_lecture_map()
    inv = defaultdict(set)
    for row in image_index.itertuples(index=False):
        feature_lecture = str(row.lecture_id).strip()
        lecture = reverse.get(feature_lecture, feature_lecture)
        try:
            slide = int(row.slide_id)
        except Exception:
            continue
        if (lecture, slide) in ocr_lookup and (lecture, slide) in image_lookup:
            inv[lecture].add(slide)
    return {k: sorted(v) for k, v in inv.items()}


class LectureSenseCrossModal(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(TEXT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.15))
        self.ocr_proj = nn.Sequential(nn.Linear(OCR_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.15))
        self.image_proj = nn.Sequential(nn.Linear(IMAGE_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.15))
        self.scorer = nn.Sequential(
            nn.Linear(2308, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, text, ocr, image):
        t = self.text_proj(text)
        o = self.ocr_proj(ocr)
        i = self.image_proj(image)
        t_norm = F.normalize(t, dim=-1)
        o_norm = F.normalize(o, dim=-1)
        i_norm = F.normalize(i, dim=-1)
        native_to = torch.sum(F.normalize(text, dim=-1) * F.normalize(ocr, dim=-1), dim=-1, keepdim=True)
        projected_to = torch.sum(t_norm * o_norm, dim=-1, keepdim=True)
        projected_ti = torch.sum(t_norm * i_norm, dim=-1, keepdim=True)
        projected_oi = torch.sum(o_norm * i_norm, dim=-1, keepdim=True)
        fusion = torch.cat([
            t, o, i, torch.abs(t - o), t * o,
            torch.abs(t - i), t * i,
            torch.abs(o - i), o * i,
            native_to, projected_to, projected_ti, projected_oi,
        ], dim=-1)
        return self.scorer(fusion).squeeze(-1)


def load_base_model(device):
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_CHECKPOINT}")
    model = LectureSenseCrossModal().to(device)
    checkpoint = torch.load(BASE_CHECKPOINT, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_positive_sequences(df):
    positives = df[df["label"] == 1].copy()
    sequences = {}
    for lecture, part in positives.groupby("lecture_id"):
        part = part.sort_values(["sentence_idx", "timestamp"], kind="stable")
        seen = set()
        rows = []
        for row in part.itertuples(index=False):
            key = (int(row.sentence_idx), normalize_sentence(row.sentence_text))
            if key in seen:
                continue
            seen.add(key)
            ts = float(row.timestamp) if np.isfinite(row.timestamp) else float(len(rows) * 8.0)
            rows.append({
                "lecture_id": str(lecture),
                "feature_lecture_id": str(row.feature_lecture_id),
                "sentence_idx": int(row.sentence_idx),
                "timestamp": ts,
                "text_norm": normalize_sentence(row.sentence_text),
                "slide_id": int(row.slide_id),
            })
        if rows:
            sequences[str(lecture)] = rows
    return sequences


def score_lecture(lecture, rows, slides, text_lookup, ocr_lookup, image_lookup,
                  text_embeddings, ocr_embeddings, image_embeddings, base_model,
                  device, batch_size):
    text_indices = []
    valid_rows = []
    feature_lecture = rows[0]["feature_lecture_id"]
    for r in rows:
        idx = text_lookup.get((feature_lecture, r["text_norm"]))
        if idx is not None:
            text_indices.append(idx)
            valid_rows.append(r)
    if not valid_rows:
        raise RuntimeError(f"No text embeddings mapped for {lecture}")

    ocr_indices = [ocr_lookup[(lecture, s)] for s in slides]
    image_indices = [image_lookup[(lecture, s)] for s in slides]
    ocr_mat = torch.from_numpy(ocr_embeddings[ocr_indices]).float().to(device)
    image_mat = torch.from_numpy(image_embeddings[image_indices]).float().to(device)
    scores = np.zeros((len(valid_rows), len(slides)), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(valid_rows), batch_size):
            end = min(start + batch_size, len(valid_rows))
            text = torch.from_numpy(text_embeddings[text_indices[start:end]]).float().to(device)
            count = end - start
            tb = text.unsqueeze(1).expand(count, len(slides), TEXT_DIM).reshape(-1, TEXT_DIM)
            ob = ocr_mat.unsqueeze(0).expand(count, -1, -1).reshape(-1, OCR_DIM)
            ib = image_mat.unsqueeze(0).expand(count, -1, -1).reshape(-1, IMAGE_DIM)
            scores[start:end] = base_model(tb, ob, ib).reshape(count, len(slides)).cpu().numpy()
            print(f"  {lecture}: scored {end}/{len(valid_rows)}")

    timestamps = np.array([r["timestamp"] for r in valid_rows], dtype=np.float32)
    if len(timestamps) > 1:
        total = max(1.0, float(timestamps[-1] - timestamps[0]))
        time_norm = (timestamps - timestamps[0]) / total
    else:
        time_norm = np.zeros(len(timestamps), dtype=np.float32)
    labels = np.array([slides.index(r["slide_id"]) for r in valid_rows], dtype=np.int64)
    return {
        "scores": scores,
        "time_norm": time_norm,
        "labels": labels,
        "slide_ids": np.array(slides, dtype=np.int64),
        "timestamps": timestamps,
    }


class TemporalResidualGRU(nn.Module):
    def __init__(self, input_dim, max_slides, hidden=128, layers=2):
        super().__init__()
        self.max_slides = max_slides
        self.input_proj = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.10))
        self.gru = nn.GRU(128, hidden, num_layers=layers, batch_first=True,
                           dropout=0.20 if layers > 1 else 0.0, bidirectional=True)
        self.delta = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.10), nn.Linear(hidden, max_slides))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x):
        z = self.input_proj(x)
        h, _ = self.gru(z)
        return self.delta(h)


def make_features(scores, time_norm, max_slides, temperature):
    n, s = scores.shape
    if s > max_slides:
        raise ValueError(f"Lecture has {s} slides but checkpoint supports max {max_slides}")
    logits = torch.from_numpy(scores.astype(np.float32))
    probs = F.softmax(logits / temperature, dim=-1).numpy()
    x = np.zeros((n, max_slides + 2), dtype=np.float32)
    x[:, :s] = probs
    x[:, max_slides] = time_norm
    if n > 1:
        x[:, max_slides + 1] = np.clip(np.diff(np.r_[time_norm, 1.0]), 0.0, 1.0)
    return x


class WindowDataset(Dataset):
    def __init__(self, lectures, window, stride, max_slides, temperature):
        self.samples = []
        for item in lectures.values():
            features = make_features(item["scores"], item["time_norm"], max_slides, temperature)
            labels = item["labels"]
            n = len(labels)
            starts = list(range(0, max(1, n - window + 1), stride))
            last = max(0, n - window)
            if not starts or starts[-1] != last:
                starts.append(last)
            for start in sorted(set(starts)):
                end = min(start + window, n)
                x = np.zeros((window, features.shape[1]), dtype=np.float32)
                y = np.zeros(window, dtype=np.int64)
                mask = np.zeros(window, dtype=np.float32)
                length = end - start
                x[:length] = features[start:end]
                y[:length] = labels[start:end]
                mask[:length] = 1.0
                self.samples.append((x, y, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, mask = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def evaluate_base(lectures):
    top1, top3, mrr = [], [], []
    for item in lectures.values():
        order = np.argsort(-item["scores"], axis=1)
        target = item["labels"]
        t1 = (order[:, 0] == target).astype(np.float32)
        t3 = np.any(order[:, :min(3, order.shape[1])] == target[:, None], axis=1).astype(np.float32)
        ranks = [1.0 / (int(np.where(order[i] == target[i])[0][0]) + 1) for i in range(len(target))]
        top1.extend(t1.tolist()); top3.extend(t3.tolist()); mrr.extend(ranks)
    return float(np.mean(top1)), float(np.mean(top3)), float(np.mean(mrr))


def evaluate_model(model, lectures, device, max_slides, temperature):
    model.eval()
    top1, top3, mrr = [], [], []
    with torch.no_grad():
        for lecture, item in lectures.items():
            x = torch.from_numpy(make_features(item["scores"], item["time_norm"], max_slides, temperature)).unsqueeze(0).to(device)
            delta = model(x)[0].cpu()
            base_scores = torch.from_numpy(item["scores"])
            fused = base_scores + delta[:, :base_scores.shape[1]]
            order = torch.argsort(fused, dim=1, descending=True)
            target = torch.from_numpy(item["labels"])
            t1 = (order[:, 0] == target).float().numpy()
            t3 = (order[:, :min(3, order.shape[1])] == target[:, None]).any(dim=1).float().numpy()
            ranks = [1.0 / (int((order[i] == target[i]).nonzero(as_tuple=False)[0].item()) + 1) for i in range(len(target))]
            top1.extend(t1.tolist()); top3.extend(t3.tolist()); mrr.extend(ranks)
            print(f"  {lecture}: Top1={np.mean(t1):.4f} Top3={np.mean(t3):.4f} MRR={np.mean(ranks):.4f}")
    return float(np.mean(top1)), float(np.mean(top3)), float(np.mean(mrr))


def train_epoch(model, loader, optimizer, device, residual_reg, max_slides):
    model.train()
    total = 0.0
    count = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        delta = model(x)
        probs = x[:, :, :max_slides].clamp_min(1e-6)
        base_logits = torch.log(probs)
        fused = base_logits + delta
        loss_raw = F.cross_entropy(fused.reshape(-1, max_slides), y.reshape(-1), reduction="none")
        valid = mask.reshape(-1)
        loss = (loss_raw * valid).sum() / valid.sum().clamp_min(1.0)
        loss = loss + residual_reg * (delta.pow(2) * mask.unsqueeze(-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(1, count)


def load_cached_or_score(cache, *args):
    if cache.exists():
        data = np.load(cache)
        print(f"  Loaded cache: {cache.name}")
        return {k: data[k] for k in data.files}
    item = score_lecture(*args)
    np.savez_compressed(cache, **item)
    print(f"  Saved cache: {cache.name}")
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--residual-reg", type=float, default=1e-3)
    parser.add_argument("--max-slides", type=int, default=MAX_SLIDES_DEFAULT)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading feature files...")
    text_embeddings = np.load(TEXT_EMB).astype(np.float32)
    ocr_embeddings = np.load(OCR_EMB).astype(np.float32)
    image_embeddings = np.load(IMAGE_EMB).astype(np.float32)
    text_index = load_csv(TEXT_INDEX)
    ocr_index = load_csv(OCR_INDEX)
    image_index = load_csv(IMAGE_INDEX)
    pairs = normalize_pairs(load_csv(PAIRS))
    print(f"Text embeddings : {text_embeddings.shape}")
    print(f"OCR embeddings  : {ocr_embeddings.shape}")
    print(f"Image embeddings: {image_embeddings.shape}")

    text_lookup = make_text_lookup(text_index, len(text_embeddings))
    ocr_lookup = make_slide_lookup(ocr_index, "embedding_row", len(ocr_embeddings))
    image_lookup = make_slide_lookup(image_index, "embedding_idx", len(image_embeddings))
    slides_by_lecture = build_slide_inventory(image_index, ocr_lookup, image_lookup)
    print(f"Text lookup entries : {len(text_lookup)}")
    print(f"OCR lookup entries  : {len(ocr_lookup)}")
    print(f"Image lookup entries: {len(image_lookup)}")

    train_df = pairs[pairs["split"] == "train"].copy()
    val_df = pairs[pairs["split"].isin(["val", "validation"])].copy()
    test_df = pairs[pairs["split"] == "test"].copy()
    print(f"Pairs: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("TEST SET: never used for training or model selection.")

    train_seq = build_positive_sequences(train_df)
    val_seq = build_positive_sequences(val_df)
    max_train_val_slides = max([len(v) for v in slides_by_lecture.values()] + [1])
    if max_train_val_slides > args.max_slides:
        raise RuntimeError(f"Need --max-slides >= {max_train_val_slides}")

    base_model = load_base_model(device)
    train_lectures, val_lectures = {}, {}

    print("\nBuilding frozen base score matrices for TRAIN lectures...")
    for lecture, rows in train_seq.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            print(f"  SKIP {lecture}: no slide inventory")
            continue
        cache = CACHE_DIR / f"train_{lecture}.npz"
        item = load_cached_or_score(
            cache, lecture, rows, slides, text_lookup, ocr_lookup, image_lookup,
            text_embeddings, ocr_embeddings, image_embeddings, base_model, device, args.score_batch
        )
        train_lectures[lecture] = item

    print("\nBuilding frozen base score matrices for VALIDATION lectures...")
    for lecture, rows in val_seq.items():
        slides = slides_by_lecture.get(lecture, [])
        if not slides:
            print(f"  SKIP {lecture}: no slide inventory")
            continue
        cache = CACHE_DIR / f"val_{lecture}.npz"
        item = load_cached_or_score(
            cache, lecture, rows, slides, text_lookup, ocr_lookup, image_lookup,
            text_embeddings, ocr_embeddings, image_embeddings, base_model, device, args.score_batch
        )
        val_lectures[lecture] = item

    if not train_lectures or not val_lectures:
        raise RuntimeError("Train or validation sequences are empty")

    base_top1, base_top3, base_mrr = evaluate_base(val_lectures)
    print(f"\nBASE VALIDATION | Top1={base_top1:.4f} Top3={base_top3:.4f} MRR={base_mrr:.4f}")

    model = TemporalResidualGRU(args.max_slides + 2, args.max_slides, args.hidden, args.layers).to(device)
    dataset = WindowDataset(train_lectures, args.window, args.stride, args.max_slides, args.temperature)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"Temporal training windows: {len(dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    best_mrr = -1.0
    best_top1 = -1.0
    stale = 0

    print("\nStarting V3 temporal residual training...\n")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, args.residual_reg, args.max_slides)
        val_top1, val_top3, val_mrr = evaluate_model(model, val_lectures, device, args.max_slides, args.temperature)
        scheduler.step(val_mrr)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss {loss:.4f} | Val Top1 {val_top1:.4f} | Val Top3 {val_top3:.4f} | Val MRR {val_mrr:.4f} | LR {lr:.2e}")

        improved = (val_mrr > best_mrr + 1e-6) or (abs(val_mrr - best_mrr) <= 1e-6 and val_top1 > best_top1 + 1e-6)
        if improved:
            best_mrr, best_top1, stale = val_mrr, val_top1, 0
            torch.save({
                "model_type": "TemporalResidualGRU",
                "model_state_dict": model.state_dict(),
                "base_model_state_dict": base_model.state_dict(),
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top3": val_top3,
                "val_mrr": val_mrr,
                "base_val_top1": base_top1,
                "base_val_top3": base_top3,
                "base_val_mrr": base_mrr,
                "max_slides": args.max_slides,
                "hidden": args.hidden,
                "layers": args.layers,
                "temperature": args.temperature,
                "window": args.window,
                "stride": args.stride,
                "residual_reg": args.residual_reg,
                "seed": args.seed,
            }, OUT_CHECKPOINT)
            print(f"  Saved best checkpoint -> {OUT_CHECKPOINT}")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("\n================================")
    print("V3 TEMPORAL TRAINING COMPLETE")
    print("================================")
    print(f"Base validation Top-1  : {base_top1:.4f}")
    print(f"Best V3 validation Top-1: {best_top1:.4f}")
    print(f"Best V3 validation MRR  : {best_mrr:.4f}")
    print(f"Checkpoint: {OUT_CHECKPOINT}")
    print("TEST SET: NOT USED for model selection.")


if __name__ == "__main__":
    main()
