"""
LectureSense V2: Hard-Negative + Ranking-Loss training for the 4-modal fusion model.

Purpose
-------
Train the existing LectureSense cross-modal scoring architecture using grouped
ranking examples instead of independent binary pairs.  For each transcript
sentence with a known ground-truth slide, the model sees:

    [positive slide] + [hard negatives from the SAME lecture]

Hard negatives are selected using:
    1) immediate neighboring slides (slide-1 / slide+1), and
    2) high frozen multimodal similarity negatives.

Loss:
    CrossEntropy over the candidate slides
    + margin-ranking loss between the positive and each negative.

Validation is lecture-wise and TEST IS NOT USED for model selection.
The output checkpoint keeps the same parameter names/shapes as the current
LectureSenseCrossModal model used by inference.py.

Run from project root:
    python src\train_hard_negative_4modal.py

Recommended first run on RTX 2050 4GB:
    python src\train_hard_negative_4modal.py --epochs 20 --batch-size 32

Output:
    checkpoints/ranking_4modal_hardneg_best_model.pt
"""

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

TEXT_EMB = DATA_DIR / "text_embeddings.npy"
OCR_EMB = DATA_DIR / "ocr_embeddings.npy"
IMAGE_EMB = DATA_DIR / "image_embeddings.npy"
TEXT_INDEX = DATA_DIR / "text_embedding_index.csv"
OCR_INDEX = DATA_DIR / "ocr_embedding_index.csv"
IMAGE_INDEX = DATA_DIR / "image_embedding_index.csv"
PAIRS = DATA_DIR / "pairs.csv"

OUT_CKPT = CHECKPOINT_DIR / "ranking_4modal_hardneg_best_model.pt"

TEXT_DIM = 384
OCR_DIM = 384
IMAGE_DIM = 512
HIDDEN_DIM = 256

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


# ---------------------------------------------------------------------------
# Exact compatible model architecture
# ---------------------------------------------------------------------------
class LectureSenseCrossModal(nn.Module):
    """Architecture compatible with the current ranking_4modal checkpoint."""

    def __init__(self):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.ocr_proj = nn.Sequential(
            nn.Linear(OCR_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # 9 * 256 + 4 cosine features = 2308
        self.scorer = nn.Sequential(
            nn.Linear(2308, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, text: torch.Tensor, ocr: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text)
        o = self.ocr_proj(ocr)
        i = self.image_proj(image)

        t_norm = F.normalize(t, dim=-1)
        o_norm = F.normalize(o, dim=-1)
        i_norm = F.normalize(i, dim=-1)

        native_to = torch.sum(
            F.normalize(text, dim=-1) * F.normalize(ocr, dim=-1),
            dim=-1,
            keepdim=True,
        )
        projected_to = torch.sum(t_norm * o_norm, dim=-1, keepdim=True)
        projected_ti = torch.sum(t_norm * i_norm, dim=-1, keepdim=True)
        projected_oi = torch.sum(o_norm * i_norm, dim=-1, keepdim=True)

        fusion = torch.cat(
            [
                t,
                o,
                i,
                torch.abs(t - o),
                t * o,
                torch.abs(t - i),
                t * i,
                torch.abs(o - i),
                o * i,
                native_to,
                projected_to,
                projected_ti,
                projected_oi,
            ],
            dim=-1,
        )

        return self.scorer(fusion).squeeze(-1)


def normalize_sentence(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s]", "", x)
    return x.strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def make_text_lookup(text_index: pd.DataFrame, n_text: int):
    df = text_index.copy()
    df["lecture_id"] = df["lecture_id"].astype(str).str.strip()

    sentence_col = "sentence_norm" if "sentence_norm" in df.columns else "sentence_text"
    if sentence_col not in df.columns:
        raise ValueError("text_embedding_index.csv must contain sentence_norm or sentence_text")

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
    """Build (lecture, slide) -> embedding-row lookup.

    OCR metadata stores an explicit ``embedding_row`` column.
    Image metadata in this project does not; its row order is the
    embedding order used by the original extraction/inference code.
    Therefore, for image lookups we derive the embedding index from
    the CSV row index when ``embedding_idx`` is absent.
    """
    df = index.copy()

    if "lecture_id" not in df.columns or "slide_id" not in df.columns:
        raise ValueError(
            "Slide embedding index must contain lecture_id and slide_id columns."
        )

    if emb_key not in df.columns:
        if emb_key == "embedding_idx":
            # The existing image extraction/inference pipeline uses the
            # dataframe row order as the image_embeddings row index.
            df = df.copy()
            df["__derived_embedding_idx"] = np.arange(len(df), dtype=np.int64)
            actual_emb_key = "__derived_embedding_idx"
        else:
            raise ValueError(
                f"{Path(str(IMAGE_INDEX)).name if 'IMAGE_INDEX' in globals() else 'index'} "
                f"does not contain required embedding column '{emb_key}'. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        actual_emb_key = emb_key

    reverse = build_reverse_lecture_map()
    df["lecture_id"] = (
        df["lecture_id"]
        .astype(str)
        .str.strip()
        .map(lambda x: reverse.get(x, x))
    )
    df["slide_id"] = pd.to_numeric(df["slide_id"], errors="coerce")
    df[actual_emb_key] = pd.to_numeric(df[actual_emb_key], errors="coerce")
    df = df.dropna(subset=["slide_id", actual_emb_key]).copy()
    df["slide_id"] = df["slide_id"].astype(int)
    df[actual_emb_key] = df[actual_emb_key].astype(int)
    df = df[(df[actual_emb_key] >= 0) & (df[actual_emb_key] < n_emb)]

    lookup = {}
    # Use iterrows here because pandas itertuples() can rename columns whose
    # names begin with underscores, which would break getattr() for the
    # derived image embedding index.
    for _, row in df.iterrows():
        lookup[(str(row["lecture_id"]), int(row["slide_id"]))] = int(row[actual_emb_key])
    return lookup


def build_reverse_lecture_map() -> Dict[str, str]:
    return {v: k for k, v in LECTURE_ID_MAP.items()}


def normalize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs.copy()
    df["lecture_id"] = df["lecture_id"].astype(str).str.strip()
    df["sentence_text"] = df["sentence_text"].fillna("").astype(str)
    df["sentence_norm"] = df["sentence_text"].map(normalize_sentence)
    df["slide_id"] = pd.to_numeric(df["slide_id"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["slide_id", "label"]).copy()
    df["slide_id"] = df["slide_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df["feature_lecture_id"] = df["lecture_id"].map(LECTURE_ID_MAP).fillna(df["lecture_id"])
    return df


def build_slide_inventory(image_index: pd.DataFrame) -> Dict[str, List[int]]:
    inv = defaultdict(set)
    reverse = build_reverse_lecture_map()
    for row in image_index.itertuples(index=False):
        feature_lecture = str(row.lecture_id).strip()
        pair_lecture = reverse.get(feature_lecture, feature_lecture)
        try:
            slide_id = int(row.slide_id)
        except Exception:
            continue
        inv[pair_lecture].add(slide_id)
    return {k: sorted(v) for k, v in inv.items()}


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def build_hard_negative_groups(
    df: pd.DataFrame,
    text_lookup: Dict[Tuple[str, str], int],
    ocr_lookup: Dict[Tuple[str, int], int],
    image_lookup: Dict[Tuple[str, int], int],
    text_embeddings: np.ndarray,
    ocr_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    slides_by_lecture: Dict[str, List[int]],
    hard_k: int,
    semantic_pool: int,
    seed: int,
) -> List[dict]:
    """Build one group per positive ground-truth sentence.

    Negatives include neighboring slides plus the highest frozen similarity
    non-matching slides.  All negatives are from the same lecture.
    """
    rng = random.Random(seed)
    positives = df[df["label"] == 1].copy()

    groups = []
    seen = set()

    for row in positives.itertuples(index=False):
        pair_lecture = str(row.lecture_id)
        feature_lecture = str(row.feature_lecture_id)
        sentence_norm = str(row.sentence_norm)
        true_slide = int(row.slide_id)

        text_idx = text_lookup.get((feature_lecture, sentence_norm))
        if text_idx is None:
            continue

        ocr_true_idx = ocr_lookup.get((pair_lecture, true_slide))
        image_true_idx = image_lookup.get((pair_lecture, true_slide))
        candidates = slides_by_lecture.get(pair_lecture, [])

        if ocr_true_idx is None or image_true_idx is None or true_slide not in candidates:
            continue

        # Prefer sentence_idx to distinguish repeated phrases if available.
        sentence_key = (
            pair_lecture,
            int(getattr(row, "sentence_idx", -1)),
            sentence_norm,
            true_slide,
        )
        if sentence_key in seen:
            continue
        seen.add(sentence_key)

        neighboring = []
        for s in (true_slide - 1, true_slide + 1):
            if s in candidates and s != true_slide:
                neighboring.append(s)

        # Compute frozen multimodal similarity for non-neighbors.
        text_vec = text_embeddings[text_idx]
        scored = []
        pool = [s for s in candidates if s != true_slide]

        if len(pool) > semantic_pool:
            # Deterministic subsample to bound preprocessing time.
            pool = rng.sample(pool, semantic_pool)

        for s in pool:
            oi = ocr_lookup.get((pair_lecture, s))
            ii = image_lookup.get((pair_lecture, s))
            if oi is None or ii is None:
                continue
            text_ocr = cosine_np(text_vec, ocr_embeddings[oi])
            # Text and image embeddings live in different vector spaces
            # (384-D vs 512-D), so they cannot be compared directly.
            # Use the true slide's image as the visual anchor and measure
            # visual similarity against the candidate slide image instead.
            image_true_vec = image_embeddings[image_true_idx]
            image_img = cosine_np(image_true_vec, image_embeddings[ii])
            # Text/OCR semantic similarity is the primary signal; visual
            # similarity supplies an additional hard-negative signal.
            hard_score = 0.65 * text_ocr + 0.35 * image_img
            scored.append((hard_score, s))

        scored.sort(reverse=True)
        semantic_hard = [s for _, s in scored]

        negatives = []
        for s in neighboring + semantic_hard:
            if s not in negatives:
                negatives.append(s)
            if len(negatives) >= hard_k:
                break

        # Fill remaining slots randomly from the lecture if necessary.
        if len(negatives) < hard_k:
            remaining = [s for s in candidates if s != true_slide and s not in negatives]
            rng.shuffle(remaining)
            negatives.extend(remaining[: hard_k - len(negatives)])

        negatives = negatives[:hard_k]
        if not negatives:
            continue

        groups.append(
            {
                "lecture_id": pair_lecture,
                "feature_lecture_id": feature_lecture,
                "sentence_norm": sentence_norm,
                "text_idx": int(text_idx),
                "true_slide": true_slide,
                "neg_slides": negatives,
            }
        )

    return groups


class RankingGroupDataset(Dataset):
    def __init__(
        self,
        groups: List[dict],
        text_embeddings: np.ndarray,
        ocr_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        ocr_lookup: Dict[Tuple[str, int], int],
        image_lookup: Dict[Tuple[str, int], int],
    ):
        self.groups = groups
        self.text_embeddings = text_embeddings
        self.ocr_embeddings = ocr_embeddings
        self.image_embeddings = image_embeddings
        self.ocr_lookup = ocr_lookup
        self.image_lookup = image_lookup

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        lecture = g["lecture_id"]
        slides = [g["true_slide"]] + list(g["neg_slides"])

        text = self.text_embeddings[g["text_idx"]].astype(np.float32)
        ocr_rows = []
        image_rows = []
        valid_slides = []

        for slide in slides:
            oi = self.ocr_lookup.get((lecture, slide))
            ii = self.image_lookup.get((lecture, slide))
            if oi is None or ii is None:
                continue
            ocr_rows.append(self.ocr_embeddings[oi].astype(np.float32))
            image_rows.append(self.image_embeddings[ii].astype(np.float32))
            valid_slides.append(int(slide))

        if g["true_slide"] not in valid_slides or len(valid_slides) < 2:
            raise RuntimeError(f"Invalid candidate group for {lecture}, slide {g['true_slide']}")

        target = valid_slides.index(g["true_slide"])
        return {
            "text": torch.tensor(text, dtype=torch.float32),
            "ocr": torch.tensor(np.asarray(ocr_rows), dtype=torch.float32),
            "image": torch.tensor(np.asarray(image_rows), dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
            "slides": valid_slides,
        }


def collate_groups(batch):
    if not batch:
        raise RuntimeError("Empty batch")

    b = len(batch)
    k = max(x["ocr"].shape[0] for x in batch)

    text = torch.stack([x["text"] for x in batch], dim=0)
    ocr = torch.zeros(b, k, OCR_DIM, dtype=torch.float32)
    image = torch.zeros(b, k, IMAGE_DIM, dtype=torch.float32)
    mask = torch.zeros(b, k, dtype=torch.bool)
    target = torch.zeros(b, dtype=torch.long)
    slide_lists = []

    for i, x in enumerate(batch):
        n = x["ocr"].shape[0]
        ocr[i, :n] = x["ocr"]
        image[i, :n] = x["image"]
        mask[i, :n] = True
        target[i] = x["target"]
        slide_lists.append(x["slides"])

    return {
        "text": text,
        "ocr": ocr,
        "image": image,
        "mask": mask,
        "target": target,
        "slides": slide_lists,
    }


def pairwise_ranking_loss(scores: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    """Positive score must exceed each valid negative by margin."""
    losses = []
    for i in range(scores.shape[0]):
        pos_idx = int(target[i].item())
        pos = scores[i, pos_idx]
        neg = torch.cat([scores[i, :pos_idx], scores[i, pos_idx + 1 :]], dim=0)
        if neg.numel() == 0:
            continue
        y = torch.ones_like(neg)
        losses.append(F.margin_ranking_loss(pos.expand_as(neg), neg, y, margin=margin))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def masked_scores(model, batch, device):
    b, k, _ = batch["ocr"].shape
    text = batch["text"].unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1).to(device)
    ocr = batch["ocr"].reshape(b * k, -1).to(device)
    image = batch["image"].reshape(b * k, -1).to(device)
    logits = model(text, ocr, image).reshape(b, k)

    mask = batch["mask"].to(device)
    logits = logits.masked_fill(~mask, -1e9)
    return logits


def ranking_metrics(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    top1_hits = []
    top3_hits = []
    reciprocal_ranks = []

    with torch.no_grad():
        for batch in loader:
            scores = masked_scores(model, batch, device)
            targets = batch["target"].to(device)

            sorted_idx = torch.argsort(scores, dim=1, descending=True)
            top1 = sorted_idx[:, 0]
            top1_hits.extend((top1 == targets).float().cpu().tolist())

            top3 = sorted_idx[:, : min(3, scores.shape[1])]
            top3_hits.extend((top3 == targets.unsqueeze(1)).any(dim=1).float().cpu().tolist())

            for row in range(scores.shape[0]):
                order = sorted_idx[row].tolist()
                target_idx = int(targets[row].item())
                rank = order.index(target_idx) + 1
                reciprocal_ranks.append(1.0 / rank)

    if not top1_hits:
        return 0.0, 0.0, 0.0
    return float(np.mean(top1_hits)), float(np.mean(top3_hits)), float(np.mean(reciprocal_ranks))


def train_one_epoch(model, loader, optimizer, device, margin, rank_weight):
    model.train()
    running = 0.0
    batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        scores = masked_scores(model, batch, device)
        targets = batch["target"].to(device)

        ce = F.cross_entropy(scores, targets)
        rank = pairwise_ranking_loss(scores, targets, margin)
        loss = ce + rank_weight * rank

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running += float(loss.item())
        batches += 1

    return running / max(1, batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-k", type=int, default=4,
                        help="number of hard negatives per positive")
    parser.add_argument("--semantic-pool", type=int, default=80,
                        help="max non-positive slides scored during hard-negative construction")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warm-start", action="store_true",
                        help="initialize from checkpoints/ranking_4modal_best_model.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------
    # Load features + metadata
    # ------------------------------------------------------------
    print("Loading feature files...")
    text_embeddings = np.load(TEXT_EMB).astype(np.float32)
    ocr_embeddings = np.load(OCR_EMB).astype(np.float32)
    image_embeddings = np.load(IMAGE_EMB).astype(np.float32)

    text_index = load_table(TEXT_INDEX)
    ocr_index = load_table(OCR_INDEX)
    image_index = load_table(IMAGE_INDEX)
    pairs = normalize_pairs(load_table(PAIRS))

    print(f"Text embeddings : {text_embeddings.shape}")
    print(f"OCR embeddings  : {ocr_embeddings.shape}")
    print(f"Image embeddings: {image_embeddings.shape}")

    text_lookup = make_text_lookup(text_index, len(text_embeddings))
    ocr_lookup = make_slide_lookup(ocr_index, "embedding_row", len(ocr_embeddings))
    image_lookup = make_slide_lookup(image_index, "embedding_idx", len(image_embeddings))
    slides_by_lecture = build_slide_inventory(image_index)

    print(f"Text lookup entries : {len(text_lookup)}")
    print(f"OCR lookup entries  : {len(ocr_lookup)}")
    print(f"Image lookup entries: {len(image_lookup)}")

    # ------------------------------------------------------------
    # Lecture-wise split. TEST IS NEVER USED FOR TRAINING/SELECTION.
    # ------------------------------------------------------------
    train_df = pairs[pairs["split"] == "train"].copy()
    val_df = pairs[pairs["split"].isin(["val", "validation"])].copy()
    test_df = pairs[pairs["split"] == "test"].copy()

    print(f"Pairs: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("\nBuilding hard-negative TRAIN groups...")
    train_groups = build_hard_negative_groups(
        train_df,
        text_lookup,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings,
        slides_by_lecture,
        hard_k=args.hard_k,
        semantic_pool=args.semantic_pool,
        seed=args.seed,
    )

    print("Building hard-negative VALIDATION groups...")
    val_groups = build_hard_negative_groups(
        val_df,
        text_lookup,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings,
        slides_by_lecture,
        hard_k=args.hard_k,
        semantic_pool=args.semantic_pool,
        seed=args.seed + 1,
    )

    if not train_groups:
        raise RuntimeError("No training ranking groups were built.")
    if not val_groups:
        raise RuntimeError("No validation ranking groups were built.")

    print(f"Train ranking groups: {len(train_groups)}")
    print(f"Val ranking groups  : {len(val_groups)}")

    train_ds = RankingGroupDataset(
        train_groups, text_embeddings, ocr_embeddings, image_embeddings, ocr_lookup, image_lookup
    )
    val_ds = RankingGroupDataset(
        val_groups, text_embeddings, ocr_embeddings, image_embeddings, ocr_lookup, image_lookup
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_groups,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_groups,
    )

    model = LectureSenseCrossModal().to(device)

    if args.warm_start:
        warm = CHECKPOINT_DIR / "ranking_4modal_best_model.pt"
        if not warm.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {warm}")
        checkpoint = torch.load(warm, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Warm-started from: {warm}")
        if missing or unexpected:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_mrr = -1.0
    best_top1 = -1.0
    epochs_without_improvement = 0

    print("\nStarting V2 hard-negative ranking training...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            margin=args.margin,
            rank_weight=args.rank_weight,
        )

        val_top1, val_top3, val_mrr = ranking_metrics(model, val_loader, device)
        scheduler.step(val_mrr)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Val Top1 {val_top1:.4f} | "
            f"Val Top3 {val_top3:.4f} | "
            f"Val MRR {val_mrr:.4f} | LR {lr_now:.2e}"
        )

        improved = (val_mrr > best_mrr + 1e-6) or (
            abs(val_mrr - best_mrr) <= 1e-6 and val_top1 > best_top1 + 1e-6
        )

        if improved:
            best_mrr = val_mrr
            best_top1 = val_top1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "val_mrr": val_mrr,
                    "text_dim": TEXT_DIM,
                    "ocr_dim": OCR_DIM,
                    "image_dim": IMAGE_DIM,
                    "loss": "cross_entropy + weighted_margin_ranking",
                    "hard_k": args.hard_k,
                    "margin": args.margin,
                    "rank_weight": args.rank_weight,
                    "seed": args.seed,
                },
                OUT_CKPT,
            )
            print(f"  Saved best checkpoint -> {OUT_CKPT}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("\n=============================")
    print("V2 TRAINING COMPLETE")
    print("=============================")
    print(f"Best validation Top-1: {best_top1:.4f}")
    print(f"Best validation MRR  : {best_mrr:.4f}")
    print(f"Checkpoint: {OUT_CKPT}")
    print("TEST SET: NOT USED for model selection.")


if __name__ == "__main__":
    main()
