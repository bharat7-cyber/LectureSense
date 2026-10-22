import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import LectureSenseRanking


# ============================================================
# Lecture ID mapping
# ============================================================

LECTURE_ID_MAP = {

    "climate_science_policy_MIT":
        "climate_science_policy_MIT2",

    "creating_breakthrough_products":
        "creating_breakthrough_products_MIT",

    "cryptocurrency":
        "cryptocurrency_MIT",

    "deeplearning_goodfellow":
        "deeplearning",

    "image_processing_high_res":
        "image_processing",

    "numerics_hennig":
        "numerics",

    "physics_intro_02":
        "physics",

    "psychology_MIT":
        "psychology",

    "reinforcement_learning_silver":
        "reinforcement_learning",

    "visual_system_MIT":
        "sensory_systems",

    "short_range_mit":
        "short_range",

    "team_dynamics_game_design_MIT":
        "team_dynamics_game_design_mit",

    "ML_for_health_MIT":
        "ML_for_health_MIT",

    "computer_vision_2_2":
        "computer_vision_2_2",

    "cities_and_decarbonization":
        "cities_and_decarbonization",

    "climate_and_cities":
        "climate_and_cities",

    "cognitive_robotics_MIT":
        "cognitive_robotics_MIT",

    "phonetics":
        "phonetics",

    "solar_resource":
        "solar_resource",

    "theory_of_computation":
        "theory_of_computation",
}


# ============================================================
# Ranking Dataset
# ============================================================

class RankingDataset(Dataset):

    def __init__(
        self,
        groups,
        text_embeddings,
        image_embeddings,
        image_lookup
    ):

        self.groups = groups
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.image_lookup = image_lookup

    def __len__(self):

        return len(self.groups)

    def __getitem__(self, idx):

        group = self.groups[idx]

        text_idx = group["embedding_idx"]

        correct_slide = group["correct_slide"]

        candidate_slides = group["candidate_slides"]

        feature_lecture_id = (
            group["feature_lecture_id"]
        )

        image_indices = []

        for slide_id in candidate_slides:

            key = (
                feature_lecture_id,
                int(slide_id)
            )

            if key not in self.image_lookup:
                raise KeyError(
                    f"Missing image embedding for {key}"
                )

            image_indices.append(
                self.image_lookup[key]
            )

        text = torch.tensor(
            self.text_embeddings[text_idx],
            dtype=torch.float32
        )

        images = torch.tensor(
            self.image_embeddings[image_indices],
            dtype=torch.float32
        )

        target = candidate_slides.index(
            correct_slide
        )

        return (
            text,
            images,
            target
        )


# ============================================================
# Prepare data
# ============================================================

def prepare_data(
    pairs,
    text_index,
    image_index
):

    pairs = pairs.copy()

    text_index = text_index.copy()

    image_index = image_index.copy()

    # --------------------------------------------------------
    # Normalize lecture IDs
    # --------------------------------------------------------

    pairs["lecture_id"] = (
        pairs["lecture_id"]
        .astype(str)
        .str.strip()
    )

    text_index["lecture_id"] = (
        text_index["lecture_id"]
        .astype(str)
        .str.strip()
    )

    image_index["lecture_id"] = (
        image_index["lecture_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Normalize sentence text
    # --------------------------------------------------------

    pairs["sentence_text"] = (
        pairs["sentence_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    text_index["sentence_text"] = (
        text_index["sentence_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Normalize slide IDs
    # --------------------------------------------------------

    pairs["slide_id"] = pd.to_numeric(
        pairs["slide_id"],
        errors="coerce"
    )

    image_index["slide_id"] = pd.to_numeric(
        image_index["slide_id"],
        errors="coerce"
    )

    pairs = pairs.dropna(
        subset=["slide_id"]
    ).copy()

    image_index = image_index.dropna(
        subset=["slide_id"]
    ).copy()

    pairs["slide_id"] = (
        pairs["slide_id"].astype(int)
    )

    image_index["slide_id"] = (
        image_index["slide_id"].astype(int)
    )

    # --------------------------------------------------------
    # Convert original lecture ID
    # to feature lecture ID
    # --------------------------------------------------------

    pairs["feature_lecture_id"] = (
        pairs["lecture_id"]
        .map(LECTURE_ID_MAP)
        .fillna(pairs["lecture_id"])
    )

    # --------------------------------------------------------
    # Clean text metadata
    # --------------------------------------------------------

    text_index = (
        text_index[
            [
                "lecture_id",
                "sentence_text",
                "embedding_idx"
            ]
        ]
        .drop_duplicates(
            subset=[
                "lecture_id",
                "sentence_text"
            ]
        )
    )

    # --------------------------------------------------------
    # Clean image metadata
    # --------------------------------------------------------

    image_index = (
        image_index[
            [
                "lecture_id",
                "slide_id"
            ]
        ]
        .drop_duplicates(
            subset=[
                "lecture_id",
                "slide_id"
            ]
        )
        .reset_index(drop=True)
    )

    # Create sequential image embedding indices.
    image_index[
        "image_embedding_idx"
    ] = np.arange(
        len(image_index)
    )

    # --------------------------------------------------------
    # Match sentence → text embedding
    # --------------------------------------------------------

    pairs = pairs.merge(
        text_index,

        left_on=[
            "feature_lecture_id",
            "sentence_text"
        ],

        right_on=[
            "lecture_id",
            "sentence_text"
        ],

        how="left",

        suffixes=(
            "",
            "_text"
        )
    )

    # --------------------------------------------------------
    # Match slide → image embedding
    # --------------------------------------------------------

    pairs = pairs.merge(
        image_index,

        left_on=[
            "feature_lecture_id",
            "slide_id"
        ],

        right_on=[
            "lecture_id",
            "slide_id"
        ],

        how="left",

        suffixes=(
            "",
            "_image"
        )
    )

    # --------------------------------------------------------
    # Check mapping
    # --------------------------------------------------------

    before = len(pairs)

    pairs = pairs.dropna(
        subset=[
            "embedding_idx",
            "image_embedding_idx"
        ]
    ).copy()

    after = len(pairs)

    print(
        f"Embedding mapping: "
        f"{after}/{before} rows matched."
    )

    if after == 0:
        raise RuntimeError(
            "No pairs could be mapped to embeddings."
        )

    pairs["embedding_idx"] = (
        pairs["embedding_idx"]
        .astype(int)
    )

    pairs["image_embedding_idx"] = (
        pairs["image_embedding_idx"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Build image lookup
    # --------------------------------------------------------

    image_lookup = {}

    for _, row in image_index.iterrows():

        key = (
            row["lecture_id"],
            int(row["slide_id"])
        )

        image_lookup[key] = int(
            row["image_embedding_idx"]
        )

    return (
        pairs,
        image_lookup
    )


# ============================================================
# Build sentence-level ranking groups
# ============================================================

def build_groups(
    pairs,
    image_lookup
):

    groups = []

    grouped = pairs.groupby(
        [
            "lecture_id",
            "sentence_idx",
            "sentence_text",
            "split"
        ]
    )

    for (
        lecture_id,
        sentence_idx,
        sentence_text,
        split
    ), group in grouped:

        # A ranking example needs one known
        # correct slide.
        positives = group[
            group["label"] == 1
        ]

        if len(positives) == 0:
            continue

        correct_slide = int(
            positives.iloc[0]["slide_id"]
        )

        feature_lecture_id = (
            group.iloc[0]["feature_lecture_id"]
        )

        # Find all available slides for this lecture.
        available_slides = sorted(
            [
                slide_id

                for (
                    lec,
                    slide_id
                ) in image_lookup.keys()

                if lec == feature_lecture_id
            ]
        )

        if len(available_slides) == 0:
            continue

        if correct_slide not in available_slides:
            continue

        groups.append(
            {
                "lecture_id":
                    lecture_id,

                "feature_lecture_id":
                    feature_lecture_id,

                "sentence_idx":
                    int(sentence_idx),

                "sentence_text":
                    sentence_text,

                "embedding_idx":
                    int(
                        group.iloc[0][
                            "embedding_idx"
                        ]
                    ),

                "correct_slide":
                    correct_slide,

                "candidate_slides":
                    available_slides,

                "split":
                    split
            }
        )

    return groups


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):
    """
    Different lectures contain different numbers of slides.

    Example:

        sample 1 → 60 slides
        sample 2 → 22 slides
        sample 3 → 15 slides

    Therefore normal torch.stack() cannot be used.

    We pad every batch to its maximum slide count and
    create a mask indicating which slide positions are real.
    """

    # --------------------------------------------------------
    # Text embeddings
    # --------------------------------------------------------

    texts = torch.stack(
        [
            item[0]
            for item in batch
        ]
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = torch.tensor(
        [
            item[2]
            for item in batch
        ],
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Maximum number of slides in this batch
    # --------------------------------------------------------

    max_slides = max(
        item[1].shape[0]
        for item in batch
    )

    image_dim = (
        batch[0][1].shape[1]
    )

    # --------------------------------------------------------
    # Padded image tensor
    # --------------------------------------------------------

    images = torch.zeros(
        len(batch),
        max_slides,
        image_dim,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Valid slide mask
    # --------------------------------------------------------

    valid_mask = torch.zeros(
        len(batch),
        max_slides,
        dtype=torch.bool
    )

    # --------------------------------------------------------
    # Fill actual slides
    # --------------------------------------------------------

    for i, item in enumerate(batch):

        sample_images = item[1]

        num_slides = (
            sample_images.shape[0]
        )

        images[
            i,
            :num_slides
        ] = sample_images

        valid_mask[
            i,
            :num_slides
        ] = True

    return (
        texts,
        images,
        targets,
        valid_mask
    )


# ============================================================
# Score a batch
# ============================================================

def calculate_scores(
    model,
    texts,
    images
):

    batch_size = texts.size(0)

    num_slides = images.size(1)

    # Repeat each sentence for every slide.
    text_flat = (
        texts
        .unsqueeze(1)
        .expand(
            -1,
            num_slides,
            -1
        )
        .reshape(
            batch_size * num_slides,
            -1
        )
    )

    image_flat = (
        images
        .reshape(
            batch_size * num_slides,
            -1
        )
    )

    scores = model(
        text_flat,
        image_flat
    )

    scores = scores.reshape(
        batch_size,
        num_slides
    )

    return scores


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total = 0

    top1 = 0

    top3 = 0

    for (
        texts,
        images,
        targets,
        valid_mask
    ) in loader:

        texts = texts.to(
            device,
            non_blocking=True
        )

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        valid_mask = valid_mask.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Calculate slide scores
        # ----------------------------------------------------

        scores = calculate_scores(
            model,
            texts,
            images
        )

        # ----------------------------------------------------
        # Remove padded slide positions
        # ----------------------------------------------------

        scores = scores.masked_fill(
            ~valid_mask,
            -1e9
        )

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        predictions = scores.argmax(
            dim=1
        )

        top1 += (
            predictions == targets
        ).sum().item()

        # ----------------------------------------------------
        # Top-3
        # ----------------------------------------------------

        k = min(
            3,
            scores.size(1)
        )

        top3_predictions = (
            scores.topk(
                k=k,
                dim=1
            ).indices
        )

        top3 += (
            (
                top3_predictions
                == targets.unsqueeze(1)
            )
            .any(dim=1)
            .sum()
            .item()
        )

        total += texts.size(0)

    if total == 0:
        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    return (
        top1 / total,
        top3 / total
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "LectureSense multimodal "
            "ranking training"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    data_dir = Path(
        "data"
    )

    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir /
        "ranking_best_model.pt"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load text embeddings
    # --------------------------------------------------------

    text_embeddings = np.load(
        data_dir /
        "text_embeddings.npy"
    )

    print(
        "Text embeddings:",
        text_embeddings.shape
    )

    # --------------------------------------------------------
    # Load image embeddings
    # --------------------------------------------------------

    image_embeddings = np.load(
        data_dir /
        "image_embeddings.npy"
    )

    print(
        "Image embeddings:",
        image_embeddings.shape
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    text_index = pd.read_csv(
        data_dir /
        "text_embedding_index.csv"
    )

    image_index = pd.read_csv(
        data_dir /
        "image_embedding_index.csv"
    )

    pairs = pd.read_csv(
        data_dir /
        "pairs.csv"
    )

    print(
        "Total pairs:",
        len(pairs)
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    pairs, image_lookup = prepare_data(
        pairs,
        text_index,
        image_index
    )

    # --------------------------------------------------------
    # Build ranking groups
    # --------------------------------------------------------

    groups = build_groups(
        pairs,
        image_lookup
    )

    print()
    print(
        "Ranking groups:",
        len(groups)
    )

    train_groups = [
        g
        for g in groups
        if g["split"] == "train"
    ]

    val_groups = [
        g
        for g in groups
        if g["split"] == "val"
    ]

    test_groups = [
        g
        for g in groups
        if g["split"] == "test"
    ]

    print(
        "Train groups:",
        len(train_groups)
    )

    print(
        "Validation groups:",
        len(val_groups)
    )

    print(
        "Test groups:",
        len(test_groups)
    )

    if len(train_groups) == 0:
        raise RuntimeError(
            "Training groups are empty."
        )

    if len(val_groups) == 0:
        raise RuntimeError(
            "Validation groups are empty."
        )

    if len(test_groups) == 0:
        raise RuntimeError(
            "Test groups are empty."
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RankingDataset(
        train_groups,
        text_embeddings,
        image_embeddings,
        image_lookup
    )

    val_dataset = RankingDataset(
        val_groups,
        text_embeddings,
        image_embeddings,
        image_lookup
    )

    test_dataset = RankingDataset(
        test_groups,
        text_embeddings,
        image_embeddings,
        image_lookup
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LectureSenseRanking().to(
        device
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # Best validation score
    # --------------------------------------------------------

    best_val_top1 = -1.0

    print()
    print(
        "Starting ranking training..."
    )

    print()

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        running_loss = 0.0

        total_samples = 0

        for (
            texts,
            images,
            targets,
            valid_mask
        ) in train_loader:

            texts = texts.to(
                device,
                non_blocking=True
            )

            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            valid_mask = valid_mask.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            scores = calculate_scores(
                model,
                texts,
                images
            )

            # ------------------------------------------------
            # Mask padded slides
            # ------------------------------------------------

            scores = scores.masked_fill(
                ~valid_mask,
                -1e9
            )

            # ------------------------------------------------
            # Ranking loss
            #
            # Cross entropy over candidate slides means
            # the correct slide is explicitly encouraged
            # to have the highest score.
            # ------------------------------------------------

            loss = F.cross_entropy(
                scores,
                targets
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_size = texts.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss /
            total_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_top1, val_top3 = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(
            val_top1
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Top-1: "
            f"{val_top1:.4f} | "
            f"Val Top-3: "
            f"{val_top3:.4f} | "
            f"LR: "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_top1 > best_val_top1:

            best_val_top1 = val_top1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "text_dim":
                        384,

                    "image_dim":
                        512,

                    "hidden_dim":
                        256
                },
                checkpoint_path
            )

            print(
                "  Saved "
                "ranking_best_model.pt"
            )

    # ========================================================
    # Final test evaluation
    # ========================================================

    print()

    print(
        "Loading best ranking model..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_top1, test_top3 = evaluate(
        model,
        test_loader,
        device
    )

    print()

    print(
        "=" * 70
    )

    print(
        "LECTURESENSE RANKING TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Top-1 Accuracy: "
        f"{test_top1:.4f}"
    )

    print(
        f"Top-3 Accuracy: "
        f"{test_top3:.4f}"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Model saved to:"
    )

    print(
        checkpoint_path
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()