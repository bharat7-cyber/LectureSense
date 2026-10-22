import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import LectureSenseFusion


# ------------------------------------------------------------
# Lecture ID mapping
# pairs.csv uses the original MaViLS/SRT names.
# Feature metadata uses the normalized names created by
# extract_features.py.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class PairDataset(Dataset):

    def __init__(
        self,
        df,
        text_embeddings,
        image_embeddings
    ):

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

        self.text_indices = (
            df["embedding_idx"]
            .astype(int)
            .to_numpy()
        )

        self.image_indices = (
            df["image_embedding_idx"]
            .astype(int)
            .to_numpy()
        )

        self.labels = (
            df["label"]
            .astype(np.float32)
            .to_numpy()
        )

        print(
            "Usable samples:",
            len(self.labels)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        text = torch.tensor(
            self.text_embeddings[
                self.text_indices[idx]
            ],
            dtype=torch.float32
        )

        image = torch.tensor(
            self.image_embeddings[
                self.image_indices[idx]
            ],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return text, image, label


# ------------------------------------------------------------
# Prepare embedding indices
# ------------------------------------------------------------

def prepare_pairs(
    pairs,
    text_index,
    image_index
):

    pairs = pairs.copy()
    text_index = text_index.copy()
    image_index = image_index.copy()

    # -----------------------------
    # Normalize lecture IDs
    # -----------------------------

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

    # -----------------------------
    # Normalize sentence text
    # -----------------------------

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

    # -----------------------------
    # Normalize slide IDs
    # -----------------------------

    pairs["slide_id"] = (
        pd.to_numeric(
            pairs["slide_id"],
            errors="coerce"
        )
    )

    image_index["slide_id"] = (
        pd.to_numeric(
            image_index["slide_id"],
            errors="coerce"
        )
    )

    # Remove invalid slide IDs.
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

    # -----------------------------
    # Convert pairs lecture ID
    # to feature lecture ID
    # -----------------------------

    pairs["feature_lecture_id"] = (
        pairs["lecture_id"]
        .map(LECTURE_ID_MAP)
        .fillna(pairs["lecture_id"])
    )

    # -----------------------------
    # Remove duplicate metadata
    # -----------------------------

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

    # -----------------------------
    # Assign image embedding row
    # -----------------------------

    image_index["image_embedding_idx"] = (
        np.arange(len(image_index))
    )

    # -----------------------------
    # Text embedding merge
    # -----------------------------

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

    # -----------------------------
    # Image embedding merge
    # -----------------------------

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

    # -----------------------------
    # Check matching
    # -----------------------------

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

    # Convert indices to integers.
    pairs["embedding_idx"] = (
        pairs["embedding_idx"]
        .astype(int)
    )

    pairs["image_embedding_idx"] = (
        pairs["image_embedding_idx"]
        .astype(int)
    )

    # -----------------------------
    # Check split distribution
    # -----------------------------

    print()
    print("Mapped split distribution:")

    print(
        pairs["split"]
        .value_counts()
        .to_string()
    )

    print()

    return pairs


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():

        for (
            text,
            image,
            labels
        ) in loader:

            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            logits = model(
                text,
                image
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * len(labels)
            )

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += len(labels)

    if total == 0:

        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    return (
        total_loss / total,
        correct / total
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    # -----------------------------
    # Paths
    # -----------------------------

    data_dir = Path("data")

    checkpoint_dir = Path(
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    # -----------------------------
    # Device
    # -----------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # -----------------------------
    # Load embeddings
    # -----------------------------

    text_embeddings = np.load(
        data_dir /
        "text_embeddings.npy"
    )

    image_embeddings = np.load(
        data_dir /
        "image_embeddings.npy"
    )

    # -----------------------------
    # Load metadata
    # -----------------------------

    text_index = pd.read_csv(
        data_dir /
        "text_embedding_index.csv"
    )

    image_index = pd.read_csv(
        data_dir /
        "image_embedding_index.csv"
    )

    # -----------------------------
    # Load training pairs
    # -----------------------------

    pairs = pd.read_csv(
        data_dir /
        "pairs.csv"
    )

    print(
        "Text embeddings:",
        text_embeddings.shape
    )

    print(
        "Image embeddings:",
        image_embeddings.shape
    )

    print(
        "Total pairs:",
        len(pairs)
    )

    print(
        "Positive:",
        int(pairs["label"].sum())
    )

    print(
        "Negative:",
        int(
            (pairs["label"] == 0)
            .sum()
        )
    )

    print()

    # -----------------------------
    # Map pairs to embeddings
    # -----------------------------

    pairs = prepare_pairs(
        pairs,
        text_index,
        image_index
    )

    # -----------------------------
    # Split
    # -----------------------------

    train_df = pairs[
        pairs["split"] == "train"
    ].copy()

    val_df = pairs[
        pairs["split"] == "val"
    ].copy()

    test_df = pairs[
        pairs["split"] == "test"
    ].copy()

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # Safety checks.
    if len(train_df) == 0:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_df) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if len(test_df) == 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    # -----------------------------
    # Datasets
    # -----------------------------

    train_dataset = PairDataset(
        train_df,
        text_embeddings,
        image_embeddings
    )

    val_dataset = PairDataset(
        val_df,
        text_embeddings,
        image_embeddings
    )

    test_dataset = PairDataset(
        test_df,
        text_embeddings,
        image_embeddings
    )

    # -----------------------------
    # DataLoaders
    # -----------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------
    # Model
    # -----------------------------

    model = (
        LectureSenseFusion()
        .to(device)
    )

    # -----------------------------
    # Class imbalance
    # -----------------------------

    positive = (
        train_df["label"]
        .sum()
    )

    negative = (
        len(train_df)
        - positive
    )

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # -----------------------------
    # Optimizer
    # -----------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # -----------------------------
    # Training
    # -----------------------------

    best_val_loss = float(
        "inf"
    )

    print()
    print(
        "Starting training..."
    )
    print()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        running_loss = 0.0
        total = 0

        for (
            text,
            image,
            labels
        ) in train_loader:

            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                text,
                image
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * len(labels)
            )

            total += len(labels)

        train_loss = (
            running_loss /
            total
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{args.epochs} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_acc:.4f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "text_dim":
                        384,

                    "image_dim":
                        512
                },
                checkpoint_dir /
                "best_model.pt"
            )

            print(
                "  Saved "
                "best_model.pt"
            )

    # -----------------------------
    # Final test evaluation
    # -----------------------------

    print()
    print(
        "Loading best model..."
    )

    checkpoint = torch.load(
        checkpoint_dir /
        "best_model.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print(
        "Final Test Results"
    )

    print(
        "------------------"
    )

    print(
        "Test Loss:",
        round(test_loss, 4)
    )

    print(
        "Test Accuracy:",
        round(test_acc, 4)
    )

    print()

    print(
        "Model saved to:"
    )

    print(
        "checkpoints/"
        "best_model.pt"
    )


if __name__ == "__main__":
    main()
