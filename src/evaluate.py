import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import LectureSenseFusion


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


# ============================================================
# LECTURE ID MAPPING
# ============================================================

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
    "team_dynamics_game_design_MIT": "team_dynamics_game_design_mit",
    "theory_of_computation": "theory_of_computation",
    "visual_system_MIT": "sensory_systems",
}


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_slide(value):
    try:
        return int(float(value))
    except Exception:
        return int(value)


# ============================================================
# DATASET
# ============================================================

class EvalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        text_embeddings,
        image_embeddings,
    ):
        self.df = dataframe.reset_index(drop=True)

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        text_vector = self.text_embeddings[
            int(row["embedding_idx"])
        ]

        image_vector = self.image_embeddings[
            int(row["image_embedding_idx"])
        ]

        label = float(row["label"])

        return (
            torch.tensor(
                text_vector,
                dtype=torch.float32,
            ),
            torch.tensor(
                image_vector,
                dtype=torch.float32,
            ),
            torch.tensor(
                label,
                dtype=torch.float32,
            ),
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, labels):

    predictions = np.asarray(
        predictions
    ).astype(int)

    labels = np.asarray(
        labels
    ).astype(int)

    tp = int(
        (
            (predictions == 1)
            & (labels == 1)
        ).sum()
    )

    fp = int(
        (
            (predictions == 1)
            & (labels == 0)
        ).sum()
    )

    fn = int(
        (
            (predictions == 0)
            & (labels == 1)
        ).sum()
    )

    tn = int(
        (
            (predictions == 0)
            & (labels == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    accuracy = (
        (tp + tn) / len(labels)
        if len(labels) > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
        accuracy,
    )


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_pairs(
    pairs,
    text_index,
    image_index,
):

    pairs = pairs.copy()

    # --------------------------------------------------------
    # Normalize pair data
    # --------------------------------------------------------

    pairs["lecture_id"] = (
        pairs["lecture_id"]
        .astype(str)
        .str.strip()
    )

    pairs["sentence_text"] = (
        pairs["sentence_text"]
        .apply(normalize_text)
    )

    pairs["slide_id"] = (
        pairs["slide_id"]
        .apply(normalize_slide)
    )

    # --------------------------------------------------------
    # Map pair lecture IDs to feature lecture IDs
    # --------------------------------------------------------

    pairs["feature_lecture_id"] = (
        pairs["lecture_id"]
        .map(LECTURE_ID_MAP)
        .fillna(
            pairs["lecture_id"]
        )
    )

    # ========================================================
    # TEXT INDEX
    # ========================================================

    text_index = text_index.copy()

    text_index["lecture_id"] = (
        text_index["lecture_id"]
        .astype(str)
        .str.strip()
    )

    text_index["sentence_text"] = (
        text_index["sentence_text"]
        .apply(normalize_text)
    )

    text_index = (
        text_index
        .drop_duplicates(
            subset=[
                "lecture_id",
                "sentence_text",
            ]
        )
        .copy()
    )

    text_index = text_index[
        [
            "lecture_id",
            "sentence_text",
            "embedding_idx",
        ]
    ]

    # ========================================================
    # IMAGE INDEX
    # ========================================================

    image_index = image_index.copy()

    image_index["lecture_id"] = (
        image_index["lecture_id"]
        .astype(str)
        .str.strip()
    )

    image_index["slide_id"] = (
        image_index["slide_id"]
        .apply(normalize_slide)
    )

    image_index = (
        image_index
        .drop_duplicates(
            subset=[
                "lecture_id",
                "slide_id",
            ]
        )
        .copy()
    )

    # IMPORTANT:
    # image_embeddings.npy rows correspond to the rows
    # of image_embedding_index.csv.
    image_index = (
        image_index
        .reset_index(drop=True)
    )

    image_index[
        "image_embedding_idx"
    ] = np.arange(
        len(image_index)
    )

    image_index = image_index[
        [
            "lecture_id",
            "slide_id",
            "image_embedding_idx",
        ]
    ]

    # ========================================================
    # MERGE TEXT EMBEDDINGS
    # ========================================================

    pairs = pairs.merge(
        text_index,
        how="left",
        left_on=[
            "feature_lecture_id",
            "sentence_text",
        ],
        right_on=[
            "lecture_id",
            "sentence_text",
        ],
        suffixes=(
            "",
            "_text",
        ),
    )

    # ========================================================
    # MERGE IMAGE EMBEDDINGS
    # ========================================================

    pairs = pairs.merge(
        image_index,
        how="left",
        left_on=[
            "feature_lecture_id",
            "slide_id",
        ],
        right_on=[
            "lecture_id",
            "slide_id",
        ],
        suffixes=(
            "",
            "_image",
        ),
    )

    before_count = len(pairs)

    # --------------------------------------------------------
    # Remove samples without embeddings
    # --------------------------------------------------------

    pairs = pairs.dropna(
        subset=[
            "embedding_idx",
            "image_embedding_idx",
        ]
    ).copy()

    pairs["embedding_idx"] = (
        pairs["embedding_idx"]
        .astype(int)
    )

    pairs["image_embedding_idx"] = (
        pairs["image_embedding_idx"]
        .astype(int)
    )

    after_count = len(pairs)

    print(
        f"Test pairs before mapping: {before_count}"
    )

    print(
        f"Test pairs after mapping:  {after_count}"
    )

    print(
        f"Test pairs removed:        "
        f"{before_count - after_count}"
    )

    return pairs


# ============================================================
# FUSION MODEL EVALUATION
# ============================================================

def evaluate_fusion(
    model,
    test_df,
    text_embeddings,
    image_embeddings,
    device,
):

    dataset = EvalDataset(
        test_df,
        text_embeddings,
        image_embeddings,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    all_probabilities = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for (
            text,
            image,
            labels,
        ) in loader:

            text = text.to(device)

            image = image.to(device)

            logits = model(
                text,
                image,
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

            all_labels.extend(
                labels
                .numpy()
                .tolist()
            )

    probabilities = np.asarray(
        all_probabilities
    )

    labels = np.asarray(
        all_labels
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return calculate_metrics(
        predictions,
        labels,
    )


# ============================================================
# MAJORITY REFERENCE
# ============================================================

def evaluate_majority_reference(
    test_df,
):

    labels = (
        test_df["label"]
        .astype(int)
        .to_numpy()
    )

    if len(labels) == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    majority_class = int(
        pd.Series(labels)
        .mode()[0]
    )

    predictions = np.full_like(
        labels,
        majority_class,
    )

    return calculate_metrics(
        predictions,
        labels,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            CHECKPOINT_DIR
            / "best_model.pt"
        ),
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    pairs = pd.read_csv(
        DATA_DIR / "pairs.csv"
    )

    text_embeddings = np.load(
        DATA_DIR
        / "text_embeddings.npy"
    )

    image_embeddings = np.load(
        DATA_DIR
        / "image_embeddings.npy"
    )

    text_index = pd.read_csv(
        DATA_DIR
        / "text_embedding_index.csv"
    )

    image_index = pd.read_csv(
        DATA_DIR
        / "image_embedding_index.csv"
    )

    print(
        f"Text embeddings:  "
        f"{text_embeddings.shape}"
    )

    print(
        f"Image embeddings: "
        f"{image_embeddings.shape}"
    )

    print(
        f"All pairs:        "
        f"{len(pairs)}"
    )

    # ========================================================
    # TEST SPLIT
    # ========================================================

    test_df = pairs[
        pairs["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    print(
        f"Raw test pairs:    "
        f"{len(test_df)}"
    )

    test_df = prepare_test_pairs(
        test_df,
        text_index,
        image_index,
    )

    if len(test_df) == 0:

        raise RuntimeError(
            "No usable test samples "
            "remain after mapping."
        )

    print(
        f"Usable test pairs: "
        f"{len(test_df)}"
    )

    print(
        "\nTest label distribution:"
    )

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = LectureSenseFusion().to(
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Your training script saved a dictionary
    # containing model_state_dict.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        "\nLoaded checkpoint:"
    )

    print(
        checkpoint_path
    )

    # ========================================================
    # FUSION EVALUATION
    # ========================================================

    print(
        "\nEvaluating multimodal "
        "fusion model..."
    )

    (
        precision_fusion,
        recall_fusion,
        f1_fusion,
        accuracy_fusion,
    ) = evaluate_fusion(
        model,
        test_df,
        text_embeddings,
        image_embeddings,
        device,
    )

    # ========================================================
    # REFERENCE BASELINE
    # ========================================================

    print(
        "Evaluating reference baseline..."
    )

    (
        precision_reference,
        recall_reference,
        f1_reference,
        accuracy_reference,
    ) = evaluate_majority_reference(
        test_df
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "LECTURESENSE TEST RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Model':<28}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
        f"{'Accuracy':>12}"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Multimodal Fusion':<28}"
        f"{precision_fusion:>12.4f}"
        f"{recall_fusion:>12.4f}"
        f"{f1_fusion:>10.4f}"
        f"{accuracy_fusion:>12.4f}"
    )

    print(
        f"{'Majority Reference':<28}"
        f"{precision_reference:>12.4f}"
        f"{recall_reference:>12.4f}"
        f"{f1_reference:>10.4f}"
        f"{accuracy_reference:>12.4f}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    results = pd.DataFrame(
        [
            {
                "model": "Multimodal Fusion",
                "precision": precision_fusion,
                "recall": recall_fusion,
                "f1": f1_fusion,
                "accuracy": accuracy_fusion,
            },
            {
                "model": "Majority Reference",
                "precision": precision_reference,
                "recall": recall_reference,
                "f1": f1_reference,
                "accuracy": accuracy_reference,
            },
        ]
    )

    results_path = (
        RESULTS_DIR
        / "evaluation_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )


if __name__ == "__main__":
    main()