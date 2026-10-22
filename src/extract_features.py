from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms


ROOT = Path("data/MaViLS")

GT_DIR = ROOT / "data" / "ground_truth_files"

SLIDE_DIR = ROOT / "slide_images"

DATA_DIR = Path("data")


LECTURE_FOLDER_MAP = {
    "cities_and_decarbonization": "cities&decarbonization",
    "climate_and_cities": "cities&climate",
    "climate_science_policy_MIT2": "climate_science_policy_MIT",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT",
    "computer_vision_2_2": "computer_vision_2_2_geiger",
    "creating_breakthrough_products_MIT": "creating_breakthrough_products",
    "cryptocurrency_MIT": "cryptocurrency",
    "deeplearning": "deeplearning_goodfellow",
    "image_processing": "image_processing",
    "image_processing_high_res": "image_processing",
    "ML_for_health_MIT": "ML_for_health_care_MIT",
    "numerics": "numerics",
    "numerics_hennig": "numerics",
    "phonetics": "phonetics",
    "physics": "physics_intro_02",
    "physics_intro_02": "physics_intro_02",
    "psychology": "psychology",
    "psychology_MIT": "psychology",
    "reinforcement_learning": "reinforcement_learning",
    "reinforcement_learning_silver": "reinforcement_learning",
    "sensory_systems": "sensory_systems",
    "visual_system_MIT": "sensory_systems",
    "short_range": "short_range_mit",
    "short_range_mit": "short_range_mit",
    "solar_resource": "solar_resource",
    "team_dynamics": "team_dynamics_game_design_mit",
    "team_dynamics_game_design_mit": "team_dynamics_game_design_mit",
    "theory_of_computation": "theory_of_computation",
}


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_ground_truth():

    files = sorted(
        GT_DIR.glob("ground_truth_*.xlsx")
    )

    records = []

    for path in files:

        name = path.stem

        lecture_id = None

        for key in LECTURE_FOLDER_MAP:

            key_clean = key.lower()

            name_clean = name.lower()

            if name_clean.endswith(key_clean):
                lecture_id = key
                break

        if lecture_id is None:
            continue

        try:
            df = pd.read_excel(path)
        except Exception:
            continue

        time_col = None

        if "Key" in df.columns:
            time_col = "Key"
        elif "Time" in df.columns:
            time_col = "Time"

        if time_col is None:
            continue

        if "Slidenumber" not in df.columns:
            continue

        if "Value" not in df.columns:
            continue

        for _, row in df.iterrows():

            timestamp = row[time_col]

            slide_id = row["Slidenumber"]

            sentence = clean_text(
                row["Value"]
            )

            if pd.isna(timestamp):
                continue

            try:
                timestamp = float(timestamp)
            except Exception:
                continue

            try:
                slide_id = int(slide_id)
            except Exception:
                continue

            records.append(
                {
                    "lecture_id": lecture_id,
                    "sentence_text": sentence,
                    "timestamp": timestamp,
                    "slide_id": slide_id,
                }
            )

    return pd.DataFrame(records)


def encode_text(gt):

    print(
        f"Encoding {gt['sentence_text'].nunique()} "
        "unique transcript sentences with Sentence-BERT..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    unique_sentences = (
        gt["sentence_text"]
        .drop_duplicates()
        .tolist()
    )

    embeddings = model.encode(
        unique_sentences,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    sentence_to_idx = {
        sentence: i
        for i, sentence in enumerate(
            unique_sentences
        )
    }

        text_index = []

    for idx, row in gt.iterrows():

        text_index.append(
            {
                "lecture_id": row["lecture_id"],
                "sentence_text": row["sentence_text"],
                "embedding_idx":
                    sentence_to_idx[row["sentence_text"]]
            }
        )

    text_index = pd.DataFrame(
        text_index
    ).drop_duplicates(
        subset=["lecture_id", "sentence_text"]
    )

    np.save(
        DATA_DIR / "text_embeddings.npy",
        embeddings
    )

    text_index.to_csv(
        DATA_DIR / "text_embedding_index.csv",
        index=False
    )

    print(
        "Saved text_embeddings.npy",
        embeddings.shape
    )

    print(
        "Saved text_embedding_index.csv"
    )


def get_slide_path(
    lecture_id,
    slide_id
):

    folder_name = LECTURE_FOLDER_MAP.get(
        lecture_id
    )

    if folder_name is None:
        return None

    folder = SLIDE_DIR / folder_name

    if not folder.exists():
        return None

    candidates = [
        folder / f"slide_{slide_id}.png",
        folder / f"slide_{slide_id}.jpg",
        folder / f"slide_{slide_id}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def encode_images(gt):

    print("Loading ResNet-18...")

    weights = (
        models.ResNet18_Weights.DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    model.fc = torch.nn.Identity()

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    preprocess = weights.transforms()

    unique_slides = (
        gt[
            ["lecture_id", "slide_id"]
        ]
        .drop_duplicates()
    )

    embeddings = []

    metadata = []

    missing = []

    for _, row in unique_slides.iterrows():

        lecture_id = row["lecture_id"]

        slide_id = int(
            row["slide_id"]
        )

        if slide_id < 1:
            continue

        path = get_slide_path(
            lecture_id,
            slide_id
        )

        if path is None:

            missing.append(
                f"{lecture_id}::{slide_id}"
            )

            embeddings.append(
                np.zeros(512,
                         dtype=np.float32)
            )

            metadata.append(
                {
                    "lecture_id":
                        lecture_id,
                    "slide_id":
                        slide_id
                }
            )

            continue

        try:

            image = Image.open(
                path
            ).convert("RGB")

            tensor = preprocess(
                image
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                feature = model(
                    tensor
                )

            feature = (
                feature
                .cpu()
                .numpy()[0]
            )

            embeddings.append(
                feature.astype(
                    np.float32
                )
            )

            metadata.append(
                {
                    "lecture_id":
                        lecture_id,
                    "slide_id":
                        slide_id
                }
            )

        except Exception as e:

            print(
                "Image error:",
                path,
                e
            )

            embeddings.append(
                np.zeros(
                    512,
                    dtype=np.float32
                )
            )

            metadata.append(
                {
                    "lecture_id":
                        lecture_id,
                    "slide_id":
                        slide_id
                }
            )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    metadata = pd.DataFrame(
        metadata
    )

    np.save(
        DATA_DIR / "image_embeddings.npy",
        embeddings
    )

    metadata.to_csv(
        DATA_DIR /
        "image_embedding_index.csv",
        index=False
    )

    print(
        "Saved image_embeddings.npy",
        embeddings.shape
    )

    print(
        "Saved image_embedding_index.csv"
    )

    print(
        "Missing slide images:",
        len(missing)
    )

    if missing:

        print(
            "First missing examples:"
        )

        for x in missing[:20]:
            print(" ", x)

    else:

        print(
            "SUCCESS: All slide images were found."
        )


def main():

    DATA_DIR.mkdir(
        exist_ok=True
    )

    print("Loading MaViLS ground truth...")

    gt = load_ground_truth()

    print(
        "Ground-truth rows:",
        len(gt)
    )

    print(
        "Lectures:",
        gt["lecture_id"].nunique()
    )

    encode_text(gt)

    encode_images(gt)


if __name__ == "__main__":
    main()