# ============================================================
# LectureSense
# Improved Cross-Modal Ranking Model
#
# Transcript + Slide OCR + Slide Image
#
# Main improvement:
#   - Native SBERT Transcript <-> OCR cosine similarity
#   - Learned projected cross-modal similarities
#   - Multimodal learned fusion
#
# ============================================================

import os
import re
import random
import argparse
import unicodedata

from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TEXT_DIM = 384
OCR_DIM = 384
IMAGE_DIM = 512

HIDDEN_DIM = 256

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 4

CHECKPOINT_PATH = (
    "checkpoints/ranking_4modal_best_model.pt"
)


# ============================================================
# LECTURE ID MAPPING
# ============================================================

LECTURE_MAP = {

    "ML_for_health_MIT":
        "ML_for_health_MIT",

    "cities_and_decarbonization":
        "cities_and_decarbonization",

    "climate_and_cities":
        "climate_and_cities",

    "climate_science_policy_MIT":
        "climate_science_policy_MIT2",

    "cognitive_robotics_MIT":
        "cognitive_robotics_MIT",

    "computer_vision_2_2":
        "computer_vision_2_2",

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

    "phonetics":
        "phonetics",

    "physics_intro_02":
        "physics",

    "psychology_MIT":
        "psychology",

    "reinforcement_learning_silver":
        "reinforcement_learning",

    "short_range_mit":
        "short_range",

    "solar_resource":
        "solar_resource",

    "team_dynamics_game_design_MIT":
        "team_dynamics_game_design_mit",

    "theory_of_computation":
        "theory_of_computation",

    "visual_system_MIT":
        "sensory_systems",
}


# Feature lecture ID -> pairs lecture ID

REVERSE_LECTURE_MAP = {
    feature_id: pair_id
    for pair_id, feature_id
    in LECTURE_MAP.items()
}


# ============================================================
# NORMALIZE SENTENCE
# ============================================================

def normalize_sentence(text):

    if pd.isna(text):
        return None

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("Device: cuda")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device("cpu")

        print("Device: cpu")

    return device


# ============================================================
# MODEL
# ============================================================

class LectureSenseCrossModal(
    nn.Module
):

    def __init__(
        self,
        text_dim=TEXT_DIM,
        ocr_dim=OCR_DIM,
        image_dim=IMAGE_DIM,
        hidden_dim=HIDDEN_DIM
    ):

        super().__init__()

        # ----------------------------------------------------
        # Transcript projection
        # ----------------------------------------------------

        self.text_proj = nn.Sequential(

            nn.Linear(
                text_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )


        # ----------------------------------------------------
        # OCR projection
        # ----------------------------------------------------

        self.ocr_proj = nn.Sequential(

            nn.Linear(
                ocr_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )


        # ----------------------------------------------------
        # Image projection
        # ----------------------------------------------------

        self.image_proj = nn.Sequential(

            nn.Linear(
                image_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )


        # ----------------------------------------------------
        # Fusion features
        #
        # t
        # o
        # i
        # |t-o|
        # t*o
        # |t-i|
        # t*i
        # |o-i|
        # o*i
        #
        # = 9 x 256
        #
        # Plus:
        #
        # native SBERT cosine(t,o)
        # projected cosine(t,o)
        # projected cosine(t,i)
        # projected cosine(o,i)
        #
        # = 4 scalar features
        #
        # Total = 2308
        # ----------------------------------------------------

        fusion_dim = (
            hidden_dim * 9
            + 4
        )


        # ----------------------------------------------------
        # Learned compatibility scorer
        # ----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.Linear(
                fusion_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(
                128,
                1
            )
        )


    def forward(
        self,
        text,
        ocr,
        image
    ):

        # ====================================================
        # Native transcript-OCR semantic similarity
        #
        # This is calculated BEFORE projection because both
        # transcript and OCR use Sentence-BERT embeddings.
        # ====================================================

        text_native = F.normalize(
            text,
            p=2,
            dim=-1
        )

        ocr_native = F.normalize(
            ocr,
            p=2,
            dim=-1
        )

        native_text_ocr_cosine = torch.sum(
            text_native * ocr_native,
            dim=-1,
            keepdim=True
        )


        # ====================================================
        # Learned projections
        # ====================================================

        t = self.text_proj(
            text
        )

        o = self.ocr_proj(
            ocr
        )

        i = self.image_proj(
            image
        )


        # ====================================================
        # Normalize projected representations
        # ====================================================

        t_norm = F.normalize(
            t,
            p=2,
            dim=-1
        )

        o_norm = F.normalize(
            o,
            p=2,
            dim=-1
        )

        i_norm = F.normalize(
            i,
            p=2,
            dim=-1
        )


        # ====================================================
        # Learned cross-modal cosine similarities
        # ====================================================

        text_ocr_cosine = torch.sum(
            t_norm * o_norm,
            dim=-1,
            keepdim=True
        )

        text_image_cosine = torch.sum(
            t_norm * i_norm,
            dim=-1,
            keepdim=True
        )

        ocr_image_cosine = torch.sum(
            o_norm * i_norm,
            dim=-1,
            keepdim=True
        )


        # ====================================================
        # Multimodal feature fusion
        # ====================================================

        fusion = torch.cat(
            [

                # Original representations
                t,
                o,
                i,

                # Transcript <-> OCR
                torch.abs(
                    t - o
                ),

                t * o,

                # Transcript <-> Image
                torch.abs(
                    t - i
                ),

                t * i,

                # OCR <-> Image
                torch.abs(
                    o - i
                ),

                o * i,

                # Similarity features
                native_text_ocr_cosine,

                text_ocr_cosine,

                text_image_cosine,

                ocr_image_cosine

            ],
            dim=-1
        )


        # ====================================================
        # Compatibility score
        # ====================================================

        score = self.scorer(
            fusion
        ).squeeze(-1)


        return score


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\nLoading embeddings..."
    )


    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    text_embeddings = np.load(
        "data/text_embeddings.npy"
    )

    ocr_embeddings = np.load(
        "data/ocr_embeddings.npy"
    )

    image_embeddings = np.load(
        "data/image_embeddings.npy"
    )


    print(
        "Text embeddings :",
        text_embeddings.shape
    )

    print(
        "OCR embeddings  :",
        ocr_embeddings.shape
    )

    print(
        "Image embeddings:",
        image_embeddings.shape
    )


    # ========================================================
    # TEXT INDEX
    # ========================================================

    text_index = pd.read_csv(
        "data/text_embedding_index.csv"
    )


    print(
        "Text index rows :",
        len(text_index)
    )


    # --------------------------------------------------------
    # Important:
    #
    # text index has more rows than embedding rows.
    # embedding_idx is authoritative.
    # --------------------------------------------------------

    text_index = text_index.dropna(
        subset=[
            "lecture_id",
            "sentence_text",
            "embedding_idx"
        ]
    ).copy()


    text_index["lecture_id"] = (
        text_index[
            "lecture_id"
        ].astype(str)
    )


    text_index["sentence_norm"] = (
        text_index[
            "sentence_text"
        ].map(
            normalize_sentence
        )
    )


    text_index["embedding_idx"] = (
        text_index[
            "embedding_idx"
        ].astype(int)
    )


    # Keep valid embedding rows only

    text_index = text_index[
        (
            text_index[
                "embedding_idx"
            ] >= 0
        )
        &
        (
            text_index[
                "embedding_idx"
            ] < len(
                text_embeddings
            )
        )
    ].copy()


    # --------------------------------------------------------
    # Build transcript lookup
    # --------------------------------------------------------

    text_lookup = {}


    for row in text_index.itertuples(
        index=False
    ):

        key = (
            row.lecture_id,
            row.sentence_norm
        )

        text_lookup[key] = int(
            row.embedding_idx
        )


    print(
        "Usable text lookup entries:",
        len(text_lookup)
    )


    # ========================================================
    # OCR INDEX
    # ========================================================

    ocr_index = pd.read_csv(
        "data/ocr_embedding_index.csv"
    )


    # Feature ID -> pair ID

    ocr_index["lecture_id"] = (
        ocr_index[
            "lecture_id"
        ]
        .astype(str)
        .map(
            lambda x:
            REVERSE_LECTURE_MAP.get(
                x,
                x
            )
        )
    )


    ocr_index["slide_id"] = (
        ocr_index[
            "slide_id"
        ].astype(int)
    )


    ocr_index["embedding_row"] = (
        ocr_index[
            "embedding_row"
        ].astype(int)
    )


    ocr_lookup = {}


    for row in ocr_index.itertuples(
        index=False
    ):

        key = (
            row.lecture_id,
            int(row.slide_id)
        )

        ocr_lookup[key] = int(
            row.embedding_row
        )


    print(
        "OCR lookup entries:",
        len(ocr_lookup)
    )


    # ========================================================
    # IMAGE INDEX
    # ========================================================

    image_index = pd.read_csv(
        "data/image_embedding_index.csv"
    )


    # Feature ID -> pair ID

    image_index["lecture_id"] = (
        image_index[
            "lecture_id"
        ]
        .astype(str)
        .map(
            lambda x:
            REVERSE_LECTURE_MAP.get(
                x,
                x
            )
        )
    )


    image_index["slide_id"] = (
        image_index[
            "slide_id"
        ].astype(int)
    )


    image_lookup = {}


    for idx, row in image_index.iterrows():

        key = (
            row["lecture_id"],
            int(row["slide_id"])
        )

        image_lookup[key] = int(
            idx
        )


    print(
        "Image lookup entries:",
        len(image_lookup)
    )


    # ========================================================
    # PAIRS
    # ========================================================

    pairs = pd.read_csv(
        "data/pairs.csv"
    )


    print(
        "Pairs rows:",
        len(pairs)
    )


    pairs["lecture_id"] = (
        pairs[
            "lecture_id"
        ].astype(str)
    )


    pairs["sentence_text"] = (
        pairs[
            "sentence_text"
        ].astype(str)
    )


    pairs["slide_id"] = (
        pairs[
            "slide_id"
        ].astype(int)
    )


    pairs["label"] = (
        pairs[
            "label"
        ].astype(int)
    )


    return (
        text_embeddings,
        ocr_embeddings,
        image_embeddings,
        text_lookup,
        ocr_lookup,
        image_lookup,
        pairs
    )


# ============================================================
# BUILD RANKING GROUPS
# ============================================================

def build_ranking_groups(
    pairs,
    text_lookup,
    ocr_lookup,
    image_lookup
):

    print(
        "\nBuilding ranking groups..."
    )


    # --------------------------------------------------------
    # Find all slides available for each lecture
    # --------------------------------------------------------

    slides_by_lecture = (
        defaultdict(set)
    )


    for lecture_id, slide_id in (
        image_lookup.keys()
    ):

        slides_by_lecture[
            lecture_id
        ].add(
            int(slide_id)
        )


    for lecture_id in slides_by_lecture:

        slides_by_lecture[
            lecture_id
        ] = sorted(
            slides_by_lecture[
                lecture_id
            ]
        )


    # --------------------------------------------------------
    # Positive examples only
    # --------------------------------------------------------

    positives = pairs[
        pairs["label"] == 1
    ].copy()


    groups = []

    skipped_text = 0

    skipped_slide = 0

    seen = set()


    # --------------------------------------------------------
    # One ranking group per positive sentence
    # --------------------------------------------------------

    for row in positives.itertuples(
        index=False
    ):

        pair_lecture_id = str(
            row.lecture_id
        )

        sentence_text = str(
            row.sentence_text
        )

        true_slide = int(
            row.slide_id
        )


        # ----------------------------------------------------
        # Pair ID -> feature ID
        # ----------------------------------------------------

        feature_lecture_id = (
            LECTURE_MAP.get(
                pair_lecture_id,
                pair_lecture_id
            )
        )


        # ----------------------------------------------------
        # Normalize transcript
        # ----------------------------------------------------

        sentence_norm = (
            normalize_sentence(
                sentence_text
            )
        )


        # ----------------------------------------------------
        # Transcript lookup
        # ----------------------------------------------------

        text_key = (
            feature_lecture_id,
            sentence_norm
        )


        if text_key not in text_lookup:

            skipped_text += 1

            continue


        text_idx = text_lookup[
            text_key
        ]


        # ----------------------------------------------------
        # True slide image
        # ----------------------------------------------------

        image_key = (
            pair_lecture_id,
            true_slide
        )


        if image_key not in image_lookup:

            skipped_slide += 1

            continue


        # ----------------------------------------------------
        # True slide OCR
        # ----------------------------------------------------

        ocr_key = (
            pair_lecture_id,
            true_slide
        )


        if ocr_key not in ocr_lookup:

            skipped_slide += 1

            continue


        # ----------------------------------------------------
        # Candidate slides
        # ----------------------------------------------------

        candidates = (
            slides_by_lecture.get(
                pair_lecture_id,
                []
            )
        )


        if true_slide not in candidates:

            skipped_slide += 1

            continue


        # ----------------------------------------------------
        # Remove duplicate groups
        # ----------------------------------------------------

        group_key = (
            pair_lecture_id,
            sentence_norm,
            true_slide
        )


        if group_key in seen:

            continue


        seen.add(
            group_key
        )


        groups.append(
            {
                "lecture_id":
                    pair_lecture_id,

                "sentence_text":
                    sentence_text,

                "sentence_norm":
                    sentence_norm,

                "text_idx":
                    text_idx,

                "true_slide":
                    true_slide,

                "candidate_slides":
                    candidates
            }
        )


    print(
        "Ranking groups:",
        len(groups)
    )


    print(
        "Skipped because of text:",
        skipped_text
    )


    print(
        "Skipped because of slide:",
        skipped_slide
    )


    return groups


# ============================================================
# DATASET
# ============================================================

class RankingDataset(
    Dataset
):

    def __init__(
        self,
        groups,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings
    ):

        self.groups = groups

        self.ocr_lookup = (
            ocr_lookup
        )

        self.image_lookup = (
            image_lookup
        )

        self.text_embeddings = (
            text_embeddings
        )

        self.ocr_embeddings = (
            ocr_embeddings
        )

        self.image_embeddings = (
            image_embeddings
        )


    def __len__(self):

        return len(
            self.groups
        )


    def __getitem__(
        self,
        idx
    ):

        group = self.groups[
            idx
        ]


        lecture_id = group[
            "lecture_id"
        ]


        text_idx = group[
            "text_idx"
        ]


        candidate_slides = group[
            "candidate_slides"
        ]


        true_slide = group[
            "true_slide"
        ]


        # ----------------------------------------------------
        # Transcript embedding
        # ----------------------------------------------------

        text_embedding = (
            self.text_embeddings[
                text_idx
            ]
            .astype(
                np.float32
            )
        )


        ocr_list = []

        image_list = []

        valid_slides = []


        # ----------------------------------------------------
        # Candidate slide features
        # ----------------------------------------------------

        for slide_id in candidate_slides:

            ocr_key = (
                lecture_id,
                slide_id
            )

            image_key = (
                lecture_id,
                slide_id
            )


            if ocr_key not in self.ocr_lookup:

                continue


            if image_key not in self.image_lookup:

                continue


            ocr_idx = self.ocr_lookup[
                ocr_key
            ]


            image_idx = self.image_lookup[
                image_key
            ]


            ocr_list.append(
                self.ocr_embeddings[
                    ocr_idx
                ].astype(
                    np.float32
                )
            )


            image_list.append(
                self.image_embeddings[
                    image_idx
                ].astype(
                    np.float32
                )
            )


            valid_slides.append(
                slide_id
            )


        # ----------------------------------------------------
        # Ensure true slide exists
        # ----------------------------------------------------

        if true_slide not in valid_slides:

            raise RuntimeError(
                f"True slide {true_slide} "
                f"missing for lecture "
                f"{lecture_id}"
            )


        # Target index

        target = valid_slides.index(
            true_slide
        )


        return {

            "text":
                torch.tensor(
                    text_embedding,
                    dtype=torch.float32
                ),

            "ocr":
                torch.tensor(
                    np.asarray(
                        ocr_list,
                        dtype=np.float32
                    ),
                    dtype=torch.float32
                ),

            "image":
                torch.tensor(
                    np.asarray(
                        image_list,
                        dtype=np.float32
                    ),
                    dtype=torch.float32
                ),

            "target":
                torch.tensor(
                    target,
                    dtype=torch.long
                ),

            "slides":
                valid_slides
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    batch_size = len(
        batch
    )


    max_slides = max(
        item["ocr"].shape[0]
        for item in batch
    )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text_batch = torch.stack(
        [
            item["text"]
            for item in batch
        ]
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    ocr_batch = torch.zeros(
        batch_size,
        max_slides,
        OCR_DIM,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_batch = torch.zeros(
        batch_size,
        max_slides,
        IMAGE_DIM,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Candidate mask
    # --------------------------------------------------------

    valid_mask = torch.zeros(
        batch_size,
        max_slides,
        dtype=torch.bool
    )


    targets = []

    slide_lists = []


    # --------------------------------------------------------
    # Fill batch
    # --------------------------------------------------------

    for i, item in enumerate(
        batch
    ):

        n_slides = (
            item["ocr"].shape[0]
        )


        ocr_batch[
            i,
            :n_slides
        ] = item["ocr"]


        image_batch[
            i,
            :n_slides
        ] = item["image"]


        valid_mask[
            i,
            :n_slides
        ] = True


        targets.append(
            item["target"]
        )


        slide_lists.append(
            item["slides"]
        )


    targets = torch.stack(
        targets
    )


    return {

        "text":
            text_batch,

        "ocr":
            ocr_batch,

        "image":
            image_batch,

        "valid_mask":
            valid_mask,

        "target":
            targets,

        "slides":
            slide_lists
    }


# ============================================================
# CALCULATE SCORES
# ============================================================

def calculate_scores(
    model,
    text,
    ocr,
    image,
    valid_mask
):

    batch_size = (
        text.shape[0]
    )

    num_slides = (
        ocr.shape[1]
    )


    # --------------------------------------------------------
    # Repeat transcript for every slide candidate
    # --------------------------------------------------------

    text_expanded = (
        text
        .unsqueeze(1)
        .expand(
            batch_size,
            num_slides,
            -1
        )
        .reshape(
            batch_size * num_slides,
            -1
        )
    )


    # Flatten OCR

    ocr_flat = (
        ocr.reshape(
            batch_size * num_slides,
            -1
        )
    )


    # Flatten images

    image_flat = (
        image.reshape(
            batch_size * num_slides,
            -1
        )
    )


    # --------------------------------------------------------
    # Model scoring
    # --------------------------------------------------------

    scores = model(
        text_expanded,
        ocr_flat,
        image_flat
    )


    scores = scores.reshape(
        batch_size,
        num_slides
    )


    # --------------------------------------------------------
    # Mask padding
    # --------------------------------------------------------

    scores = scores.masked_fill(
        ~valid_mask,
        -1e9
    )


    return scores


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True
):

    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    total_samples = 0

    top1_count = 0

    top3_count = 0

    reciprocal_rank_sum = 0.0


    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for batch in loader:

        text = batch[
            "text"
        ].to(
            device,
            non_blocking=True
        )


        ocr = batch[
            "ocr"
        ].to(
            device,
            non_blocking=True
        )


        image = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )


        valid_mask = batch[
            "valid_mask"
        ].to(
            device
        )


        targets = batch[
            "target"
        ].to(
            device
        )


        batch_size = (
            text.shape[0]
        )


        # ----------------------------------------------------
        # Zero gradients
        # ----------------------------------------------------

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )


        # ----------------------------------------------------
        # Forward + loss
        # ----------------------------------------------------

        with torch.set_grad_enabled(
            training
        ):

            scores = calculate_scores(
                model,
                text,
                ocr,
                image,
                valid_mask
            )


            loss = criterion(
                scores,
                targets
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            if training:

                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )


                optimizer.step()


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * batch_size
        )


        total_samples += (
            batch_size
        )


        # ----------------------------------------------------
        # Ranking metrics
        # ----------------------------------------------------

        with torch.no_grad():

            rankings = torch.argsort(
                scores,
                dim=1,
                descending=True
            )


            for i in range(
                batch_size
            ):

                target = targets[
                    i
                ].item()


                ranking = rankings[
                    i
                ]


                positions = (
                    ranking == target
                ).nonzero(
                    as_tuple=False
                )


                if positions.numel() == 0:

                    continue


                rank = (
                    positions[
                        0
                    ].item()
                    + 1
                )


                if rank == 1:

                    top1_count += 1


                if rank <= 3:

                    top3_count += 1


                reciprocal_rank_sum += (
                    1.0 / rank
                )


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if total_samples == 0:

        raise RuntimeError(
            "No samples were processed."
        )


    # --------------------------------------------------------
    # Epoch metrics
    # --------------------------------------------------------

    average_loss = (
        total_loss
        / total_samples
    )


    top1 = (
        top1_count
        / total_samples
    )


    top3 = (
        top3_count
        / total_samples
    )


    mrr = (
        reciprocal_rank_sum
        / total_samples
    )


    return (
        average_loss,
        top1,
        top3,
        mrr
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Train LectureSense "
            "improved cross-modal ranking model."
        )
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed()


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()


    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        text_embeddings,
        ocr_embeddings,
        image_embeddings,
        text_lookup,
        ocr_lookup,
        image_lookup,
        pairs
    ) = load_data()


    # ========================================================
    # BUILD GROUPS
    # ========================================================

    groups = build_ranking_groups(
        pairs,
        text_lookup,
        ocr_lookup,
        image_lookup
    )


    if len(groups) == 0:

        raise RuntimeError(
            "No ranking groups were created."
        )


    # ========================================================
    # LECTURE-WISE SPLIT
    # ========================================================

    lecture_split = (
        pairs[
            [
                "lecture_id",
                "split"
            ]
        ]
        .drop_duplicates(
            "lecture_id"
        )
        .set_index(
            "lecture_id"
        )[
            "split"
        ]
        .to_dict()
    )


    train_groups = []

    val_groups = []

    test_groups = []


    for group in groups:

        lecture_id = group[
            "lecture_id"
        ]


        split = lecture_split.get(
            lecture_id
        )


        if split == "train":

            train_groups.append(
                group
            )

        elif split == "val":

            val_groups.append(
                group
            )

        elif split == "test":

            test_groups.append(
                group
            )


    print(
        "\nTrain groups:",
        len(train_groups)
    )


    print(
        "Val groups:",
        len(val_groups)
    )


    print(
        "Test groups:",
        len(test_groups)
    )


    if len(train_groups) == 0:

        raise RuntimeError(
            "Training groups = 0."
        )


    if len(val_groups) == 0:

        raise RuntimeError(
            "Validation groups = 0."
        )


    if len(test_groups) == 0:

        raise RuntimeError(
            "Test groups = 0."
        )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = RankingDataset(
        train_groups,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings
    )


    val_dataset = RankingDataset(
        val_groups,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings
    )


    test_dataset = RankingDataset(
        test_groups,
        ocr_lookup,
        image_lookup,
        text_embeddings,
        ocr_embeddings,
        image_embeddings
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    pin_memory = (
        torch.cuda.is_available()
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = LectureSenseCrossModal().to(
        device
    )


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )


    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_val_top1 = -1.0

    epochs_without_improvement = 0


    print(
        "\nStarting training..."
    )


    print(
        "Epochs:",
        args.epochs
    )


    print(
        "Batch size:",
        args.batch_size
    )


    print()


    for epoch in range(
        1,
        args.epochs + 1
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            train_loss,
            train_top1,
            train_top3,
            train_mrr
        ) = run_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device,

            training=True
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_top1,
            val_top3,
            val_mrr
        ) = run_epoch(

            model,

            val_loader,

            optimizer,

            criterion,

            device,

            training=False
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs} "
            f"| Train Loss {train_loss:.4f} "
            f"| Train Top1 {train_top1:.4f} "
            f"| Train Top3 {train_top3:.4f} "
            f"| Train MRR {train_mrr:.4f} "
            f"| Val Loss {val_loss:.4f} "
            f"| Val Top1 {val_top1:.4f} "
            f"| Val Top3 {val_top3:.4f} "
            f"| Val MRR {val_mrr:.4f} "
            f"| LR {current_lr:.6f}"
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_top1
        )


        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_top1 > best_val_top1:

            best_val_top1 = (
                val_top1
            )

            epochs_without_improvement = 0


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_top1":
                        val_top1,

                    "val_top3":
                        val_top3,

                    "val_mrr":
                        val_mrr,

                    "text_dim":
                        TEXT_DIM,

                    "ocr_dim":
                        OCR_DIM,

                    "image_dim":
                        IMAGE_DIM,

                    "hidden_dim":
                        HIDDEN_DIM
                },
                CHECKPOINT_PATH
            )


            print(
                "  -> Saved best model "
                f"(Val Top-1 = "
                f"{val_top1:.4f})"
            )


        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise RuntimeError(
            "Best checkpoint was not created."
        )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        "\nLoaded best checkpoint "
        f"from epoch "
        f"{checkpoint['epoch']}"
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    (
        test_loss,
        test_top1,
        test_top3,
        test_mrr
    ) = run_epoch(

        model,

        test_loader,

        optimizer,

        criterion,

        device,

        training=False
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Test Loss : {test_loss:.4f}"
    )

    print(
        f"Test Top-1: {test_top1:.4f}"
    )

    print(
        f"Test Top-3: {test_top3:.4f}"
    )

    print(
        f"Test MRR  : {test_mrr:.4f}"
    )

    print(
        "========================================"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()