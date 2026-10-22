# ============================================================
# LectureSense
# Controlled Modality Ablation
#
# Experiments:
#
# 1. transcript_ocr
#    Transcript + Slide OCR
#
# 2. transcript_image
#    Transcript + Slide Image
#
# 3. full
#    Transcript + Slide OCR + Slide Image
#
# Controlled comparison:
# - Same MaViLS data
# - Same lecture mapping
# - Same sentence normalization
# - Same ranking groups
# - Same candidate slides
# - Same train/validation/test split
# - Same ranking loss
# - Same optimizer
# - Same training configuration
#
# Metrics:
# - Top-1
# - Top-3
# - MRR
#
# ============================================================


import os
import re
import json
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

CHECKPOINT_DIR = "checkpoints"

RESULT_PATH = "data/ablation_results.json"


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


REVERSE_LECTURE_MAP = {

    feature_id: pair_id

    for pair_id, feature_id
    in LECTURE_MAP.items()
}


# ============================================================
# SENTENCE NORMALIZATION
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
# SEED
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

        device = torch.device(
            "cuda"
        )

        print(
            "Device: cuda"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "Device: cpu"
        )

    return device


# ============================================================
# ABLATION MODEL
# ============================================================

class AblationModel(nn.Module):

    def __init__(
        self,
        mode,
        hidden_dim=HIDDEN_DIM
    ):

        super().__init__()

        self.mode = mode


        # ====================================================
        # TRANSCRIPT PROJECTION
        # ====================================================

        self.text_proj = nn.Sequential(

            nn.Linear(
                TEXT_DIM,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )


        # ====================================================
        # OCR PROJECTION
        # ====================================================

        if mode in [

            "transcript_ocr",

            "full"

        ]:

            self.ocr_proj = nn.Sequential(

                nn.Linear(
                    OCR_DIM,
                    hidden_dim
                ),

                nn.ReLU(),

                nn.Dropout(0.10)
            )

        else:

            self.ocr_proj = None


        # ====================================================
        # IMAGE PROJECTION
        # ====================================================

        if mode in [

            "transcript_image",

            "full"

        ]:

            self.image_proj = nn.Sequential(

                nn.Linear(
                    IMAGE_DIM,
                    hidden_dim
                ),

                nn.ReLU(),

                nn.Dropout(0.10)
            )

        else:

            self.image_proj = None


        # ====================================================
        # FUSION DIMENSION
        # ====================================================

        if mode == "transcript_ocr":

            # t       = 256
            # o       = 256
            # |t-o|   = 256
            # t*o     = 256
            # cosine  = 1
            # cosine  = 1
            #
            # Total = 1026

            fusion_dim = (

                hidden_dim * 4

                + 2
            )


        elif mode == "transcript_image":

            # t       = 256
            # i       = 256
            # |t-i|   = 256
            # t*i     = 256
            # cosine  = 1
            #
            # Total = 1025

            fusion_dim = (

                hidden_dim * 4

                + 1
            )


        elif mode == "full":

            # t         = 256
            # o         = 256
            # i         = 256
            # |t-o|     = 256
            # t*o       = 256
            # |t-i|     = 256
            # t*i       = 256
            # |o-i|     = 256
            # o*i       = 256
            #
            # 9 * 256 = 2304
            #
            # 4 cosine scalars = 4
            #
            # Total = 2308

            fusion_dim = (

                hidden_dim * 9

                + 4
            )


        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )


        print(
            f"{mode} fusion dimension:",
            fusion_dim
        )


        # ====================================================
        # RANKING SCORER
        # ====================================================

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


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        text,
        ocr,
        image
    ):

        # ====================================================
        # TRANSCRIPT
        # ====================================================

        t = self.text_proj(
            text
        )


        t_norm = F.normalize(

            t,

            p=2,

            dim=-1
        )


        # ====================================================
        # TRANSCRIPT + OCR
        # ====================================================

        if self.mode in [

            "transcript_ocr",

            "full"

        ]:

            o = self.ocr_proj(
                ocr
            )


            o_norm = F.normalize(

                o,

                p=2,

                dim=-1
            )


            # ------------------------------------------------
            # Projected cosine
            # ------------------------------------------------

            projected_text_ocr_cosine = torch.sum(

                t_norm * o_norm,

                dim=-1,

                keepdim=True
            )


            # ------------------------------------------------
            # Native SBERT cosine
            # ------------------------------------------------

            native_text = F.normalize(

                text,

                p=2,

                dim=-1
            )


            native_ocr = F.normalize(

                ocr,

                p=2,

                dim=-1
            )


            native_text_ocr_cosine = torch.sum(

                native_text * native_ocr,

                dim=-1,

                keepdim=True
            )


        # ====================================================
        # TRANSCRIPT + IMAGE
        # ====================================================

        if self.mode in [

            "transcript_image",

            "full"

        ]:

            i = self.image_proj(
                image
            )


            i_norm = F.normalize(

                i,

                p=2,

                dim=-1
            )


            # ------------------------------------------------
            # Projected transcript-image cosine
            # ------------------------------------------------

            projected_text_image_cosine = torch.sum(

                t_norm * i_norm,

                dim=-1,

                keepdim=True
            )


        # ====================================================
        # FULL MODEL
        # ====================================================

        if self.mode == "full":

            # ------------------------------------------------
            # OCR-image cosine
            # ------------------------------------------------

            projected_ocr_image_cosine = torch.sum(

                o_norm * i_norm,

                dim=-1,

                keepdim=True
            )


            # ------------------------------------------------
            # 9 vector blocks
            # ------------------------------------------------

            fusion = [

                # 1. Transcript
                t,

                # 2. OCR
                o,

                # 3. Image
                i,

                # 4. Transcript-OCR difference
                torch.abs(
                    t - o
                ),

                # 5. Transcript-OCR product
                t * o,

                # 6. Transcript-Image difference
                torch.abs(
                    t - i
                ),

                # 7. Transcript-Image product
                t * i,

                # 8. OCR-Image difference
                torch.abs(
                    o - i
                ),

                # 9. OCR-Image product
                o * i,

                # 10. Native Transcript-OCR cosine
                native_text_ocr_cosine,

                # 11. Projected Transcript-OCR cosine
                projected_text_ocr_cosine,

                # 12. Projected Transcript-Image cosine
                projected_text_image_cosine,

                # 13. Projected OCR-Image cosine
                projected_ocr_image_cosine
            ]


        # ====================================================
        # TRANSCRIPT + OCR ONLY
        # ====================================================

        elif self.mode == "transcript_ocr":

            fusion = [

                # Transcript
                t,

                # OCR
                o,

                # Difference
                torch.abs(
                    t - o
                ),

                # Element-wise product
                t * o,

                # Native cosine
                native_text_ocr_cosine,

                # Projected cosine
                projected_text_ocr_cosine
            ]


        # ====================================================
        # TRANSCRIPT + IMAGE ONLY
        # ====================================================

        elif self.mode == "transcript_image":

            fusion = [

                # Transcript
                t,

                # Image
                i,

                # Difference
                torch.abs(
                    t - i
                ),

                # Element-wise product
                t * i,

                # Projected cosine
                projected_text_image_cosine
            ]


        # ====================================================
        # CONCATENATE
        # ====================================================

        fused = torch.cat(

            fusion,

            dim=-1
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        expected_dim = (

            self.scorer[0]
            .in_features
        )


        if fused.shape[-1] != expected_dim:

            raise RuntimeError(

                "Fusion dimension mismatch: "

                f"mode={self.mode}, "

                f"actual={fused.shape[-1]}, "

                f"expected={expected_dim}"
            )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        score = self.scorer(
            fused
        ).squeeze(-1)


        return score


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\nLoading embeddings..."
    )


    # ========================================================
    # EMBEDDINGS
    # ========================================================

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
        ]
        .astype(str)
    )


    text_index["sentence_norm"] = (

        text_index[
            "sentence_text"
        ]
        .map(
            normalize_sentence
        )
    )


    text_index["embedding_idx"] = (

        text_index[
            "embedding_idx"
        ]
        .astype(int)
    )


    # --------------------------------------------------------
    # Keep valid embedding indices
    # --------------------------------------------------------

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
            ]
            <
            len(text_embeddings)
        )

    ].copy()


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Text index contains FEATURE lecture IDs.
    # Keep them as feature IDs.
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


    # --------------------------------------------------------
    # Convert feature lecture ID -> pair lecture ID
    # --------------------------------------------------------

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
        ]
        .astype(int)
    )


    ocr_index["embedding_row"] = (

        ocr_index[
            "embedding_row"
        ]
        .astype(int)
    )


    ocr_lookup = {}


    for row in ocr_index.itertuples(

        index=False
    ):

        key = (

            row.lecture_id,

            int(
                row.slide_id
            )
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


    # --------------------------------------------------------
    # Convert feature lecture ID -> pair lecture ID
    # --------------------------------------------------------

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
        ]
        .astype(int)
    )


    image_lookup = {}


    for idx, row in image_index.iterrows():

        key = (

            row["lecture_id"],

            int(
                row["slide_id"]
            )
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
        ]
        .astype(str)
    )


    pairs["sentence_text"] = (

        pairs[
            "sentence_text"
        ]
        .astype(str)
    )


    pairs["sentence_norm"] = (

        pairs[
            "sentence_text"
        ]
        .map(
            normalize_sentence
        )
    )


    pairs["slide_id"] = (

        pairs[
            "slide_id"
        ]
        .astype(int)
    )


    pairs["label"] = (

        pairs[
            "label"
        ]
        .astype(int)
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


    # ========================================================
    # AVAILABLE SLIDES
    # ========================================================

    slides_by_lecture = defaultdict(
        set
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


    # ========================================================
    # POSITIVE PAIRS
    # ========================================================

    positives = pairs[

        pairs["label"] == 1

    ].copy()


    groups = []


    skipped_text = 0

    skipped_slide = 0


    seen = set()


    # ========================================================
    # CREATE ONE GROUP PER POSITIVE SENTENCE
    # ========================================================

    for row in positives.itertuples(

        index=False
    ):

        pair_lecture_id = str(

            row.lecture_id
        )


        sentence_text = str(

            row.sentence_text
        )


        sentence_norm = normalize_sentence(

            sentence_text
        )


        true_slide = int(

            row.slide_id
        )


        # ----------------------------------------------------
        # Pair ID -> Feature ID
        # ----------------------------------------------------

        feature_lecture_id = (

            LECTURE_MAP.get(

                pair_lecture_id,

                pair_lecture_id
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
        # OCR true slide
        # ----------------------------------------------------

        ocr_key = (

            pair_lecture_id,

            true_slide
        )


        if ocr_key not in ocr_lookup:

            skipped_slide += 1

            continue


        # ----------------------------------------------------
        # Image true slide
        # ----------------------------------------------------

        image_key = (

            pair_lecture_id,

            true_slide
        )


        if image_key not in image_lookup:

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
        # Duplicate group protection
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


        # ----------------------------------------------------
        # Preserve split
        # ----------------------------------------------------

        split = str(

            row.split
        ).strip().lower()


        groups.append({

            "lecture_id":
                pair_lecture_id,

            "sentence_text":
                sentence_text,

            "sentence_norm":
                sentence_norm,

            "text_idx":
                int(
                    text_idx
                ),

            "true_slide":
                true_slide,

            "candidate_slides":
                candidates,

            "split":
                split
        })


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

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


    split_counts = defaultdict(
        int
    )


    for group in groups:

        split_counts[
            group["split"]
        ] += 1


    print(
        "\nGroup split counts:"
    )


    print(

        "train:",

        split_counts["train"]
    )


    print(

        "val:",

        split_counts["val"]
    )


    print(

        "test:",

        split_counts["test"]
    )


    return groups


# ============================================================
# DATASET
# ============================================================

class AblationDataset(
    Dataset
):

    def __init__(

        self,

        groups,

        ocr_lookup,

        image_lookup,

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        mode

    ):

        self.groups = groups

        self.ocr_lookup = ocr_lookup

        self.image_lookup = image_lookup

        self.text_embeddings = text_embeddings

        self.ocr_embeddings = ocr_embeddings

        self.image_embeddings = image_embeddings

        self.mode = mode


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


        # ====================================================
        # TRANSCRIPT
        # ====================================================

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


        # ====================================================
        # CANDIDATE SLIDES
        #
        # Keep only slides for which BOTH OCR and image
        # features exist.
        #
        # This guarantees that every ablation sees the same
        # candidate-slide universe.
        # ====================================================

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


        # ====================================================
        # TARGET MUST EXIST
        # ====================================================

        if true_slide not in valid_slides:

            raise RuntimeError(

                f"True slide {true_slide} "

                f"missing for lecture "

                f"{lecture_id}"
            )


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
# COLLATE
# ============================================================

def collate_fn(batch):

    batch_size = len(
        batch
    )


    max_slides = max(

        item["ocr"].shape[0]

        for item in batch
    )


    # ========================================================
    # TRANSCRIPT
    # ========================================================

    text_batch = torch.stack(

        [

            item["text"]

            for item in batch

        ]
    )


    # ========================================================
    # OCR
    # ========================================================

    ocr_batch = torch.zeros(

        batch_size,

        max_slides,

        OCR_DIM,

        dtype=torch.float32
    )


    # ========================================================
    # IMAGE
    # ========================================================

    image_batch = torch.zeros(

        batch_size,

        max_slides,

        IMAGE_DIM,

        dtype=torch.float32
    )


    # ========================================================
    # VALID MASK
    # ========================================================

    valid_mask = torch.zeros(

        batch_size,

        max_slides,

        dtype=torch.bool
    )


    targets = []

    slide_lists = []


    # ========================================================
    # FILL BATCH
    # ========================================================

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


    # ========================================================
    # REPEAT TRANSCRIPT FOR EACH CANDIDATE SLIDE
    # ========================================================

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


    # ========================================================
    # FLATTEN OCR
    # ========================================================

    ocr_flat = (

        ocr.reshape(

            batch_size * num_slides,

            -1
        )
    )


    # ========================================================
    # FLATTEN IMAGE
    # ========================================================

    image_flat = (

        image.reshape(

            batch_size * num_slides,

            -1
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    scores = model(

        text_expanded,

        ocr_flat,

        image_flat
    )


    scores = scores.reshape(

        batch_size,

        num_slides
    )


    # ========================================================
    # MASK PADDING
    # ========================================================

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


    # ========================================================
    # BATCHES
    # ========================================================

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


        # ====================================================
        # ZERO GRADIENTS
        # ====================================================

        if training:

            optimizer.zero_grad(

                set_to_none=True
            )


        # ====================================================
        # FORWARD
        # ====================================================

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


            # =================================================
            # BACKPROPAGATION
            # =================================================

            if training:

                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    GRAD_CLIP
                )


                optimizer.step()


        # ====================================================
        # LOSS
        # ====================================================

        total_loss += (

            loss.item()

            *

            batch_size
        )


        total_samples += (

            batch_size
        )


        # ====================================================
        # RANKING METRICS
        # ====================================================

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


    # ========================================================
    # SAFETY
    # ========================================================

    if total_samples == 0:

        raise RuntimeError(

            "No samples were processed."
        )


    # ========================================================
    # METRICS
    # ========================================================

    average_loss = (

        total_loss

        /

        total_samples
    )


    top1 = (

        top1_count

        /

        total_samples
    )


    top3 = (

        top3_count

        /

        total_samples
    )


    mrr = (

        reciprocal_rank_sum

        /

        total_samples
    )


    return (

        average_loss,

        top1,

        top3,

        mrr
    )


# ============================================================
# TRAIN ONE ABLATION
# ============================================================

def train_ablation(

    mode,

    train_groups,

    val_groups,

    test_groups,

    ocr_lookup,

    image_lookup,

    text_embeddings,

    ocr_embeddings,

    image_embeddings,

    device,

    epochs,

    batch_size

):

    print(

        "\n"

        + "=" * 75
    )


    print(

        "ABLATION:",

        mode
    )


    print(

        "=" * 75
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = AblationDataset(

        train_groups,

        ocr_lookup,

        image_lookup,

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        mode
    )


    val_dataset = AblationDataset(

        val_groups,

        ocr_lookup,

        image_lookup,

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        mode
    )


    test_dataset = AblationDataset(

        test_groups,

        ocr_lookup,

        image_lookup,

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        mode
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=collate_fn
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=collate_fn
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        collate_fn=collate_fn
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = AblationModel(

        mode

    ).to(

        device
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_val_top1 = -1.0

    best_epoch = 0

    best_state = None

    history = []


    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(

        1,

        epochs + 1

    ):

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


        scheduler.step(

            val_top1
        )


        current_lr = optimizer.param_groups[

            0

        ][

            "lr"
        ]


        print(

            f"Epoch {epoch:02d}/{epochs} | "

            f"Train Loss {train_loss:.4f} | "

            f"Train Top1 {train_top1:.4f} | "

            f"Train Top3 {train_top3:.4f} | "

            f"Train MRR {train_mrr:.4f} | "

            f"Val Loss {val_loss:.4f} | "

            f"Val Top1 {val_top1:.4f} | "

            f"Val Top3 {val_top3:.4f} | "

            f"Val MRR {val_mrr:.4f} | "

            f"LR {current_lr:.6f}"
        )


        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_top1":
                train_top1,

            "train_top3":
                train_top3,

            "train_mrr":
                train_mrr,

            "val_loss":
                val_loss,

            "val_top1":
                val_top1,

            "val_top3":
                val_top3,

            "val_mrr":
                val_mrr,

            "lr":
                current_lr
        })


        # ====================================================
        # SAVE BEST STATE
        # ====================================================

        if val_top1 > best_val_top1:

            best_val_top1 = val_top1

            best_epoch = epoch

            best_state = {

                key:

                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }


    # ========================================================
    # RESTORE BEST
    # ========================================================

    if best_state is not None:

        model.load_state_dict(

            best_state
        )


    # ========================================================
    # TEST
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
    # RESULTS
    # ========================================================

    print(

        "\nBest epoch:",

        best_epoch
    )


    print(

        "Best validation Top-1:",

        f"{best_val_top1:.4f}"
    )


    print(
        "\nFINAL TEST"
    )


    print(

        "Test Loss:",

        f"{test_loss:.4f}"
    )


    print(

        "Test Top-1:",

        f"{test_top1:.4f}"
    )


    print(

        "Test Top-3:",

        f"{test_top3:.4f}"
    )


    print(

        "Test MRR:",

        f"{test_mrr:.4f}"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    os.makedirs(

        CHECKPOINT_DIR,

        exist_ok=True
    )


    checkpoint_path = os.path.join(

        CHECKPOINT_DIR,

        f"ablation_{mode}_best.pt"
    )


    torch.save(

        {

            "mode":
                mode,

            "best_epoch":
                best_epoch,

            "best_val_top1":
                best_val_top1,

            "test_loss":
                test_loss,

            "test_top1":
                test_top1,

            "test_top3":
                test_top3,

            "test_mrr":
                test_mrr,

            "model_state_dict":
                model.state_dict(),

            "history":
                history
        },

        checkpoint_path
    )


    print(

        "\nCheckpoint saved:",

        checkpoint_path
    )


    return {

        "mode":
            mode,

        "best_epoch":
            best_epoch,

        "best_val_top1":
            best_val_top1,

        "test_loss":
            test_loss,

        "test_top1":
            test_top1,

        "test_top3":
            test_top3,

        "test_mrr":
            test_mrr,

        "checkpoint":
            checkpoint_path
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "LectureSense controlled modality ablation"
    )


    parser.add_argument(

        "--mode",

        choices=[

            "transcript_ocr",

            "transcript_image",

            "full",

            "all"
        ],

        default="all"
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


    # ========================================================
    # SEED
    # ========================================================

    set_seed()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


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


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(groups) == 0:

        raise RuntimeError(

            "Ranking groups = 0. "

            "Check lecture/text mapping."
        )


    # ========================================================
    # SPLIT
    # ========================================================

    train_groups = [

        group

        for group in groups

        if group["split"] == "train"
    ]


    val_groups = [

        group

        for group in groups

        if group["split"] == "val"
    ]


    test_groups = [

        group

        for group in groups

        if group["split"] == "test"
    ]


    print(

        "\n"

        + "=" * 75
    )


    print(
        "FINAL DATA SPLIT"
    )


    print(
        "=" * 75
    )


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


    # ========================================================
    # SAFETY
    # ========================================================

    if (

        len(train_groups) == 0

        or

        len(val_groups) == 0

        or

        len(test_groups) == 0

    ):

        raise RuntimeError(

            "One or more splits are empty."
        )


    # ========================================================
    # EXPERIMENTS
    # ========================================================

    if args.mode == "all":

        modes = [

            "transcript_ocr",

            "transcript_image",

            "full"
        ]

    else:

        modes = [

            args.mode
        ]


    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    results = []


    for mode in modes:

        result = train_ablation(

            mode,

            train_groups,

            val_groups,

            test_groups,

            ocr_lookup,

            image_lookup,

            text_embeddings,

            ocr_embeddings,

            image_embeddings,

            device,

            args.epochs,

            args.batch_size
        )


        results.append(
            result
        )


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print(

        "\n\n"

        + "=" * 75
    )


    print(
        "CONTROLLED ABLATION COMPARISON"
    )


    print(
        "=" * 75
    )


    print(

        f"{'Model':24}"

        f"{'Top-1':>12}"

        f"{'Top-3':>12}"

        f"{'MRR':>12}"
    )


    print(
        "-" * 60
    )


    for result in results:

        print(

            f"{result['mode']:24}"

            f"{result['test_top1'] * 100:>11.2f}%"

            f"{result['test_top3'] * 100:>11.2f}%"

            f"{result['test_mrr'] * 100:>11.2f}%"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(

        "data",

        exist_ok=True
    )


    with open(

        RESULT_PATH,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            {

                "configuration": {

                    "seed":
                        SEED,

                    "epochs":
                        args.epochs,

                    "batch_size":
                        args.batch_size,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "gradient_clip":
                        GRAD_CLIP
                },

                "groups": {

                    "total":
                        len(groups),

                    "train":
                        len(train_groups),

                    "validation":
                        len(val_groups),

                    "test":
                        len(test_groups)
                },

                "results":
                    results

            },

            f,

            indent=2
        )


    print(

        "\nResults saved:",

        RESULT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()