# ============================================================
# LectureSense
# Ground-Truth Ablation Evaluation
#
# Evaluates:
#   1. Transcript + OCR
#   2. Transcript + Image
#   3. Transcript + OCR + Image
#
# Uses the same MaViLS ground-truth protocol for all models.
#
# Output:
#   data/ablation_ground_truth_results.json
# ============================================================

import os
import re
import json
import unicodedata
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_DIM = 384
OCR_DIM = 384
IMAGE_DIM = 512

HIDDEN_DIM = 256

RESULT_PATH = "data/ablation_ground_truth_results.json"


CHECKPOINTS = {

    "transcript_ocr":
        "checkpoints/ablation_transcript_ocr_best.pt",

    "transcript_image":
        "checkpoints/ablation_transcript_image_best.pt",

    "full":
        "checkpoints/ablation_full_best.pt"
}


# ============================================================
# LECTURE ID MAPPING
#
# Pair/data lecture IDs -> feature lecture IDs
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
        "sensory_systems"
}


REVERSE_LECTURE_MAP = {

    feature_id: pair_id

    for pair_id, feature_id
    in LECTURE_MAP.items()
}


# ============================================================
# EXACT MaViLS GROUND-TRUTH FILE MAPPING
#
# These filenames were checked against the files present in:
# data/MaViLS/data/ground_truth_files
# ============================================================

GROUND_TRUTH_MAP = {

    "ML_for_health_MIT":
        "ground_truth_ML_for_health_MIT.xlsx",

    "cities_and_decarbonization":
        "ground_truth_cities_and_decarbonization.xlsx",

    "climate_and_cities":
        "ground_truth_climate_and_cities.xlsx",

    "climate_science_policy_MIT":
        "ground_truth_climate_science_policy_MIT2.xlsx",

    "cognitive_robotics_MIT":
        "ground_truth_cognitive_robotics_MIT.xlsx",

    "computer_vision_2_2":
        "ground_truth_computer_vision_2_2.xlsx",

    "creating_breakthrough_products":
        "ground_truth_creating_breakthrough_products_MIT.xlsx",

    "cryptocurrency":
        "ground_truth_cryptocurrency_MIT.xlsx",

    "deeplearning_goodfellow":
        "ground_truth_deeplearning.xlsx",

    "image_processing_high_res":
        "ground_truth_image_processing.xlsx",

    "numerics_hennig":
        "ground_truth_numerics.xlsx",

    "phonetics":
        "ground_truth_phonetics.xlsx",

    "physics_intro_02":
        "ground_truth_physics.xlsx",

    "psychology_MIT":
        "ground_truth_psychology.xlsx",

    "reinforcement_learning_silver":
        "ground_truth_reinforcement_learning.xlsx",

    "short_range_mit":
        "ground_truth_short_range.xlsx",

    "solar_resource":
        "ground_truth_solar_resource.xlsx",

    "team_dynamics_game_design_MIT":
        "ground_truth_team_dynamics_game_design_MIT.xlsx",

    "theory_of_computation":
        "ground_truth_theory_of_computation.xlsx",

    "visual_system_MIT":
        "ground_truth_sensory_systems.xlsx"
}


# ============================================================
# NORMALIZE SENTENCE
# ============================================================

def normalize_sentence(text):

    if text is None:
        return ""

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

class AblationModel(nn.Module):

    def __init__(self, mode):

        super().__init__()

        self.mode = mode


        # ----------------------------------------------------
        # Transcript projection
        # ----------------------------------------------------

        self.text_proj = nn.Sequential(

            nn.Linear(
                TEXT_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )


        # ----------------------------------------------------
        # OCR projection
        # ----------------------------------------------------

        if mode in [
            "transcript_ocr",
            "full"
        ]:

            self.ocr_proj = nn.Sequential(

                nn.Linear(
                    OCR_DIM,
                    HIDDEN_DIM
                ),

                nn.ReLU(),

                nn.Dropout(0.10)
            )

        else:

            self.ocr_proj = None


        # ----------------------------------------------------
        # Image projection
        # ----------------------------------------------------

        if mode in [
            "transcript_image",
            "full"
        ]:

            self.image_proj = nn.Sequential(

                nn.Linear(
                    IMAGE_DIM,
                    HIDDEN_DIM
                ),

                nn.ReLU(),

                nn.Dropout(0.10)
            )

        else:

            self.image_proj = None


        # ----------------------------------------------------
        # Fusion dimension
        # ----------------------------------------------------

        if mode == "transcript_ocr":

            fusion_dim = (
                HIDDEN_DIM * 4
                + 2
            )

        elif mode == "transcript_image":

            fusion_dim = (
                HIDDEN_DIM * 4
                + 1
            )

        elif mode == "full":

            fusion_dim = (
                HIDDEN_DIM * 9
                + 4
            )

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )


        # ----------------------------------------------------
        # Ranking scorer
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


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        text,
        ocr,
        image
    ):

        t = self.text_proj(text)

        t_norm = F.normalize(
            t,
            p=2,
            dim=-1
        )


        # ====================================================
        # OCR branch
        # ====================================================

        if self.mode in [
            "transcript_ocr",
            "full"
        ]:

            o = self.ocr_proj(ocr)

            o_norm = F.normalize(
                o,
                p=2,
                dim=-1
            )


            projected_text_ocr_cosine = torch.sum(
                t_norm * o_norm,
                dim=-1,
                keepdim=True
            )


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
        # Image branch
        # ====================================================

        if self.mode in [
            "transcript_image",
            "full"
        ]:

            i = self.image_proj(image)

            i_norm = F.normalize(
                i,
                p=2,
                dim=-1
            )


            projected_text_image_cosine = torch.sum(
                t_norm * i_norm,
                dim=-1,
                keepdim=True
            )


        # ====================================================
        # FULL MODEL
        # ====================================================

        if self.mode == "full":

            projected_ocr_image_cosine = torch.sum(
                o_norm * i_norm,
                dim=-1,
                keepdim=True
            )


            fusion = [

                t,

                o,

                i,

                torch.abs(
                    t - o
                ),

                t * o,

                torch.abs(
                    t - i
                ),

                t * i,

                torch.abs(
                    o - i
                ),

                o * i,

                native_text_ocr_cosine,

                projected_text_ocr_cosine,

                projected_text_image_cosine,

                projected_ocr_image_cosine
            ]


        # ====================================================
        # TRANSCRIPT + OCR
        # ====================================================

        elif self.mode == "transcript_ocr":

            fusion = [

                t,

                o,

                torch.abs(
                    t - o
                ),

                t * o,

                native_text_ocr_cosine,

                projected_text_ocr_cosine
            ]


        # ====================================================
        # TRANSCRIPT + IMAGE
        # ====================================================

        elif self.mode == "transcript_image":

            fusion = [

                t,

                i,

                torch.abs(
                    t - i
                ),

                t * i,

                projected_text_image_cosine
            ]


        fused = torch.cat(
            fusion,
            dim=-1
        )


        return self.scorer(
            fused
        ).squeeze(-1)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    print("\nLoading features...")


    # --------------------------------------------------------
    # Embeddings
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
        "Text:",
        text_embeddings.shape
    )

    print(
        "OCR:",
        ocr_embeddings.shape
    )

    print(
        "Image:",
        image_embeddings.shape
    )


    # ========================================================
    # TEXT INDEX
    # ========================================================

    text_index = pd.read_csv(
        "data/text_embedding_index.csv"
    )


    text_index = text_index.dropna(
        subset=[
            "lecture_id",
            "sentence_text",
            "embedding_idx"
        ]
    ).copy()


    text_index["lecture_id"] = (
        text_index["lecture_id"]
        .astype(str)
    )


    text_index["sentence_norm"] = (
        text_index["sentence_text"]
        .map(normalize_sentence)
    )


    text_index["embedding_idx"] = (
        text_index["embedding_idx"]
        .astype(int)
    )


    text_lookup = {}


    for row in text_index.itertuples(
        index=False
    ):

        idx = int(
            row.embedding_idx
        )


        if idx < 0:
            continue


        if idx >= len(text_embeddings):
            continue


        key = (
            row.lecture_id,
            row.sentence_norm
        )


        text_lookup[key] = idx


    # ========================================================
    # OCR INDEX
    # ========================================================

    ocr_index = pd.read_csv(
        "data/ocr_embedding_index.csv"
    )


    ocr_index["lecture_id"] = (
        ocr_index["lecture_id"]
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
        ocr_index["slide_id"]
        .astype(int)
    )


    ocr_index["embedding_row"] = (
        ocr_index["embedding_row"]
        .astype(int)
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


    # ========================================================
    # IMAGE INDEX
    # ========================================================

    image_index = pd.read_csv(
        "data/image_embedding_index.csv"
    )


    image_index["lecture_id"] = (
        image_index["lecture_id"]
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
        image_index["slide_id"]
        .astype(int)
    )


    image_lookup = {}


    for idx, row in image_index.iterrows():

        key = (
            row["lecture_id"],
            int(row["slide_id"])
        )


        image_lookup[key] = int(idx)


    print(
        "Text lookup:",
        len(text_lookup)
    )

    print(
        "OCR lookup:",
        len(ocr_lookup)
    )

    print(
        "Image lookup:",
        len(image_lookup)
    )


    return (

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        text_lookup,

        ocr_lookup,

        image_lookup
    )


# ============================================================
# FIND EXACT GROUND-TRUTH FILE
# ============================================================

def find_ground_truth_file(lecture_id):

    gt_dir = (
        "data/MaViLS/data/"
        "ground_truth_files"
    )


    filename = GROUND_TRUTH_MAP.get(
        lecture_id
    )


    if filename is None:

        print(
            f"Ground truth mapping missing: "
            f"{lecture_id}"
        )

        return None


    path = os.path.join(
        gt_dir,
        filename
    )


    if not os.path.exists(path):

        print(
            f"Ground truth file does not exist:"
            f"\n{path}"
        )

        return None


    return path


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(lecture_id):

    path = find_ground_truth_file(
        lecture_id
    )


    if path is None:
        return []


    print(
        f"Ground truth: {lecture_id}"
    )

    print(
        f"File: {path}"
    )


    df = pd.read_excel(
        path
    )


    columns = [
        str(c)
        for c in df.columns
    ]


    # ========================================================
    # COLUMN DETECTION
    # ========================================================

    time_col = None
    slide_col = None
    sentence_col = None


    for col in columns:

        low = col.lower().strip()


        if low in [
            "key",
            "time",
            "timestamp",
            "start"
        ]:

            time_col = col


        if low in [
            "slidenumber",
            "slide_number",
            "slide",
            "slideid",
            "slide_id"
        ]:

            slide_col = col


        if low in [
            "value",
            "sentence",
            "text",
            "transcript"
        ]:

            sentence_col = col


    # --------------------------------------------------------
    # Fallback matching
    # --------------------------------------------------------

    if time_col is None:

        for col in columns:

            if "key" in col.lower():

                time_col = col

                break


    if slide_col is None:

        for col in columns:

            if "slide" in col.lower():

                slide_col = col

                break


    if sentence_col is None:

        for col in columns:

            low = col.lower()

            if (
                "value" in low
                or
                "sentence" in low
                or
                "text" in low
            ):

                sentence_col = col

                break


    if (
        time_col is None
        or
        slide_col is None
        or
        sentence_col is None
    ):

        print(
            "Available columns:",
            columns
        )

        raise RuntimeError(
            "Could not identify MaViLS "
            f"ground-truth columns for {lecture_id}"
        )


    records = []


    for _, row in df.iterrows():

        timestamp = row[
            time_col
        ]

        slide_id = row[
            slide_col
        ]

        sentence = row[
            sentence_col
        ]


        if pd.isna(timestamp):
            continue


        if pd.isna(slide_id):
            continue


        if pd.isna(sentence):
            continue


        try:

            slide_id = int(
                slide_id
            )

        except Exception:

            continue


        # MaViLS uses -1 for samples where
        # no unique slide can be identified.

        if slide_id < 1:
            continue


        records.append({

            "timestamp":
                timestamp,

            "slide_id":
                slide_id,

            "sentence":
                str(sentence)
        })


    print(
        f"GT usable rows: {len(records)}"
    )


    return records


# ============================================================
# GET CANDIDATE SLIDES
# ============================================================

def get_candidate_slides(
    lecture_id,
    ocr_lookup,
    image_lookup
):

    slides = set()


    for (
        lid,
        slide_id
    ) in image_lookup.keys():

        if lid != lecture_id:
            continue


        key = (
            lecture_id,
            slide_id
        )


        # Keep exactly the same candidate universe
        # for all three ablation models.

        if key not in ocr_lookup:
            continue


        slides.add(
            int(slide_id)
        )


    return sorted(
        slides
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    mode,
    device
):

    checkpoint_path = CHECKPOINTS[
        mode
    ]


    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    model = AblationModel(
        mode
    ).to(
        device
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )


    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model.eval()


    print(
        f"Loaded {mode}: "
        f"{checkpoint_path}"
    )


    if isinstance(
        checkpoint,
        dict
    ):

        if "best_epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint[
                    "best_epoch"
                ]
            )


    return model


# ============================================================
# SCORE ONE SENTENCE AGAINST ALL SLIDES
# ============================================================

@torch.no_grad()
def score_sentence(

    model,

    text_embedding,

    lecture_id,

    candidate_slides,

    ocr_lookup,

    image_lookup,

    ocr_embeddings,

    image_embeddings,

    device

):

    text_list = []
    ocr_list = []
    image_list = []

    valid_slides = []


    for slide_id in candidate_slides:

        ocr_key = (
            lecture_id,
            slide_id
        )

        image_key = (
            lecture_id,
            slide_id
        )


        if ocr_key not in ocr_lookup:
            continue


        if image_key not in image_lookup:
            continue


        ocr_idx = ocr_lookup[
            ocr_key
        ]

        image_idx = image_lookup[
            image_key
        ]


        text_list.append(
            text_embedding
        )

        ocr_list.append(
            ocr_embeddings[
                ocr_idx
            ]
        )

        image_list.append(
            image_embeddings[
                image_idx
            ]
        )

        valid_slides.append(
            slide_id
        )


    if not valid_slides:
        return {}


    text_tensor = torch.tensor(
        np.asarray(
            text_list,
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=device
    )


    ocr_tensor = torch.tensor(
        np.asarray(
            ocr_list,
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=device
    )


    image_tensor = torch.tensor(
        np.asarray(
            image_list,
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=device
    )


    scores = model(
        text_tensor,
        ocr_tensor,
        image_tensor
    )


    scores = (
        scores
        .detach()
        .cpu()
        .numpy()
    )


    return {

        int(slide_id):
            float(score)

        for slide_id, score
        in zip(
            valid_slides,
            scores
        )
    }


# ============================================================
# EVALUATE ONE MODEL ON ONE LECTURE
# ============================================================

def evaluate_model(

    model,

    ground_truth,

    lecture_id,

    text_lookup,

    text_embeddings,

    ocr_lookup,

    image_lookup,

    ocr_embeddings,

    image_embeddings,

    device

):

    candidate_slides = get_candidate_slides(

        lecture_id,

        ocr_lookup,

        image_lookup
    )


    if not candidate_slides:

        return None


    total = 0

    top1_count = 0

    top3_count = 0

    reciprocal_rank_sum = 0.0


    for gt in ground_truth:

        sentence = gt[
            "sentence"
        ]

        true_slide = int(
            gt["slide_id"]
        )


        normalized_sentence = (
            normalize_sentence(
                sentence
            )
        )


        feature_lecture_id = (
            LECTURE_MAP.get(
                lecture_id,
                lecture_id
            )
        )


        text_key = (

            feature_lecture_id,

            normalized_sentence
        )


        if text_key not in text_lookup:

            continue


        text_idx = text_lookup[
            text_key
        ]


        text_embedding = (
            text_embeddings[
                text_idx
            ]
        )


        scores = score_sentence(

            model,

            text_embedding,

            lecture_id,

            candidate_slides,

            ocr_lookup,

            image_lookup,

            ocr_embeddings,

            image_embeddings,

            device
        )


        if not scores:
            continue


        if true_slide not in scores:
            continue


        ranked_slides = sorted(

            scores.keys(),

            key=lambda slide:

            scores[slide],

            reverse=True
        )


        try:

            rank = (
                ranked_slides.index(
                    true_slide
                )
                + 1
            )

        except ValueError:

            continue


        total += 1


        if rank == 1:

            top1_count += 1


        if rank <= 3:

            top3_count += 1


        reciprocal_rank_sum += (
            1.0 / rank
        )


    if total == 0:

        return None


    top1 = (
        top1_count
        /
        total
    )


    top3 = (
        top3_count
        /
        total
    )


    mrr = (
        reciprocal_rank_sum
        /
        total
    )


    return {

        "lecture_id":
            lecture_id,

        "samples":
            total,

        "top1":
            top1,

        "top3":
            top3,

        "mrr":
            mrr,

        # Single-label exact-match:
        # Precision = Recall = F1 = Top-1
        "precision":
            top1,

        "recall":
            top1,

        "f1":
            top1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "Evaluate LectureSense modality "
        "ablations on MaViLS ground truth."
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


    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    # ========================================================
    # LOAD FEATURES
    # ========================================================

    (

        text_embeddings,

        ocr_embeddings,

        image_embeddings,

        text_lookup,

        ocr_lookup,

        image_lookup

    ) = load_features()


    # ========================================================
    # LECTURES
    # ========================================================

    lecture_ids = list(
        LECTURE_MAP.keys()
    )


    print(
        "\nLectures requested:",
        len(lecture_ids)
    )


    # ========================================================
    # MODES
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


    all_results = {}


    # ========================================================
    # EVALUATE EACH MODEL
    # ========================================================

    for mode in modes:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "EVALUATING:",
            mode
        )

        print(
            "=" * 75
        )


        model = load_model(
            mode,
            device
        )


        mode_results = []


        for lecture_id in lecture_ids:

            print(
                f"\n[{mode}] {lecture_id}"
            )


            ground_truth = load_ground_truth(
                lecture_id
            )


            if not ground_truth:

                print(
                    "SKIPPED: no ground truth"
                )

                continue


            result = evaluate_model(

                model,

                ground_truth,

                lecture_id,

                text_lookup,

                text_embeddings,

                ocr_lookup,

                image_lookup,

                ocr_embeddings,

                image_embeddings,

                device
            )


            if result is None:

                print(
                    "SKIPPED: no usable samples"
                )

                continue


            mode_results.append(
                result
            )


            print(

                f"Samples={result['samples']} | "

                f"Top1={result['top1']:.4f} | "

                f"Top3={result['top3']:.4f} | "

                f"MRR={result['mrr']:.4f}"
            )


        all_results[mode] = mode_results


        # Free model before loading next one.

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    aggregate = {}


    for mode, results in all_results.items():

        if not results:
            continue


        total_samples = sum(

            r["samples"]

            for r in results
        )


        weighted_top1 = sum(

            r["top1"]
            *
            r["samples"]

            for r in results

        ) / total_samples


        weighted_top3 = sum(

            r["top3"]
            *
            r["samples"]

            for r in results

        ) / total_samples


        weighted_mrr = sum(

            r["mrr"]
            *
            r["samples"]

            for r in results

        ) / total_samples


        aggregate[mode] = {

            "lectures_evaluated":
                len(results),

            "samples":
                total_samples,

            "top1":
                weighted_top1,

            "top3":
                weighted_top3,

            "mrr":
                weighted_mrr,

            "precision":
                weighted_top1,

            "recall":
                weighted_top1,

            "f1":
                weighted_top1
        }


    # ========================================================
    # CHECK CONSISTENCY
    # ========================================================

    sample_counts = [

        result["samples"]

        for result in aggregate.values()
    ]


    lecture_counts = [

        result["lectures_evaluated"]

        for result in aggregate.values()
    ]


    if len(set(sample_counts)) > 1:

        print(
            "\nWARNING:"
            "\nDifferent models were evaluated "
            "on different numbers of samples."
        )


    if len(set(lecture_counts)) > 1:

        print(
            "\nWARNING:"
            "\nDifferent models were evaluated "
            "on different numbers of lectures."
        )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "FINAL GROUND-TRUTH ABLATION RESULTS"
    )

    print(
        "=" * 80
    )


    print(

        f"{'Model':24}"

        f"{'Lectures':>10}"

        f"{'Samples':>10}"

        f"{'Top-1':>12}"

        f"{'Top-3':>12}"

        f"{'MRR':>12}"
    )


    print(
        "-" * 80
    )


    display_names = {

        "transcript_ocr":
            "Transcript + OCR",

        "transcript_image":
            "Transcript + Image",

        "full":
            "Full"
    }


    for mode in modes:

        if mode not in aggregate:
            continue


        r = aggregate[
            mode
        ]


        print(

            f"{display_names[mode]:24}"

            f"{r['lectures_evaluated']:>10}"

            f"{r['samples']:>10}"

            f"{r['top1'] * 100:>11.2f}%"

            f"{r['top3'] * 100:>11.2f}%"

            f"{r['mrr'] * 100:>11.2f}%"
        )


    # ========================================================
    # COMPARISONS
    # ========================================================

    comparisons = {}


    # --------------------------------------------------------
    # OCR vs Image
    # --------------------------------------------------------

    if (

        "transcript_ocr" in aggregate

        and

        "transcript_image" in aggregate

    ):

        ocr = aggregate[
            "transcript_ocr"
        ]

        image = aggregate[
            "transcript_image"
        ]


        comparisons[
            "transcript_ocr_vs_transcript_image"
        ] = {

            "top1_difference":
                ocr["top1"]
                -
                image["top1"],

            "top3_difference":
                ocr["top3"]
                -
                image["top3"],

            "mrr_difference":
                ocr["mrr"]
                -
                image["mrr"]
        }


    # --------------------------------------------------------
    # Full vs OCR
    # --------------------------------------------------------

    if (

        "full" in aggregate

        and

        "transcript_ocr" in aggregate

    ):

        full = aggregate[
            "full"
        ]

        ocr = aggregate[
            "transcript_ocr"
        ]


        comparisons[
            "full_vs_transcript_ocr"
        ] = {

            "top1_difference":
                full["top1"]
                -
                ocr["top1"],

            "top3_difference":
                full["top3"]
                -
                ocr["top3"],

            "mrr_difference":
                full["mrr"]
                -
                ocr["mrr"]
        }


    # --------------------------------------------------------
    # Full vs Image
    # --------------------------------------------------------

    if (

        "full" in aggregate

        and

        "transcript_image" in aggregate

    ):

        full = aggregate[
            "full"
        ]

        image = aggregate[
            "transcript_image"
        ]


        comparisons[
            "full_vs_transcript_image"
        ] = {

            "top1_difference":
                full["top1"]
                -
                image["top1"],

            "top3_difference":
                full["top3"]
                -
                image["top3"],

            "mrr_difference":
                full["mrr"]
                -
                image["mrr"]
        }


    # ========================================================
    # SAVE JSON
    # ========================================================

    os.makedirs(
        "data",
        exist_ok=True
    )


    output = {

        "evaluation":
            "MaViLS ground-truth "
            "controlled modality ablation",

        "ground_truth_directory":
            "data/MaViLS/data/"
            "ground_truth_files",

        "lecture_mapping":
            LECTURE_MAP,

        "ground_truth_mapping":
            GROUND_TRUTH_MAP,

        "models":
            CHECKPOINTS,

        "aggregate":
            aggregate,

        "per_lecture":
            all_results,

        "comparisons":
            comparisons
    }


    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )


    print(
        "\nSaved:",
        RESULT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()