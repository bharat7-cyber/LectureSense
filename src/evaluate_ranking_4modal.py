# ============================================================
# LectureSense
# MaViLS Ground-Truth Evaluation
#
# Actual feature index formats:
#
# text_embedding_index.csv:
#   lecture_id
#   sentence_text
#   embedding_idx
#
# ocr_embedding_index.csv:
#   lecture_id
#   slide_id
#   image_path
#   embedding_row
#
# image_embedding_index.csv:
#   lecture_id
#   slide_id
#
# Evaluation:
#   Top-1 Accuracy
#   Top-3 Accuracy
#   MRR
#   Precision
#   Recall
#   F1
# ============================================================

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"

MAVILS_ROOT = DATA / "MaViLS"

OCR_EMB_PATH = (
    DATA / "ocr_embeddings.npy"
)

IMAGE_EMB_PATH = (
    DATA / "image_embeddings.npy"
)

OCR_INDEX_PATH = (
    DATA / "ocr_embedding_index.csv"
)

IMAGE_INDEX_PATH = (
    DATA / "image_embedding_index.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "ranking_4modal_best_model.pt"
)

GT_DIR = (
    MAVILS_ROOT
    / "data"
    / "ground_truth_files"
)

SRT_DIR = (
    MAVILS_ROOT
    / "data"
    / "audioscripts"
)

OUTPUT_PATH = (
    DATA / "evaluation_4modal.json"
)


# ============================================================
# LECTURE MAPPING
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


# ============================================================
# GROUND TRUTH FILE ALIASES
# ============================================================

GT_ALIASES = {

    "deeplearning_goodfellow":
        "ground_truth_deeplearning.xlsx",

    "image_processing_high_res":
        "ground_truth_image_processing.xlsx",

    "numerics_hennig":
        "ground_truth_numerics.xlsx",

    "physics_intro_02":
        "ground_truth_physics.xlsx",

    "psychology_MIT":
        "ground_truth_psychology.xlsx",

    "reinforcement_learning_silver":
        "ground_truth_reinforcement_learning.xlsx",

    "short_range_mit":
        "ground_truth_short_range.xlsx",

    "visual_system_MIT":
        "ground_truth_sensory_systems.xlsx",
}


# ============================================================
# MODEL
# ============================================================

class LectureSenseCrossModal(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Transcript
        # 384 -> 256
        # ----------------------------------------------------

        self.text_proj = nn.Sequential(

            nn.Linear(
                384,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            )
        )

        # ----------------------------------------------------
        # OCR
        # 384 -> 256
        # ----------------------------------------------------

        self.ocr_proj = nn.Sequential(

            nn.Linear(
                384,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            )
        )

        # ----------------------------------------------------
        # Image
        # 512 -> 256
        # ----------------------------------------------------

        self.image_proj = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            )
        )

        # ----------------------------------------------------
        # Fusion dimensions
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
        # 9 * 256 = 2304
        #
        # + 4 cosine similarities
        #
        # = 2308
        # ----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.Linear(
                2308,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

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

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        t = self.text_proj(
            text
        )

        o = self.ocr_proj(
            ocr
        )

        i = self.image_proj(
            image
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        t_norm = torch.nn.functional.normalize(
            t,
            dim=-1
        )

        o_norm = torch.nn.functional.normalize(
            o,
            dim=-1
        )

        i_norm = torch.nn.functional.normalize(
            i,
            dim=-1
        )

        # ----------------------------------------------------
        # Native text-OCR similarity
        # ----------------------------------------------------

        native_to = torch.sum(

            torch.nn.functional.normalize(
                text,
                dim=-1
            )
            *
            torch.nn.functional.normalize(
                ocr,
                dim=-1
            ),

            dim=-1,

            keepdim=True
        )

        # ----------------------------------------------------
        # Projected similarities
        # ----------------------------------------------------

        projected_to = torch.sum(

            t_norm * o_norm,

            dim=-1,

            keepdim=True
        )

        projected_ti = torch.sum(

            t_norm * i_norm,

            dim=-1,

            keepdim=True
        )

        projected_oi = torch.sum(

            o_norm * i_norm,

            dim=-1,

            keepdim=True
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fusion = torch.cat(

            [

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

                native_to,
                projected_to,
                projected_ti,
                projected_oi,

            ],

            dim=-1
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        return self.scorer(
            fusion
        ).squeeze(-1)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# NORMALIZE LECTURE ID
# ============================================================

def normalize_lecture_id(
    value
):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(
        value
    ).strip().lower()

    value = value.replace(
        "&",
        " and "
    )

    value = value.replace(
        "-",
        "_"
    )

    value = re.sub(
        r"\s+",
        "_",
        value
    )

    value = re.sub(
        r"_+",
        "_",
        value
    )

    return value.strip(
        "_"
    )


# ============================================================
# NORMALIZE SLIDE ID
# ============================================================

def normalize_slide_id(
    value
):

    if value is None:
        return None

    if pd.isna(value):
        return None

    try:

        return int(
            float(value)
        )

    except Exception:

        pass

    value = str(
        value
    ).strip()

    match = re.search(
        r"\d+",
        value
    )

    if match:

        return int(
            match.group()
        )

    return None


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(
    text
):

    if text is None:
        return ""

    if pd.isna(text):
        return ""

    text = str(
        text
    ).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# ============================================================
# PARSE TIME
# ============================================================

def parse_time(
    value
):

    if value is None:
        return None

    if isinstance(

        value,

        (
            int,
            float,
            np.integer,
            np.floating
        )

    ):

        if pd.isna(value):
            return None

        return float(
            value
        )

    value = str(
        value
    ).strip()

    if not value:
        return None

    # --------------------------------------------------------
    # HH:MM:SS.milliseconds
    # --------------------------------------------------------

    match = re.match(

        r"^(\d+):(\d{2}):(\d{2})"
        r"(?:[.,](\d+))?$",

        value

    )

    if match:

        h = int(
            match.group(1)
        )

        m = int(
            match.group(2)
        )

        s = int(
            match.group(3)
        )

        fraction = 0.0

        if match.group(4):

            fraction = float(
                "0."
                +
                match.group(4)
            )

        return (

            h * 3600
            +
            m * 60
            +
            s
            +
            fraction

        )

    # --------------------------------------------------------
    # MM:SS
    # --------------------------------------------------------

    match = re.match(

        r"^(\d+):(\d{2})"
        r"(?:[.,](\d+))?$",

        value

    )

    if match:

        m = int(
            match.group(1)
        )

        s = int(
            match.group(2)
        )

        fraction = 0.0

        if match.group(3):

            fraction = float(
                "0."
                +
                match.group(3)
            )

        return (

            m * 60
            +
            s
            +
            fraction

        )

    try:

        return float(
            value
        )

    except Exception:

        return None


# ============================================================
# FIND GROUND TRUTH FILE
# ============================================================

def find_ground_truth_file(
    lecture_id
):

    if not GT_DIR.exists():

        return None

    # --------------------------------------------------------
    # Exact
    # --------------------------------------------------------

    exact = (

        GT_DIR
        /
        f"ground_truth_{lecture_id}.xlsx"

    )

    if exact.exists():

        return exact

    # --------------------------------------------------------
    # Known alias
    # --------------------------------------------------------

    if lecture_id in GT_ALIASES:

        alias_file = (

            GT_DIR
            /
            GT_ALIASES[
                lecture_id
            ]

        )

        if alias_file.exists():

            return alias_file

    # --------------------------------------------------------
    # Normalized filename matching
    # --------------------------------------------------------

    target = normalize_lecture_id(

        f"ground_truth_{lecture_id}"

    )

    for file in GT_DIR.glob(
        "*.xlsx"
    ):

        if (

            normalize_lecture_id(
                file.stem
            )
            ==
            target

        ):

            return file

    # --------------------------------------------------------
    # Token overlap
    # --------------------------------------------------------

    target_tokens = set(

        normalize_lecture_id(
            lecture_id
        )
        .replace(
            "_",
            " "
        )
        .split()

    )

    best_file = None
    best_score = 0

    for file in GT_DIR.glob(
        "*.xlsx"
    ):

        stem = normalize_lecture_id(
            file.stem
        )

        stem = stem.replace(
            "ground_truth_",
            ""
        )

        file_tokens = set(

            stem
            .replace(
                "_",
                " "
            )
            .split()

        )

        score = len(

            target_tokens
            &
            file_tokens

        )

        if score > best_score:

            best_score = score

            best_file = file

    if best_score >= 2:

        return best_file

    return None


# ============================================================
# FIND SRT FILE
# ============================================================

def find_srt_file(
    lecture_id
):

    if not SRT_DIR.exists():

        return None

    exact = (

        SRT_DIR
        /
        f"{lecture_id}.srt"

    )

    if exact.exists():

        return exact

    target = normalize_lecture_id(
        lecture_id
    )

    srt_files = list(
        SRT_DIR.rglob(
            "*.srt"
        )
    )

    for file in srt_files:

        if (

            normalize_lecture_id(
                file.stem
            )
            ==
            target

        ):

            return file

    for file in srt_files:

        stem = normalize_lecture_id(
            file.stem
        )

        if target in stem:

            return file

    return None


# ============================================================
# READ SRT
# ============================================================

def read_srt(
    path
):

    text = path.read_text(

        encoding="utf-8-sig",

        errors="ignore"

    )

    segments = []

    # ========================================================
    # MaViLS FORMAT
    #
    # [00:00:10.420 --> 00:00:10.880] Hi,
    # ========================================================

    pattern = re.compile(

        r"\[\s*"
        r"([0-9:. ,]+)"
        r"\s*-->\s*"
        r"([0-9:. ,]+)"
        r"\s*\]"
        r"\s*(.*)"

    )

    matches = list(

        pattern.finditer(
            text
        )

    )

    if matches:

        for match in matches:

            start = parse_time(

                match.group(1)
                .strip()
                .replace(
                    ",",
                    "."
                )

            )

            end = parse_time(

                match.group(2)
                .strip()
                .replace(
                    ",",
                    "."
                )

            )

            word = match.group(
                3
            ).strip()

            if (

                start is None
                or
                not word

            ):

                continue

            segments.append({

                "start":
                    start,

                "end":
                    end,

                "text":
                    word,

            })

        return segments

    # ========================================================
    # STANDARD SRT FALLBACK
    # ========================================================

    blocks = re.split(

        r"\n\s*\n",

        text.strip()

    )

    for block in blocks:

        lines = (

            block
            .strip()
            .splitlines()

        )

        if len(lines) < 2:

            continue

        time_line = None

        text_lines = []

        for line in lines:

            if "-->" in line:

                time_line = line

            elif not re.match(

                r"^\d+$",

                line.strip()

            ):

                text_lines.append(
                    line.strip()
                )

        if time_line is None:

            continue

        parts = time_line.split(
            "-->"
        )

        if len(parts) != 2:

            continue

        start = parse_time(

            parts[0]
            .strip()
            .replace(
                ",",
                "."
            )

        )

        end = parse_time(

            parts[1]
            .strip()
            .replace(
                ",",
                "."
            )

        )

        sentence = " ".join(
            text_lines
        ).strip()

        if (

            start is None
            or
            not sentence

        ):

            continue

        segments.append({

            "start":
                start,

            "end":
                end,

            "text":
                sentence,

        })

    return segments


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(
    path
):

    df = pd.read_excel(
        path
    )

    columns = {

        str(column)
        .lower()
        .strip():
        column

        for column in df.columns

    }

    time_col = None
    slide_col = None
    text_col = None

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    for key, original in columns.items():

        if key in {

            "key",
            "time",
            "timestamp",
            "start",
            "starttime"

        }:

            time_col = original

            break

    # --------------------------------------------------------
    # Slide
    # --------------------------------------------------------

    for key, original in columns.items():

        if key in {

            "slidenumber",
            "slide_number",
            "slide",
            "slideid"

        }:

            slide_col = original

            break

    # --------------------------------------------------------
    # Sentence
    # --------------------------------------------------------

    for key, original in columns.items():

        if key in {

            "value",
            "sentence",
            "text",
            "transcript"

        }:

            text_col = original

            break

    if slide_col is None:

        raise ValueError(

            f"Could not find slide column "
            f"in {path}"

        )

    rows = []

    for _, row in df.iterrows():

        slide_id = normalize_slide_id(

            row[slide_col]

        )

        if slide_id is None:

            continue

        # MaViLS -1 = no unique slide

        if slide_id < 1:

            continue

        timestamp = None

        if time_col is not None:

            timestamp = parse_time(

                row[time_col]

            )

        sentence = ""

        if text_col is not None:

            value = row[text_col]

            if not pd.isna(value):

                sentence = str(
                    value
                ).strip()

        rows.append({

            "timestamp":
                timestamp,

            "slide_id":
                slide_id,

            "text":
                sentence,

        })

    return rows


# ============================================================
# TEXT SIMILARITY
# ============================================================

def token_jaccard(
    a,
    b
):

    a = set(

        normalize_text(a)
        .split()

    )

    b = set(

        normalize_text(b)
        .split()

    )

    if not a or not b:

        return 0.0

    return (

        len(a & b)
        /
        len(a | b)

    )


# ============================================================
# MATCH GT TO SRT
# ============================================================

def match_ground_truth_to_srt(

    gt_rows,

    srt_segments

):

    matched = []

    normalized_segments = [

        normalize_text(
            segment["text"]
        )

        for segment
        in srt_segments

    ]

    used = set()

    for gt in gt_rows:

        gt_text = normalize_text(
            gt["text"]
        )

        best_idx = None

        best_score = 0.0

        # ----------------------------------------------------
        # Exact text
        # ----------------------------------------------------

        if gt_text:

            for i, segment_text in enumerate(

                normalized_segments

            ):

                if i in used:

                    continue

                if segment_text == gt_text:

                    best_idx = i

                    best_score = 1.0

                    break

        # ----------------------------------------------------
        # Fuzzy text
        # ----------------------------------------------------

        if (

            best_idx is None
            and
            gt_text

        ):

            for i, segment_text in enumerate(

                normalized_segments

            ):

                if i in used:

                    continue

                score = token_jaccard(

                    gt_text,

                    segment_text

                )

                if score > best_score:

                    best_score = score

                    best_idx = i

        # ----------------------------------------------------
        # Timestamp fallback
        # ----------------------------------------------------

        if (

            best_idx is None
            and
            gt["timestamp"] is not None

        ):

            best_distance = float(
                "inf"
            )

            for i, segment in enumerate(
                srt_segments
            ):

                if i in used:

                    continue

                distance = abs(

                    segment["start"]
                    -
                    gt["timestamp"]

                )

                if distance < best_distance:

                    best_distance = distance

                    best_idx = i

            if best_distance > 10:

                best_idx = None

        # ----------------------------------------------------
        # Save match
        # ----------------------------------------------------

        if best_idx is not None:

            used.add(
                best_idx
            )

            final_text = gt[
                "text"
            ]

            if not final_text.strip():

                final_text = (

                    srt_segments[
                        best_idx
                    ]["text"]

                )

            final_timestamp = (

                gt["timestamp"]

                if
                gt["timestamp"] is not None

                else

                srt_segments[
                    best_idx
                ]["start"]

            )

            matched.append({

                "timestamp":
                    final_timestamp,

                "text":
                    final_text,

                "slide_id":
                    gt["slide_id"],

                "match_score":
                    best_score,

            })

    return matched


# ============================================================
# GET SLIDE EMBEDDINGS
#
# ACTUAL FILE FORMAT:
#
# image_embedding_index.csv
#     lecture_id
#     slide_id
#
# image_embeddings.npy
#     one embedding per row of image index
#
# OCR:
#     lecture_id
#     slide_id
#     image_path
#     embedding_row
#
# Therefore:
#
# image embedding row = ORIGINAL image-index DataFrame index
#
# OCR embedding row   = embedding_row column
# ============================================================

def get_slide_embeddings(

    lecture_feature_id,

    image_index,

    image_embeddings,

    ocr_index,

    ocr_embeddings

):

    target = normalize_lecture_id(
        lecture_feature_id
    )

    print(
        "\nLooking for slide features "
        f"using lecture ID: "
        f"{lecture_feature_id}"
    )

    print(
        f"Normalized lecture ID: "
        f"{target}"
    )

    # ========================================================
    # IMAGE INDEX
    # ========================================================

    image_ids = (

        image_index[
            "lecture_id"
        ]
        .apply(
            normalize_lecture_id
        )

    )

    image_rows = image_index[
        image_ids == target
    ].copy()

    # IMPORTANT:
    # Do NOT reset index here.
    #
    # The original row number corresponds to
    # image_embeddings.npy.
    # ========================================================

    image_rows[
        "_embedding_row"
    ] = image_rows.index

    # ========================================================
    # OCR INDEX
    # ========================================================

    ocr_ids = (

        ocr_index[
            "lecture_id"
        ]
        .apply(
            normalize_lecture_id
        )

    )

    ocr_rows = ocr_index[
        ocr_ids == target
    ].copy()

    print(
        f"Image rows for "
        f"'{lecture_feature_id}': "
        f"{len(image_rows)}"
    )

    print(
        f"OCR rows for "
        f"'{lecture_feature_id}': "
        f"{len(ocr_rows)}"
    )

    # ========================================================
    # IMAGE MAP
    # ========================================================

    image_map = {}

    for _, row in image_rows.iterrows():

        slide_id = normalize_slide_id(

            row["slide_id"]

        )

        if slide_id is None:

            continue

        embedding_row = int(

            row[
                "_embedding_row"
            ]

        )

        if (

            embedding_row < 0
            or
            embedding_row >= len(
                image_embeddings
            )

        ):

            print(

                f"Warning: image embedding "
                f"row {embedding_row} "
                f"out of range."

            )

            continue

        image_map[
            slide_id
        ] = embedding_row

    # ========================================================
    # OCR MAP
    # ========================================================

    ocr_map = {}

    for _, row in ocr_rows.iterrows():

        slide_id = normalize_slide_id(

            row["slide_id"]

        )

        if slide_id is None:

            continue

        try:

            embedding_row = int(

                row[
                    "embedding_row"
                ]

            )

        except Exception:

            continue

        if (

            embedding_row < 0
            or
            embedding_row >= len(
                ocr_embeddings
            )

        ):

            print(

                f"Warning: OCR embedding "
                f"row {embedding_row} "
                f"out of range."

            )

            continue

        ocr_map[
            slide_id
        ] = embedding_row

    # ========================================================
    # COMMON SLIDES
    # ========================================================

    common_slide_ids = sorted(

        set(
            image_map.keys()
        )

        &

        set(
            ocr_map.keys()
        )

    )

    print(
        f"Image slides: "
        f"{len(image_map)}"
    )

    print(
        f"OCR slides: "
        f"{len(ocr_map)}"
    )

    print(
        f"Common image+OCR slides: "
        f"{len(common_slide_ids)}"
    )

    # ========================================================
    # CREATE SLIDE LIST
    # ========================================================

    slides = []

    for slide_id in common_slide_ids:

        image_row = image_map[
            slide_id
        ]

        ocr_row = ocr_map[
            slide_id
        ]

        slides.append({

            "slide_id":
                slide_id,

            "image":
                image_embeddings[
                    image_row
                ],

            "ocr":
                ocr_embeddings[
                    ocr_row
                ],

        })

    return slides


# ============================================================
# SCORE SAMPLES
# ============================================================

def score_lecture_samples(

    matched,

    slides,

    model,

    text_model

):

    if not matched:

        return None

    if not slides:

        return None

    # ========================================================
    # ENCODE GROUND-TRUTH SENTENCES
    # ========================================================

    texts = [

        item["text"]

        for item in matched

    ]

    print(
        "\nEncoding transcript sentences..."
    )

    text_vectors = text_model.encode(

        texts,

        convert_to_numpy=True,

        normalize_embeddings=False,

        show_progress_bar=True,

        batch_size=64

    ).astype(
        np.float32
    )

    # ========================================================
    # SLIDE FEATURES
    # ========================================================

    slide_ocr = np.stack(

        [

            slide["ocr"]

            for slide in slides

        ]

    ).astype(
        np.float32
    )

    slide_image = np.stack(

        [

            slide["image"]

            for slide in slides

        ]

    ).astype(
        np.float32
    )

    # ========================================================
    # TENSORS
    # ========================================================

    text_tensor = torch.from_numpy(
        text_vectors
    ).to(
        DEVICE
    )

    ocr_tensor = torch.from_numpy(
        slide_ocr
    ).to(
        DEVICE
    )

    image_tensor = torch.from_numpy(
        slide_image
    ).to(
        DEVICE
    )

    predictions = []

    top3_hits = []

    reciprocal_ranks = []

    model.eval()

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for start in range(

            0,

            len(matched),

            64

        ):

            end = min(

                start + 64,

                len(matched)

            )

            batch_text = text_tensor[
                start:end
            ]

            batch_size = (
                batch_text.shape[0]
            )

            num_slides = (
                ocr_tensor.shape[0]
            )

            # ------------------------------------------------
            # Transcript repeated across candidate slides
            # ------------------------------------------------

            repeated_text = (

                batch_text

                .unsqueeze(1)

                .expand(

                    batch_size,

                    num_slides,

                    384

                )

                .reshape(

                    -1,

                    384

                )

            )

            # ------------------------------------------------
            # OCR candidates
            # ------------------------------------------------

            repeated_ocr = (

                ocr_tensor

                .unsqueeze(0)

                .expand(

                    batch_size,

                    num_slides,

                    384

                )

                .reshape(

                    -1,

                    384

                )

            )

            # ------------------------------------------------
            # Image candidates
            # ------------------------------------------------

            repeated_image = (

                image_tensor

                .unsqueeze(0)

                .expand(

                    batch_size,

                    num_slides,

                    512

                )

                .reshape(

                    -1,

                    512

                )

            )

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            scores = model(

                repeated_text,

                repeated_ocr,

                repeated_image

            )

            scores = scores.reshape(

                batch_size,

                num_slides

            )

            # ------------------------------------------------
            # Each sentence
            # ------------------------------------------------

            for j in range(
                batch_size
            ):

                row = scores[j]

                sorted_indices = (

                    torch.argsort(

                        row,

                        descending=True

                    )

                    .cpu()

                    .numpy()

                )

                predicted_slide = (

                    slides[
                        sorted_indices[0]
                    ]["slide_id"]

                )

                true_slide = matched[
                    start + j
                ]["slide_id"]

                # ------------------------------------------------
                # Top 3
                # ------------------------------------------------

                top3_ids = [

                    slides[index][
                        "slide_id"
                    ]

                    for index
                    in sorted_indices[:3]

                ]

                hit3 = (

                    true_slide
                    in
                    top3_ids

                )

                # ------------------------------------------------
                # MRR
                # ------------------------------------------------

                reciprocal_rank = 0.0

                for rank, index in enumerate(

                    sorted_indices,

                    start=1

                ):

                    candidate = (

                        slides[index][
                            "slide_id"
                        ]

                    )

                    if candidate == true_slide:

                        reciprocal_rank = (

                            1.0
                            /
                            rank

                        )

                        break

                predictions.append(
                    predicted_slide
                )

                top3_hits.append(
                    int(hit3)
                )

                reciprocal_ranks.append(
                    reciprocal_rank
                )

    return (

        predictions,

        top3_hits,

        reciprocal_ranks

    )


# ============================================================
# EVALUATE ONE LECTURE
# ============================================================

def evaluate_lecture(

    lecture_id,

    model,

    text_model,

    ocr_embeddings,

    ocr_index,

    image_embeddings,

    image_index

):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"LECTURE: {lecture_id}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    gt_file = find_ground_truth_file(
        lecture_id
    )

    if gt_file is None:

        print(
            "Ground truth file not found."
        )

        return None

    print(
        f"Ground truth: {gt_file}"
    )

    gt_rows = read_ground_truth(
        gt_file
    )

    print(
        f"Ground truth rows: "
        f"{len(gt_rows)}"
    )

    print(
        f"Usable GT rows: "
        f"{len(gt_rows)}"
    )

    if not gt_rows:

        return None

    # ========================================================
    # SRT
    # ========================================================

    srt_file = find_srt_file(
        lecture_id
    )

    if srt_file is None:

        print(
            "SRT file not found."
        )

        return None

    print(
        f"SRT: {srt_file}"
    )

    srt_segments = read_srt(
        srt_file
    )

    print(
        f"SRT word entries: "
        f"{len(srt_segments)}"
    )

    if not srt_segments:

        return None

    # ========================================================
    # MATCH
    # ========================================================

    matched = match_ground_truth_to_srt(

        gt_rows,

        srt_segments

    )

    print(
        f"Matched GT/SRT rows: "
        f"{len(matched)}"
    )

    if not matched:

        return None

    # ========================================================
    # FEATURE ID
    # ========================================================

    feature_lecture_id = (

        LECTURE_MAP.get(

            lecture_id,

            lecture_id

        )

    )

    print(
        f"Pair lecture ID: "
        f"{lecture_id}"
    )

    print(
        f"Feature lecture ID: "
        f"{feature_lecture_id}"
    )

    # ========================================================
    # SLIDE FEATURES
    # ========================================================

    slides = get_slide_embeddings(

        feature_lecture_id,

        image_index,

        image_embeddings,

        ocr_index,

        ocr_embeddings

    )

    print(
        f"Candidate slides: "
        f"{len(slides)}"
    )

    if not slides:

        print(
            "No candidate slide features."
        )

        return None

    # ========================================================
    # REMOVE GT SLIDES NOT IN CANDIDATES
    # ========================================================

    candidate_slide_ids = {

        slide["slide_id"]

        for slide in slides

    }

    valid_matched = [

        item

        for item in matched

        if item["slide_id"]
        in candidate_slide_ids

    ]

    removed = (

        len(matched)
        -
        len(valid_matched)

    )

    print(
        "Removed GT rows without "
        f"candidate slide: {removed}"
    )

    print(
        f"Evaluation samples: "
        f"{len(valid_matched)}"
    )

    if not valid_matched:

        return None

    # ========================================================
    # SCORE
    # ========================================================

    result = score_lecture_samples(

        valid_matched,

        slides,

        model,

        text_model

    )

    if result is None:

        return None

    (
        predictions,
        top3_hits,
        reciprocal_ranks
    ) = result

    # ========================================================
    # METRICS
    # ========================================================

    true_labels = np.array(

        [

            item["slide_id"]

            for item in valid_matched

        ]

    )

    predicted_labels = np.array(
        predictions
    )

    top1 = float(

        np.mean(

            predicted_labels
            ==
            true_labels

        )

    )

    top3 = float(

        np.mean(
            top3_hits
        )

    )

    mrr = float(

        np.mean(
            reciprocal_ranks
        )

    )

    # For single-label multiclass prediction:
    #
    # micro precision = accuracy
    # micro recall    = accuracy
    # micro F1        = accuracy

    precision = top1

    recall = top1

    f1 = top1

    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(
        f"Top-1 Accuracy : "
        f"{top1:.4f}"
    )

    print(
        f"Top-3 Accuracy : "
        f"{top3:.4f}"
    )

    print(
        f"MRR            : "
        f"{mrr:.4f}"
    )

    print(
        f"Precision      : "
        f"{precision:.4f}"
    )

    print(
        f"Recall         : "
        f"{recall:.4f}"
    )

    print(
        f"F1             : "
        f"{f1:.4f}"
    )

    return {

        "lecture_id":
            lecture_id,

        "feature_lecture_id":
            feature_lecture_id,

        "ground_truth_file":
            str(gt_file),

        "srt_file":
            str(srt_file),

        "ground_truth_rows":
            len(gt_rows),

        "matched_rows":
            len(matched),

        "evaluated_rows":
            len(valid_matched),

        "removed_rows":
            removed,

        "candidate_slides":
            len(slides),

        "top1_accuracy":
            top1,

        "top3_accuracy":
            top3,

        "mrr":
            mrr,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate LectureSense "
            "ranking model on MaViLS."
        )

    )

    parser.add_argument(

        "--lecture",

        type=str,

        default=None,

        help=(
            "Evaluate one lecture only. "
            "Without this argument, evaluate all lectures."
        )

    )

    args = parser.parse_args()

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required_files = [

        CHECKPOINT_PATH,

        OCR_EMB_PATH,

        IMAGE_EMB_PATH,

        OCR_INDEX_PATH,

        IMAGE_INDEX_PATH,

    ]

    print(
        "\nChecking required files..."
    )

    for path in required_files:

        if path.exists():

            print(
                f"OK: {path}"
            )

        else:

            print(
                f"MISSING: {path}"
            )

            sys.exit(1)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print(
        "\nLoading trained model..."
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE

    )

    model = (
        LectureSenseCrossModal()
        .to(DEVICE)
    )

    if (

        isinstance(
            checkpoint,
            dict
        )

        and

        "model_state_dict"
        in checkpoint

    ):

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]

        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    # ========================================================
    # CHECKPOINT INFO
    # ========================================================

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            print(

                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"

            )

        if "val_top1" in checkpoint:

            print(

                f"Checkpoint Val Top-1: "
                f"{checkpoint['val_top1']:.4f}"

            )

        if "val_top3" in checkpoint:

            print(

                f"Checkpoint Val Top-3: "
                f"{checkpoint['val_top3']:.4f}"

            )

        if "val_mrr" in checkpoint:

            print(

                f"Checkpoint Val MRR: "
                f"{checkpoint['val_mrr']:.4f}"

            )

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    print(
        "\nLoading feature files..."
    )

    ocr_embeddings = np.load(
        OCR_EMB_PATH
    )

    image_embeddings = np.load(
        IMAGE_EMB_PATH
    )

    # ========================================================
    # LOAD INDEXES
    # ========================================================

    ocr_index = pd.read_csv(
        OCR_INDEX_PATH
    )

    image_index = pd.read_csv(
        IMAGE_INDEX_PATH
    )

    print(
        f"OCR embeddings  : "
        f"{ocr_embeddings.shape}"
    )

    print(
        f"Image embeddings: "
        f"{image_embeddings.shape}"
    )

    print(
        f"OCR index rows  : "
        f"{len(ocr_index)}"
    )

    print(
        f"Image index rows: "
        f"{len(image_index)}"
    )

    # ========================================================
    # VALIDATE DIMENSIONS
    # ========================================================

    if (

        ocr_embeddings.ndim != 2
        or
        ocr_embeddings.shape[1] != 384

    ):

        raise ValueError(

            "Expected OCR embeddings "
            "shape [N,384], got "
            f"{ocr_embeddings.shape}"

        )

    if (

        image_embeddings.ndim != 2
        or
        image_embeddings.shape[1] != 512

    ):

        raise ValueError(

            "Expected image embeddings "
            "shape [N,512], got "
            f"{image_embeddings.shape}"

        )

    # ========================================================
    # VALIDATE INDEX SCHEMAS
    # ========================================================

    expected_ocr_columns = {

        "lecture_id",

        "slide_id",

        "image_path",

        "embedding_row"

    }

    expected_image_columns = {

        "lecture_id",

        "slide_id"

    }

    missing_ocr = (

        expected_ocr_columns
        -
        set(
            ocr_index.columns
        )

    )

    missing_image = (

        expected_image_columns
        -
        set(
            image_index.columns
        )

    )

    if missing_ocr:

        raise ValueError(

            "OCR index missing columns: "
            f"{missing_ocr}"

        )

    if missing_image:

        raise ValueError(

            "Image index missing columns: "
            f"{missing_image}"

        )

    # ========================================================
    # LOAD SENTENCE-BERT
    # ========================================================

    print(
        "\nLoading Sentence-BERT..."
    )

    text_model = SentenceTransformer(

        "all-MiniLM-L6-v2",

        device=str(
            DEVICE
        )

    )

    # ========================================================
    # SELECT LECTURES
    # ========================================================

    if args.lecture:

        requested = args.lecture.strip()

        if requested in LECTURE_MAP:

            lectures = [
                requested
            ]

        else:

            normalized = normalize_lecture_id(
                requested
            )

            matched = None

            for lecture in LECTURE_MAP:

                if (

                    normalize_lecture_id(
                        lecture
                    )
                    ==
                    normalized

                ):

                    matched = lecture

                    break

            if matched is None:

                print(
                    f"\nUnknown lecture: "
                    f"{args.lecture}"
                )

                print(
                    "\nAvailable lectures:"
                )

                for lecture in LECTURE_MAP:

                    print(
                        f"  {lecture}"
                    )

                sys.exit(1)

            lectures = [
                matched
            ]

    else:

        lectures = list(
            LECTURE_MAP.keys()
        )

    print(
        f"\nLectures to evaluate: "
        f"{len(lectures)}"
    )

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    results = []

    for lecture_id in lectures:

        try:

            result = evaluate_lecture(

                lecture_id,

                model,

                text_model,

                ocr_embeddings,

                ocr_index,

                image_embeddings,

                image_index

            )

            if result is not None:

                results.append(
                    result
                )

        except Exception as error:

            print()

            print(
                f"ERROR while evaluating "
                f"{lecture_id}:"
            )

            print(
                repr(error)
            )

            continue

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n\n"
        + "=" * 60
    )

    print(
        "FINAL MaViLS EVALUATION"
    )

    print(
        "=" * 60
    )

    if not results:

        print(
            "Lectures evaluated : 0"
        )

        print(
            "Samples evaluated  : 0"
        )

        print(
            "Top-1 Accuracy     : 0.0000"
        )

        print(
            "Top-3 Accuracy     : 0.0000"
        )

        print(
            "MRR                : 0.0000"
        )

        print(
            "Precision          : 0.0000"
        )

        print(
            "Recall             : 0.0000"
        )

        print(
            "F1                 : 0.0000"
        )

    else:

        total_samples = sum(

            result[
                "evaluated_rows"
            ]

            for result in results

        )

        top1 = (

            sum(

                result[
                    "top1_accuracy"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        top3 = (

            sum(

                result[
                    "top3_accuracy"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        mrr = (

            sum(

                result[
                    "mrr"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        print(
            f"Lectures evaluated : "
            f"{len(results)}"
        )

        print(
            f"Samples evaluated  : "
            f"{total_samples}"
        )

        print(
            f"Top-1 Accuracy     : "
            f"{top1:.4f}"
        )

        print(
            f"Top-3 Accuracy     : "
            f"{top3:.4f}"
        )

        print(
            f"MRR                : "
            f"{mrr:.4f}"
        )

        print(
            f"Precision          : "
            f"{top1:.4f}"
        )

        print(
            f"Recall             : "
            f"{top1:.4f}"
        )

        print(
            f"F1                 : "
            f"{top1:.4f}"
        )

        # ----------------------------------------------------
        # Per lecture
        # ----------------------------------------------------

        print(
            "\nPer-lecture results:"
        )

        for result in results:

            print(

                f"{result['lecture_id']}: "

                f"Top1="
                f"{result['top1_accuracy']:.4f}, "

                f"Top3="
                f"{result['top3_accuracy']:.4f}, "

                f"MRR="
                f"{result['mrr']:.4f}, "

                f"F1="
                f"{result['f1']:.4f}, "

                f"N="
                f"{result['evaluated_rows']}"

            )

    print(
        "=" * 60
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "evaluation_type":
            "MaViLS ground-truth "
            "slide ranking",

        "model":
            "LectureSenseCrossModal",

        "modalities": [

            "transcript_text",

            "slide_ocr",

            "slide_image",

        ],

        "lectures_requested":
            len(lectures),

        "lectures_evaluated":
            len(results),

        "results":
            results,

    }

    if results:

        total_samples = sum(

            result[
                "evaluated_rows"
            ]

            for result in results

        )

        top1 = (

            sum(

                result[
                    "top1_accuracy"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        top3 = (

            sum(

                result[
                    "top3_accuracy"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        mrr = (

            sum(

                result[
                    "mrr"
                ]
                *
                result[
                    "evaluated_rows"
                ]

                for result in results

            )
            /
            total_samples

        )

        output[
            "aggregate"
        ] = {

            "samples":
                int(
                    total_samples
                ),

            "top1_accuracy":
                float(
                    top1
                ),

            "top3_accuracy":
                float(
                    top3
                ),

            "mrr":
                float(
                    mrr
                ),

            "precision":
                float(
                    top1
                ),

            "recall":
                float(
                    top1
                ),

            "f1":
                float(
                    top1
                ),

        }

    OUTPUT_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        OUTPUT_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            output,

            file,

            indent=2

        )

    print(
        "\nSaved evaluation:"
    )

    print(
        OUTPUT_PATH.resolve()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()