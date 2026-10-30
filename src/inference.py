"""
LectureSense
============

Offline inference using:

    MaViLS SRT transcript
            +
    Lecture PDF slides
            |
            v
    Transcript SBERT embeddings
    Slide OCR SBERT embeddings
    Slide ResNet-18 embeddings
            |
            v
    Trained 4-modal LectureSense model
            |
            v
    Transcript-Slide score matrix
            |
            v
    Dynamic Programming alignment
            |
            v
    Timeline JSON

Optional inference-time visual evidence:
    Lecture video frames -> ResNet-18 -> video/slide visual similarity
    -> fused with the learned multimodal score before alignment

Example:

python src\\inference.py ^
  --transcript data\\MaViLS\\data\\audioscripts\\computer_vision_2_2_high_res.srt ^
  --slides data\\MaViLS\\data\\lectures\\computer_vision_2_2_geiger.pdf ^
  --checkpoint checkpoints\\ranking_4modal_best_model.pt ^
  --out demo\\timeline.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from dp_alignment import dp_align, build_timeline
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os
import shutil
import pytesseract

TESSERACT_PATH = (
    shutil.which("tesseract")
    or (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        else None
    )
)

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print("WARNING: Tesseract OCR is not installed.")
from sentence_transformers import SentenceTransformer

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from video_visual_alignment import (
    create_video_frame_embeddings,
    build_video_slide_score_matrix,
    fuse_model_and_video_scores,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_DIM = 384
OCR_DIM = 384
IMAGE_DIM = 512
PROJECTED_DIM = 256
FUSION_DIM = 2308

DEFAULT_CHECKPOINT = (
    "checkpoints/ranking_4modal_best_model.pt"
)

DEFAULT_OUTPUT = (
    "demo/timeline.json"
)

DEFAULT_RENDER_DIR = (
    "demo/rendered_slides"
)

DEFAULT_SCORE_BATCH = 4
DEFAULT_VIDEO_SAMPLE_SECONDS = 1.0
DEFAULT_VIDEO_WEIGHT = 0.30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean transcript/OCR text.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove subtitle tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Common subtitle line break marker
    text = text.replace(
        "\\N",
        " ",
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TIME CONVERSION
# ============================================================

def parse_time_to_seconds(value):
    """
    Convert:

        HH:MM:SS.mmm

    into seconds.

    Also supports:

        HH:MM:SS,mmm
    """

    if value is None:
        return None

    value = str(value).strip()

    value = value.replace(
        ",",
        ".",
    )

    parts = value.split(":")

    try:

        if len(parts) == 3:

            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            return (
                hours * 3600.0
                + minutes * 60.0
                + seconds
            )

        elif len(parts) == 2:

            minutes = float(parts[0])
            seconds = float(parts[1])

            return (
                minutes * 60.0
                + seconds
            )

        else:

            return float(parts[0])

    except (ValueError, TypeError):

        return None


# ============================================================
# MaViLS SRT READER
# ============================================================

def read_mavils_srt(path):
    """
    Read the MaViLS word-level transcript format.

    Typical MaViLS lines look like:

        [00:00:01.070 --> 00:00:01.730] In
        [00:00:01.730 --> 00:00:02.110] the
        [00:00:02.110 --> 00:00:02.370] second

    Each line represents a timestamped word/token.

    We first parse all word entries and then combine them
    into longer transcript segments.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {path}"
        )

    print(
        f"Reading MaViLS transcript: {path}"
    )

    raw_text = path.read_text(
        encoding="utf-8-sig",
        errors="ignore",
    )

    raw_text = raw_text.replace(
        "\r\n",
        "\n",
    )

    raw_text = raw_text.replace(
        "\r",
        "\n",
    )

    lines = raw_text.splitlines()

    # --------------------------------------------------------
    # IMPORTANT:
    # MaViLS uses square brackets.
    #
    # [00:00:01.070 --> 00:00:01.730] In
    # --------------------------------------------------------

    word_pattern = re.compile(
        r"^\s*\[\s*"
        r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d+)"
        r"\s*-->\s*"
        r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d+)"
        r"\s*\]\s*"
        r"(?P<word>.*)"
        r"$"
    )

    words = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        match = word_pattern.match(line)

        if match is None:

            # Some MaViLS versions may have extra
            # spaces before/after brackets.
            alternate = re.match(
                r"\[?\s*"
                r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
                r"\s*-->\s*"
                r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
                r"\s*\]?\s*(.*)$",
                line,
            )

            if alternate is None:
                continue

            start_text = alternate.group(1)
            end_text = alternate.group(2)
            word_text = alternate.group(3)

        else:

            start_text = match.group(
                "start"
            )

            end_text = match.group(
                "end"
            )

            word_text = match.group(
                "word"
            )

        start = parse_time_to_seconds(
            start_text
        )

        end = parse_time_to_seconds(
            end_text
        )

        word = clean_text(
            word_text
        )

        if (
            start is None
            or end is None
            or not word
        ):
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
                "text": word,
            }
        )

    print(
        f"Raw MaViLS word entries: "
        f"{len(words)}"
    )

    if not words:

        raise RuntimeError(
            "No MaViLS transcript entries were parsed. "
            "Expected lines similar to:\n"
            "[00:00:01.070 --> 00:00:01.730] In"
        )

    # ========================================================
    # MERGE WORDS
    # ========================================================

    segments = []

    current_words = []
    current_start = None
    current_end = None

    # Approximately 8 seconds per transcript segment.
    MAX_SEGMENT_SECONDS = 8.0

    for word_item in words:

        if current_start is None:

            current_start = word_item[
                "start"
            ]

        current_words.append(
            word_item["text"]
        )

        current_end = word_item[
            "end"
        ]

        current_text = clean_text(
            " ".join(current_words)
        )

        duration = (
            current_end
            - current_start
        )

        sentence_boundary = (
            current_text.endswith(".")
            or current_text.endswith("?")
            or current_text.endswith("!")
            or current_text.endswith(":")
            or current_text.endswith(";")
        )

        time_boundary = (
            duration >= MAX_SEGMENT_SECONDS
        )

        if (
            sentence_boundary
            or time_boundary
        ):

            if current_text:

                segments.append(
                    {
                        "start":
                            current_start,

                        "end":
                            current_end,

                        "text":
                            current_text,
                    }
                )

            current_words = []
            current_start = None
            current_end = None

    # Add remaining words
    if current_words:

        current_text = clean_text(
            " ".join(current_words)
        )

        if current_text:

            segments.append(
                {
                    "start":
                        current_start,

                    "end":
                        current_end,

                    "text":
                        current_text,
                }
            )

    print(
        f"Merged transcript segments: "
        f"{len(segments)}"
    )

    # Final validation
    final_segments = []

    for item in segments:

        text = clean_text(
            item["text"]
        )

        if not text:
            continue

        if re.fullmatch(
            r"[\W_]+",
            text,
        ):
            continue

        final_segments.append(
            {
                "start":
                    float(item["start"]),

                "end":
                    float(item["end"]),

                "text":
                    text,
            }
        )

    print(
        f"Final transcript segments: "
        f"{len(final_segments)}"
    )

    return final_segments


# ============================================================
# PDF RENDERING
# ============================================================

def render_pdf(
    pdf_path,
    output_dir,
):
    """
    Render PDF pages to PNG slide images.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"Slides PDF not found: {pdf_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    print(
        f"Opening PDF: {pdf_path}"
    )

    document = fitz.open(
        str(pdf_path)
    )

    slide_paths = []

    for index in range(
        len(document)
    ):

        page = document[index]

        matrix = fitz.Matrix(
            2.0,
            2.0,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        slide_path = (
            output_dir
            / f"slide_{index + 1:04d}.png"
        )

        pixmap.save(
            str(slide_path)
        )

        slide_paths.append(
            slide_path
        )

    document.close()

    print(
        f"Rendered slides: "
        f"{len(slide_paths)}"
    )

    return slide_paths


# ============================================================
# OCR
# ============================================================

def extract_ocr(
    slide_paths,
):
    """
    Extract text from each rendered slide.
    """

    print(
        "Running OCR on slides..."
    )

    ocr_texts = []

    total = len(
        slide_paths
    )

    for index, slide_path in enumerate(
        slide_paths,
        start=1,
    ):

        try:

            image = Image.open(
                slide_path
            ).convert(
                "RGB"
            )

            text = pytesseract.image_to_string(
                image
            )

            text = clean_text(
                text
            )

        except Exception as exc:

            print(
                f"OCR warning on slide "
                f"{index}: {exc}"
            )

            text = ""

        ocr_texts.append(
            text
        )

        if (
            index == 1
            or index % 10 == 0
            or index == total
        ):

            print(
                f"OCR: "
                f"{index}/{total}"
            )

    return ocr_texts


# ============================================================
# SENTENCE-BERT
# ============================================================

def load_sbert():

    print(
        "Loading Sentence-BERT..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=str(DEVICE),
    )

    return model


def create_text_embeddings(
    model,
    texts,
):
    """
    Produce 384-dimensional embeddings.
    """

    cleaned = []

    for text in texts:

        value = clean_text(
            text
        )

        if not value:
            value = " "

        cleaned.append(
            value
        )

    if not cleaned:

        return np.empty(
            (
                0,
                TEXT_DIM,
            ),
            dtype=np.float32,
        )

    embeddings = model.encode(
        cleaned,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# ============================================================
# RESNET-18
# ============================================================

def create_image_embeddings(
    slide_paths,
):
    """
    Generate 512-dimensional ResNet-18 features.
    """

    print(
        "Loading ResNet-18..."
    )

    weights = (
        ResNet18_Weights.DEFAULT
    )

    model = resnet18(
        weights=weights
    )

    model.fc = nn.Identity()

    model = model.to(
        DEVICE
    )

    model.eval()

    preprocess = (
        weights.transforms()
    )

    batch_size = 8

    all_features = []

    total = len(
        slide_paths
    )

    with torch.no_grad():

        for start in range(
            0,
            total,
            batch_size,
        ):

            batch_paths = slide_paths[
                start:
                start + batch_size
            ]

            batch_images = []

            for slide_path in batch_paths:

                image = Image.open(
                    slide_path
                ).convert(
                    "RGB"
                )

                image = preprocess(
                    image
                )

                batch_images.append(
                    image
                )

            batch = torch.stack(
                batch_images
            ).to(
                DEVICE
            )

            features = model(
                batch
            )

            all_features.append(
                features
                .cpu()
            )

            processed = min(
                start + batch_size,
                total,
            )

            print(
                f"Image features: "
                f"{processed}/{total}"
            )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_features:

        return np.empty(
            (
                0,
                IMAGE_DIM,
            ),
            dtype=np.float32,
        )

    features = torch.cat(
        all_features,
        dim=0,
    )

    return features.numpy().astype(
        np.float32
    )


# ============================================================
# EXACT TRAINED MODEL
# ============================================================

class LectureSenseCrossModal(
    nn.Module
):
    """
    Exact architecture used by the trained
    4-modal LectureSense model.
    """

    def __init__(self):

        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(
                384,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15,
            ),
        )

        self.ocr_proj = nn.Sequential(
            nn.Linear(
                384,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15,
            ),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(
                512,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15,
            ),
        )

        self.scorer = nn.Sequential(
            nn.Linear(
                2308,
                512,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15,
            ),

            nn.Linear(
                512,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.15,
            ),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        text,
        ocr,
        image,
    ):

        t = self.text_proj(
            text
        )

        o = self.ocr_proj(
            ocr
        )

        i = self.image_proj(
            image
        )

        t_norm = F.normalize(
            t,
            dim=-1,
        )

        o_norm = F.normalize(
            o,
            dim=-1,
        )

        i_norm = F.normalize(
            i,
            dim=-1,
        )

        native_to = torch.sum(
            F.normalize(
                text,
                dim=-1,
            )
            *
            F.normalize(
                ocr,
                dim=-1,
            ),
            dim=-1,
            keepdim=True,
        )

        projected_to = torch.sum(
            t_norm
            * o_norm,
            dim=-1,
            keepdim=True,
        )

        projected_ti = torch.sum(
            t_norm
            * i_norm,
            dim=-1,
            keepdim=True,
        )

        projected_oi = torch.sum(
            o_norm
            * i_norm,
            dim=-1,
            keepdim=True,
        )

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
            dim=-1,
        )

        return self.scorer(
            fusion
        ).squeeze(-1)


# ============================================================
# MODEL LOADING
# ============================================================

def load_trained_model(
    checkpoint_path,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    model = (
        LectureSenseCrossModal()
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # --------------------------------------------------------
    # Determine state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    # --------------------------------------------------------
    # Load exact model weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # Checkpoint metadata
    if isinstance(
        checkpoint,
        dict,
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

    return model


# ============================================================
# SCORE MATRIX
# ============================================================

def compute_score_matrix(
    model,
    transcript_embeddings,
    ocr_embeddings,
    image_embeddings,
    batch_size=4,
):
    """
    Compute model score for every:

        transcript x slide

    pair.

    Returns:
        [num_transcript_segments, num_slides]
    """

    num_transcripts = len(
        transcript_embeddings
    )

    num_slides = len(
        ocr_embeddings
    )

    if num_transcripts == 0:

        raise RuntimeError(
            "Transcript embedding array is empty."
        )

    if num_slides == 0:

        raise RuntimeError(
            "Slide embedding arrays are empty."
        )

    if len(image_embeddings) != num_slides:

        raise RuntimeError(
            "OCR and image embeddings have "
            "different numbers of slides."
        )

    print(
        "Computing transcript-slide scores..."
    )

    score_matrix = np.zeros(
        (
            num_transcripts,
            num_slides,
        ),
        dtype=np.float32,
    )

    # Keep slide tensors on GPU.
    ocr_tensor = torch.from_numpy(
        ocr_embeddings
    ).to(
        DEVICE
    )

    image_tensor = torch.from_numpy(
        image_embeddings
    ).to(
        DEVICE
    )

    with torch.no_grad():

        for start in range(
            0,
            num_transcripts,
            batch_size,
        ):

            end = min(
                start + batch_size,
                num_transcripts,
            )

            current_count = (
                end - start
            )

            text_tensor = torch.from_numpy(
                transcript_embeddings[
                    start:end
                ]
            ).to(
                DEVICE
            )

            # ----------------------------------------------
            # Repeat transcript vectors once per slide.
            # ----------------------------------------------

            text_batch = (
                text_tensor
                .unsqueeze(1)
                .expand(
                    current_count,
                    num_slides,
                    TEXT_DIM,
                )
                .reshape(
                    -1,
                    TEXT_DIM,
                )
            )

            # ----------------------------------------------
            # Repeat OCR slide vectors for each transcript.
            # ----------------------------------------------

            ocr_batch = (
                ocr_tensor
                .unsqueeze(0)
                .expand(
                    current_count,
                    -1,
                    -1,
                )
                .reshape(
                    -1,
                    OCR_DIM,
                )
            )

            # ----------------------------------------------
            # Repeat image vectors for each transcript.
            # ----------------------------------------------

            image_batch = (
                image_tensor
                .unsqueeze(0)
                .expand(
                    current_count,
                    -1,
                    -1,
                )
                .reshape(
                    -1,
                    IMAGE_DIM,
                )
            )

            batch_scores = model(
                text_batch,
                ocr_batch,
                image_batch,
            )

            batch_scores = (
                batch_scores
                .reshape(
                    current_count,
                    num_slides,
                )
                .cpu()
                .numpy()
            )

            score_matrix[
                start:end
            ] = batch_scores

            print(
                f"Scored transcript segments: "
                f"{end}/{num_transcripts}"
            )

    # Release GPU tensors
    del ocr_tensor
    del image_tensor

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return score_matrix


# ============================================================
# DYNAMIC PROGRAMMING
# ============================================================

def dp_align_legacy(
    score_matrix,
    jump_penalty=0.15,
):
    """
    Temporally consistent transcript-slide alignment.

    Same slide:
        penalty = 0

    Adjacent slide:
        penalty = 0

    Larger slide jump:
        penalty increases with distance.
    """

    scores = np.asarray(
        score_matrix,
        dtype=np.float32,
    )

    num_sentences, num_slides = (
        scores.shape
    )

    if (
        num_sentences == 0
        or num_slides == 0
    ):

        return []

    dp = np.full(
        (
            num_sentences,
            num_slides,
        ),
        -np.inf,
        dtype=np.float32,
    )

    back = np.zeros(
        (
            num_sentences,
            num_slides,
        ),
        dtype=np.int32,
    )

    # First transcript segment
    dp[0] = scores[0]

    # --------------------------------------------------------
    # Dynamic programming
    # --------------------------------------------------------

    for i in range(
        1,
        num_sentences,
    ):

        previous = dp[
            i - 1
        ]

        for current_slide in range(
            num_slides
        ):

            best_value = -np.inf
            best_previous = 0

            for previous_slide in range(
                num_slides
            ):

                distance = abs(
                    previous_slide
                    - current_slide
                )

                if distance <= 1:

                    penalty = 0.0

                else:

                    penalty = (
                        jump_penalty
                        * distance
                    )

                candidate = (
                    previous[
                        previous_slide
                    ]
                    - penalty
                )

                if candidate > best_value:

                    best_value = candidate

                    best_previous = (
                        previous_slide
                    )

            dp[
                i,
                current_slide
            ] = (
                scores[
                    i,
                    current_slide
                ]
                + best_value
            )

            back[
                i,
                current_slide
            ] = best_previous

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    path = np.zeros(
        num_sentences,
        dtype=np.int32,
    )

    path[-1] = int(
        np.argmax(
            dp[-1]
        )
    )

    for i in range(
        num_sentences - 1,
        0,
        -1,
    ):

        path[i - 1] = back[
            i,
            path[i],
        ]

    return path.tolist()


# ============================================================
# TIMELINE
# ============================================================

def make_timeline(
    transcript_segments,
    slide_paths,
    alignment,
):
    """
    Convert alignment into a compact timeline.

    Consecutive transcript segments assigned to the
    same slide are merged.
    """

    raw = []

    for i, slide_index in enumerate(
        alignment
    ):

        slide_index = int(
            slide_index
        )

        transcript = (
            transcript_segments[i]
        )

        raw.append(
            {
                "slide_id":
                    slide_index + 1,

                "slide_index":
                    slide_index,

                "start_time":
                    round(
                        transcript["start"],
                        3,
                    ),

                "end_time":
                    round(
                        transcript["end"],
                        3,
                    ),

                "text":
                    transcript["text"],
            }
        )

    # --------------------------------------------------------
    # Merge consecutive same-slide entries
    # --------------------------------------------------------

    timeline = []

    for item in raw:

        if (
            timeline
            and
            timeline[-1]["slide_id"]
            == item["slide_id"]
        ):

            timeline[-1]["end_time"] = (
                item["end_time"]
            )

            timeline[-1]["text"] = (
                timeline[-1]["text"]
                + " "
                + item["text"]
            )

        else:

            timeline.append(
                dict(item)
            )

    # --------------------------------------------------------
    # Add slide names and timeline indices
    # --------------------------------------------------------

    for index, item in enumerate(
        timeline
    ):

        item[
            "timeline_index"
        ] = index

        item[
            "slide_title"
        ] = Path(
            slide_paths[
                item["slide_index"]
            ]
        ).stem

        item[
            "text"
        ] = clean_text(
            item["text"]
        )

    return timeline


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    output_path,
    timeline,
    metadata,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {

        "project":
            "LectureSense",

        "mode":
            "offline_srt_pdf",

        "metadata":
            metadata,

        "timeline":
            timeline,
    }

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Timeline saved to:"
    )

    print(
        output_path.resolve()
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    transcript_segments,
    slide_paths,
    timeline,
):

    print()
    print(
        "=" * 60
    )

    print(
        "LECTURESENSE INFERENCE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Transcript segments : "
        f"{len(transcript_segments)}"
    )

    print(
        f"Slides              : "
        f"{len(slide_paths)}"
    )

    print(
        f"Timeline entries    : "
        f"{len(timeline)}"
    )

    unique_slides = len(
        set(
            item["slide_id"]
            for item in timeline
        )
    )

    print(
        f"Unique slides used  : "
        f"{unique_slides}"
    )

    print()
    print(
        "First timeline entries:"
    )

    for item in timeline[:10]:

        print(
            f"["
            f"{item['start_time']:.1f}s - "
            f"{item['end_time']:.1f}s"
            f"] "
            f"Slide {item['slide_id']}: "
            f"{item['text'][:100]}"
        )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "LectureSense offline "
            "MaViLS SRT + PDF inference"
        )
    )

    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to MaViLS SRT file",
    )

    parser.add_argument(
        "--slides",
        required=True,
        help="Path to lecture PDF",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to trained checkpoint",
    )

    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Output timeline JSON",
    )

    parser.add_argument(
        "--render-dir",
        default=DEFAULT_RENDER_DIR,
        help="Temporary rendered slide directory",
    )

    parser.add_argument(
        "--score-batch",
        type=int,
        default=DEFAULT_SCORE_BATCH,
        help="Inference batch size",
    )

    parser.add_argument(
        "--jump-penalty",
        type=float,
        default=0.05,
        help="Forward slide transition penalty",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=120.0,
        help="Preferred maximum duration of one slide in seconds",
    )

    parser.add_argument(
        "--duration-penalty",
        type=float,
        default=0.01,
        help="Strength of duration penalty beyond max-duration",
    )

    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=480.0,
        help="Maximum slide-run duration considered by the decoder",
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Optional lecture video for inference-time visual alignment",
    )

    parser.add_argument(
        "--video-sample-seconds",
        type=float,
        default=DEFAULT_VIDEO_SAMPLE_SECONDS,
        help="Video frame sampling interval in seconds",
    )

    parser.add_argument(
        "--video-weight",
        type=float,
        default=DEFAULT_VIDEO_WEIGHT,
        help="Weight of video visual evidence during score fusion (0-1)",
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "LECTURESENSE OFFLINE INFERENCE"
    )

    print(
        "=" * 60
    )

    print(
        f"Transcript : {args.transcript}"
    )

    print(
        f"Slides     : {args.slides}"
    )

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    if args.video:
        print(
            f"Video      : {args.video}"
        )
        print(
            f"Video sample interval: {args.video_sample_seconds:.2f}s"
        )
        print(
            f"Video fusion weight: {args.video_weight:.2f}"
        )

    # ========================================================
    # 1. TRANSCRIPT
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "1/6  READING TRANSCRIPT"
    )

    print(
        "=" * 60
    )

    transcript_segments = (
        read_mavils_srt(
            args.transcript
        )
    )

    if not transcript_segments:

        raise RuntimeError(
            "No transcript segments found."
        )

    transcript_texts = [
        item["text"]
        for item in transcript_segments
    ]

    print(
        f"Transcript segments: "
        f"{len(transcript_segments)}"
    )

    # ========================================================
    # 2. PDF
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "2/6  RENDERING SLIDES"
    )

    print(
        "=" * 60
    )

    slide_paths = render_pdf(
        args.slides,
        args.render_dir,
    )

    if not slide_paths:

        raise RuntimeError(
            "No slides were rendered."
        )

    # ========================================================
    # 3. OCR
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "3/6  EXTRACTING OCR"
    )

    print(
        "=" * 60
    )

    ocr_texts = extract_ocr(
        slide_paths
    )

    if len(ocr_texts) != len(
        slide_paths
    ):

        raise RuntimeError(
            "OCR result count does not match "
            "slide count."
        )

    # ========================================================
    # 4. EMBEDDINGS
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "4/6  CREATING MULTIMODAL FEATURES"
    )

    print(
        "=" * 60
    )

    sentence_model = load_sbert()

    print(
        "\nTranscript embeddings..."
    )

    transcript_embeddings = (
        create_text_embeddings(
            sentence_model,
            transcript_texts,
        )
    )

    print(
        f"Transcript embeddings: "
        f"{transcript_embeddings.shape}"
    )

    print(
        "\nOCR embeddings..."
    )

    ocr_embeddings = (
        create_text_embeddings(
            sentence_model,
            ocr_texts,
        )
    )

    print(
        f"OCR embeddings: "
        f"{ocr_embeddings.shape}"
    )

    # Free SBERT
    del sentence_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "\nImage embeddings..."
    )

    image_embeddings = (
        create_image_embeddings(
            slide_paths
        )
    )

    print(
        f"Image embeddings: "
        f"{image_embeddings.shape}"
    )

    # Sanity checks
    if transcript_embeddings.shape[1] != TEXT_DIM:

        raise RuntimeError(
            "Unexpected transcript embedding dimension."
        )

    if ocr_embeddings.shape[1] != OCR_DIM:

        raise RuntimeError(
            "Unexpected OCR embedding dimension."
        )

    if image_embeddings.shape[1] != IMAGE_DIM:

        raise RuntimeError(
            "Unexpected image embedding dimension."
        )

    # ========================================================
    # 5. MODEL + ALIGNMENT
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "5/6  MULTIMODAL ALIGNMENT"
    )

    print(
        "=" * 60
    )

    model = load_trained_model(
        args.checkpoint
    )

    score_matrix = (
        compute_score_matrix(
            model,
            transcript_embeddings,
            ocr_embeddings,
            image_embeddings,
            batch_size=max(
                1,
                args.score_batch,
            ),
        )
    )

    print(
        f"Score matrix shape: "
        f"{score_matrix.shape}"
    )

    # Persist the learned model scores for debugging and evaluation.
    score_output = Path("runs/session/score_matrix.npy")
    score_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_output, score_matrix)
    print(
        f"Saved model score matrix: {score_output.resolve()}"
    )

    # --------------------------------------------------------
    # Optional video-frame visual evidence. This is an
    # inference-time fusion layer; the trained checkpoint is
    # unchanged.
    # --------------------------------------------------------
    final_score_matrix = score_matrix
    video_score_matrix = None

    if args.video:

        print()
        print("Extracting actual lecture-video visual evidence...")

        video_embeddings, video_timestamps, video_duration = (
            create_video_frame_embeddings(
                args.video,
                device=DEVICE,
                sample_every_seconds=max(0.25, args.video_sample_seconds),
                batch_size=8,
            )
        )

        video_score_matrix = build_video_slide_score_matrix(
            video_embeddings,
            video_timestamps,
            image_embeddings,
            segment_starts=[item["start"] for item in transcript_segments],
            segment_ends=[item["end"] for item in transcript_segments],
        )

        video_score_output = Path("runs/session/video_score_matrix.npy")
        video_score_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(video_score_output, video_score_matrix)

        final_score_matrix = fuse_model_and_video_scores(
            score_matrix,
            video_score_matrix,
            video_weight=args.video_weight,
        )

        fused_output = Path("runs/session/fused_score_matrix.npy")
        np.save(fused_output, final_score_matrix)

        print(
            f"Video score matrix shape: {video_score_matrix.shape}"
        )
        print(
            f"Saved video score matrix: {video_score_output.resolve()}"
        )
        print(
            f"Saved fused score matrix: {fused_output.resolve()}"
        )
        print(
            f"Video frames sampled: {len(video_timestamps)}"
        )
        if video_duration > 0:
            print(
                f"Video duration: {video_duration:.1f}s"
            )

    print(
        f"Final alignment score matrix shape: {final_score_matrix.shape}"
    )

    print(
        "\nRunning duration-aware dynamic programming..."
    )

    sentence_starts = [
        item["start"]
        for item in transcript_segments
    ]

    sentence_ends = [
        item["end"]
        for item in transcript_segments
    ]

    alignment = dp_align(
        final_score_matrix,
        jump_penalty=args.jump_penalty,
        max_forward_jump=1,
        segment_starts=sentence_starts,
        segment_ends=sentence_ends,
        max_duration_seconds=args.max_duration,
        duration_penalty=args.duration_penalty,
        max_run_seconds=args.max_run_seconds,
    )

    if not alignment:

        raise RuntimeError(
            "Dynamic-programming alignment "
            "returned no result."
        )

    print(
        f"Alignment entries: "
        f"{len(alignment)}"
    )

    # ========================================================
    # 6. TIMELINE
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "6/6  BUILDING TIMELINE"
    )

    print(
        "=" * 60
    )

    timeline = make_timeline(
        transcript_segments,
        slide_paths,
        alignment,
    )

    if not timeline:

        raise RuntimeError(
            "Timeline is empty."
        )

    metadata = {

        "transcript_file":
            str(
                Path(
                    args.transcript
                )
            ),

        "slides_file":
            str(
                Path(
                    args.slides
                )
            ),

        "checkpoint":
            str(
                Path(
                    args.checkpoint
                )
            ),

        "num_transcript_segments":
            len(
                transcript_segments
            ),

        "num_slides":
            len(
                slide_paths
            ),

        "timeline_entries":
            len(
                timeline
            ),

        "jump_penalty":
            args.jump_penalty,

        "duration_aware_alignment":
            True,

        "max_duration_seconds":
            args.max_duration,

        "duration_penalty":
            args.duration_penalty,

        "max_run_seconds":
            args.max_run_seconds,

        "video_file":
            str(Path(args.video)) if args.video else None,

        "video_enabled":
            bool(args.video),

        "video_sample_seconds":
            args.video_sample_seconds if args.video else None,

        "video_weight":
            args.video_weight if args.video else 0.0,

        "score_fusion":
            "learned_model_plus_video_visual_evidence"
            if args.video
            else "learned_model_only",

        "model":
            "LectureSenseCrossModal",

        "text_embedding_model":
            "all-MiniLM-L6-v2",

        "ocr_embedding_model":
            "all-MiniLM-L6-v2",

        "image_embedding_model":
            "ResNet-18",

        "fusion_dimension":
            FUSION_DIM,
    }

    save_result(
        args.out,
        timeline,
        metadata,
    )

    print_summary(
        transcript_segments,
        slide_paths,
        timeline,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()



