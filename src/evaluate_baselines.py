from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAVILS = DATA / "MaViLS"

OCR_EMB_PATH = DATA / "ocr_embeddings.npy"
IMAGE_EMB_PATH = DATA / "image_embeddings.npy"

OCR_INDEX_PATH = DATA / "ocr_embedding_index.csv"
IMAGE_INDEX_PATH = DATA / "image_embedding_index.csv"

GT_DIR = MAVILS / "data" / "ground_truth_files"
SRT_DIR = MAVILS / "data" / "audioscripts"

OUTPUT = DATA / "baseline_evaluation.json"


LECTURE_MAP = {
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


GT_ALIASES = {
    "deeplearning_goodfellow": "ground_truth_deeplearning.xlsx",
    "image_processing_high_res": "ground_truth_image_processing.xlsx",
    "numerics_hennig": "ground_truth_numerics.xlsx",
    "physics_intro_02": "ground_truth_physics.xlsx",
    "psychology_MIT": "ground_truth_psychology.xlsx",
    "reinforcement_learning_silver": "ground_truth_reinforcement_learning.xlsx",
    "short_range_mit": "ground_truth_short_range.xlsx",
    "visual_system_MIT": "ground_truth_sensory_systems.xlsx",
}


def norm_id(x):

    if x is None or pd.isna(x):
        return ""

    x = str(x).strip().lower()
    x = x.replace("&", " and ")
    x = x.replace("-", "_")
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"_+", "_", x)

    return x.strip("_")


def norm_text(x):

    if x is None or pd.isna(x):
        return ""

    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s]", "", x)

    return x.strip()


def slide_id(x):

    try:
        return int(float(x))
    except Exception:

        match = re.search(
            r"\d+",
            str(x)
        )

        if match:
            return int(match.group())

        return None


def parse_time(x):

    if x is None:
        return None

    try:

        if not isinstance(x, str):
            return float(x)

    except Exception:
        pass

    x = str(x).strip().replace(
        ",",
        "."
    )

    m = re.match(
        r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$",
        x
    )

    if m:

        h = int(m.group(1))
        mi = int(m.group(2))
        s = int(m.group(3))

        frac = 0.0

        if m.group(4):
            frac = float(
                "0." + m.group(4)
            )

        return (
            h * 3600
            + mi * 60
            + s
            + frac
        )

    m = re.match(
        r"^(\d+):(\d{2})(?:\.(\d+))?$",
        x
    )

    if m:

        mi = int(m.group(1))
        s = int(m.group(2))

        frac = 0.0

        if m.group(3):
            frac = float(
                "0." + m.group(3)
            )

        return (
            mi * 60
            + s
            + frac
        )

    return None


def find_gt(lecture):

    exact = (
        GT_DIR /
        f"ground_truth_{lecture}.xlsx"
    )

    if exact.exists():
        return exact

    if lecture in GT_ALIASES:

        alias = (
            GT_DIR /
            GT_ALIASES[lecture]
        )

        if alias.exists():
            return alias

    target = norm_id(lecture)

    for path in GT_DIR.glob("*.xlsx"):

        name = norm_id(path.stem)

        if name.startswith("ground_truth_"):
            name = name[len("ground_truth_"):]

        if name == target:
            return path

    return None


def find_srt(lecture):

    exact = (
        SRT_DIR /
        f"{lecture}.srt"
    )

    if exact.exists():
        return exact

    target = norm_id(lecture)

    for path in SRT_DIR.glob("*.srt"):

        if norm_id(path.stem) == target:
            return path

    return None


def read_gt(path):

    df = pd.read_excel(path)

    columns = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    slide_col = None
    time_col = None
    text_col = None

    for name in [
        "slidenumber",
        "slide_number",
        "slide",
        "slideid"
    ]:

        if name in columns:
            slide_col = columns[name]
            break

    for name in [
        "key",
        "time",
        "timestamp",
        "start",
        "starttime"
    ]:

        if name in columns:
            time_col = columns[name]
            break

    for name in [
        "value",
        "sentence",
        "text",
        "transcript"
    ]:

        if name in columns:
            text_col = columns[name]
            break

    rows = []

    for _, row in df.iterrows():

        sid = slide_id(
            row[slide_col]
        )

        if sid is None or sid < 1:
            continue

        text = ""

        if (
            text_col is not None
            and not pd.isna(row[text_col])
        ):

            text = str(
                row[text_col]
            ).strip()

        timestamp = None

        if time_col is not None:

            timestamp = parse_time(
                row[time_col]
            )

        rows.append({
            "slide_id": sid,
            "text": text,
            "timestamp": timestamp
        })

    return rows


def read_srt(path):

    text = path.read_text(
        encoding="utf-8-sig",
        errors="ignore"
    )

    pattern = re.compile(
        r"\[\s*"
        r"([0-9:. ,]+)"
        r"\s*-->\s*"
        r"([0-9:. ,]+)"
        r"\s*\]"
        r"\s*(.*)"
    )

    segments = []

    for match in pattern.finditer(text):

        start = parse_time(
            match.group(1)
        )

        end = parse_time(
            match.group(2)
        )

        word = match.group(3).strip()

        if start is None or not word:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": word
        })

    return segments


def match_gt_srt(gt, srt):

    normalized_srt = [
        norm_text(x["text"])
        for x in srt
    ]

    used = set()
    output = []

    for item in gt:

        target = norm_text(
            item["text"]
        )

        best = None

        if target:

            for i, text in enumerate(
                normalized_srt
            ):

                if i in used:
                    continue

                if text == target:

                    best = i
                    break

        if (
            best is None
            and item["timestamp"] is not None
        ):

            best_distance = float("inf")

            for i, seg in enumerate(srt):

                if i in used:
                    continue

                distance = abs(
                    seg["start"]
                    -
                    item["timestamp"]
                )

                if distance < best_distance:

                    best_distance = distance
                    best = i

            if best_distance > 10:
                best = None

        if best is not None:

            used.add(best)

            timestamp = item["timestamp"]

            if timestamp is None:
                timestamp = srt[best]["start"]

            output.append({
                "slide_id": item["slide_id"],
                "text": item["text"],
                "timestamp": timestamp
            })

    return output


def get_features(
    feature_id,
    image_index,
    image_embeddings,
    ocr_index,
    ocr_embeddings
):

    target = norm_id(feature_id)

    image_ids = (
        image_index["lecture_id"]
        .apply(norm_id)
    )

    ocr_ids = (
        ocr_index["lecture_id"]
        .apply(norm_id)
    )

    image_rows = image_index[
        image_ids == target
    ].copy()

    ocr_rows = ocr_index[
        ocr_ids == target
    ].copy()

    image_rows["_row"] = image_rows.index

    image_map = {}

    for _, row in image_rows.iterrows():

        sid = slide_id(
            row["slide_id"]
        )

        if sid is None:
            continue

        row_id = int(
            row["_row"]
        )

        if row_id < len(image_embeddings):

            image_map[sid] = row_id

    ocr_map = {}

    for _, row in ocr_rows.iterrows():

        sid = slide_id(
            row["slide_id"]
        )

        if sid is None:
            continue

        try:
            row_id = int(
                row["embedding_row"]
            )
        except Exception:
            continue

        if row_id < len(ocr_embeddings):

            ocr_map[sid] = row_id

    common = sorted(
        set(image_map)
        &
        set(ocr_map)
    )

    slides = []

    for sid in common:

        slides.append({
            "slide_id": sid,

            "ocr": ocr_embeddings[
                ocr_map[sid]
            ],

            "image": image_embeddings[
                image_map[sid]
            ]
        })

    return slides


def normalize_vectors(x):

    return x / (
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True
        ) + 1e-8
    )


def evaluate_similarity(
    query,
    candidates,
    candidate_ids,
    true_ids
):

    query = normalize_vectors(
        query
    )

    candidates = normalize_vectors(
        candidates
    )

    scores = query @ candidates.T

    top1 = []
    top3 = []
    reciprocal_rank = []

    for i in range(
        len(true_ids)
    ):

        ranking = np.argsort(
            scores[i]
        )[::-1]

        ranked_ids = [
            candidate_ids[j]
            for j in ranking
        ]

        true_id = true_ids[i]

        top1.append(
            int(
                ranked_ids[0]
                ==
                true_id
            )
        )

        top3.append(
            int(
                true_id
                in
                ranked_ids[:3]
            )
        )

        rank = None

        for r, sid in enumerate(
            ranked_ids,
            start=1
        ):

            if sid == true_id:

                rank = r
                break

        reciprocal_rank.append(
            1.0 / rank
            if rank is not None
            else 0.0
        )

    return {
        "top1": float(
            np.mean(top1)
        ),

        "top3": float(
            np.mean(top3)
        ),

        "mrr": float(
            np.mean(
                reciprocal_rank
            )
        ),

        "n": len(true_ids)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lecture",
        default=None
    )

    args = parser.parse_args()

    print(
        "Loading features..."
    )

    ocr_embeddings = np.load(
        OCR_EMB_PATH
    )

    image_embeddings = np.load(
        IMAGE_EMB_PATH
    )

    ocr_index = pd.read_csv(
        OCR_INDEX_PATH
    )

    image_index = pd.read_csv(
        IMAGE_INDEX_PATH
    )

    print(
        f"OCR: {ocr_embeddings.shape}"
    )

    print(
        f"Image: {image_embeddings.shape}"
    )

    print(
        "\nLoading Sentence-BERT..."
    )

    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    if args.lecture:

        lectures = [
            args.lecture
        ]

    else:

        lectures = list(
            LECTURE_MAP.keys()
        )

    results = []

    for lecture in lectures:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"LECTURE: {lecture}"
        )

        print(
            "=" * 60
        )

        gt_file = find_gt(
            lecture
        )

        if gt_file is None:

            print(
                "Ground truth not found."
            )

            continue

        srt_file = find_srt(
            lecture
        )

        if srt_file is None:

            print(
                "SRT not found."
            )

            continue

        gt = read_gt(
            gt_file
        )

        srt = read_srt(
            srt_file
        )

        matched = match_gt_srt(
            gt,
            srt
        )

        print(
            f"Evaluation samples: "
            f"{len(matched)}"
        )

        if not matched:
            continue

        feature_id = LECTURE_MAP.get(
            lecture,
            lecture
        )

        slides = get_features(
            feature_id,
            image_index,
            image_embeddings,
            ocr_index,
            ocr_embeddings
        )

        if not slides:

            print(
                "No slide features."
            )

            continue

        candidate_ids = [
            x["slide_id"]
            for x in slides
        ]

        valid = [
            x for x in matched
            if x["slide_id"]
            in candidate_ids
        ]

        if not valid:
            continue

        true_ids = [
            x["slide_id"]
            for x in valid
        ]

        texts = [
            x["text"]
            for x in valid
        ]

        text_vectors = text_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64
        ).astype(np.float32)

        ocr_vectors = np.stack([
            x["ocr"]
            for x in slides
        ]).astype(np.float32)

        image_vectors = np.stack([
            x["image"]
            for x in slides
        ]).astype(np.float32)

        # ====================================================
        # BASELINE 1:
        # Transcript text -> OCR text
        # ====================================================

        text_ocr = evaluate_similarity(
            text_vectors,
            ocr_vectors,
            candidate_ids,
            true_ids
        )

        # ====================================================
        # BASELINE 2:
        # Transcript text -> image
        #
        # Image is 512-D and text is 384-D.
        # PCA is fitted ONLY on candidate slide images
        # for this lecture.
        #
        # This is diagnostic only.
        # ====================================================

        n_components = min(
            384,
            image_vectors.shape[0],
            image_vectors.shape[1]
        )

        pca = PCA(
            n_components=n_components
        )

        image_reduced = pca.fit_transform(
            image_vectors
        ).astype(np.float32)

        if n_components < 384:

            padded = np.zeros(
                (
                    len(image_reduced),
                    384
                ),
                dtype=np.float32
            )

            padded[
                :,
                :n_components
            ] = image_reduced

            image_reduced = padded

        text_image = evaluate_similarity(
            text_vectors,
            image_reduced,
            candidate_ids,
            true_ids
        )

        print(
            "\nText -> OCR"
        )

        print(
            f"Top-1: {text_ocr['top1']:.4f}"
        )

        print(
            f"Top-3: {text_ocr['top3']:.4f}"
        )

        print(
            f"MRR:   {text_ocr['mrr']:.4f}"
        )

        print(
            "\nText -> Image"
        )

        print(
            f"Top-1: {text_image['top1']:.4f}"
        )

        print(
            f"Top-3: {text_image['top3']:.4f}"
        )

        print(
            f"MRR:   {text_image['mrr']:.4f}"
        )

        results.append({

            "lecture_id":
                lecture,

            "n":
                len(valid),

            "text_ocr":
                text_ocr,

            "text_image":
                text_image

        })

    # ========================================================
    # AGGREGATE
    # ========================================================

    print(
        "\n\n"
        + "=" * 60
    )

    print(
        "BASELINE RESULTS"
    )

    print(
        "=" * 60
    )

    if not results:

        print(
            "No results."
        )

        return

    total = sum(
        r["n"]
        for r in results
    )

    for name in [
        "text_ocr",
        "text_image"
    ]:

        top1 = sum(
            r[name]["top1"] * r["n"]
            for r in results
        ) / total

        top3 = sum(
            r[name]["top3"] * r["n"]
            for r in results
        ) / total

        mrr = sum(
            r[name]["mrr"] * r["n"]
            for r in results
        ) / total

        print(
            f"\n{name}"
        )

        print(
            f"Top-1: {top1:.4f}"
        )

        print(
            f"Top-3: {top3:.4f}"
        )

        print(
            f"MRR:   {mrr:.4f}"
        )

    output = {
        "results": results,
        "samples": total
    }

    with open(
        OUTPUT,
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
        OUTPUT
    )


if __name__ == "__main__":
    main()