
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# LectureSense V5 — SINGLE-FILE, ROBUST TRAIN-DATA BUILDER
# ============================================================
# Purpose:
#   1) Reuse completed V4 inference whenever available.
#   2) Run the REAL Visual-First branch on training lectures.
#   3) Align both branches to the exact MaViLS transcript segmentation.
#   4) Create timestamp-level multimodal training features.
#
# IMPORTANT:
#   This file does NOT touch validation/test lectures.
#   It does NOT promise or manufacture 95% accuracy.
#   It is designed to create clean training data for the next
#   learned V5 fusion model.
# ============================================================

PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"

# Fix imports BEFORE importing project modules.
for path in (PROJECT, SRC):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

TRAIN_LECTURES = [
    "cities_and_decarbonization",
    "climate_and_cities",
    "climate_science_policy_MIT2",
    "cognitive_robotics_MIT",
    "computer_vision_2_2",
    "creating_breakthrough_products_MIT",
    "cryptocurrency_MIT",
    "deeplearning",
    "image_processing",
    "ML_for_health_MIT",
    "numerics",
    "phonetics",
]

# Known exact aliases from the MaViLS dataset used in this project.
SRT_ALIASES = {
    "cities_and_decarbonization": [
        "cities_and_decarbonization.srt",
    ],
    "climate_and_cities": [
        "climate_and_cities.srt",
    ],
    "climate_science_policy_MIT2": [
        "climate_science_policy_MIT2.srt",
        "climate_science_policy_MIT.srt",
    ],
    "cognitive_robotics_MIT": [
        "cognitive_robotics_MIT.srt",
    ],
    "computer_vision_2_2": [
        "computer_vision_2_2_high_res.srt",
        "computer_vision_2_2.srt",
    ],
    "creating_breakthrough_products_MIT": [
        "creating_breakthrough_products_MIT.srt",
    ],
    "cryptocurrency_MIT": [
        "cryptocurrency.srt",
        "cryptocurrency_MIT.srt",
    ],
    "deeplearning": [
        "deeplearning_goodfellow.srt",
        "deeplearning.srt",
    ],
    "image_processing": [
        "image_processing_high_res.srt",
        "image_processing.srt",
    ],
    "ML_for_health_MIT": [
        "ML_for_health_MIT.srt",
    ],
    "numerics": [
        "numerics_hennig.srt",
        "numerics.srt",
    ],
    "phonetics": [
        "phonetics.srt",
    ],
}

PDF_ALIASES = {
    "cities_and_decarbonization": [
        "cities&decarbonization.pdf",
        "cities_and_decarbonization.pdf",
    ],
    "climate_and_cities": [
        "climate_and_cities.pdf",
    ],
    "climate_science_policy_MIT2": [
        "climate_science_policy_MIT.pdf",
        "climate_science_policy_MIT2.pdf",
    ],
    "cognitive_robotics_MIT": [
        "cognitive_robotics_MIT.pdf",
    ],
    "computer_vision_2_2": [
        "computer_vision_2_2_geiger.pdf",
    ],
    "creating_breakthrough_products_MIT": [
        "creating_breakthrough_products.pdf",
    ],
    "cryptocurrency_MIT": [
        "cryptocurrency.pdf",
        "cryptocurrency_MIT.pdf",
    ],
    "deeplearning": [
        "deeplearning_goodfellow.pdf",
    ],
    "image_processing": [
        "image_processing.pdf",
    ],
    "ML_for_health_MIT": [
        "ML_for_health_care_MIT.pdf",
        "ML_for_health_MIT.pdf",
    ],
    "numerics": [
        "numerics.pdf",
        "numerics_hennig.pdf",
    ],
    "phonetics": [
        "phonetics.pdf",
    ],
}

VIDEO_ALIASES = {
    "cities_and_decarbonization": [
        "cities_and_decarbonization_standard_res.mp4",
        "cities_and_decarbonization_high_res.mp4",
    ],
    "climate_and_cities": [
        "climate_and_cities_high_res.mp4",
        "climate_and_cities.mp4",
    ],
    "climate_science_policy_MIT2": [
        "climate_science_policy_MIT_high_res.mp4",
        "climate_science_policy_MIT2_high_res.mp4",
    ],
    "cognitive_robotics_MIT": [
        "cognitive_robotics_high_res.mp4",
        "cognitive_robotics_MIT_high_res.mp4",
    ],
    "computer_vision_2_2": [
        "computer_vision_2_2_high_res.mp4",
    ],
    "creating_breakthrough_products_MIT": [
        "creating_breakthrough_products_MIT.mp4",
        "creating_breakthrough_products_high_res.mp4",
    ],
    "cryptocurrency_MIT": [
        "cryptocurrency_high_res.mp4",
        "cryptocurrency.mp4",
    ],
    "deeplearning": [
        "deep_learning_high_res.mp4",
        "deeplearning_high_res.mp4",
    ],
    "image_processing": [
        "image_processing_high_res.mp4",
    ],
    "ML_for_health_MIT": [
        "ML_for_health_high_res.mp4",
        "ML_for_health_MIT_high_res.mp4",
    ],
    "numerics": [
        "numerics_high_res.mp4",
    ],
    "phonetics": [
        "phonetics_high_res.mp4",
    ],
}

GROUND_TRUTH_ALIASES = {
    name: [f"ground_truth_{name}.xlsx"] for name in TRAIN_LECTURES
}

VIDEO_DIR = Path(r"C:\Users\Bharat\Downloads\archive\video")


def norm_name(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def existing_alias(folder, aliases):
    if not folder.exists():
        return None

    for name in aliases:
        p = folder / name
        if p.exists():
            return p

    return None


def fuzzy_candidates(folder, lecture, extensions):
    """
    Conservative fallback:
    compares normalized lecture/source tokens against filenames.
    Returns files with the strongest token overlap, but only accepts
    an unambiguous top candidate.
    """
    if not folder.exists():
        return []

    keys = set()
    raw = lecture.lower()

    variants = {
        lecture,
        raw.replace("_mit2", ""),
        raw.replace("_mit", ""),
        raw.replace("_high_res", ""),
        raw.replace("deeplearning", "deep_learning"),
        raw.replace("ml_for_health", "ml_for_health_care"),
        raw.replace("cities_and_decarbonization", "cities&decarbonization"),
        raw.replace("creating_breakthrough_products_mit",
                    "creating_breakthrough_products"),
    }

    for v in variants:
        n = norm_name(v)
        if n:
            keys.add(n)

    scored = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in extensions:
            continue

        stem = norm_name(f.stem)
        score = 0
        for k in keys:
            if k in stem or stem in k:
                score = max(score, min(len(k), len(stem)))
            else:
                # Shared character-prefix gives a weak score only.
                prefix = 0
                for a, b in zip(k, stem):
                    if a != b:
                        break
                    prefix += 1
                score = max(score, prefix)

        if score > 5:
            scored.append((score, f))

    scored.sort(key=lambda x: (-x[0], x[1].name.lower()))
    return [f for _, f in scored]


def find_asset(folder, aliases, lecture, extensions):
    exact = existing_alias(folder, aliases)
    if exact is not None:
        return exact

    candidates = fuzzy_candidates(folder, lecture, extensions)
    if len(candidates) == 1:
        return candidates[0]

    # Accept only a clearly stronger candidate.
    if len(candidates) >= 2:
        c0 = candidates[0]
        c1 = candidates[1]

        # Recompute a simple confidence ratio from names.
        a = norm_name(c0.stem)
        b = norm_name(c1.stem)
        if a and b:
            ratio = len(set(a) & set(norm_name(lecture))) / max(
                1, len(set(b) & set(norm_name(lecture)))
            )
            if ratio >= 1.35:
                return c0

    return None


def find_assets(root, lecture):
    data = root / "data" / "MaViLS" / "data"

    srt = find_asset(
        data / "audioscripts",
        SRT_ALIASES.get(lecture, []),
        lecture,
        {".srt"},
    )

    pdf = find_asset(
        data / "lectures",
        PDF_ALIASES.get(lecture, []),
        lecture,
        {".pdf"},
    )

    gt = find_asset(
        data / "ground_truth_files",
        GROUND_TRUTH_ALIASES.get(lecture, []),
        lecture,
        {".xlsx", ".xls"},
    )

    video = find_asset(
        VIDEO_DIR,
        VIDEO_ALIASES.get(lecture, []),
        lecture,
        {".mp4", ".mov", ".mkv", ".avi"},
    )

    return srt, pdf, gt, video


def get_field(obj, *names, default=None):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def normalize_timeline(payload):
    """
    Supports the actual LectureSense timeline JSON shape:
    {
        "timeline": [
            {"start_time": ..., "end_time": ..., "slide_id": ...}
        ],
        ...
    }

    Also supports common equivalent dict/list variants.
    """
    items = payload

    if isinstance(payload, dict):
        for key in ("timeline", "segments", "entries", "data"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break

    if not isinstance(items, list):
        raise ValueError("timeline.json does not contain a timeline list")

    result = []
    for item in items:
        start = get_field(item, "start_time", "start", "begin", "from")
        end = get_field(item, "end_time", "end", "finish", "to")
        slide = get_field(
            item,
            "slide_id",
            "slide",
            "slide_number",
            "slidenumber",
        )

        if start is None or end is None or slide is None:
            continue

        result.append(
            {
                "start": float(start),
                "end": float(end),
                "slide": int(slide),
            }
        )

    if not result:
        raise ValueError("No usable timeline entries found")

    result.sort(key=lambda x: (x["start"], x["end"]))
    return result


def ensure_v4(root, lecture, srt, pdf, video, v4_dir):
    """
    Reuse V4 if its timeline + core matrices already exist.
    Otherwise run the official V4 inference script.
    """
    timeline = v4_dir / "timeline.json"
    core_matrix = v4_dir / "v4_final_score_matrix.npy"

    if timeline.exists() and core_matrix.exists():
        print(f"[V4] Reusing existing completed V4 output:\n     {v4_dir}")
        return timeline

    checkpoint = root / "checkpoints" / "temporal_v4_best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found: {checkpoint}"
        )

    v4_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(root / "src" / "inference_temporal_v4.py"),
        "--transcript", str(srt),
        "--slides", str(pdf),
        "--video", str(video),
        "--checkpoint", str(checkpoint),
        "--out", str(timeline),
        "--render-dir", str(v4_dir / "rendered_slides"),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", "0.4",
        "--video-sample-seconds", "1.0",
    ]

    print("[V4] Running official V4 inference...")
    subprocess.run(cmd, check=True, cwd=root)

    if not timeline.exists():
        raise RuntimeError(
            f"V4 finished without creating {timeline}"
        )

    return timeline


def get_exact_transcript_segments(srt):
    """
    Uses the exact MaViLS segmentation function used by V4.
    Adds both PROJECT/src and PROJECT to sys.path before import.
    """
    from inference import read_mavils_srt

    segments = read_mavils_srt(str(srt))

    if not segments:
        raise RuntimeError("No transcript segments found")

    return segments


def segment_midpoints(segments):
    mids = []
    for s in segments:
        start = get_field(s, "start", "start_time")
        end = get_field(s, "end", "end_time")

        if start is None and end is None:
            raise ValueError(
                "Transcript segment has neither start nor end timestamp"
            )

        if start is None:
            start = end
        if end is None:
            end = start

        mids.append((float(start) + float(end)) / 2.0)

    return np.asarray(mids, dtype=np.float32)


def load_v4_predictions(v4_dir, segments):
    score_path = v4_dir / "v4_final_score_matrix.npy"
    boundary_path = v4_dir / "v4_boundary_probability.npy"

    if not score_path.exists():
        raise FileNotFoundError(score_path)

    scores = np.load(score_path)

    mids = segment_midpoints(segments)

    if scores.shape[0] != len(segments):
        raise ValueError(
            f"V4 score rows ({scores.shape[0]}) != "
            f"MaViLS transcript segments ({len(segments)})"
        )

    with open(v4_dir / "timeline.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    timeline = normalize_timeline(payload)

    v4_pred = np.full(len(mids), 1, dtype=np.int32)

    for i, t in enumerate(mids):
        active = [
            x for x in timeline
            if x["start"] <= float(t) <= x["end"]
        ]

        if active:
            v4_pred[i] = int(active[-1]["slide"])
        else:
            # Safe fallback: use the learned score for this transcript segment.
            v4_pred[i] = int(np.argmax(scores[i])) + 1

    boundary = (
        np.load(boundary_path)
        if boundary_path.exists()
        else np.zeros(len(segments), dtype=np.float32)
    )

    return scores, boundary, v4_pred, mids


def run_real_visual_first(video, pdf, cache_path, sample_seconds):
    """
    Uses the actual visual_first_alignment implementation.

    IMPORTANT FIX:
    viterbi_monotonic signature is:
        viterbi_monotonic(score, max_jump=3, jump_penalty=0.015)

    It does NOT take timestamps as the second positional argument.
    """
    if cache_path.exists():
        print(f"[VISUAL] Reusing cache:\n         {cache_path}")
        z = np.load(cache_path)
        return (
            np.asarray(z["times"]),
            np.asarray(z["scores"]),
            np.asarray(z["path"]).astype(np.int32),
        )

    from visual_first_alignment import run_visual_first, viterbi_monotonic

    print(
        f"[VISUAL] Running real Visual-First "
        f"(sample_every={sample_seconds:.2f}s)..."
    )

    _, scores, times, duration = run_visual_first(
        Path(video),
        Path(pdf),
        sample_every=float(sample_seconds),
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )

    scores = np.asarray(scores, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)

    if scores.ndim != 2:
        raise ValueError(
            f"Visual score matrix must be 2-D, got {scores.shape}"
        )

    if len(times) != scores.shape[0]:
        raise ValueError(
            f"Visual timestamps ({len(times)}) != "
            f"visual score rows ({scores.shape[0]})"
        )

    # THE CORRECT CALL — no timestamps positional argument.
    path, _ = viterbi_monotonic(
        scores,
        max_jump=2,
        jump_penalty=0.05,
    )

    path = np.asarray(path, dtype=np.int32)

    if len(path) != len(times):
        raise ValueError(
            f"Visual Viterbi path ({len(path)}) != timestamps ({len(times)})"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        times=times,
        scores=scores,
        path=path,
    )

    print(f"[VISUAL] Cache saved:\n         {cache_path}")
    return times, scores, path


def load_ground_truth(gt_path):
    """
    Robust MaViLS GT loader.

    Supported time/timestamp columns:
        time, timestamp, start, key

    Supported slide columns:
        slidenumber, slide_number, slide, slideid

    The current MaViLS files can use:
        key | slidenumber | value

    Here `key` is treated as the timestamp and `slidenumber` as
    the ground-truth slide id. The `value` column is not required.
    """
    df = pd.read_excel(gt_path)
    df.columns = [
        str(c).strip().lower().replace(" ", "").replace("-", "_")
        for c in df.columns
    ]

    time_candidates = [
        "time",
        "timestamp",
        "start",
        "key",
    ]
    slide_candidates = [
        "slidenumber",
        "slide_number",
        "slide",
        "slideid",
        "slide_id",
    ]

    time_col = next(
        (c for c in time_candidates if c in df.columns),
        None,
    )
    slide_col = next(
        (c for c in slide_candidates if c in df.columns),
        None,
    )

    # Fallback for unusual capitalization/spelling after normalization.
    if time_col is None:
        time_col = next(
            (c for c in df.columns if "time" in c or "timestamp" in c),
            None,
        )

    if slide_col is None:
        slide_col = next(
            (
                c for c in df.columns
                if "slide" in c or c == "slidenumber"
            ),
            None,
        )

    if time_col is None or slide_col is None:
        raise ValueError(
            f"Could not identify GT columns in {gt_path}. "
            f"Columns={list(df.columns)}"
        )

    out = df[[time_col, slide_col]].copy()
    out.columns = ["time", "gt"]

    # Some MaViLS files store numeric values directly; others may
    # contain strings that can be converted to numbers.
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["gt"] = pd.to_numeric(out["gt"], errors="coerce")

    out = out.dropna()

    # Match the project evaluation convention: invalid slide IDs
    # (0 and -1) are excluded.
    out = out[out["gt"] > 0]

    out["gt"] = out["gt"].astype(int)
    out = out.sort_values("time")

    if out.empty:
        raise ValueError(
            f"Ground truth has no valid timestamp/slide rows: {gt_path}"
        )

    return out.reset_index(drop=True)


def safe_entropy(row):
    x = np.asarray(row, dtype=np.float64)
    if x.size == 0:
        return 0.0

    x = x - np.max(x)
    p = np.exp(x)
    p /= p.sum() + 1e-12
    return float(-(p * np.log(p + 1e-12)).sum())


def build_one(root, lecture, sample_seconds):
    print("\n" + "=" * 78)
    print(lecture)
    print("=" * 78)

    srt, pdf, gt, video = find_assets(root, lecture)

    print("SRT :", srt)
    print("PDF :", pdf)
    print("GT  :", gt)
    print("VIDEO:", video)

    missing = [
        name for name, obj in [
            ("SRT", srt),
            ("PDF", pdf),
            ("GT", gt),
            ("VIDEO", video),
        ]
        if obj is None
    ]

    if missing:
        print("[SKIP] Missing:", missing)

        # Help with future mappings.
        if "PDF" in missing:
            folder = root / "data" / "MaViLS" / "data" / "lectures"
            print("\nPDF candidates containing lecture keywords:")
            for f in fuzzy_candidates(folder, lecture, {".pdf"})[:10]:
                print("   ", f.name)

        if "VIDEO" in missing:
            print("\nVIDEO candidates containing lecture keywords:")
            for f in fuzzy_candidates(VIDEO_DIR, lecture, {".mp4", ".mov", ".mkv", ".avi"})[:10]:
                print("   ", f.name)

        return None

    v4_dir = root / "runs" / "v5_gate_train_v4" / lecture
    out_dir = root / "runs" / "v5_train_gate_dataset_FINAL_V2" / lecture
    visual_cache = (
        root / "runs" / "v5_train_gate_dataset_FINAL_V2"
        / "_visual_cache" / f"{lecture}.npz"
    )

    ensure_v4(root, lecture, srt, pdf, video, v4_dir)

    segments = get_exact_transcript_segments(srt)
    print("Exact V4 transcript segments:", len(segments))

    v4_scores, boundary, v4_pred, seg_mids = load_v4_predictions(
        v4_dir,
        segments,
    )

    visual_times, visual_scores, visual_path = run_real_visual_first(
        video,
        pdf,
        visual_cache,
        sample_seconds,
    )

    gt = load_ground_truth(gt)

    rows = []

    for _, record in gt.iterrows():
        t = float(record["time"])
        truth = int(record["gt"])

        # Nearest actual video sample.
        visual_idx = int(
            np.argmin(np.abs(visual_times - t))
        )

        # Transcript segment containing the GT time.
        seg_idx = int(
            np.searchsorted(seg_mids, t, side="right") - 1
        )
        seg_idx = max(0, min(seg_idx, len(seg_mids) - 1))

        # -------------------------
        # Visual branch
        # -------------------------
        vis_row = visual_scores[visual_idx]
        visual_top_raw = int(np.argmax(vis_row)) + 1
        visual_viterbi_pred = int(visual_path[visual_idx]) + 1

        vis_order = np.argsort(vis_row)[::-1]
        if len(vis_order) >= 2:
            visual_margin = float(
                vis_row[vis_order[0]] - vis_row[vis_order[1]]
            )
        else:
            visual_margin = 0.0

        visual_score = float(vis_row[visual_viterbi_pred - 1])

        # -------------------------
        # V4 branch
        # -------------------------
        v4_row = v4_scores[seg_idx]
        v4_pred_now = int(v4_pred[seg_idx])

        v4_order = np.argsort(v4_row)[::-1]
        if len(v4_order) >= 2:
            v4_margin = float(
                v4_row[v4_order[0]] - v4_row[v4_order[1]]
            )
        else:
            v4_margin = 0.0

        v4_score = float(v4_row[v4_pred_now - 1])

        # Cross-modal scores.
        visual_on_v4 = (
            float(vis_row[v4_pred_now - 1])
            if 1 <= v4_pred_now <= len(vis_row)
            else 0.0
        )

        v4_on_visual = (
            float(v4_row[visual_viterbi_pred - 1])
            if 1 <= visual_viterbi_pred <= len(v4_row)
            else 0.0
        )

        # Temporal context.
        prev_v4 = (
            int(v4_pred[seg_idx - 1])
            if seg_idx > 0 else v4_pred_now
        )
        next_v4 = (
            int(v4_pred[seg_idx + 1])
            if seg_idx + 1 < len(v4_pred)
            else v4_pred_now
        )

        rows.append({
            "lecture": lecture,
            "time": t,
            "gt": truth,

            "v4_pred": v4_pred_now,
            "visual_pred": visual_viterbi_pred,
            "visual_raw_top1": visual_top_raw,

            "v4_score": v4_score,
            "v4_margin": v4_margin,
            "v4_entropy": safe_entropy(v4_row),
            "v4_boundary": (
                float(boundary[seg_idx])
                if seg_idx < len(boundary)
                else 0.0
            ),

            "visual_score": visual_score,
            "visual_margin": visual_margin,

            "cross_visual_on_v4": visual_on_v4,
            "cross_v4_on_visual": v4_on_visual,

            "prediction_agree": int(
                v4_pred_now == visual_viterbi_pred
            ),
            "prediction_distance": abs(
                v4_pred_now - visual_viterbi_pred
            ),

            "v4_stable": int(v4_pred_now == prev_v4 == next_v4),
            "visual_stable": (
                int(
                    visual_idx > 0
                    and visual_idx + 1 < len(visual_path)
                    and visual_path[visual_idx - 1]
                    == visual_path[visual_idx]
                    == visual_path[visual_idx + 1]
                )
            ),

            "v4_transition_pressure": abs(
                next_v4 - v4_pred_now
            ),

            "v4_is_correct": int(v4_pred_now == truth),
            "visual_is_correct": int(
                visual_viterbi_pred == truth
            ),

            # Training target:
            # visual branch can recover this V4 error.
            "gate_target": int(
                visual_viterbi_pred == truth
                and v4_pred_now != truth
            ),
        })

    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = out_dir / "features.csv"
    summary_path = out_dir / "summary.json"

    df.to_csv(feature_path, index=False)

    v4_acc = float(df["v4_is_correct"].mean()) if len(df) else 0.0
    visual_acc = (
        float(df["visual_is_correct"].mean())
        if len(df) else 0.0
    )
    oracle_acc = (
        float(
            (
                df["v4_is_correct"].astype(bool)
                | df["visual_is_correct"].astype(bool)
            ).mean()
        )
        if len(df) else 0.0
    )

    summary = {
        "lecture": lecture,
        "gt_points": int(len(df)),
        "v4_accuracy": v4_acc,
        "visual_accuracy": visual_acc,
        "oracle_gate_accuracy": oracle_acc,
        "v4_only_correct": int(
            (
                (df["v4_is_correct"] == 1)
                & (df["visual_is_correct"] == 0)
            ).sum()
        ),
        "visual_only_correct": int(
            (
                (df["v4_is_correct"] == 0)
                & (df["visual_is_correct"] == 1)
            ).sum()
        ),
        "both_correct": int(
            (
                (df["v4_is_correct"] == 1)
                & (df["visual_is_correct"] == 1)
            ).sum()
        ),
        "both_wrong": int(
            (
                (df["v4_is_correct"] == 0)
                & (df["visual_is_correct"] == 0)
            ).sum()
        ),
        "gate_positive_count": int(
            df["gate_target"].sum()
        ),
        "feature_count": int(len(df.columns)),
        "sample_seconds_visual": float(sample_seconds),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRESULT")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(" ", feature_path)
    print(" ", summary_path)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LectureSense V5 single-file training dataset builder"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="LectureSense project root",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Process one training lecture",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=2.0,
        help="Visual sampling interval. Default=2s.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.only:
        if args.only not in TRAIN_LECTURES:
            raise ValueError(
                f"--only must be one of: {TRAIN_LECTURES}"
            )
        lectures = [args.only]
    else:
        lectures = TRAIN_LECTURES

    print("=" * 78)
    print("LECTURESENSE V5 — SINGLE-FILE MULTIMODAL DATA BUILDER V2")
    print("=" * 78)
    print("Project:", root)
    print("Training lectures:", len(lectures))
    print("Visual sample seconds:", args.sample_seconds)
    print("Validation/test lectures: NOT USED")
    print("=" * 78)

    completed = 0
    summaries = []

    for lecture in lectures:
        try:
            result = build_one(
                root,
                lecture,
                args.sample_seconds,
            )
            if result is not None:
                completed += 1
                summaries.append(result)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as exc:
            print(
                f"\n[FAILED] {lecture}: "
                f"{type(exc).__name__}: {exc}"
            )
            print("Continuing to next training lecture.")

    # Combined training set.
    output_root = (
        root / "runs" / "v5_train_gate_dataset_FINAL_V2"
    )

    feature_files = list(
        output_root.glob("*/features.csv")
    )

    if feature_files:
        combined = pd.concat(
            [pd.read_csv(f) for f in feature_files],
            ignore_index=True,
        )
        combined.to_csv(
            output_root / "ALL_TRAIN_FEATURES.csv",
            index=False,
        )

        combined_summary = {
            "lectures_completed": completed,
            "lectures_requested": len(lectures),
            "rows": int(len(combined)),
            "v4_accuracy": float(
                combined["v4_is_correct"].mean()
            ),
            "visual_accuracy": float(
                combined["visual_is_correct"].mean()
            ),
            "oracle_gate_accuracy": float(
                (
                    combined["v4_is_correct"].astype(bool)
                    | combined["visual_is_correct"].astype(bool)
                ).mean()
            ),
            "gate_positive_count": int(
                combined["gate_target"].sum()
            ),
            "validation_used": False,
            "test_used": False,
        }

        (output_root / "ALL_TRAIN_SUMMARY.json").write_text(
            json.dumps(combined_summary, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("COMBINED TRAINING DATA")
        print("=" * 78)
        print(json.dumps(combined_summary, indent=2))
        print(
            "\nCombined file:",
            output_root / "ALL_TRAIN_FEATURES.csv"
        )

    print("\n" + "=" * 78)
    print("BUILD COMPLETE")
    print("=" * 78)
    print(f"Completed lectures: {completed}/{len(lectures)}")
    print("Validation/test lectures were not used.")

if __name__ == "__main__":
    main()
