import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference import read_mavils_srt
from visual_first_alignment import load_ground_truth, run_visual_first, viterbi_monotonic


VIDEO_ALIASES = {
    "computer_vision_2_2": "computer_vision_2_2_high_res.mp4",
}


def locate(folder, stem, ext):
    exact = folder / f"{stem}{ext}"
    if exact.exists():
        return exact
    matches = list(folder.glob(f"*{stem}*{ext}"))
    return matches[0] if matches else None


def get_assets(root, lecture):
    data = root / "data" / "MaViLS" / "data"

    srt = locate(data / "audioscripts", lecture, ".srt")
    pdf = locate(data / "lectures", lecture, ".pdf")
    gt = locate(data / "ground_truth_files", lecture, ".xlsx")

    video_dir = Path(r"C:\Users\Bharat\Downloads\archive\video")
    video = None

    alias = VIDEO_ALIASES.get(lecture)
    if alias:
        p = video_dir / alias
        if p.exists():
            video = p

    if video is None:
        key = lecture.lower().replace("_", "")
        for p in video_dir.glob("*.mp4"):
            if key in p.stem.lower().replace("_", ""):
                video = p
                break

    missing = []
    for name, value in [
        ("SRT", srt),
        ("PDF", pdf),
        ("GT", gt),
        ("VIDEO", video),
    ]:
        if value is None:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"{lecture}: missing {', '.join(missing)}"
        )

    return srt, pdf, gt, video


def load_v4_timeline(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        timeline = data.get("timeline")
        if isinstance(timeline, list):
            return timeline

        for key in ("segments", "entries", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported timeline.json structure: {type(data).__name__}"
    )


def get_timeline_keys(timeline):
    first = timeline[0]
    lower = {str(k).lower(): k for k in first}

    def pick(names):
        for name in names:
            if name in lower:
                return lower[name]
        return None

    start_key = pick(["start_time", "start", "timestamp_start"])
    end_key = pick(["end_time", "end", "timestamp_end"])
    slide_key = pick(
        ["slide_id", "slide", "slide_number", "slidenumber"]
    )

    if not all([start_key, end_key, slide_key]):
        raise ValueError(
            f"Could not identify timeline fields: {list(first.keys())}"
        )

    return start_key, end_key, slide_key


def create_v4_csv(root, lecture):
    v4_dir = (
        root
        / "runs"
        / "v5_gate_dataset_fixed"
        / "v4"
        / lecture
    )

    csv_path = v4_dir / "v4_segment_predictions.csv"

    if csv_path.exists():
        print(f"Using existing V4 CSV: {csv_path}")
        return csv_path

    timeline_path = v4_dir / "timeline.json"
    score_path = v4_dir / "v4_final_score_matrix.npy"
    boundary_path = v4_dir / "v4_boundary_probability.npy"
    next_path = v4_dir / "v4_next_slide_logits.npy"

    required = [
        timeline_path,
        score_path,
        boundary_path,
        next_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required existing V4 output missing: {path}"
            )

    timeline = load_v4_timeline(timeline_path)
    start_key, end_key, slide_key = get_timeline_keys(timeline)

    score_matrix = np.load(score_path)
    boundary = np.load(boundary_path)
    next_logits = np.load(next_path)

    srt, _, _, _ = get_assets(root, lecture)
    transcript = read_mavils_srt(srt)

    print(f"Transcript segments: {len(transcript)}")
    print(f"Score rows: {score_matrix.shape[0]}")

    if len(transcript) != score_matrix.shape[0]:
        raise ValueError(
            f"Transcript/score mismatch: "
            f"{len(transcript)} vs {score_matrix.shape[0]}"
        )

    timeline_start = np.array(
        [float(x[start_key]) for x in timeline]
    )
    timeline_end = np.array(
        [float(x[end_key]) for x in timeline]
    )
    timeline_slide = np.array(
        [int(x[slide_key]) for x in timeline]
    )

    rows = []

    for i, segment in enumerate(transcript):
        start = float(segment["start"])
        end = float(segment["end"])

        matches = np.flatnonzero(
            (timeline_start <= start)
            & (start < timeline_end)
        )

        if len(matches):
            timeline_index = int(matches[0])
        else:
            centers = (timeline_start + timeline_end) / 2.0
            timeline_index = int(
                np.argmin(np.abs(centers - start))
            )

        slide = int(timeline_slide[timeline_index])

        scores = np.asarray(score_matrix[i]).reshape(-1)

        if 1 <= slide <= len(scores):
            selected = float(scores[slide - 1])
            others = np.delete(scores, slide - 1)
            second = (
                float(np.max(others))
                if len(others)
                else selected
            )
        else:
            order = np.argsort(scores)[::-1]
            slide = int(order[0]) + 1
            selected = float(scores[order[0]])
            second = (
                float(scores[order[1]])
                if len(order) > 1
                else selected
            )

        boundary_value = float(
            np.asarray(boundary[i]).reshape(-1)[0]
        )

        next_row = np.asarray(next_logits[i]).reshape(-1)
        next_slide = (
            int(np.argmax(next_row)) + 1
            if len(next_row)
            else -1
        )

        rows.append(
            {
                "segment_index": i,
                "start": start,
                "end": end,
                "v4_pred": slide,
                "v4_score": selected,
                "v4_margin": selected - second,
                "boundary_probability": boundary_value,
                "next_slide_pred": next_slide,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(csv_path, index=False)

    print(f"Created: {csv_path}")
    print(f"V4 CSV rows: {len(result)}")

    return csv_path


def get_visual_cache(root, lecture, video, pdf):
    cache_dir = (
        root
        / "runs"
        / "v5_gate_dataset_from_existing_v4"
        / "_visual_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{lecture}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)

        print(f"Using existing Visual-First cache: {cache_path}")

        return (
            cached["times"],
            cached["scores"],
            cached["path"],
            True,
        )

    print(f"[{lecture}] Running actual Visual-First...")

    _, scores, times, _ = run_visual_first(
        video,
        pdf,
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )

    # IMPORTANT:
    # viterbi_monotonic returns (path, dp_score).
    path, _ = viterbi_monotonic(
        scores,
        max_jump=2,
        jump_penalty=0.05,
    )

    path = np.asarray(path, dtype=np.int16)
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float32)

    np.savez_compressed(
        cache_path,
        times=times,
        scores=scores,
        path=path,
    )

    print(f"Saved Visual-First cache: {cache_path}")

    return times, scores, path, False


def find_v4_segment(v4, timestamp):
    starts = v4["start"].to_numpy(dtype=float)
    ends = v4["end"].to_numpy(dtype=float)

    matches = np.flatnonzero(
        (starts <= timestamp)
        & (timestamp < ends)
    )

    if len(matches):
        return int(matches[0])

    centers = (starts + ends) / 2.0
    return int(
        np.argmin(np.abs(centers - timestamp))
    )


def nearest_visual_frame(times, scores, path, timestamp):
    index = int(np.searchsorted(times, timestamp))

    if index <= 0:
        index = 0
    elif index >= len(times):
        index = len(times) - 1
    elif abs(times[index - 1] - timestamp) <= abs(
        times[index] - timestamp
    ):
        index -= 1

    prediction = int(path[index]) + 1

    row = scores[index]
    order = np.argsort(row)[::-1]

    selected = float(row[prediction - 1])

    second = (
        float(row[order[1]])
        if len(order) > 1
        else selected
    )

    return (
        prediction,
        float(times[index]),
        selected,
        selected - second,
        index,
    )


def run_comparison(root, lecture):
    v4_dir = (
        root
        / "runs"
        / "v5_gate_dataset_fixed"
        / "v4"
        / lecture
    )

    v4_csv = create_v4_csv(root, lecture)
    v4 = pd.read_csv(v4_csv)

    _, pdf, gt_path, video = get_assets(root, lecture)

    ground_truth = load_ground_truth(gt_path)

    times, visual_scores, visual_path, cache_reused = (
        get_visual_cache(
            root,
            lecture,
            video,
            pdf,
        )
    )

    rows = []

    for timestamp, ground_truth_slide in ground_truth:
        timestamp = float(timestamp)
        ground_truth_slide = int(ground_truth_slide)

        v4_index = find_v4_segment(
            v4,
            timestamp,
        )

        v4_row = v4.iloc[v4_index]

        (
            visual_prediction,
            visual_time,
            visual_score,
            visual_margin,
            visual_index,
        ) = nearest_visual_frame(
            times,
            visual_scores,
            visual_path,
            timestamp,
        )

        v4_prediction = int(v4_row["v4_pred"])

        rows.append(
            {
                "timestamp": timestamp,
                "gt_slide": ground_truth_slide,
                "v4_pred": v4_prediction,
                "visual_pred": visual_prediction,
                "visual_frame_time": visual_time,
                "v4_correct": int(
                    v4_prediction == ground_truth_slide
                ),
                "visual_correct": int(
                    visual_prediction == ground_truth_slide
                ),
                "v4_score": float(v4_row["v4_score"]),
                "v4_margin": float(v4_row["v4_margin"]),
                "boundary_probability": float(
                    v4_row["boundary_probability"]
                ),
                "next_slide_pred": int(
                    v4_row["next_slide_pred"]
                ),
                "visual_score": visual_score,
                "visual_margin": visual_margin,
                "visual_frame_index": visual_index,
            }
        )

    result = pd.DataFrame(rows)

    v4_predictions = result["v4_pred"].to_numpy()
    visual_predictions = result["visual_pred"].to_numpy()

    result["v4_stable"] = np.r_[
        1.0,
        (
            v4_predictions[1:]
            == v4_predictions[:-1]
        ).astype(float),
    ]

    result["v4_boundary_proxy"] = np.r_[
        0.0,
        (
            v4_predictions[1:]
            != v4_predictions[:-1]
        ).astype(float),
    ]

    result["visual_stable"] = np.r_[
        1.0,
        (
            visual_predictions[1:]
            == visual_predictions[:-1]
        ).astype(float),
    ]

    result["prediction_agree"] = (
        v4_predictions == visual_predictions
    ).astype(float)

    result["prediction_distance"] = np.abs(
        v4_predictions - visual_predictions
    ).astype(float)

    result["gate_target"] = (
        (result["visual_correct"] == 1)
        & (result["v4_correct"] == 0)
    ).astype(int)

    output_dir = (
        root
        / "runs"
        / "v5_gate_dataset_from_existing_v4"
        / lecture
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        output_dir / "timestamp_comparison.csv"
    )

    result.to_csv(
        comparison_path,
        index=False,
    )

    summary = {
        "lecture": lecture,
        "gt_points": int(len(result)),
        "v4_accuracy": float(
            result["v4_correct"].mean()
        ),
        "visual_accuracy": float(
            result["visual_correct"].mean()
        ),
        "both_correct": int(
            (
                (result["v4_correct"] == 1)
                & (result["visual_correct"] == 1)
            ).sum()
        ),
        "v4_only_correct": int(
            (
                (result["v4_correct"] == 1)
                & (result["visual_correct"] == 0)
            ).sum()
        ),
        "visual_only_correct": int(
            (
                (result["v4_correct"] == 0)
                & (result["visual_correct"] == 1)
            ).sum()
        ),
        "both_wrong": int(
            (
                (result["v4_correct"] == 0)
                & (result["visual_correct"] == 0)
            ).sum()
        ),
        "oracle_gate_accuracy": float(
            (
                (result["v4_correct"] == 1)
                | (result["visual_correct"] == 1)
            ).mean()
        ),
        "gate_positive_count": int(
            result["gate_target"].sum()
        ),
        "visual_cache_reused": bool(cache_reused),
    }

    summary_path = output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== RESULT ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print(f"Saved: {comparison_path}")
    print(f"Saved: {summary_path}")

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
    )

    parser.add_argument(
        "--only",
        required=True,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    try:
        run_comparison(
            root,
            args.only,
        )
    except Exception as exc:
        print(
            f"\n[{args.only}] FAILED: "
            f"{type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    main()
