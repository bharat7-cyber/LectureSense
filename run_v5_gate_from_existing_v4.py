import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visual_first_alignment import (
    load_ground_truth,
    run_visual_first,
    viterbi_monotonic,
)

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

VIDEO_ALIASES = {
    "computer_vision_2_2": "computer_vision_2_2_high_res.mp4",
    "climate_science_policy_MIT2": "climate_science_policy_MIT2_high_res.mp4",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT_high_res.mp4",
    "creating_breakthrough_products_MIT": "creating_breakthrough_products_MIT_high_res.mp4",
    "cryptocurrency_MIT": "cryptocurrency_MIT_high_res.mp4",
    "ML_for_health_MIT": "ML_for_health_MIT_high_res.mp4",
}

def locate_file(folder, stem, ext):
    exact = folder / f"{stem}{ext}"
    if exact.exists():
        return exact
    candidates = list(folder.glob(f"*{stem}*{ext}"))
    if candidates:
        return candidates[0]
    candidates = list(folder.glob(f"{stem}*{ext}"))
    return candidates[0] if candidates else None

def locate_assets(root, lecture):
    data = root / "data" / "MaViLS" / "data"
    srt_dir = data / "audioscripts"
    pdf_dir = data / "lectures"
    gt_dir = data / "ground_truth_files"
    video_dir = Path(r"C:\Users\Bharat\Downloads\archive\video")

    srt = locate_file(srt_dir, lecture, ".srt")
    pdf = locate_file(pdf_dir, lecture, ".pdf")
    gt = locate_file(gt_dir, lecture, ".xlsx")

    video = None
    alias = VIDEO_ALIASES.get(lecture)
    if alias and (video_dir / alias).exists():
        video = video_dir / alias
    else:
        candidates = list(video_dir.glob("*.mp4"))
        key = lecture.lower().replace("_", "")
        for p in candidates:
            if key in p.stem.lower().replace("_", ""):
                video = p
                break

    missing = []
    for name, p in [("SRT", srt), ("PDF", pdf), ("GT", gt), ("VIDEO", video)]:
        if p is None:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"{lecture}: missing {', '.join(missing)}")

    return srt, pdf, gt, video

def find_time_columns(timeline):
    if not isinstance(timeline, list):
        raise ValueError("timeline.json is not a list")

    if not timeline:
        raise ValueError("timeline.json is empty")

    sample = timeline[0]
    keys = {str(k).lower(): k for k in sample.keys()}

    start = next((keys[k] for k in ["start", "start_time", "timestamp_start"] if k in keys), None)
    end = next((keys[k] for k in ["end", "end_time", "timestamp_end"] if k in keys), None)
    slide = next((keys[k] for k in ["slide", "slide_id", "slidenumber", "slide_number"] if k in keys), None)

    if start is None or end is None or slide is None:
        raise ValueError(f"Could not identify timeline columns. First entry: {sample}")

    return start, end, slide

def make_v4_csv(v4_dir):
    csv_path = v4_dir / "v4_segment_predictions.csv"
    if csv_path.exists():
        return csv_path

    timeline_path = v4_dir / "timeline.json"
    final_path = v4_dir / "v4_final_score_matrix.npy"
    boundary_path = v4_dir / "v4_boundary_probability.npy"
    next_path = v4_dir / "v4_next_slide_logits.npy"

    for p in [timeline_path, final_path, boundary_path, next_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing existing V4 output: {p}")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    start_key, end_key, slide_key = find_time_columns(timeline)

    score = np.load(final_path)
    boundary = np.load(boundary_path)
    next_logits = np.load(next_path)

    n = len(timeline)
    if score.ndim != 2 or score.shape[0] != n:
        raise ValueError(
            f"V4 score matrix shape {score.shape} does not match timeline length {n}"
        )

    if len(boundary) != n:
        raise ValueError(
            f"Boundary length {len(boundary)} does not match timeline length {n}"
        )

    # Existing timeline is the authoritative V4 segment timing/prediction.
    # Use the final score matrix only to recover score/margin features.
    rows = []
    for i, item in enumerate(timeline):
        s = int(item[slide_key])

        row = np.asarray(score[i]).reshape(-1)
        order = np.argsort(row)[::-1]

        if 1 <= s <= len(row):
            v4_score = float(row[s - 1])
            remaining = row[np.arange(len(row)) != (s - 1)]
            second = float(np.max(remaining)) if len(remaining) else v4_score
            margin = v4_score - second
        else:
            best_idx = int(order[0])
            v4_score = float(row[best_idx])
            second = float(row[order[1]]) if len(order) > 1 else v4_score
            margin = v4_score - second

        b = float(np.asarray(boundary[i]).reshape(-1)[0])

        next_pred = -1
        if next_logits.ndim >= 2 and i < next_logits.shape[0]:
            nr = np.asarray(next_logits[i]).reshape(-1)
            if len(nr):
                next_pred = int(np.argmax(nr)) + 1

        rows.append({
            "segment_index": i,
            "start": float(item[start_key]),
            "end": float(item[end_key]),
            "v4_pred": s,
            "v4_score": v4_score,
            "v4_margin": margin,
            "boundary_probability": b,
            "next_slide_pred": next_pred,
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Created existing-V4 CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    return csv_path

def interval_index(v4, t):
    starts = v4["start"].to_numpy()
    ends = v4["end"].to_numpy()

    idx = np.flatnonzero((starts <= t) & (t < ends))
    if len(idx):
        if len(idx) > 1:
            lengths = ends[idx] - starts[idx]
            return int(idx[np.argmin(lengths)])
        return int(idx[0])

    mids = (starts + ends) / 2.0
    return int(np.argmin(np.abs(mids - t)))

def get_visual_cache(root, lecture, video, pdf):
    cache_dir = root / "runs" / "v5_gate_dataset_from_existing_v4" / "_visual_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{lecture}.npz"

    if cache.exists():
        z = np.load(cache)
        return z["times"], z["scores"], z["path"], True

    print(f"[{lecture}] Running actual Visual-First...")
    _, scores, times, _ = run_visual_first(
        video,
        pdf,
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )

    path = viterbi_monotonic(
        scores,
        max_jump=2,
        jump_penalty=0.05,
    ).astype(np.int16)

    np.savez_compressed(
        cache,
        times=np.asarray(times, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float32),
        path=path,
    )
    return np.asarray(times), np.asarray(scores), path, False

def nearest_visual(times, scores, path, t):
    j = int(np.searchsorted(times, t))

    if j <= 0:
        j = 0
    elif j >= len(times):
        j = len(times) - 1
    elif abs(times[j - 1] - t) <= abs(times[j] - t):
        j -= 1

    row = scores[j]
    order = np.argsort(row)[::-1]

    pred = int(path[j]) + 1
    score = float(row[pred - 1])

    if len(order) > 1:
        second = float(row[order[1]])
    else:
        second = score

    return pred, float(times[j]), score, score - second, j

def process(root, lecture):
    v4_dir = (
        root
        / "runs"
        / "v5_gate_dataset_fixed"
        / "v4"
        / lecture
    )

    if not v4_dir.exists():
        raise FileNotFoundError(f"Existing V4 directory not found: {v4_dir}")

    # This creates only the missing CSV. It does NOT run V4.
    v4_csv = make_v4_csv(v4_dir)
    v4 = pd.read_csv(v4_csv)

    _, pdf, gt_path, video = locate_assets(root, lecture)
    gt = load_ground_truth(gt_path)

    times, scores, path, cache_reused = get_visual_cache(
        root, lecture, video, pdf
    )

    rows = []
    for t, gt_slide in gt:
        t = float(t)
        gt_slide = int(gt_slide)

        vi = interval_index(v4, t)
        vr = v4.iloc[vi]

        visual_pred, visual_time, visual_score, visual_margin, frame_idx = nearest_visual(
            times, scores, path, t
        )

        v4_pred = int(vr["v4_pred"])

        rows.append({
            "timestamp": t,
            "gt_slide": gt_slide,
            "v4_pred": v4_pred,
            "visual_pred": visual_pred,
            "visual_frame_time": visual_time,
            "v4_correct": int(v4_pred == gt_slide),
            "visual_correct": int(visual_pred == gt_slide),
            "v4_score": float(vr["v4_score"]),
            "v4_margin": float(vr["v4_margin"]),
            "boundary_probability": float(vr["boundary_probability"]),
            "next_slide_pred": int(vr["next_slide_pred"]),
            "visual_score": visual_score,
            "visual_margin": visual_margin,
            "visual_frame_index": frame_idx,
        })

    df = pd.DataFrame(rows)

    if len(df):
        vp = df["v4_pred"].to_numpy()
        xp = df["visual_pred"].to_numpy()

        df["v4_stable"] = np.r_[1.0, (vp[1:] == vp[:-1]).astype(float)]
        df["v4_boundary_proxy"] = np.r_[0.0, (vp[1:] != vp[:-1]).astype(float)]
        df["visual_stable"] = np.r_[1.0, (xp[1:] == xp[:-1]).astype(float)]
        df["prediction_agree"] = (vp == xp).astype(float)
        df["prediction_distance"] = np.abs(vp - xp).astype(float)
    else:
        for c in [
            "v4_stable", "v4_boundary_proxy", "visual_stable",
            "prediction_agree", "prediction_distance"
        ]:
            df[c] = []

    # V5 gate training target:
    # 1 = switch from V4 to Visual-First at this timestamp.
    df["gate_target"] = (
        (df["visual_correct"] == 1) &
        (df["v4_correct"] == 0)
    ).astype(int)

    out_dir = root / "runs" / "v5_gate_dataset_from_existing_v4" / lecture
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "timestamp_comparison.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "lecture": lecture,
        "gt_points": int(len(df)),
        "v4_correct": int(df["v4_correct"].sum()),
        "visual_correct": int(df["visual_correct"].sum()),
        "both_correct": int(((df.v4_correct == 1) & (df.visual_correct == 1)).sum()),
        "v4_only_correct": int(((df.v4_correct == 1) & (df.visual_correct == 0)).sum()),
        "visual_only_correct": int(((df.v4_correct == 0) & (df.visual_correct == 1)).sum()),
        "both_wrong": int(((df.v4_correct == 0) & (df.visual_correct == 0)).sum()),
        "v4_accuracy": float(df.v4_correct.mean()) if len(df) else 0.0,
        "visual_accuracy": float(df.visual_correct.mean()) if len(df) else 0.0,
        "oracle_gate_accuracy": float(
            ((df.v4_correct == 1) | (df.visual_correct == 1)).mean()
        ) if len(df) else 0.0,
        "gate_positive_count": int(df.gate_target.sum()),
        "visual_cache_reused": bool(cache_reused),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"=== {lecture} ===")
    print(f"GT points:             {summary['gt_points']}")
    print(f"V4 accuracy:           {summary['v4_accuracy']:.4%}")
    print(f"Visual-First accuracy: {summary['visual_accuracy']:.4%}")
    print(f"Both correct:          {summary['both_correct']}")
    print(f"V4-only correct:       {summary['v4_only_correct']}")
    print(f"Visual-only correct:   {summary['visual_only_correct']}")
    print(f"Both wrong:            {summary['both_wrong']}")
    print(f"Oracle gate accuracy:  {summary['oracle_gate_accuracy']:.4%}")
    print(f"Gate positive samples: {summary['gate_positive_count']}")
    print(f"Saved: {out_csv}")

    return df, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lectures = [args.only] if args.only else TRAIN_LECTURES

    all_df = []
    summaries = []

    for lecture in lectures:
        try:
            df, summary = process(root, lecture)
            df["lecture"] = lecture
            all_df.append(df)
            summaries.append(summary)
        except Exception as e:
            print(f"\n[{lecture}] FAILED: {type(e).__name__}: {e}")

    if not all_df:
        print("\nNo lecture completed.")
        return

    out_root = root / "runs" / "v5_gate_dataset_from_existing_v4"
    combined = pd.concat(all_df, ignore_index=True)
    combined.to_csv(out_root / "gate_dataset.csv", index=False)

    combined_summary = {
        "lectures_completed": [x["lecture"] for x in summaries],
        "total_rows": int(len(combined)),
        "total_gate_positive": int(combined["gate_target"].sum()),
    }

    (out_root / "summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )

    print("\nCOMBINED DATASET")
    print(f"Rows: {len(combined)}")
    print(f"Gate-positive rows: {int(combined['gate_target'].sum())}")
    print(f"Saved: {out_root / 'gate_dataset.csv'}")

if __name__ == "__main__":
    main()
