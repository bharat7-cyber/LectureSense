import argparse
import json
import re
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

ALIASES = {
    "computer_vision_2_2": {
        "srt": "computer_vision_2_2_high_res.srt",
        "pdf": "computer_vision_2_2_geiger.pdf",
        "gt": "ground_truth_computer_vision_2_2.xlsx",
        "video": "computer_vision_2_2_high_res.mp4",
    },
    "climate_science_policy_MIT2": {
        "video_names": ["climate_science_policy_MIT2_high_res.mp4", "climate_science_policy_MIT_high_res.mp4"],
    },
    "cognitive_robotics_MIT": {
        "video_names": ["cognitive_robotics_MIT_high_res.mp4"],
    },
    "creating_breakthrough_products_MIT": {
        "video_names": ["creating_breakthrough_products_MIT_high_res.mp4"],
    },
    "cryptocurrency_MIT": {
        "video_names": ["cryptocurrency_MIT_high_res.mp4"],
    },
    "ML_for_health_MIT": {
        "video_names": ["ML_for_health_MIT_high_res.mp4"],
    },
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def first_match(files, key, preferred_ext=None):
    nk = norm(key)
    exact = [p for p in files if norm(p.stem) == nk]
    if exact:
        return exact[0]
    contains = [p for p in files if nk in norm(p.stem) or norm(p.stem) in nk]
    if preferred_ext:
        contains = [p for p in contains if p.suffix.lower() == preferred_ext.lower()]
    return contains[0] if contains else None

def locate_assets(root, lecture):
    mavils = root / "data" / "MaViLS" / "data"
    srt_dir = mavils / "audioscripts"
    pdf_dir = mavils / "lectures"
    gt_dir = mavils / "ground_truth_files"
    video_dir = Path(r"C:\Users\Bharat\Downloads\archive\video")

    srt_files = list(srt_dir.glob("*.srt"))
    pdf_files = list(pdf_dir.glob("*.pdf"))
    gt_files = list(gt_dir.glob("*.xlsx"))
    video_files = list(video_dir.glob("*.mp4"))

    a = ALIASES.get(lecture, {})

    def exact_name(folder, name):
        if not name:
            return None
        p = folder / name
        return p if p.exists() else None

    srt = exact_name(srt_dir, a.get("srt"))
    pdf = exact_name(pdf_dir, a.get("pdf"))
    gt = exact_name(gt_dir, a.get("gt"))

    if srt is None:
        srt = first_match(srt_files, lecture, ".srt")
    if pdf is None:
        pdf = first_match(pdf_files, lecture, ".pdf")
    if gt is None:
        gt = first_match(gt_files, lecture, ".xlsx")

    video = None
    for name in a.get("video_names", []):
        p = video_dir / name
        if p.exists():
            video = p
            break
    if video is None and a.get("video"):
        p = video_dir / a["video"]
        if p.exists():
            video = p
    if video is None:
        # Prefer filenames containing the normalized lecture key.
        nk = norm(lecture)
        candidates = [p for p in video_files if nk in norm(p.stem)]
        if candidates:
            video = candidates[0]

    missing = []
    for name, value in [("SRT", srt), ("PDF", pdf), ("GT", gt), ("VIDEO", video)]:
        if value is None:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"{lecture}: could not locate {', '.join(missing)}. "
            f"Use --only and inspect the filenames if this is a new alias."
        )

    return srt, pdf, gt, video

def find_col(df, names, contains=()):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(x in lc for x in contains):
            return c
    return None

def load_v4_csv(path):
    df = pd.read_csv(path)
    start = find_col(df, ["start", "start_time", "segment_start"], ("start",))
    end = find_col(df, ["end", "end_time", "segment_end"], ("end",))
    pred = find_col(df, ["v4_pred", "pred", "slide"], ("pred",))
    score = find_col(df, ["v4_score", "score"], ("score",))
    margin = find_col(df, ["v4_margin", "margin"], ("margin",))
    boundary = find_col(df, ["boundary_probability", "boundary_prob"], ("boundary",))

    missing = [x for x, c in [("start", start), ("end", end), ("pred", pred)] if c is None]
    if missing:
        raise ValueError(f"V4 CSV {path} is missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "start": pd.to_numeric(df[start], errors="coerce"),
        "end": pd.to_numeric(df[end], errors="coerce"),
        "v4_pred": pd.to_numeric(df[pred], errors="coerce"),
    })
    out["v4_score"] = pd.to_numeric(df[score], errors="coerce") if score else 0.0
    out["v4_margin"] = pd.to_numeric(df[margin], errors="coerce") if margin else 0.0
    out["boundary_probability"] = pd.to_numeric(df[boundary], errors="coerce") if boundary else 0.0
    out = out.dropna(subset=["start", "end", "v4_pred"]).reset_index(drop=True)
    return out

def interval_lookup(v4, t):
    mask = (v4["start"].to_numpy() <= t) & (t < v4["end"].to_numpy())
    idxs = np.flatnonzero(mask)
    if len(idxs):
        # If intervals overlap, choose the shortest containing interval.
        if len(idxs) > 1:
            lens = v4.iloc[idxs]["end"].to_numpy() - v4.iloc[idxs]["start"].to_numpy()
            return int(idxs[np.argmin(lens)])
        return int(idxs[0])
    mids = (v4["start"].to_numpy() + v4["end"].to_numpy()) / 2.0
    return int(np.argmin(np.abs(mids - t)))

def visual_cache(root, lecture, video, pdf, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{lecture}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["times"], z["scores"], z["path"], True

    print(f"[{lecture}] Running ACTUAL Visual-First branch...")
    timeline, scores, times, duration = run_visual_first(
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
    else:
        if abs(times[j - 1] - t) <= abs(times[j] - t):
            j -= 1

    row = scores[j]
    order = np.argsort(row)[::-1]
    best = int(path[j]) + 1
    best_score = float(row[best - 1])
    second = float(row[order[1]]) if len(order) > 1 else best_score
    margin = best_score - second
    return best, float(times[j]), best_score, margin, j

def add_stability_features(df):
    preds = df["v4_pred"].to_numpy(dtype=np.int64)
    vp = df["visual_pred"].to_numpy(dtype=np.int64)

    v4_stable = np.ones(len(df), dtype=np.float32)
    v4_boundary = np.zeros(len(df), dtype=np.float32)
    visual_stable = np.ones(len(df), dtype=np.float32)

    if len(df) > 1:
        v4_stable[1:] = (preds[1:] == preds[:-1]).astype(np.float32)
        v4_boundary[1:] = (preds[1:] != preds[:-1]).astype(np.float32)
        visual_stable[1:] = (vp[1:] == vp[:-1]).astype(np.float32)

    df["v4_stable"] = v4_stable
    df["v4_boundary_proxy"] = v4_boundary
    df["visual_stable"] = visual_stable
    df["prediction_agree"] = (preds == vp).astype(np.float32)
    df["prediction_distance"] = np.abs(preds - vp).astype(np.float32)
    return df

def process(root, lecture):
    out_root = root / "runs" / "v5_gate_dataset_existing_v4"
    out_dir = out_root / lecture
    out_dir.mkdir(parents=True, exist_ok=True)

    srt, pdf, gt_path, video = locate_assets(root, lecture)

    # Reuse V4 output. This script NEVER reruns V4.
    v4_csv = root / "runs" / "v5_gate_dataset_fixed" / "v4" / lecture / "v4_segment_predictions.csv"
    if not v4_csv.exists():
        raise FileNotFoundError(
            f"Existing V4 CSV not found:\n{v4_csv}\n"
            "Run the V4 exporter for this lecture first."
        )

    v4 = load_v4_csv(v4_csv)
    gt_rows = load_ground_truth(gt_path)

    times, scores, path, reused = visual_cache(
        root, lecture, video, pdf, out_root / "_visual_cache"
    )

    records = []
    for t, gt_slide in gt_rows:
        t = float(t)
        gt_slide = int(gt_slide)

        vi = interval_lookup(v4, t)
        vr = v4.iloc[vi]

        visual_pred, visual_time, visual_score, visual_margin, frame_idx = nearest_visual(
            times, scores, path, t
        )

        v4_pred = int(vr["v4_pred"])
        records.append({
            "timestamp": t,
            "gt_slide": gt_slide,
            "v4_pred": v4_pred,
            "visual_pred": int(visual_pred),
            "visual_frame_time": visual_time,
            "v4_correct": int(v4_pred == gt_slide),
            "visual_correct": int(visual_pred == gt_slide),
            "v4_score": float(vr["v4_score"]),
            "v4_margin": float(vr["v4_margin"]),
            "boundary_probability": float(vr["boundary_probability"]),
            "visual_score": visual_score,
            "visual_margin": visual_margin,
            "visual_frame_index": frame_idx,
        })

    df = pd.DataFrame(records)
    df = add_stability_features(df)

    # Target: Visual-First is correct exactly where V4 is wrong.
    df["gate_target"] = (
        (df["visual_correct"] == 1) & (df["v4_correct"] == 0)
    ).astype(np.int64)

    out_csv = out_dir / "timestamp_comparison.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "lecture": lecture,
        "gt_points": int(len(df)),
        "v4_correct": int(df.v4_correct.sum()),
        "visual_correct": int(df.visual_correct.sum()),
        "both_correct": int(((df.v4_correct == 1) & (df.visual_correct == 1)).sum()),
        "v4_only_correct": int(((df.v4_correct == 1) & (df.visual_correct == 0)).sum()),
        "visual_only_correct": int(((df.v4_correct == 0) & (df.visual_correct == 1)).sum()),
        "both_wrong": int(((df.v4_correct == 0) & (df.visual_correct == 0)).sum()),
        "oracle_gate_accuracy": float(
            ((df.v4_correct == 1) | (df.visual_correct == 1)).mean()
        ),
        "v4_accuracy": float(df.v4_correct.mean()),
        "visual_accuracy": float(df.visual_correct.mean()),
        "gate_positive_count": int(df.gate_target.sum()),
        "visual_cache_reused": bool(reused),
        "v4_csv": str(v4_csv),
        "visual_cache": str(out_root / "_visual_cache" / f"{lecture}.npz"),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"=== {lecture} ===")
    print(f"GT points:              {summary['gt_points']}")
    print(f"V4 accuracy:            {summary['v4_accuracy']:.4%}")
    print(f"Visual-First accuracy:  {summary['visual_accuracy']:.4%}")
    print(f"Both correct:           {summary['both_correct']}")
    print(f"V4-only correct:        {summary['v4_only_correct']}")
    print(f"Visual-only correct:    {summary['visual_only_correct']}")
    print(f"Both wrong:             {summary['both_wrong']}")
    print(f"Oracle gate accuracy:   {summary['oracle_gate_accuracy']:.4%}")
    print(f"Gate positive samples:  {summary['gate_positive_count']}")
    print(f"Saved: {out_csv}")

    return df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lectures = [args.only] if args.only else TRAIN_LECTURES

    all_frames = []
    summaries = []
    for lecture in lectures:
        try:
            df, summary = process(root, lecture)
            all_frames.append(df.assign(lecture=lecture))
            summaries.append(summary)
        except Exception as e:
            print(f"\n[{lecture}] FAILED: {type(e).__name__}: {e}")

    out_root = root / "runs" / "v5_gate_dataset_existing_v4"
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(out_root / "gate_dataset.csv", index=False)
        (out_root / "summary.json").write_text(
            json.dumps({
                "lectures_completed": [x["lecture"] for x in summaries],
                "total_rows": int(len(combined)),
                "total_gate_positive": int(combined["gate_target"].sum()),
            }, indent=2),
            encoding="utf-8",
        )
        print("\nCOMBINED DATASET")
        print(f"Rows: {len(combined)}")
        print(f"Gate-positive rows: {int(combined['gate_target'].sum())}")
        print(f"Saved: {out_root / 'gate_dataset.csv'}")
    else:
        print("\nNo lecture completed.")

if __name__ == "__main__":
    main()
