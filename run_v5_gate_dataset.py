import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

# Known MaViLS naming aliases. The matcher also falls back to normalized
# filename similarity, so this list is deliberately conservative.
ALIASES = {
    "ML_for_health_MIT": ["ML_for_health_MIT", "ML_for_health"],
    "deeplearning": ["deeplearning_goodfellow", "deeplearning"],
    "image_processing": ["image_processing_high_res", "image_processing"],
    "computer_vision_2_2": ["computer_vision_2_2"],
    "creating_breakthrough_products_MIT": [
        "creating_breakthrough_products_MIT",
        "creating_breakthrough_products",
    ],
    "climate_science_policy_MIT2": [
        "climate_science_policy_MIT2",
        "climate_science_policy",
    ],
    "cognitive_robotics_MIT": ["cognitive_robotics_MIT", "cognitive_robotics"],
    "cryptocurrency_MIT": ["cryptocurrency_MIT", "cryptocurrency"],
}

VAL_LECTURES = [
    "physics_intro_02",
    "psychology_MIT",
    "reinforcement_learning_silver",
]

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def find_asset(folder, suffixes, aliases):
    candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    if not candidates:
        return None

    n_aliases = [norm(x) for x in aliases]
    # exact normalized stem
    for p in candidates:
        if norm(p.stem) in n_aliases:
            return p

    # alias contained in stem
    for a in n_aliases:
        for p in candidates:
            if a and (a in norm(p.stem) or norm(p.stem) in a):
                return p

    # token overlap fallback
    target = set(re.findall(r"[a-z0-9]+", aliases[0].lower()))
    best, best_score = None, 0
    for p in candidates:
        toks = set(re.findall(r"[a-z0-9]+", p.stem.lower()))
        score = len(target & toks)
        if score > best_score:
            best, best_score = p, score
    return best

def discover(root, lecture):
    srt_dir = root / "data" / "MaViLS" / "data" / "audioscripts"
    pdf_dir = root / "data" / "MaViLS" / "data" / "lectures"
    gt_dir = root / "data" / "MaViLS" / "data" / "ground_truth_files"
    video_dir = Path(r"C:\Users\Bharat\Downloads\archive\video")

    aliases = [lecture] + ALIASES.get(lecture, [])
    srt = find_asset(srt_dir, {".srt"}, aliases)
    pdf = find_asset(pdf_dir, {".pdf"}, aliases)

    gt_aliases = [lecture] + ALIASES.get(lecture, [])
    gt = find_asset(gt_dir, {".xlsx", ".xls"}, ["ground_truth_" + x for x in gt_aliases])

    video = find_asset(video_dir, {".mp4", ".mkv", ".avi"}, aliases)

    return srt, pdf, gt, video

def run_v4_export(root, lecture, srt, pdf, video, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v4_segment_predictions.csv"
    if csv_path.exists():
        return csv_path

    script = root / "run_v5_v4_export_fixed.py"
    if not script.exists():
        raise FileNotFoundError("run_v5_v4_export_fixed.py not found in project root.")

    # The existing fixed exporter is intentionally reused so V5 sees the
    # exact same V4 segmentation/decoder implementation as the verified V4 run.
    # It has explicit validation mappings, so arbitrary training lectures may
    # require a small mapping addition if the exporter refuses the lecture.
    cmd = [
        sys.executable, str(script),
        "--root", str(root),
        "--only", lecture,
    ]
    print("\n" + "=" * 80)
    print("V4 PREPARATION:", lecture)
    print("=" * 80)
    print("SRT :", srt)
    print("PDF :", pdf)
    print("VIDEO:", video)
    p = subprocess.run(cmd)
    if p.returncode != 0 or not csv_path.exists():
        return None
    return csv_path

def load_gt(gt_path):
    df = pd.read_excel(gt_path)
    time_col = next((c for c in df.columns if str(c).lower() == "time"), df.columns[0])
    slide_col = next((c for c in df.columns if "slide" in str(c).lower()), df.columns[1])
    out = pd.DataFrame({
        "timestamp": pd.to_numeric(df[time_col], errors="coerce"),
        "gt_slide": pd.to_numeric(df[slide_col], errors="coerce"),
    }).dropna()
    out = out[out["gt_slide"] > 0].copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    return out.reset_index(drop=True)

def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(e.sum(), 1e-12)

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def add_features(comp):
    # Conservative, model-independent gate features.
    v4_score = comp["v4_score"].to_numpy(float)
    v4_margin = comp["v4_margin"].to_numpy(float)
    visual_score = comp["visual_score"].to_numpy(float)
    visual_margin = comp["visual_margin"].to_numpy(float)

    comp["score_gap_visual_v4"] = visual_score - v4_score
    comp["margin_gap_visual_v4"] = visual_margin - v4_margin
    comp["prediction_agree"] = (comp["v4_pred"] == comp["visual_pred"]).astype(int)
    comp["prediction_distance"] = np.abs(comp["v4_pred"] - comp["visual_pred"])

    # Temporal stability around each row.
    vp = comp["visual_pred"].to_numpy(int)
    tp = comp["v4_pred"].to_numpy(int)
    for name, arr in [("v4_pred", tp), ("visual_pred", vp)]:
        prev = np.r_[arr[0], arr[:-1]]
        nxt = np.r_[arr[1:], arr[-1]]
        comp[name + "_stable"] = ((arr == prev) & (arr == nxt)).astype(int)

    # Gate target:
    # 1 only when Visual is correct and V4 is wrong.
    # 0 otherwise. This makes the gate a correction detector rather than
    # a generic classifier that learns to prefer Visual.
    comp["gate_target"] = (
        (comp["visual_correct"] == 1) &
        (comp["v4_correct"] == 0)
    ).astype(int)

    return comp

def compare_with_visual(root, lecture, v4_csv, gt_path, video, pdf, out_dir):
    # Reuse the exact actual Visual-First implementation used by the
    # previously verified V5 real-gate experiment.
    sys.path.insert(0, str(root))
    from src.visual_first_alignment import run_visual_first, viterbi_monotonic

    gt = load_gt(gt_path)
    v4 = pd.read_csv(v4_csv)

    def col(cands, required=True):
        for c in v4.columns:
            n = norm(c)
            if n in [norm(x) for x in cands]:
                return c
        if required:
            raise KeyError(f"Missing V4 column: {cands}; columns={list(v4.columns)}")
        return None

    start_c = col(["start", "start_time", "timestamp_start"])
    end_c = col(["end", "end_time", "timestamp_end"])
    pred_c = col(["v4_pred", "pred", "slide"])
    score_c = col(["v4_score", "score"])
    margin_c = col(["v4_margin", "margin"])
    boundary_c = col(["boundary_probability", "boundary_prob"], False)

    timeline, scores, times, duration = run_visual_first(
        video,
        pdf,
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )
    path = viterbi_monotonic(scores, max_jump=2, jump_penalty=0.05)

    def nearest_visual(t):
        i = int(np.argmin(np.abs(times - t)))
        row = np.asarray(scores[i], dtype=float)
        order = np.argsort(row)[::-1]
        best = int(path[i]) + 1
        top = float(row[order[0]])
        second = float(row[order[1]]) if len(order) > 1 else top
        return best, float(times[i]), top, top - second

    v4 = v4.sort_values(start_c).reset_index(drop=True)

    def v4_at(t):
        hit = v4[(v4[start_c].astype(float) <= t) & (v4[end_c].astype(float) >= t)]
        if len(hit):
            r = hit.iloc[0]
        else:
            j = int(np.argmin(np.abs(
                ((v4[start_c].astype(float) + v4[end_c].astype(float)) / 2).to_numpy() - t
            )))
            r = v4.iloc[j]
        return (
            int(r[pred_c]),
            float(r[score_c]),
            float(r[margin_c]),
            float(r[boundary_c]) if boundary_c else 0.0,
        )

    rows = []
    for _, g in gt.iterrows():
        t = float(g["timestamp"])
        gt_slide = int(g["gt_slide"])
        vp, vt, vs, vm = nearest_visual(t)
        ap, a_score, a_margin, boundary = v4_at(t)
        rows.append({
            "timestamp": t,
            "gt_slide": gt_slide,
            "v4_pred": ap,
            "visual_pred": vp,
            "v4_score": a_score,
            "v4_margin": a_margin,
            "boundary_probability": boundary,
            "visual_score": vs,
            "visual_margin": vm,
            "v4_correct": int(ap == gt_slide),
            "visual_correct": int(vp == gt_slide),
            "visual_frame_time": vt,
        })

    comp = add_features(pd.DataFrame(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "timestamp_comparison.csv", index=False)

    return comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    lectures = [args.only] if args.only else TRAIN_LECTURES
    base = root / "runs" / "v5_gate_dataset"
    base.mkdir(parents=True, exist_ok=True)

    all_parts = []
    manifest = []

    for lecture in lectures:
        srt, pdf, gt, video = discover(root, lecture)
        status = {"lecture": lecture, "srt": str(srt) if srt else None,
                  "pdf": str(pdf) if pdf else None,
                  "gt": str(gt) if gt else None,
                  "video": str(video) if video else None}

        if not all([srt, pdf, gt, video]):
            status["status"] = "SKIPPED_MISSING_ASSET"
            print("\nSKIP", lecture, status)
            manifest.append(status)
            continue

        v4_dir = root / "runs" / "v5_gate_v4_exports" / lecture
        v4_csv = run_v4_export(root, lecture, srt, pdf, video, v4_dir)
        if not v4_csv:
            status["status"] = "SKIPPED_V4_EXPORT_FAILED"
            manifest.append(status)
            continue

        try:
            comp = compare_with_visual(root, lecture, v4_csv, gt, video, pdf, base / lecture)
            comp["lecture"] = lecture
            all_parts.append(comp)
            status["status"] = "OK"
            status["rows"] = int(len(comp))
            status["gate_positive"] = int(comp["gate_target"].sum())
        except Exception as e:
            status["status"] = "FAILED"
            status["error"] = repr(e)

        manifest.append(status)

    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        # Keep only rows with valid numerical gate inputs.
        feature_cols = [
            "v4_score", "v4_margin", "boundary_probability",
            "visual_score", "visual_margin",
            "score_gap_visual_v4", "margin_gap_visual_v4",
            "prediction_agree", "prediction_distance",
            "v4_pred_stable", "visual_pred_stable",
        ]
        combined = combined.dropna(subset=feature_cols + ["gate_target"])
        combined.to_csv(base / "gate_dataset.csv", index=False)

        pos = int(combined["gate_target"].sum())
        print("\n" + "=" * 80)
        print("V5 GATE DATASET")
        print("=" * 80)
        print("Rows:", len(combined))
        print("Positive correction examples:", pos)
        print("Negative examples:", len(combined) - pos)
        print("Lectures:", combined["lecture"].nunique())
        print("Saved:", base / "gate_dataset.csv")

    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest:", base / "manifest.json")

if __name__ == "__main__":
    main()
