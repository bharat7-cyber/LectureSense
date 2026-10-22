
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN = [
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

# Known MaViLS naming aliases observed in the project.
SRT_ALIASES = {
    "climate_science_policy_MIT2": ["climate_science_policy_MIT2.srt", "climate_science_policy_MIT.srt"],
    "cryptocurrency_MIT": ["cryptocurrency.srt"],
    "deeplearning": ["deeplearning_goodfellow.srt"],
}
PDF_ALIASES = {
    "computer_vision_2_2": ["computer_vision_2_2_geiger.pdf"],
    "creating_breakthrough_products_MIT": ["creating_breakthrough_products.pdf"],
    "ML_for_health_MIT": ["ML_for_health_care_MIT.pdf"],
    "deeplearning": ["deeplearning_goodfellow.pdf"],
    "cities_and_decarbonization": ["cities&decarbonization.pdf"],
}
VIDEO_ALIASES = {
    "computer_vision_2_2": ["computer_vision_2_2_high_res.mp4"],
    "creating_breakthrough_products_MIT": ["creating_breakthrough_products_MIT.mp4"],
    "cryptocurrency_MIT": ["cryptocurrency_high_res.mp4"],
    "deeplearning": ["deep_learning_high_res.mp4"],
    "image_processing": ["image_processing_high_res.mp4"],
    "ML_for_health_MIT": ["ML_for_health_high_res.mp4"],
    "numerics": ["numerics_high_res.mp4"],
    "phonetics": ["phonetics_high_res.mp4"],
    "cognitive_robotics_MIT": ["cognitive_robotics_high_res.mp4"],
    "reinforcement_learning_silver": ["reinforcement_learning_high_res.mp4"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def find_srt(root, name):
    d = root / "data" / "MaViLS" / "data" / "audioscripts"
    exact = first_existing([d / x for x in SRT_ALIASES.get(name, [])])
    if exact:
        return exact
    candidates = list(d.glob("*.srt"))
    n = norm(name)
    scored = []
    for p in candidates:
        q = norm(p.stem)
        score = 0
        if q == n: score = 1000
        elif n in q or q in n: score = 800 - abs(len(q)-len(n))
        scored.append((score, p))
    return max(scored)[1] if scored and max(scored)[0] >= 500 else None

def find_pdf(root, name):
    d = root / "data" / "MaViLS" / "data" / "lectures"
    exact = first_existing([d / x for x in PDF_ALIASES.get(name, [])])
    if exact:
        return exact
    candidates = list(d.glob("*.pdf"))
    n = norm(name)
    scored = []
    for p in candidates:
        q = norm(p.stem)
        score = 0
        if q == n: score = 1000
        elif n in q or q in n: score = 800 - abs(len(q)-len(n))
        # Special semantic aliases
        if name == "ML_for_health_MIT" and "mlforhealth" in q: score = max(score, 900)
        if name == "creating_breakthrough_products_MIT" and "creatingbreakthroughproducts" in q: score = max(score, 900)
        if name == "computer_vision_2_2" and "computervision22" in q: score = max(score, 900)
        if name == "deeplearning" and "deeplearning" in q: score = max(score, 900)
        scored.append((score, p))
    return max(scored)[1] if scored and max(scored)[0] >= 500 else None

def find_gt(root, name):
    d = root / "data" / "MaViLS" / "data" / "ground_truth_files"
    candidates = list(d.glob("*.xlsx"))
    n = norm(name)
    scored = []
    for p in candidates:
        q = norm(p.stem)
        score = 0
        if n in q or q in n: score = 900 - abs(len(q)-len(n))
        if name == "climate_science_policy_MIT2" and "climatesciencepolicy" in q: score = 900
        if name == "computer_vision_2_2" and "computervision" in q: score = 900
        if name == "creating_breakthrough_products_MIT" and "creatingbreakthroughproducts" in q: score = 900
        if name == "ML_for_health_MIT" and "mlforhealth" in q: score = 900
        scored.append((score, p))
    return max(scored)[1] if scored and max(scored)[0] >= 500 else None

def find_video(root, name):
    d = Path(r"C:\Users\Bharat\Downloads\archive\video")
    if not d.exists():
        d = root / "archive" / "video"
    exact = first_existing([d / x for x in VIDEO_ALIASES.get(name, [])])
    if exact:
        return exact

    # Search by semantic token overlap. Require at least one strong token.
    aliases = {
        "cities_and_decarbonization": ["cities", "decarbonization"],
        "climate_and_cities": ["climate", "cities"],
        "climate_science_policy_MIT2": ["climate", "science", "policy"],
        "cognitive_robotics_MIT": ["cognitive", "robotics"],
        "computer_vision_2_2": ["computer", "vision"],
        "creating_breakthrough_products_MIT": ["creating", "breakthrough", "products"],
        "cryptocurrency_MIT": ["cryptocurrency"],
        "deeplearning": ["deep", "learning"],
        "image_processing": ["image", "processing"],
        "ML_for_health_MIT": ["ml", "health"],
        "numerics": ["numerics"],
        "phonetics": ["phonetics"],
    }
    toks = aliases.get(name, [name])
    best = []
    for p in d.glob("*.mp4"):
        q = p.stem.lower()
        hits = sum(t in q for t in toks)
        if hits == len(toks):
            score = 1000
        elif hits >= 1:
            score = hits * 100 - abs(len(q) - len(name))
        else:
            score = 0
        best.append((score, p))
    best.sort(reverse=True)
    return best[0][1] if best and best[0][0] >= 100 else None

def v4_dir(root, name):
    candidates = [
        root / "runs" / "v5_gate_dataset_fixed" / "v4" / name,
        root / "runs" / "v5_gate_dataset_from_existing_v4" / name,
        root / "runs" / "v5_v4_export_fixed" / name,
    ]
    for d in candidates:
        if (d / "v4_final_score_matrix.npy").exists():
            return d
    return None

def ensure_v4(root, name, srt, pdf, video):
    existing = v4_dir(root, name)
    if existing:
        print(f"[V4] Reusing existing matrices: {existing}")
        return existing

    out = root / "runs" / "v5_gate_train_v4" / name
    out.mkdir(parents=True, exist_ok=True)
    timeline = out / "timeline.json"
    render = out / "rendered_slides"
    checkpoint = root / "checkpoints" / "temporal_v4_best_model.pt"

    cmd = [
        sys.executable,
        str(root / "src" / "inference_temporal_v4.py"),
        "--transcript", str(srt),
        "--slides", str(pdf),
        "--video", str(video),
        "--checkpoint", str(checkpoint),
        "--out", str(timeline),
        "--render-dir", str(render),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", "0.4",
        "--video-sample-seconds", "1.0",
    ]
    print("\n[V4] Running:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, cwd=str(root), check=True)

    # The V4 inference writes these matrices beside/under the run directory
    required = [
        "v4_final_score_matrix.npy",
        "v4_boundary_probability.npy",
        "v4_next_slide_logits.npy",
        "timeline.json",
    ]
    missing = [x for x in required if not (out / x).exists()]
    if missing:
        # Some versions write matrices to the standard gate directory.
        alt = root / "runs" / "v5_gate_dataset_fixed" / "v4" / name
        if all((alt / x).exists() for x in required):
            return alt
        raise RuntimeError(f"V4 finished but required outputs are missing for {name}: {missing}")
    return out

def transcript_segments(root, srt):
    src = root / "src"
    sys.path.insert(0, str(src))
    from inference import read_mavils_srt
    segs = read_mavils_srt(str(srt))
    return segs

def build_v4_rows(root, name, srt, v4):
    final = np.load(v4 / "v4_final_score_matrix.npy")
    boundary = np.load(v4 / "v4_boundary_probability.npy")
    nxt = np.load(v4 / "v4_next_slide_logits.npy")
    segs = transcript_segments(root, srt)

    n = min(len(segs), final.shape[0])
    if n == 0:
        raise RuntimeError(f"No V4 rows for {name}")

    rows = []
    for i in range(n):
        start = float(segs[i].start)
        end = float(segs[i].end)
        scores = np.asarray(final[i])
        order = np.argsort(scores)[::-1]
        pred = int(order[0]) + 1
        second = float(scores[order[1]]) if len(order) > 1 else float(scores[order[0]])
        margin = float(scores[order[0]] - second)
        b = float(np.asarray(boundary).reshape(-1)[i]) if i < len(boundary.reshape(-1)) else 0.0
        ns = int(np.argmax(nxt[i])) + 1 if i < len(nxt) else pred
        rows.append({
            "segment_index": i,
            "start": start,
            "end": end,
            "v4_pred": pred,
            "v4_score": float(scores[order[0]]),
            "v4_margin": margin,
            "boundary_probability": b,
            "next_slide_pred": ns,
        })
    out = root / "runs" / "v5_train_gate_dataset" / name
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "v4_segment_predictions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv

def run_visual(root, name, srt, pdf, video, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.npz"
    if cache.exists():
        print(f"[VISUAL] Reusing cache: {cache}")
        return cache

    sys.path.insert(0, str(root / "src"))
    from visual_first_alignment import run_visual_first, viterbi_monotonic

    print(f"[VISUAL] Running actual Visual-First: {name}")
    timeline, scores, times, duration = run_visual_first(
        video_path=video,
        pdf_path=pdf,
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )
    path, _ = viterbi_monotonic(scores, max_jump=2, jump_penalty=0.05)
    np.savez_compressed(cache, times=times, scores=scores, path=path, duration=np.array([duration]))
    return cache

def load_gt(path):
    df = pd.read_excel(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((cols[k] for k in ["time", "timestamp", "start"] if k in cols), None)
    scol = next((cols[k] for k in ["slidenumber", "slide_number", "slide", "slideid"] if k in cols), None)
    if tcol is None or scol is None:
        raise ValueError(f"GT columns not found: {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        try:
            t = float(r[tcol]); s = int(r[scol])
        except Exception:
            continue
        if s > 0:
            out.append((t, s))
    return out

def make_comparison(root, name, gt_path, v4_csv, visual_cache):
    v4 = pd.read_csv(v4_csv)
    z = np.load(visual_cache)
    vt = z["times"]
    vp = z["path"].astype(int) + 1

    gt = load_gt(gt_path)
    rows = []
    for t, g in gt:
        j = int(np.argmin(np.abs(vt - t)))
        visual_pred = int(vp[j])

        k = int(np.searchsorted(v4["start"].to_numpy(), t, side="right") - 1)
        k = max(0, min(k, len(v4)-1))
        r = v4.iloc[k]
        v4_pred = int(r.v4_pred)

        v4_correct = int(v4_pred == g)
        visual_correct = int(visual_pred == g)
        rows.append({
            "timestamp": t,
            "gt": g,
            "v4_pred": v4_pred,
            "visual_pred": visual_pred,
            "v4_correct": v4_correct,
            "visual_correct": visual_correct,
            "v4_stable": float(r.v4_margin),
            "v4_boundary_proxy": float(r.boundary_probability),
            "visual_stable": float(abs(float(z["scores"][j, visual_pred-1]) - np.partition(z["scores"][j], -2)[-2])) if z["scores"].shape[1] > 1 else 0.0,
            "prediction_agree": int(v4_pred == visual_pred),
            "prediction_distance": abs(v4_pred - visual_pred),
            "gate_target": int(visual_correct and not v4_correct),
        })

    out = root / "runs" / "v5_train_gate_dataset" / name
    df = pd.DataFrame(rows)
    df.to_csv(out / "timestamp_comparison.csv", index=False)
    summary = {
        "lecture": name,
        "gt_points": len(df),
        "v4_accuracy": float(df.v4_correct.mean()) if len(df) else 0,
        "visual_accuracy": float(df.visual_correct.mean()) if len(df) else 0,
        "both_correct": int(((df.v4_correct == 1) & (df.visual_correct == 1)).sum()),
        "v4_only_correct": int(((df.v4_correct == 1) & (df.visual_correct == 0)).sum()),
        "visual_only_correct": int(((df.v4_correct == 0) & (df.visual_correct == 1)).sum()),
        "both_wrong": int(((df.v4_correct == 0) & (df.visual_correct == 0)).sum()),
        "oracle_gate_accuracy": float(((df.v4_correct == 1) | (df.visual_correct == 1)).mean()) if len(df) else 0,
        "gate_positive_count": int(df.gate_target.sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reuse-visual", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    names = args.only if args.only else TRAIN
    cache_dir = root / "runs" / "v5_train_gate_dataset" / "_visual_cache"

    all_df = []
    summaries = []

    for idx, name in enumerate(names, 1):
        print("\n" + "="*78)
        print(f"[{idx}/{len(names)}] {name}")
        print("="*78)

        srt = find_srt(root, name)
        pdf = find_pdf(root, name)
        gt = find_gt(root, name)
        video = find_video(root, name)

        print("SRT  :", srt)
        print("PDF  :", pdf)
        print("GT   :", gt)
        print("VIDEO:", video)

        missing = [k for k,v in {"SRT":srt,"PDF":pdf,"GT":gt,"VIDEO":video}.items() if v is None]
        if missing:
            print(f"[SKIP] Missing assets: {missing}")
            continue

        try:
            v4 = ensure_v4(root, name, srt, pdf, video)
            v4csv = build_v4_rows(root, name, srt, v4)
            vcache = run_visual(root, name, srt, pdf, video, cache_dir)
            df, summary = make_comparison(root, name, gt, v4csv, vcache)
            summaries.append(summary)
            all_df.append(df.assign(lecture=name))
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"[FAILED] {name}: {type(e).__name__}: {e}")
            print("Continue to next lecture.")
            continue

    base = root / "runs" / "v5_train_gate_dataset"
    base.mkdir(parents=True, exist_ok=True)
    if summaries:
        (base / "train_summary.json").write_text(json.dumps(summaries, indent=2))
    if all_df:
        pd.concat(all_df, ignore_index=True).to_csv(base / "train_timestamp_comparison_all.csv", index=False)

    print("\n" + "="*78)
    print("V5 TRAIN GATE DATASET BUILD COMPLETE")
    print("="*78)
    print(f"Completed lectures: {len(summaries)}/{len(names)}")
    print(f"Output: {base}")
    print("Validation/test lectures were not used.")

if __name__ == "__main__":
    main()
