
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

VALS = {
    "physics_intro_02": {
        "v4": "runs/v5_v4_export_fixed/physics_intro_02/v4_segment_predictions.csv",
        "gt": "data/MaViLS/data/ground_truth_files/ground_truth_physics.xlsx",
    },
    "psychology_MIT": {
        "v4": "runs/v5_v4_export_fixed/psychology_MIT/v4_segment_predictions.csv",
        "gt": "data/MaViLS/data/ground_truth_files/ground_truth_psychology.xlsx",
    },
    "reinforcement_learning_silver": {
        "v4": "runs/v5_v4_export_fixed/reinforcement_learning_silver/v4_segment_predictions.csv",
        "gt": "data/MaViLS/data/ground_truth_files/ground_truth_reinforcement_learning.xlsx",
    },
}

CACHE_PATTERNS = [
    "runs/**/val_visual_{name}.npz",
    "runs/**/val_{name}.npz",
    "runs/**/{name}.npz",
    "data/**/val_visual_{name}.npz",
]

def load_gt(path):
    df = pd.read_excel(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((cols[x] for x in ["time", "timestamp", "timestamps"] if x in cols), None)
    scol = next((cols[x] for x in ["slidenumber", "slide", "slide_number"] if x in cols), None)
    if tcol is None or scol is None:
        raise ValueError(f"GT columns not recognized: {list(df.columns)}")
    out = pd.DataFrame({"time": pd.to_numeric(df[tcol], errors="coerce"),
                        "gt": pd.to_numeric(df[scol], errors="coerce")})
    out = out.dropna()
    out = out[out["gt"] > 0].sort_values("time").drop_duplicates("time")
    return out.reset_index(drop=True)

def find_cache(root, name):
    found = []
    for pat in CACHE_PATTERNS:
        found.extend(root.glob(pat.format(name=name)))
    # de-duplicate
    seen, out = set(), []
    for p in found:
        if p.is_file() and str(p.resolve()) not in seen:
            seen.add(str(p.resolve())); out.append(p)
    return out

def choose_key(keys, words):
    low = {k.lower(): k for k in keys}
    for k in keys:
        lk = k.lower()
        if all(w in lk for w in words):
            return k
    return None

def inspect_cache(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    print(f"Visual cache: {path}")
    print("Cache keys:", keys)
    print("Interpretation: score matrix argmax is used when explicit visual predictions are absent.")
    return z

def extract_visual(z):
    keys = list(z.keys())
    tkey = choose_key(keys, ["timestamps"]) or choose_key(keys, ["timestamp"]) or choose_key(keys, ["time"])
    pkey = None
    for words in [["pred"], ["slide", "pred"], ["prediction"]]:
        pkey = choose_key(keys, words)
        if pkey: break
    if tkey is None or pkey is None:
        # temporal_v3_cache stores transcript x slide score matrices.
        # Use argmax over slides as the Visual-First/base cache prediction.
        skey = choose_key(keys, ["score"])
        if skey is None:
            skey = choose_key(keys, ["similarity"])
        if tkey is not None and skey is not None:
            scores = np.asarray(z[skey])
            if scores.ndim == 2:
                preds = np.argmax(scores, axis=1) + 1
                return np.asarray(z[tkey]).reshape(-1), preds, f"argmax({skey})"
        raise ValueError(
            "Could not identify visual timestamp/prediction arrays. "
            f"Keys are: {keys}"
        )
    return np.asarray(z[tkey]).reshape(-1), np.asarray(z[pkey]).reshape(-1), pkey

def normalize_slide(x):
    x = np.asarray(x).astype(float)
    # Handle 0-based visual predictions.
    finite = x[np.isfinite(x)]
    if len(finite) and np.nanmin(finite) == 0:
        x = x + 1
    return x.astype(int)

def nearest_preds(query_times, pred_times, preds, tolerance=2.0):
    order = np.argsort(pred_times)
    pt = np.asarray(pred_times)[order]
    pp = np.asarray(preds)[order]
    idx = np.searchsorted(pt, query_times)
    idx0 = np.clip(idx - 1, 0, len(pt)-1)
    idx1 = np.clip(idx, 0, len(pt)-1)
    d0 = np.abs(query_times - pt[idx0])
    d1 = np.abs(query_times - pt[idx1])
    choose = np.where(d0 <= d1, idx0, idx1)
    dist = np.minimum(d0, d1)
    out = pp[choose].astype(int)
    out[dist > tolerance] = -1
    return out, dist

def load_v4(path):
    df = pd.read_csv(path)
    def col(*names):
        for n in names:
            if n in df.columns: return n
        return None
    s = col("start", "start_time", "segment_start")
    e = col("end", "end_time", "segment_end")
    p = col("v4_pred", "pred", "slide_pred")
    if not all([s,e,p]):
        raise ValueError(f"V4 columns not recognized: {list(df.columns)}")
    out = pd.DataFrame({
        "time": pd.to_numeric(df[s], errors="coerce"),
        "v4": pd.to_numeric(df[p], errors="coerce")
    }).dropna()
    return out

def analyze_one(root, name, spec, tolerance):
    v4_path = root / spec["v4"]
    gt_path = root / spec["gt"]
    if not v4_path.exists():
        return {"lecture": name, "status": "missing V4 CSV"}
    caches = find_cache(root, name)
    if not caches:
        return {"lecture": name, "status": "missing Visual-First cache",
                "hint": "Run the existing Visual-First validation once; this script does not rerun it."}
    cache = caches[0]
    z = inspect_cache(cache)
    vt, vp, pkey = extract_visual(z)
    vt = np.asarray(vt, dtype=float)
    vp = normalize_slide(vp)

    v4 = load_v4(v4_path)
    gt = load_gt(gt_path)

    # Evaluate V4 at GT timestamps using the nearest V4 segment.
    v4p, v4d = nearest_preds(gt.time.to_numpy(), v4.time.to_numpy(), v4.v4.to_numpy(), tolerance=10.0)
    visp, visd = nearest_preds(gt.time.to_numpy(), vt, vp, tolerance=tolerance)

    df = gt.copy()
    df["v4"] = v4p
    df["visual"] = visp
    df["v4_correct"] = df.v4 == df.gt
    df["visual_correct"] = df.visual == df.gt
    df["v4_dist"] = v4d
    df["visual_dist"] = visd
    df["usable"] = (df.v4 >= 1) & (df.visual >= 1)

    u = df[df.usable].copy()
    both_correct = int((u.v4_correct & u.visual_correct).sum())
    v4_only = int((u.v4_correct & ~u.visual_correct).sum())
    visual_only = int((~u.v4_correct & u.visual_correct).sum())
    both_wrong = int((~u.v4_correct & ~u.visual_correct).sum())

    # Useful gate-oracle upper bound: choose the correct branch whenever either is correct.
    oracle = int((u.v4_correct | u.visual_correct).sum())

    outdir = root / "runs/v5_gate_analysis_actual" / name
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "timestamp_comparison.csv", index=False)

    summary = {
        "lecture": name,
        "cache": str(cache),
        "visual_prediction_key": pkey,
        "gt_points": len(gt),
        "usable_points": len(u),
        "v4_accuracy": float(u.v4_correct.mean()) if len(u) else None,
        "visual_accuracy": float(u.visual_correct.mean()) if len(u) else None,
        "both_correct": both_correct,
        "v4_only_correct": v4_only,
        "visual_only_correct": visual_only,
        "both_wrong": both_wrong,
        "oracle_gate_accuracy": oracle / len(u) if len(u) else None,
        "potential_recoverable_v4_errors": visual_only,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    ap.add_argument("--visual-tolerance", type=float, default=2.0)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    names = [args.only] if args.only else list(VALS)

    results = []
    for name in names:
        print("\n" + "="*80)
        print("V5 ACTUAL GATE ANALYSIS:", name)
        print("="*80)
        try:
            r = analyze_one(root, name, VALS[name], args.visual_tolerance)
        except Exception as e:
            r = {"lecture": name, "status": "ERROR", "error": repr(e)}
        results.append(r)
        print(json.dumps(r, indent=2))

    ok = [r for r in results if "v4_accuracy" in r]
    if ok:
        n = sum(r["usable_points"] for r in ok)
        agg_v4 = sum(r["v4_accuracy"] * r["usable_points"] for r in ok) / n
        agg_visual = sum(r["visual_accuracy"] * r["usable_points"] for r in ok) / n
        agg_oracle = sum(r["oracle_gate_accuracy"] * r["usable_points"] for r in ok) / n
        print("\n" + "="*80)
        print("ACTUAL V5 GATE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Usable points: {n}")
        print(f"V4 accuracy: {agg_v4:.4%}")
        print(f"Visual accuracy: {agg_visual:.4%}")
        print(f"Oracle gate accuracy: {agg_oracle:.4%}")
        print("Oracle gate is an analysis upper bound, NOT a deployable result.")
        print("It means: if we knew which branch was correct, how much could be recovered.")
        print("\nNext step: use V4 margin/confidence + visual agreement/change features to learn a gate on validation only.")
    else:
        print("\nNo complete comparison was produced. If cache keys are shown above, send that output and the loader can be adapted.")

if __name__ == "__main__":
    main()
