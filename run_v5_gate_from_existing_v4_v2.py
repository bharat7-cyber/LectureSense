
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visual_first_alignment import load_ground_truth, run_visual_first, viterbi_monotonic

TRAIN_LECTURES = [
    "cities_and_decarbonization", "climate_and_cities",
    "climate_science_policy_MIT2", "cognitive_robotics_MIT",
    "computer_vision_2_2", "creating_breakthrough_products_MIT",
    "cryptocurrency_MIT", "deeplearning", "image_processing",
    "ML_for_health_MIT", "numerics", "phonetics",
]

VIDEO_ALIASES = {
    "computer_vision_2_2": "computer_vision_2_2_high_res.mp4",
}

def locate(folder, stem, ext):
    p = folder / f"{stem}{ext}"
    if p.exists():
        return p
    xs = list(folder.glob(f"*{stem}*{ext}"))
    return xs[0] if xs else None

def assets(root, lecture):
    d = root / "data" / "MaViLS" / "data"
    srt = locate(d / "audioscripts", lecture, ".srt")
    pdf = locate(d / "lectures", lecture, ".pdf")
    gt = locate(d / "ground_truth_files", lecture, ".xlsx")
    vd = Path(r"C:\Users\Bharat\Downloads\archive\video")
    video = vd / VIDEO_ALIASES[lecture] if lecture in VIDEO_ALIASES and (vd / VIDEO_ALIASES[lecture]).exists() else None
    if video is None:
        key = lecture.lower().replace("_", "")
        for p in vd.glob("*.mp4"):
            if key in p.stem.lower().replace("_", ""):
                video = p
                break
    missing = [n for n,p in [("SRT",srt),("PDF",pdf),("GT",gt),("VIDEO",video)] if p is None]
    if missing:
        raise FileNotFoundError(f"{lecture}: missing {', '.join(missing)}")
    return srt, pdf, gt, video

def extract_timeline(obj):
    """Accept the known V4 timeline JSON variants without changing its data."""
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("timeline", "segments", "entries", "predictions", "items", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return value

        # Some exporters may store one segment per numeric key.
        numeric_values = [v for v in obj.values() if isinstance(v, dict)]
        if numeric_values and all(
            any(k in v for k in ("start", "start_time", "end", "end_time", "slide", "slide_id"))
            for v in numeric_values
        ):
            return numeric_values

    raise ValueError(
        "Unsupported timeline.json structure. "
        f"Top-level type={type(obj).__name__}; "
        f"keys={list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
    )

def key_of(d, candidates):
    low = {str(k).lower(): k for k in d}
    for c in candidates:
        if c in low:
            return low[c]
    return None

def make_v4_csv(v4dir):
    out = v4dir / "v4_segment_predictions.csv"
    if out.exists():
        return out

    timeline_path = v4dir / "timeline.json"
    score_path = v4dir / "v4_final_score_matrix.npy"
    boundary_path = v4dir / "v4_boundary_probability.npy"
    next_path = v4dir / "v4_next_slide_logits.npy"

    obj = json.loads(timeline_path.read_text(encoding="utf-8"))
    tl = extract_timeline(obj)

    if not tl:
        raise ValueError("timeline.json contains no timeline entries")

    sk = key_of(tl[0], ["start","start_time","timestamp_start"])
    ek = key_of(tl[0], ["end","end_time","timestamp_end"])
    pk = key_of(tl[0], ["slide","slide_id","slide_number","slidenumber"])
    if not all([sk,ek,pk]):
        raise ValueError(f"Cannot identify start/end/slide keys. First entry: {tl[0]}")

    score = np.load(score_path)
    boundary = np.load(boundary_path)
    nxt = np.load(next_path)

    n = len(tl)
    if score.shape[0] != n:
        raise ValueError(f"Timeline entries={n}, score rows={score.shape[0]}")

    rows=[]
    for i,e in enumerate(tl):
        slide=int(e[pk])
        r=np.asarray(score[i]).reshape(-1)
        if 1 <= slide <= len(r):
            s=float(r[slide-1])
            others=np.delete(r,slide-1)
            second=float(np.max(others)) if len(others) else s
        else:
            order=np.argsort(r)[::-1]
            slide=int(order[0])+1
            s=float(r[order[0]])
            second=float(r[order[1]]) if len(order)>1 else s

        nr=np.asarray(nxt[i]).reshape(-1) if i < nxt.shape[0] else np.array([])
        next_slide=int(np.argmax(nr))+1 if len(nr) else -1

        rows.append({
            "segment_index":i,
            "start":float(e[sk]),
            "end":float(e[ek]),
            "v4_pred":slide,
            "v4_score":s,
            "v4_margin":s-second,
            "boundary_probability":float(np.asarray(boundary[i]).reshape(-1)[0]),
            "next_slide_pred":next_slide,
        })

    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Created: {out}")
    print(f"Rows: {len(rows)}")
    return out

def v4_at(v4,t):
    s=v4.start.to_numpy(); e=v4.end.to_numpy()
    x=np.flatnonzero((s<=t)&(t<e))
    if len(x):
        return int(x[0])
    return int(np.argmin(np.abs(((s+e)/2)-t)))

def visual(root,lecture,video,pdf):
    cd=root/"runs"/"v5_gate_dataset_from_existing_v4"/"_visual_cache"
    cd.mkdir(parents=True,exist_ok=True)
    cp=cd/f"{lecture}.npz"
    if cp.exists():
        z=np.load(cp)
        return z["times"],z["scores"],z["path"],True
    print(f"[{lecture}] Running actual Visual-First...")
    _,scores,times,_=run_visual_first(video,pdf,sample_every=2.0,max_jump=2,jump_penalty=0.05,progress=True)
    path=viterbi_monotonic(scores,max_jump=2,jump_penalty=0.05).astype(np.int16)
    np.savez_compressed(cp,times=np.asarray(times),scores=np.asarray(scores,dtype=np.float32),path=path)
    return np.asarray(times),np.asarray(scores),path,False

def process(root,lecture):
    vd=root/"runs"/"v5_gate_dataset_fixed"/"v4"/lecture
    if not vd.exists():
        raise FileNotFoundError(f"V4 directory not found: {vd}")
    v4=pd.read_csv(make_v4_csv(vd))
    _,pdf,gtp,video=assets(root,lecture)
    gt=load_ground_truth(gtp)
    times,scores,path,reused=visual(root,lecture,video,pdf)

    rows=[]
    for t,g in gt:
        t=float(t); g=int(g)
        i=v4_at(v4,t); vr=v4.iloc[i]
        j=int(np.searchsorted(times,t))
        j=max(0,min(j,len(times)-1))
        if 0<j<len(times) and abs(times[j-1]-t)<=abs(times[j]-t): j-=1
        pred=int(path[j])+1
        r=scores[j]; order=np.argsort(r)[::-1]
        vs=float(r[pred-1]); second=float(r[order[1]]) if len(order)>1 else vs
        rows.append({
            "timestamp":t,"gt_slide":g,"v4_pred":int(vr.v4_pred),"visual_pred":pred,
            "visual_frame_time":float(times[j]),
            "v4_correct":int(int(vr.v4_pred)==g),"visual_correct":int(pred==g),
            "v4_score":float(vr.v4_score),"v4_margin":float(vr.v4_margin),
            "boundary_probability":float(vr.boundary_probability),
            "next_slide_pred":int(vr.next_slide_pred),
            "visual_score":vs,"visual_margin":vs-second,"visual_frame_index":j
        })
    df=pd.DataFrame(rows)
    vp=df.v4_pred.to_numpy(); xp=df.visual_pred.to_numpy()
    df["v4_stable"]=np.r_[1.,(vp[1:]==vp[:-1]).astype(float)]
    df["v4_boundary_proxy"]=np.r_[0.,(vp[1:]!=vp[:-1]).astype(float)]
    df["visual_stable"]=np.r_[1.,(xp[1:]==xp[:-1]).astype(float)]
    df["prediction_agree"]=(vp==xp).astype(float)
    df["prediction_distance"]=np.abs(vp-xp).astype(float)
    df["gate_target"]=((df.visual_correct==1)&(df.v4_correct==0)).astype(int)

    od=root/"runs"/"v5_gate_dataset_from_existing_v4"/lecture
    od.mkdir(parents=True,exist_ok=True)
    df.to_csv(od/"timestamp_comparison.csv",index=False)
    summary={
        "lecture":lecture,"gt_points":len(df),
        "v4_accuracy":float(df.v4_correct.mean()),
        "visual_accuracy":float(df.visual_correct.mean()),
        "both_correct":int(((df.v4_correct==1)&(df.visual_correct==1)).sum()),
        "v4_only_correct":int(((df.v4_correct==1)&(df.visual_correct==0)).sum()),
        "visual_only_correct":int(((df.v4_correct==0)&(df.visual_correct==1)).sum()),
        "both_wrong":int(((df.v4_correct==0)&(df.visual_correct==0)).sum()),
        "oracle_gate_accuracy":float(((df.v4_correct==1)|(df.visual_correct==1)).mean()),
        "gate_positive_count":int(df.gate_target.sum()),
        "visual_cache_reused":reused
    }
    (od/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(f"Saved: {od/'timestamp_comparison.csv'}")
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--only")
    a=ap.parse_args()
    root=Path(a.root).resolve()
    lectures=[a.only] if a.only else TRAIN_LECTURES
    frames=[]
    for l in lectures:
        try:
            d=process(root,l); d["lecture"]=l; frames.append(d)
        except Exception as e:
            print(f"\n[{l}] FAILED: {type(e).__name__}: {e}")
    if frames:
        out=root/"runs"/"v5_gate_dataset_from_existing_v4"
        pd.concat(frames,ignore_index=True).to_csv(out/"gate_dataset.csv",index=False)
        print(f"\nCombined dataset saved: {out/'gate_dataset.csv'}")
    else:
        print("\nNo lecture completed.")

if __name__=="__main__":
    main()
