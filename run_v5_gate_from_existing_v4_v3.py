
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import read_mavils_srt
from src.visual_first_alignment import load_ground_truth, run_visual_first, viterbi_monotonic

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
    video = None
    if lecture in VIDEO_ALIASES:
        p = vd / VIDEO_ALIASES[lecture]
        if p.exists():
            video = p
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

def load_timeline(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for k in ("timeline","segments","entries","items"):
            if isinstance(obj.get(k), list):
                return obj[k]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported timeline structure: {type(obj).__name__}")

def timeline_cols(tl):
    x = tl[0]
    low = {str(k).lower(): k for k in x}
    def pick(names):
        for n in names:
            if n in low:
                return low[n]
        return None
    sk = pick(["start_time","start","timestamp_start"])
    ek = pick(["end_time","end","timestamp_end"])
    pk = pick(["slide_id","slide","slide_number","slidenumber"])
    if not all([sk,ek,pk]):
        raise ValueError(f"Timeline keys not recognized: {list(x)}")
    return sk,ek,pk

def make_v4_csv(root, lecture):
    vd = root / "runs" / "v5_gate_dataset_fixed" / "v4" / lecture
    out = vd / "v4_segment_predictions.csv"
    if out.exists():
        return out

    timeline = load_timeline(vd / "timeline.json")
    sk,ek,pk = timeline_cols(timeline)

    score = np.load(vd / "v4_final_score_matrix.npy")
    boundary = np.load(vd / "v4_boundary_probability.npy")
    nxt = np.load(vd / "v4_next_slide_logits.npy")

    srt,_,_,_ = assets(root, lecture)
    transcript = read_mavils_srt(srt)

    if len(transcript) != score.shape[0]:
        raise ValueError(
            f"Exact V4 transcript segmentation mismatch: "
            f"read_mavils_srt={len(transcript)}, score rows={score.shape[0]}"
        )

    # The timeline contains slide-level intervals (18 here), while the
    # score matrix contains transcript-segment rows (273 here).
    # Assign each transcript segment to the V4 timeline interval containing
    # its start time. This preserves the exact V4 DP timeline.
    starts = np.array([float(x["start"]) for x in transcript])
    ends = np.array([float(x["end"]) for x in transcript])

    ts = np.array([float(x[sk]) for x in timeline])
    te = np.array([float(x[ek]) for x in timeline])
    tp = np.array([int(x[pk]) for x in timeline])

    rows = []
    for i,(a,b) in enumerate(zip(starts,ends)):
        candidates = np.flatnonzero((ts <= a) & (a < te))
        if len(candidates):
            j = int(candidates[0])
        else:
            # Use nearest timeline interval if a timestamp falls exactly on
            # an interval edge or outside due to rounding.
            mids = (ts + te) / 2.0
            j = int(np.argmin(np.abs(mids - a)))

        slide = int(tp[j])
        r = np.asarray(score[i]).reshape(-1)

        if 1 <= slide <= len(r):
            val = float(r[slide-1])
            other = np.delete(r, slide-1)
            second = float(np.max(other)) if len(other) else val
        else:
            order = np.argsort(r)[::-1]
            slide = int(order[0]) + 1
            val = float(r[order[0]])
            second = float(r[order[1]]) if len(order)>1 else val

        nr = np.asarray(nxt[i]).reshape(-1)
        next_slide = int(np.argmax(nr))+1 if len(nr) else -1

        rows.append({
            "segment_index": i,
            "start": float(a),
            "end": float(b),
            "v4_pred": slide,
            "v4_score": val,
            "v4_margin": val-second,
            "boundary_probability": float(np.asarray(boundary[i]).reshape(-1)[0]),
            "next_slide_pred": next_slide,
        })

    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Created: {out}")
    print(f"Transcript segments: {len(transcript)}")
    print(f"Score rows: {score.shape[0]}")
    print(f"V4 CSV rows: {len(rows)}")
    return out

def visual_cache(root,lecture,video,pdf):
    cd=root/"runs"/"v5_gate_dataset_from_existing_v4"/"_visual_cache"
    cd.mkdir(parents=True,exist_ok=True)
    cp=cd/f"{lecture}.npz"
    if cp.exists():
        z=np.load(cp)
        return z["times"],z["scores"],z["path"],True

    print(f"[{lecture}] Running actual Visual-First...")
    _,scores,times,_=run_visual_first(
        video,pdf,sample_every=2.0,max_jump=2,jump_penalty=0.05,progress=True
    )
    path=viterbi_monotonic(
        scores,max_jump=2,jump_penalty=0.05
    ).astype(np.int16)

    np.savez_compressed(
        cp,times=np.asarray(times),
        scores=np.asarray(scores,dtype=np.float32),
        path=path
    )
    return np.asarray(times),np.asarray(scores),path,False

def interval_idx(v4,t):
    s=v4.start.to_numpy(); e=v4.end.to_numpy()
    x=np.flatnonzero((s<=t)&(t<e))
    if len(x): return int(x[0])
    return int(np.argmin(np.abs(((s+e)/2)-t)))

def process(root,lecture):
    vd=root/"runs"/"v5_gate_dataset_fixed"/"v4"/lecture
    if not vd.exists():
        raise FileNotFoundError(vd)

    v4=pd.read_csv(make_v4_csv(root,lecture))
    _,pdf,gtp,video=assets(root,lecture)
    gt=load_ground_truth(gtp)
    times,scores,path,reused=visual_cache(root,lecture,video,pdf)

    rows=[]
    for t,g in gt:
        t=float(t); g=int(g)
        i=interval_idx(v4,t)
        vr=v4.iloc[i]

        j=int(np.searchsorted(times,t))
        j=max(0,min(j,len(times)-1))
        if 0<j<len(times) and abs(times[j-1]-t)<=abs(times[j]-t):
            j-=1

        pred=int(path[j])+1
        r=scores[j]
        order=np.argsort(r)[::-1]
        vs=float(r[pred-1])
        second=float(r[order[1]]) if len(order)>1 else vs

        rows.append({
            "timestamp":t,"gt_slide":g,
            "v4_pred":int(vr.v4_pred),"visual_pred":pred,
            "visual_frame_time":float(times[j]),
            "v4_correct":int(int(vr.v4_pred)==g),
            "visual_correct":int(pred==g),
            "v4_score":float(vr.v4_score),
            "v4_margin":float(vr.v4_margin),
            "boundary_probability":float(vr.boundary_probability),
            "next_slide_pred":int(vr.next_slide_pred),
            "visual_score":vs,"visual_margin":vs-second,
            "visual_frame_index":j
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
        "visual_cache_reused":bool(reused)
    }
    (od/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\n=== RESULT ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved: {od/'timestamp_comparison.csv'}")

    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--only",required=True)
    a=ap.parse_args()
    process(Path(a.root).resolve(),a.only)

if __name__=="__main__":
    main()
