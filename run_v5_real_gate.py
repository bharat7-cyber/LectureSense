from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

import numpy as np

from src.visual_first_alignment import (
    load_ground_truth,
    run_visual_first,
)

VALIDATION = [
    {
        "id": "physics_intro_02",
        "pdf": "physics_intro_02.pdf",
        "gt": "ground_truth_physics.xlsx",
        "video_aliases": ["physics_intro_02", "physics_intro", "physics"],
    },
    {
        "id": "psychology_MIT",
        "pdf": "psychology.pdf",
        "gt": "ground_truth_psychology.xlsx",
        "video_aliases": ["psychology_MIT", "psychology"],
    },
    {
        "id": "reinforcement_learning_silver",
        "pdf": "reinforcement_learning.pdf",
        "gt": "ground_truth_reinforcement_learning.xlsx",
        "video_aliases": ["reinforcement_learning_silver", "reinforcement_learning", "RL_deepmind"],
    },
]

def norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def find_video(video_dir: Path, aliases):
    files = []
    aliases = [norm(x) for x in aliases]
    for p in video_dir.rglob("*"):
        if p.suffix.lower() not in {".mp4",".mkv",".avi",".mov"}:
            continue
        stem = norm(p.stem)
        score = 0
        for a in aliases:
            if stem == a:
                score = max(score, 1000)
            elif a in stem:
                score = max(score, 800 - abs(len(stem)-len(a)))
        if score:
            files.append((score,p))
    files.sort(key=lambda x:(-x[0],str(x[1]).lower()))
    return files[0][1] if files else None

def find_col(fieldnames, candidates):
    m={norm(x):x for x in fieldnames}
    for c in candidates:
        if norm(c) in m:
            return m[norm(c)]
    return None

def load_v4_csv(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty V4 CSV: {path}")
    fields=list(rows[0].keys())
    start_col=find_col(fields,["start","start_time","segment_start"])
    end_col=find_col(fields,["end","end_time","segment_end"])
    pred_col=find_col(fields,["v4_pred","pred","slide_id","slide"])
    if not (start_col and end_col and pred_col):
        raise ValueError(f"Cannot identify V4 columns in {path}. Columns: {fields}")
    def num(x):
        return float(x)
    out=[]
    for r in rows:
        try:
            a=num(r[start_col]); b=num(r[end_col]); p=int(float(r[pred_col]))
        except Exception:
            continue
        extra={}
        for key in ["v4_score","v4_margin","boundary_probability","next_slide_pred"]:
            col=find_col(fields,[key])
            if col:
                try: extra[key]=float(r[col])
                except Exception: extra[key]=float("nan")
        out.append((a,b,p,extra))
    if not out:
        raise ValueError(f"No usable V4 rows: {path}")
    return out

def v4_at(rows,t):
    inside=[r for r in rows if r[0] <= t <= r[1]]
    if inside:
        # Prefer the narrowest interval if overlapping.
        r=min(inside,key=lambda x:(x[1]-x[0],x[0]))
        return r
    before=[r for r in rows if r[1] < t]
    after=[r for r in rows if r[0] > t]
    cand=[]
    if before:
        r=max(before,key=lambda x:x[1]); cand.append((abs(t-r[1]),r))
    if after:
        r=min(after,key=lambda x:x[0]); cand.append((abs(r[0]-t),r))
    return min(cand,key=lambda x:x[0])[1] if cand else None

def nearest_visual(times,path,scores,t):
    i=int(np.searchsorted(times,t))
    if i<=0: i=0
    elif i>=len(times): i=len(times)-1
    else:
        if abs(float(times[i-1])-t) <= abs(float(times[i])-t):
            i-=1
    pred=int(path[i])+1
    row=scores[i]
    order=np.argsort(row)[::-1]
    best=float(row[order[0]])
    second=float(row[order[1]]) if len(order)>1 else best
    return pred,best,best-second,float(times[i]),i

def compute_visual_cache(item, root, video_dir, cache_dir, sample_seconds, max_jump, jump_penalty):
    cache=cache_dir/f"{item['id']}.npz"
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        needed={"times","scores","path"}
        if needed.issubset(z.files):
            return z["times"],z["scores"],z["path"],True
    pdf=root/"data"/"MaViLS"/"data"/"lectures"/item["pdf"]
    video=find_video(video_dir,item["video_aliases"])
    if not pdf.exists(): raise FileNotFoundError(pdf)
    if video is None: raise FileNotFoundError(f"Video not found for {item['id']}")
    print(f"Running actual Visual-First: {item['id']}")
    print(f"Video: {video}")
    timeline,scores,times,duration=run_visual_first(
        video,pdf,sample_every=sample_seconds,
        max_jump=max_jump,jump_penalty=jump_penalty,progress=True)
    # Reconstruct path from timeline at each sampled timestamp.
    path=np.empty(len(times),dtype=np.int16)
    for e in timeline:
        mask=(times>=float(e["start_time"]))&(times<=float(e["end_time"])+1e-6)
        path[mask]=int(e["slide_id"])-1
    # Any uncovered sample gets nearest timeline entry.
    if np.any(path<0):
        for i in np.where(path<0)[0]:
            t=float(times[i])
            j=min(range(len(timeline)),key=lambda k:
                min(abs(t-float(timeline[k]["start_time"])),abs(t-float(timeline[k]["end_time"]))))
            path[i]=int(timeline[j]["slide_id"])-1
    np.savez_compressed(cache,times=times.astype(np.float32),scores=scores.astype(np.float32),path=path)
    return times,scores,path,False

def analyze(item,root,cache_dir,v4_dir,video_dir,args):
    gt_path=root/"data"/"MaViLS"/"data"/"ground_truth_files"/item["gt"]
    v4_path=v4_dir/item["id"]/"v4_segment_predictions.csv"
    times,scores,path,reused=compute_visual_cache(
        item,root,video_dir,cache_dir,args.sample_seconds,args.max_jump,args.jump_penalty)
    gt=load_ground_truth(gt_path)
    v4=load_v4_csv(v4_path)
    rows=[]
    both=v4only=visualonly=wrong=0
    for t,gs in gt:
        vr,vs,vm,vt,vi=nearest_visual(times,path,scores,t)
        r=v4_at(v4,t)
        vp=(r[2] if r else None)
        v4ok=vp==gs
        visok=vr==gs
        if v4ok and visok: both+=1
        elif v4ok: v4only+=1
        elif visok: visualonly+=1
        else: wrong+=1
        rows.append({
            "timestamp":float(t),"gt_slide":int(gs),
            "v4_pred":None if vp is None else int(vp),
            "visual_pred":int(vr),
            "visual_frame_time":float(vt),
            "v4_correct":int(v4ok),"visual_correct":int(visok),
            "v4_score":r[3].get("v4_score") if r else None,
            "v4_margin":r[3].get("v4_margin") if r else None,
            "boundary_probability":r[3].get("boundary_probability") if r else None,
            "visual_score":float(vs),"visual_margin":float(vm),
            "visual_frame_index":int(vi),
        })
    n=len(rows)
    summary={
        "lecture":item["id"],"gt_points":n,
        "v4_correct":sum(x["v4_correct"] for x in rows),
        "visual_correct":sum(x["visual_correct"] for x in rows),
        "both_correct":both,"v4_only_correct":v4only,
        "visual_only_correct":visualonly,"both_wrong":wrong,
        "oracle_gate_accuracy":(both+v4only+visualonly)/n if n else 0,
        "v4_accuracy":sum(x["v4_correct"] for x in rows)/n if n else 0,
        "visual_accuracy":sum(x["visual_correct"] for x in rows)/n if n else 0,
        "potential_recoverable_v4_errors":visualonly,
        "visual_cache_reused":reused,
    }
    outdir=root/"runs"/"v5_real_gate_analysis"/item["id"]
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"timestamp_comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=".")
    p.add_argument("--only",default="")
    p.add_argument("--video-dir",default=r"C:\Users\Bharat\Downloads\archive\video")
    p.add_argument("--sample-seconds",type=float,default=1.0)
    p.add_argument("--max-jump",type=int,default=3)
    p.add_argument("--jump-penalty",type=float,default=0.015)
    p.add_argument("--recompute-visual",action="store_true")
    args=p.parse_args()
    root=Path(args.root).resolve()
    video_dir=Path(args.video_dir).expanduser().resolve()
    cache_dir=root/"runs"/"v5_real_gate_analysis"/"_visual_cache"
    cache_dir.mkdir(parents=True,exist_ok=True)
    v4_dir=root/"runs"/"v5_v4_export_fixed"
    items=[x for x in VALIDATION if not args.only or x["id"]==args.only]
    if not items: raise SystemExit(f"Unknown lecture: {args.only}")
    allsum=[]
    for item in items:
        if args.recompute_visual:
            cp=cache_dir/f"{item['id']}.npz"
            if cp.exists(): cp.unlink()
        print("\n"+"="*80)
        print(f"REAL V5 GATE ANALYSIS: {item['id']}")
        s=analyze(item,root,cache_dir,v4_dir,video_dir,args)
        print(json.dumps(s,indent=2))
        allsum.append(s)
    n=sum(s["gt_points"] for s in allsum)
    def total(k): return sum(s[k] for s in allsum)
    print("\n"+"="*80)
    print("REAL V5 GATE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Usable GT points: {n}")
    print(f"V4 accuracy: {total('v4_correct')/n:.4%}")
    print(f"Visual-First accuracy: {total('visual_correct')/n:.4%}")
    print(f"Both correct: {total('both_correct')}")
    print(f"V4-only correct: {total('v4_only_correct')}")
    print(f"Visual-only correct: {total('visual_only_correct')}")
    print(f"Both wrong: {total('both_wrong')}")
    print(f"Oracle gate accuracy: {(total('both_correct')+total('v4_only_correct')+total('visual_only_correct'))/n:.4%}")
    print(f"Potentially recoverable V4 errors: {total('potential_recoverable_v4_errors')}")
    print("Validation only. Test lectures are not loaded.")

if __name__=="__main__":
    main()
