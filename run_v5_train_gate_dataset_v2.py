
import argparse, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
sys.path.insert(0, str(SRC))

TRAIN = [
    "cities_and_decarbonization","climate_and_cities",
    "climate_science_policy_MIT2","cognitive_robotics_MIT",
    "computer_vision_2_2","creating_breakthrough_products_MIT",
    "cryptocurrency_MIT","deeplearning","image_processing",
    "ML_for_health_MIT","numerics","phonetics"
]

PDF_ALIASES = {
 "cities_and_decarbonization":["cities&decarbonization.pdf","cities_and_decarbonization.pdf"],
 "computer_vision_2_2":["computer_vision_2_2_geiger.pdf"],
 "creating_breakthrough_products_MIT":["creating_breakthrough_products.pdf"],
 "cryptocurrency_MIT":["cryptocurrency.pdf"],
 "deeplearning":["deeplearning_goodfellow.pdf"],
 "ML_for_health_MIT":["ML_for_health_care_MIT.pdf"],
}
VIDEO_ALIASES = {
 "cities_and_decarbonization":["cities_and_decarbonization_standard_res.mp4"],
 "computer_vision_2_2":["computer_vision_2_2_high_res.mp4"],
 "creating_breakthrough_products_MIT":["creating_breakthrough_products_MIT.mp4"],
 "cryptocurrency_MIT":["cryptocurrency_high_res.mp4"],
 "deeplearning":["deep_learning_high_res.mp4"],
 "image_processing":["image_processing_high_res.mp4"],
 "ML_for_health_MIT":["ML_for_health_high_res.mp4"],
 "numerics":["numerics_high_res.mp4"],
 "phonetics":["phonetics_high_res.mp4"],
 "cognitive_robotics_MIT":["cognitive_robotics_high_res.mp4"],
 "climate_and_cities":["climate_and_cities_high_res.mp4"],
 "climate_science_policy_MIT2":["climate_science_policy_MIT_high_res.mp4"],
}
SRT_ALIASES = {
 "climate_science_policy_MIT2":["climate_science_policy_MIT2.srt","climate_science_policy_MIT.srt"],
 "cryptocurrency_MIT":["cryptocurrency.srt"],
 "deeplearning":["deeplearning_goodfellow.srt"],
 "numerics":["numerics_hennig.srt"],
}

def first_file(folder, aliases, fallback_contains=None):
    for n in aliases or []:
        p=folder/n
        if p.exists(): return p
    if fallback_contains:
        hits=[p for p in folder.iterdir() if p.is_file() and fallback_contains.lower() in p.stem.lower()]
        if hits: return hits[0]
    return None

def assets(root, lecture):
    data=root/"data"/"MaViLS"/"data"
    srt=first_file(data/"audioscripts",SRT_ALIASES.get(lecture),lecture)
    pdf=first_file(data/"lectures",PDF_ALIASES.get(lecture),lecture)
    vid=first_file(Path(r"C:\Users\Bharat\Downloads\archive\video"),VIDEO_ALIASES.get(lecture),lecture)
    gt=first_file(data/"ground_truth_files",[], "ground_truth_"+lecture)
    return srt,pdf,gt,vid

def norm_timeline(x):
    if isinstance(x,dict):
        for k in ("timeline","segments","entries","data"):
            if isinstance(x.get(k),list): x=x[k]; break
    if not isinstance(x,list): raise ValueError("Unsupported timeline JSON structure")
    out=[]
    for z in x:
        if not isinstance(z,dict): continue
        def g(*ks):
            for k in ks:
                if k in z: return z[k]
            return None
        a=g("start","start_time","begin","from")
        b=g("end","end_time","finish","to")
        s=g("slide","slide_id","slide_number","slidenumber")
        if a is not None and b is not None and s is not None:
            out.append({"start":float(a),"end":float(b),"slide":int(s)})
    return out

def ensure_v4(root, lecture, srt,pdf,vid,outdir):
    tl=outdir/"timeline.json"
    mats=list(outdir.glob("v4_*_score_matrix.npy"))
    if tl.exists() and mats:
        print("[V4] Reusing existing completed V4 output:", outdir)
        return tl
    ck=root/"checkpoints"/"temporal_v4_best_model.pt"
    outdir.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(root/"src"/"inference_temporal_v4.py"),
         "--transcript",str(srt),"--slides",str(pdf),"--video",str(vid),
         "--checkpoint",str(ck),"--out",str(tl),
         "--render-dir",str(outdir/"rendered_slides"),
         "--jump-penalty","0.05","--max-duration","120",
         "--duration-penalty","0.003","--video-weight","0.4",
         "--video-sample-seconds","1.0"]
    print("[V4] Running fresh inference...")
    subprocess.run(cmd,check=True,cwd=root)
    return tl

def make_visual_cache(root,lecture,vid,pdf,outdir):
    cache=outdir/"visual_cache.npz"
    if cache.exists():
        print("[VISUAL] Reusing cache")
        return cache
    from visual_first_alignment import run_visual_first, viterbi_monotonic
    timeline,scores,times,duration=run_visual_first(
        str(vid),str(pdf),sample_every=2.0,max_jump=2,jump_penalty=0.05,progress=True)
    path,_=viterbi_monotonic(scores,times,max_jump=2,jump_penalty=0.05)
    np.savez_compressed(cache,times=np.asarray(times),scores=np.asarray(scores),path=np.asarray(path))
    return cache

def build(root,lecture):
    srt,pdf,gt,vid=assets(root,lecture)
    print("\n"+"="*78+"\n"+lecture+"\n"+"="*78)
    print("SRT :",srt); print("PDF :",pdf); print("GT  :",gt); print("VIDEO:",vid)
    missing=[n for n,p in [("SRT",srt),("PDF",pdf),("GT",gt),("VIDEO",vid)] if p is None]
    if missing: print("[SKIP] Missing:",missing); return None
    out=root/"runs"/"v5_train_gate_dataset_v2"/lecture
    v4dir=root/"runs"/"v5_gate_train_v4"/lecture
    ensure_v4(root,lecture,srt,pdf,vid,v4dir)

    # Exact MaViLS segmentation used by V4.
    from inference import read_mavils_srt
    segs=read_mavils_srt(str(srt))

    score_path=v4dir/"v4_final_score_matrix.npy"
    boundary_path=v4dir/"v4_boundary_probability.npy"
    next_path=v4dir/"v4_next_slide_logits.npy"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    scores=np.load(score_path)
    boundary=np.load(boundary_path) if boundary_path.exists() else np.zeros(len(segs))
    nxt=np.load(next_path) if next_path.exists() else None

    # V4 timeline is slide-level; derive V4 slide at each transcript segment midpoint.
    with open(v4dir/"timeline.json","r",encoding="utf-8") as f: tl=norm_timeline(json.load(f))
    mids=np.array([(float(s["start"])+float(s["end"]))/2 for s in segs])
    v4pred=np.ones(len(segs),dtype=int)
    for i,t in enumerate(mids):
        cand=[z for z in tl if z["start"] <= t <= z["end"]]
        if cand: v4pred[i]=cand[-1]["slide"]
        else:
            v4pred[i]=int(np.argmax(scores[i])+1)

    visual_cache=make_visual_cache(root,lecture,vid,pdf,outdir)
    vc=np.load(visual_cache)
    vt,vs,vpath=vc["times"],vc["scores"],vc["path"]
    vp=vpath.astype(int)+1

    gt_df=pd.read_excel(gt)
    gt_df.columns=[str(c).strip().lower() for c in gt_df.columns]
    time_col=next(c for c in gt_df.columns if "time" in c)
    slide_col=next(c for c in gt_df.columns if "slide" in c)
    g=gt_df[[time_col,slide_col]].copy()
    g.columns=["time","gt"]; g=g[pd.to_numeric(g.gt,errors="coerce")>0].dropna()
    g.time=pd.to_numeric(g.time); g.gt=pd.to_numeric(g.gt).astype(int)

    rows=[]
    for _,r in g.iterrows():
        t=float(r.time); truth=int(r.gt)
        j=int(np.argmin(np.abs(vt-t)))
        k=int(np.searchsorted(mids,t,side="right")-1); k=max(0,min(k,len(mids)-1))
        a=vs[j]
        order=np.argsort(a)[::-1]
        top=int(order[0])+1; second=int(order[1])+1 if len(order)>1 else top
        vm=float(a[order[0]]-a[order[1]]) if len(order)>1 else 0.0
        va=float(a[order[0]])
        row_scores=scores[k]
        so=np.sort(row_scores)[::-1]
        v4margin=float(so[0]-so[1]) if len(so)>1 else 0.0
        ent=float(-(np.exp(row_scores-row_scores.max())/np.exp(row_scores-row_scores.max()).sum() *
                    np.log(np.exp(row_scores-row_scores.max())/np.exp(row_scores-row_scores.max()).sum()+1e-12)).sum())
        rows.append({
          "lecture":lecture,"time":t,"gt":truth,
          "v4_pred":int(v4pred[k]),"visual_pred":top,
          "v4_score":float(row_scores[v4pred[k]-1]) if 0<v4pred[k]<=len(row_scores) else 0,
          "v4_margin":v4margin,"v4_entropy":ent,
          "v4_boundary":float(boundary[k]) if k<len(boundary) else 0,
          "visual_score":va,"visual_margin":vm,
          "agree":int(v4pred[k]==top),
          "distance":abs(int(v4pred[k])-top),
          "visual_is_correct":int(top==truth),
          "v4_is_correct":int(v4pred[k]==truth),
          "gate_target":int(top==truth and v4pred[k]!=truth)
        })
    df=pd.DataFrame(rows)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"features.csv",index=False)
    summary={
      "lecture":lecture,"gt_points":len(df),
      "v4_accuracy":float(df.v4_is_correct.mean()),
      "visual_accuracy":float(df.visual_is_correct.mean()),
      "oracle_gate_accuracy":float((df.v4_is_correct|df.visual_is_correct).mean()),
      "gate_positive_count":int(df.gate_target.sum())
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--only",default=None)
    args=ap.parse_args()
    root=Path(args.root).resolve()
    lectures=[args.only] if args.only else TRAIN
    done=0
    for l in lectures:
        try:
            if build(root,l) is not None: done+=1
        except Exception as e:
            print(f"[FAILED] {l}: {type(e).__name__}: {e}")
            print("Continuing.")
    print("\nV5 DATASET BUILD COMPLETE")
    print(f"Completed: {done}/{len(lectures)}")
    print("Validation/test lectures were not used.")

if __name__=="__main__":
    main()
