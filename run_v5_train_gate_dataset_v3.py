
import argparse, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

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
    if not folder.exists():
        return None
    for n in aliases or []:
        p=folder/n
        if p.exists(): return p
    if fallback_contains:
        key=fallback_contains.lower().replace("_","").replace("&","")
        hits=[]
        for p in folder.iterdir():
            if p.is_file():
                stem=p.stem.lower().replace("_","").replace("&","")
                if key in stem or stem in key:
                    hits.append(p)
        if hits:
            return hits[0]
    return None

def assets(root, lecture):
    data=root/"data"/"MaViLS"/"data"
    video_dir=Path(r"C:\Users\Bharat\Downloads\archive\video")
    srt=first_file(data/"audioscripts",SRT_ALIASES.get(lecture),lecture)
    pdf=first_file(data/"lectures",PDF_ALIASES.get(lecture),lecture)
    vid=first_file(video_dir,VIDEO_ALIASES.get(lecture),lecture)
    gt=first_file(data/"ground_truth_files",[], "ground_truth_"+lecture)
    return srt,pdf,gt,vid

def norm_timeline(x):
    if isinstance(x,dict):
        for k in ("timeline","segments","entries","data"):
            if isinstance(x.get(k),list):
                x=x[k]
                break
    if not isinstance(x,list):
        raise ValueError("Unsupported timeline JSON structure")
    out=[]
    for z in x:
        if not isinstance(z,dict):
            continue
        def g(*ks):
            for k in ks:
                if k in z:
                    return z[k]
            return None
        a=g("start","start_time","begin","from")
        b=g("end","end_time","finish","to")
        s=g("slide","slide_id","slide_number","slidenumber")
        if a is not None and b is not None and s is not None:
            out.append({"start":float(a),"end":float(b),"slide":int(s)})
    if not out:
        raise ValueError("No usable start/end/slide entries found in timeline.json")
    return out

def ensure_v4(root,lecture,srt,pdf,vid,outdir):
    tl=outdir/"timeline.json"
    mats=list(outdir.glob("v4_*_score_matrix.npy"))
    if tl.exists() and mats:
        print("[V4] Reusing existing completed V4 output:",outdir)
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
        print("[VISUAL] Reusing cache:",cache)
        return cache
    from visual_first_alignment import run_visual_first, viterbi_monotonic
    _,scores,times,_=run_visual_first(
        str(vid),str(pdf),sample_every=2.0,max_jump=2,
        jump_penalty=0.05,progress=True)
    path,_=viterbi_monotonic(
        np.asarray(scores),np.asarray(times),
        max_jump=2,jump_penalty=0.05)
    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        cache,times=np.asarray(times),
        scores=np.asarray(scores),path=np.asarray(path))
    return cache

def load_gt(gt):
    df=pd.read_excel(gt)
    df.columns=[str(c).strip().lower() for c in df.columns]
    tc=next(c for c in df.columns if "time" in c)
    sc=next(c for c in df.columns if "slide" in c)
    g=df[[tc,sc]].copy()
    g.columns=["time","gt"]
    g["time"]=pd.to_numeric(g["time"],errors="coerce")
    g["gt"]=pd.to_numeric(g["gt"],errors="coerce")
    return g[g["gt"]>0].dropna().reset_index(drop=True)

def build(root,lecture):
    srt,pdf,gt,vid=assets(root,lecture)
    print("\n"+"="*78+"\n"+lecture+"\n"+"="*78)
    print("SRT :",srt); print("PDF :",pdf); print("GT  :",gt); print("VIDEO:",vid)
    missing=[n for n,p in [("SRT",srt),("PDF",pdf),("GT",gt),("VIDEO",vid)] if p is None]
    if missing:
        print("[SKIP] Missing:",missing)
        return None

    v4dir=root/"runs"/"v5_gate_train_v4"/lecture
    out=root/"runs"/"v5_train_gate_dataset_v3"/lecture
    ensure_v4(root,lecture,srt,pdf,vid,v4dir)

    # Import only after src has explicitly been added to sys.path.
    from inference import read_mavils_srt
    segs=read_mavils_srt(str(srt))
    print("Exact V4 transcript segments:",len(segs))

    score_path=v4dir/"v4_final_score_matrix.npy"
    boundary_path=v4dir/"v4_boundary_probability.npy"
    next_path=v4dir/"v4_next_slide_logits.npy"
    scores=np.load(score_path)
    boundary=np.load(boundary_path) if boundary_path.exists() else np.zeros(scores.shape[0])

    if scores.shape[0] != len(segs):
        raise ValueError(
            f"V4 score rows ({scores.shape[0]}) != exact transcript segments ({len(segs)})")

    with open(v4dir/"timeline.json","r",encoding="utf-8") as f:
        tl=norm_timeline(json.load(f))

    mids=np.array([(float(s["start"])+float(s["end"]))/2 for s in segs])
    v4pred=np.ones(len(segs),dtype=int)
    for i,t in enumerate(mids):
        cand=[z for z in tl if z["start"]<=t<=z["end"]]
        v4pred[i]=cand[-1]["slide"] if cand else int(np.argmax(scores[i])+1)

    visual_cache=make_visual_cache(root,lecture,vid,pdf,out)
    vc=np.load(visual_cache)
    vt=np.asarray(vc["times"])
    vs=np.asarray(vc["scores"])
    vpath=np.asarray(vc["path"]).astype(int)
    vp=vpath+1

    g=load_gt(gt)
    rows=[]
    for _,r in g.iterrows():
        t=float(r.time); truth=int(r.gt)
        j=int(np.argmin(np.abs(vt-t)))
        k=int(np.searchsorted(mids,t,side="right")-1)
        k=max(0,min(k,len(mids)-1))

        va_row=vs[j]
        vo=np.argsort(va_row)[::-1]
        visual_top=int(vo[0])+1
        visual_second=int(vo[1])+1 if len(vo)>1 else visual_top
        visual_margin=float(va_row[vo[0]]-va_row[vo[1]]) if len(vo)>1 else 0.0
        visual_score=float(va_row[vo[0]])

        v4_row=scores[k]
        so=np.sort(v4_row)[::-1]
        v4_margin=float(so[0]-so[1]) if len(so)>1 else 0.0
        exp=np.exp(v4_row-v4_row.max())
        prob=exp/(exp.sum()+1e-12)
        entropy=float(-(prob*np.log(prob+1e-12)).sum())

        v4p=int(v4pred[k])
        cross_visual_on_v4=float(
            va_row[v4p-1]) if 0<v4p<=len(va_row) else 0.0
        cross_v4_on_visual=float(
            v4_row[visual_top-1]) if 0<visual_top<=len(v4_row) else 0.0

        rows.append({
            "lecture":lecture,"time":t,"gt":truth,
            "v4_pred":v4p,"visual_pred":visual_top,
            "v4_score":float(v4_row[v4p-1]) if 0<v4p<=len(v4_row) else 0.0,
            "v4_margin":v4_margin,"v4_entropy":entropy,
            "v4_boundary":float(boundary[k]) if k<len(boundary) else 0.0,
            "visual_score":visual_score,"visual_margin":visual_margin,
            "cross_visual_on_v4":cross_visual_on_v4,
            "cross_v4_on_visual":cross_v4_on_visual,
            "agree":int(v4p==visual_top),
            "distance":abs(v4p-visual_top),
            "v4_is_correct":int(v4p==truth),
            "visual_is_correct":int(visual_top==truth),
            "gate_target":int(visual_top==truth and v4p!=truth),
        })

    df=pd.DataFrame(rows)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"features.csv",index=False)
    summary={
        "lecture":lecture,
        "gt_points":int(len(df)),
        "v4_accuracy":float(df.v4_is_correct.mean()),
        "visual_accuracy":float(df.visual_is_correct.mean()),
        "oracle_gate_accuracy":float((df.v4_is_correct|df.visual_is_correct).mean()),
        "gate_positive_count":int(df.gate_target.sum()),
        "feature_count":int(len(df.columns)),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
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
    for lecture in lectures:
        try:
            if build(root,lecture) is not None:
                done+=1
        except Exception as e:
            print(f"[FAILED] {lecture}: {type(e).__name__}: {e}")
            print("Continuing to next lecture.")
    print("\n"+"="*78)
    print("V5 DATASET BUILD COMPLETE")
    print("="*78)
    print(f"Completed: {done}/{len(lectures)}")
    print("Validation/test lectures were not used.")

if __name__=="__main__":
    main()
