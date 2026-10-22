import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN = [
    "cities_and_decarbonization","climate_and_cities",
    "climate_science_policy_MIT2","cognitive_robotics_MIT",
    "computer_vision_2_2","creating_breakthrough_products_MIT",
    "cryptocurrency_MIT","deeplearning","image_processing",
    "ML_for_health_MIT","numerics","phonetics"
]

ALIASES = {
"cities_and_decarbonization":["cities_and_decarbonization"],
"climate_and_cities":["climate_and_cities"],
"climate_science_policy_MIT2":["climate_science_policy_MIT2","climate_science_policy_MIT"],
"cognitive_robotics_MIT":["cognitive_robotics_MIT","cognitive_robotics"],
"computer_vision_2_2":["computer_vision_2_2"],
"creating_breakthrough_products_MIT":["creating_breakthrough_products_MIT","creating_breakthrough_products"],
"cryptocurrency_MIT":["cryptocurrency_MIT","cryptocurrency"],
"deeplearning":["deeplearning_goodfellow","deeplearning"],
"image_processing":["image_processing_high_res","image_processing"],
"ML_for_health_MIT":["ML_for_health_MIT","ML_for_health"],
"numerics":["numerics_hennig","numerics"],
"phonetics":["phonetics"],
}

def norm(x): return re.sub(r"[^a-z0-9]+","",str(x).lower())

def find(folder, exts, aliases):
    ps=[p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts]
    aa=[norm(x) for x in aliases]
    for p in ps:
        if norm(p.stem) in aa: return p
    for a in aa:
        for p in ps:
            if a and (a in norm(p.stem) or norm(p.stem) in a): return p
    return None

def assets(root, name):
    sdir=root/"data/MaViLS/data/audioscripts"
    pdir=root/"data/MaViLS/data/lectures"
    gdir=root/"data/MaViLS/data/ground_truth_files"
    vdir=Path(r"C:\Users\Bharat\Downloads\archive\video")
    a=ALIASES.get(name,[name])
    s=find(sdir,{".srt"},a)
    p=find(pdir,{".pdf"},a)
    v=find(vdir,{".mp4",".mkv",".avi"},a)
    gs=["ground_truth_"+x for x in a]
    g=find(gdir,{".xlsx",".xls"},gs)
    return s,p,v,g

def gt(path):
    d=pd.read_excel(path)
    tc=next((c for c in d.columns if str(c).lower()=="time"),d.columns[0])
    sc=next((c for c in d.columns if "slide" in str(c).lower()),d.columns[1])
    o=pd.DataFrame({"timestamp":pd.to_numeric(d[tc],errors="coerce"),
                    "gt_slide":pd.to_numeric(d[sc],errors="coerce")}).dropna()
    o=o[o.gt_slide>0].sort_values("timestamp").drop_duplicates("timestamp")
    return o.reset_index(drop=True)

def run_v4(root,name,srt,pdf,video,out):
    out.mkdir(parents=True,exist_ok=True)
    timeline=out/"timeline.json"
    if timeline.exists(): return timeline
    cmd=[sys.executable,str(root/"src/inference_temporal_v4.py"),
         "--transcript",str(srt),"--slides",str(pdf),"--video",str(video),
         "--checkpoint",str(root/"checkpoints/temporal_v4_best_model.pt"),
         "--out",str(timeline),"--render-dir",str(out/"rendered_slides"),
         "--video-weight","0.4","--video-sample-seconds","1.0"]
    print("\n"+"="*80+"\nV4 DIRECT:",name,"\n"+"="*80)
    r=subprocess.run(cmd)
    return timeline if r.returncode==0 and timeline.exists() else None

def load_timeline(path):
    x=json.loads(Path(path).read_text(encoding="utf-8"))
    # support either list or {"timeline":[...]}
    if isinstance(x,dict): x=x.get("timeline",x.get("segments",[]))
    rows=[]
    for r in x:
        st=r.get("start",r.get("start_time",r.get("timestamp")))
        en=r.get("end",r.get("end_time",st))
        sl=r.get("slide",r.get("slide_number",r.get("slide_id",r.get("Slidenumber"))))
        try: rows.append((float(st),float(en),int(sl)))
        except: pass
    return pd.DataFrame(rows,columns=["start","end","v4_pred"])

def compare(root,name,v4path,gtpath,pdf,video,out):
    sys.path.insert(0,str(root))
    from src.visual_first_alignment import run_visual_first, viterbi_monotonic
    g=gt(gtpath); t=load_timeline(v4path)
    if t.empty: raise RuntimeError("V4 timeline is empty")
    scores,times=None,None
    _,scores,times,_=run_visual_first(video,pdf,sample_every=2.0,
                                      max_jump=2,jump_penalty=0.05,progress=True)
    path=viterbi_monotonic(scores,max_jump=2,jump_penalty=0.05)

    # temporal V4 stability and transition proxy
    vp=t.v4_pred.to_numpy(int)
    prev=np.r_[vp[0],vp[:-1]]
    nxt=np.r_[vp[1:],vp[-1]]
    t["v4_stable"]=((vp==prev)&(vp==nxt)).astype(int)
    t["v4_boundary_proxy"]=(vp!=prev).astype(int)

    def v4at(sec):
        h=t[(t.start<=sec)&(t.end>=sec)]
        if len(h): return h.iloc[0]
        j=np.argmin(np.abs(((t.start+t.end)/2).to_numpy()-sec))
        return t.iloc[int(j)]

    rows=[]
    for _,r in g.iterrows():
        sec=float(r.timestamp); gs=int(r.gt_slide)
        vr=v4at(sec)
        i=int(np.argmin(np.abs(times-sec)))
        s=np.asarray(scores[i],float)
        order=np.argsort(s)[::-1]
        visual=int(path[i])+1
        top=float(s[order[0]])
        second=float(s[order[1]]) if len(order)>1 else top
        # local visual stability
        lo=max(0,i-1); hi=min(len(path),i+2)
        vst=int(np.all(path[lo:hi]==path[i]))
        rows.append({
          "timestamp":sec,"gt_slide":gs,
          "v4_pred":int(vr.v4_pred),
          "visual_pred":visual,
          "visual_margin":top-second,
          "visual_score":top,
          "v4_stable":int(vr.v4_stable),
          "v4_boundary_proxy":int(vr.v4_boundary_proxy),
          "visual_stable":vst,
          "prediction_agree":int(int(vr.v4_pred)==visual),
          "prediction_distance":abs(int(vr.v4_pred)-visual),
          "v4_correct":int(int(vr.v4_pred)==gs),
          "visual_correct":int(visual==gs),
        })
    d=pd.DataFrame(rows)
    d["score_gap_visual_v4"]=d["visual_score"] # V4 raw score is unavailable in timeline; retain visual evidence
    d["gate_target"]=((d.visual_correct==1)&(d.v4_correct==0)).astype(int)
    out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"timestamp_comparison.csv",index=False)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--only")
    a=ap.parse_args(); root=Path(a.root).resolve()
    names=[a.only] if a.only else TRAIN
    base=root/"runs/v5_gate_dataset_fixed"; base.mkdir(parents=True,exist_ok=True)
    parts=[]; manifest=[]
    for n in names:
        s,p,v,g=assets(root,n)
        rec={"lecture":n,"srt":str(s) if s else None,"pdf":str(p) if p else None,
             "video":str(v) if v else None,"gt":str(g) if g else None}
        if not all([s,p,v,g]):
            rec["status"]="MISSING_ASSET"; manifest.append(rec); print("SKIP",rec); continue
        tp=run_v4(root,n,s,p,v,base/"v4"/n)
        if not tp:
            rec["status"]="V4_FAILED"; manifest.append(rec); print("V4 FAILED:",n); continue
        try:
            d=compare(root,n,tp,g,p,v,base/n); d["lecture"]=n; parts.append(d)
            rec["status"]="OK"; rec["rows"]=len(d); rec["positive"]=int(d.gate_target.sum())
        except Exception as e:
            rec["status"]="COMPARE_FAILED"; rec["error"]=repr(e)
        manifest.append(rec)
    if parts:
        allx=pd.concat(parts,ignore_index=True)
        allx.to_csv(base/"gate_dataset.csv",index=False)
        print("\n"+"="*80)
        print("DATASET ROWS:",len(allx))
        print("POSITIVE VISUAL-CORRECTION EXAMPLES:",int(allx.gate_target.sum()))
        print("NEGATIVE EXAMPLES:",int((1-allx.gate_target).sum()))
        print("LECTURES:",allx.lecture.nunique())
        print("SAVED:",base/"gate_dataset.csv")
    (base/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("MANIFEST:",base/"manifest.json")
if __name__=="__main__": main()
