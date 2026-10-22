import argparse, json, os, random, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

TRAIN = [
    'cities_and_decarbonization','climate_and_cities','climate_science_policy_MIT2',
    'cognitive_robotics_MIT','computer_vision_2_2','creating_breakthrough_products_MIT',
    'cryptocurrency_MIT','deeplearning','image_processing','ML_for_health_MIT',
    'numerics','phonetics'
]
VAL = ['physics','psychology','reinforcement_learning','sensory_systems']
TEST = ['short_range','solar_resource','team_dynamics_game_design_MIT','theory_of_computation']

DATA = ROOT/'data'/'MaViLS'/'data'
VIDEO_DIR = Path(r'C:\Users\Bharat\Downloads\archive\video')
V4_ROOT = ROOT/'runs'/'v5_gate_train_v4'
OUT = ROOT/'runs'/'v5_candidate_fusion_FINAL'
MODEL_DIR = ROOT/'checkpoints'
MODEL_PATH = MODEL_DIR/'v5_candidate_fusion_best.pt'
CONFIG_PATH = MODEL_DIR/'v5_candidate_fusion_config.json'

SRT = {
'cities_and_decarbonization':'cities_and_decarbonization.srt',
'climate_and_cities':'climate_and_cities.srt',
'climate_science_policy_MIT2':'climate_science_policy_MIT.srt',
'cognitive_robotics_MIT':'cognitive_robotics_MIT.srt',
'computer_vision_2_2':'computer_vision_2_2_high_res.srt',
'creating_breakthrough_products_MIT':'creating_breakthrough_products_MIT.srt',
'cryptocurrency_MIT':'cryptocurrency.srt','deeplearning':'deeplearning_goodfellow.srt',
'image_processing':'image_processing_high_res.srt','ML_for_health_MIT':'ML_for_health_MIT.srt',
'numerics':'numerics_hennig.srt','phonetics':'phonetics.srt',
'physics':'physics_intro_02.srt','psychology':'psychology_MIT.srt',
'reinforcement_learning':'reinforcement_learning_silver.srt','sensory_systems':'visual_system_MIT.srt',
'short_range':'short_range_mit.srt','solar_resource':'solar_resource.srt',
' team_dynamics_game_design_MIT':'team_dynamics_game_design_MIT.srt',
' team_dynamics':'team_dynamics_game_design_MIT.srt',
'team_dynamics_game_design_MIT':'team_dynamics_game_design_MIT.srt',
' theory_of_computation':'theory_of_computation.srt',
'theory_of_computation':'theory_of_computation.srt'
}
PDF = {
'cities_and_decarbonization':'cities&decarbonization.pdf','climate_and_cities':'cities&climate.pdf',
'climate_science_policy_MIT2':'climate_science_policy_MIT.pdf','cognitive_robotics_MIT':'cognitive_robotics_MIT.pdf',
'computer_vision_2_2':'computer_vision_2_2_geiger.pdf','creating_breakthrough_products_MIT':'creating_breakthrough_products.pdf',
'cryptocurrency_MIT':'cryptocurrency.pdf','deeplearning':'deeplearning_goodfellow.pdf','image_processing':'image_processing.pdf',
'ML_for_health_MIT':'ML_for_health_care_MIT.pdf','numerics':'numerics.pdf','phonetics':'phonetics.pdf',
'physics':'physics_intro_02.pdf','psychology':'psychology.pdf','reinforcement_learning':'reinforcement_learning.pdf',
'sensory_systems':'sensory_systems.pdf','short_range':'short_range_mit.pdf','solar_resource':'solar_resource.pdf',
'team_dynamics_game_design_MIT':'team_dynamics_game_design_mit.pdf','theory_of_computation':'theory_of_computation.pdf'
}
VIDEO = {
'cities_and_decarbonization':'cities_and_decarbonization_standard_res.mp4',
'climate_and_cities':'cities_and_climate_high_res.mp4',
'climate_science_policy_MIT2':None,
'cognitive_robotics_MIT':'cognitive_robotics_high_res.mp4','computer_vision_2_2':'computer_vision_2_2_high_res.mp4',
'creating_breakthrough_products_MIT':'creating_breakthrough_products_MIT.mp4','cryptocurrency_MIT':'cryptocurrency_high_res.mp4',
'deeplearning':'deep_learning_high_res.mp4','image_processing':'image_processing_high_res.mp4',
'ML_for_health_MIT':'ML_for_health_high_res.mp4','numerics':'numerics_high_res.mp4','phonetics':'phonetics_high_res.mp4',
'physics':'physics_high_res.mp4','psychology':'psychology_high_res.mp4','reinforcement_learning':'reinforcement_learning_high_res.mp4',
'sensory_systems':'sensory_systems_high_res.mp4','short_range':'short_range_MIT.mp4','solar_resource':'solar_resource_high_res.mp4',
'team_dynamics_game_design_MIT':'team_dynamics_high_res.mp4','theory_of_computation':'theory_of_computation_high_res.mp4'
}
GT = {x:f'ground_truth_{x}.xlsx' for x in TRAIN+VAL+TEST}
GT['team_dynamics_game_design_MIT']='ground_truth_team_dynamics_game_design_MIT.xlsx'
GT['sensory_systems']='ground_truth_sensory_systems.xlsx'

def asset(split, lecture):
    srt=DATA/'audioscripts'/SRT.get(lecture)
    pdf=DATA/'lectures'/PDF.get(lecture)
    gt=DATA/'ground_truth_files'/GT.get(lecture)
    video=(VIDEO_DIR/VIDEO[lecture]) if VIDEO.get(lecture) else None
    return (srt if srt.exists() else None, pdf if pdf.exists() else None,
            gt if gt.exists() else None, video if video and video.exists() else None)

def get_segments(srt):
    from inference import read_mavils_srt
    return read_mavils_srt(str(srt))

def seg_mids(segs):
    out=[]
    for s in segs:
        a=float(s.get('start',s.get('start_time',0))) if isinstance(s,dict) else float(s.start)
        b=float(s.get('end',s.get('end_time',a))) if isinstance(s,dict) else float(s.end)
        out.append((a+b)/2)
    return np.asarray(out,np.float32)

def load_gt(path):
    # MaViLS GT variants include:
    #   time | slidenumber | value
    # and:
    #   key  | slidenumber | value
    d=pd.read_excel(path)
    d.columns=[
        str(c).strip().lower().replace(' ','').replace('-','_')
        for c in d.columns
    ]
    tc=next(
        (c for c in ['time','timestamp','start','key'] if c in d.columns),
        None
    )
    sc=next(
        (c for c in ['slidenumber','slide_number','slide','slideid','slide_id']
         if c in d.columns),
        None
    )
    if tc is None or sc is None:
        raise ValueError(
            f'GT columns not understood: {list(d.columns)}'
        )

    d=d[[tc,sc]].copy()
    d.columns=['time','gt']
    d['time']=pd.to_numeric(d['time'],errors='coerce')
    d['gt']=pd.to_numeric(d['gt'],errors='coerce')
    d=d.dropna()

    # IMPORTANT: use the dataframe column, not DataFrame.gt().
    d=d[d['gt']>0]
    d['gt']=d['gt'].astype(int)

    if d.empty:
        raise ValueError(f'GT contains no valid slide-labelled timestamps: {path}')

    return d.sort_values('time').reset_index(drop=True)

def run_v4(lecture,srt,pdf,video):
    d=V4_ROOT/lecture; score=d/'v4_final_score_matrix.npy'
    if score.exists(): return d
    if video is None: raise RuntimeError(f'No exact video for {lecture}; refusing to guess.')
    d.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'src'/'inference_temporal_v4.py'),
         '--transcript',str(srt),'--slides',str(pdf),'--video',str(video),
         '--checkpoint',str(ROOT/'checkpoints'/'temporal_v4_best_model.pt'),
         '--out',str(d/'timeline.json'),'--render-dir',str(d/'rendered_slides'),
         '--jump-penalty','0.05','--max-duration','120','--duration-penalty','0.003',
         '--video-weight','0.4','--video-sample-seconds','1.0']
    print('[V4] running',lecture); subprocess.run(cmd,check=True,cwd=ROOT); return d

def get_visual(lecture,pdf,video):
    OUT.mkdir(parents=True,exist_ok=True)
    new=OUT/'_visual_cache'/f'{lecture}.npz'
    old=ROOT/'runs'/'v5_train_gate_dataset_FINAL_V2'/'_visual_cache'/f'{lecture}.npz'
    if new.exists():
        z=np.load(new); return z['times'],z['scores'],z['path']
    if old.exists():
        new.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(old,new)
        z=np.load(new); print('[VISUAL] reused previous cache',lecture); return z['times'],z['scores'],z['path']
    if video is None: raise RuntimeError(f'No exact video for {lecture}; refusing to guess.')
    from visual_first_alignment import run_visual_first, viterbi_monotonic
    print('[VISUAL] running real Visual-First',lecture)
    _,scores,times,_=run_visual_first(Path(video),Path(pdf),sample_every=2.0,max_jump=2,jump_penalty=0.05,progress=True)
    path,_=viterbi_monotonic(np.asarray(scores),max_jump=2,jump_penalty=0.05)
    new.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(new,times=np.asarray(times,np.float32),scores=np.asarray(scores,np.float32),path=np.asarray(path,np.int32))
    return np.asarray(times),np.asarray(scores),np.asarray(path)

def softmax(x):
    z=x-x.max(axis=1,keepdims=True); e=np.exp(z); return e/(e.sum(axis=1,keepdims=True)+1e-12)

def rank_norm(x):
    order=np.argsort(-x,axis=1); r=np.empty_like(order,dtype=np.float32)
    for i in range(x.shape[0]): r[i,order[i]]=np.arange(x.shape[1],dtype=np.float32)
    if x.shape[1]>1: r/=x.shape[1]-1
    return 1-r

def local_features(v4,vis,boundary,si,vi):
    n=v4.shape[1]; s0=max(0,si-1); s1=min(v4.shape[0]-1,si+1)
    v0=v4[si]; vp=v4[s0]; vn=v4[s1]
    w0=vis[vi]; wprev=vis[max(0,vi-1)]; wnext=vis[min(vis.shape[0]-1,vi+1)]
    vprob=softmax(v4[[si]])[0]; wprob=softmax(vis[[vi]])[0]
    vr=rank_norm(v4[[si]])[0]; wr=rank_norm(vis[[vi]])[0]
    vt=int(np.argmax(v0)); wt=int(np.argmax(w0))
    vm=float(np.sort(v0)[-1]-np.sort(v0)[-2]) if n>1 else 0.0
    wm=float(np.sort(w0)[-1]-np.sort(w0)[-2]) if n>1 else 0.0
    allf=[]
    for slide in range(n):
        f=[
            float(v0[slide]),float(vprob[slide]),float(vr[slide]),
            float(w0[slide]),float(wprob[slide]),float(wr[slide]),
            float((vp[slide]+v0[slide]+vn[slide])/3),
            float((wprev[slide]+w0[slide]+wnext[slide])/3),
            float(v0[slide]-vp[slide]),float(v0[slide]-vn[slide]),
            float(w0[slide]-wprev[slide]),float(w0[slide]-wnext[slide]),
            float(v0[slide]*w0[slide]),float(v0[slide]-w0[slide]),
            float(boundary[si]) if si<len(boundary) else 0.0,
            vm,wm,float(slide==vt),float(slide==wt),float(slide==vt==wt),
            float(abs(slide-vt)),float(abs(slide-wt)),
            float(slide/max(1,n-1)),float(si/max(1,v4.shape[0]-1)),
        ]
        allf.append(f)
    return np.asarray(allf,np.float32)

def lecture_arrays(lecture,need_gt=True):
    srt,pdf,gt_path,video=asset('x',lecture)
    print('\n'+'='*78); print(lecture); print('='*78)
    print('SRT :',srt); print('PDF :',pdf); print('GT  :',gt_path); print('VIDEO:',video)
    if srt is None or pdf is None or video is None or (need_gt and gt_path is None):
        raise RuntimeError('Missing exact asset(s)')
    v4d=run_v4(lecture,srt,pdf,video)
    v4=np.load(v4d/'v4_final_score_matrix.npy').astype(np.float32)
    bpath=v4d/'v4_boundary_probability.npy'
    boundary=np.load(bpath).astype(np.float32) if bpath.exists() else np.zeros(v4.shape[0],np.float32)
    segs=get_segments(srt); mids=seg_mids(segs)
    if v4.shape[0]!=len(mids): raise ValueError(f'V4 rows {v4.shape[0]} != transcript segments {len(mids)}')
    vt,vis,_=get_visual(lecture,pdf,video); vis=np.asarray(vis,np.float32); vt=np.asarray(vt,np.float32)
    if vis.shape[1]!=v4.shape[1]: raise ValueError(f'V4 slides {v4.shape[1]} != visual slides {vis.shape[1]}')
    return v4,vis,boundary,mids,vt,gt_path

def build_training_lecture(lecture):
    v4,vis,boundary,mids,vt,gt_path=lecture_arrays(lecture,True)
    gt=load_gt(gt_path)
    rows=[]; nslides=v4.shape[1]
    for t,truth in zip(gt['time'],gt['gt']):
        t=float(t); truth=int(truth)
        si=max(0,min(len(mids)-1,int(np.searchsorted(mids,t,side='right')-1)))
        vi=int(np.argmin(np.abs(vt-t)))
        feats=local_features(v4,vis,boundary,si,vi)
        rows.append((feats,truth-1,t))
    X=np.stack([r[0] for r in rows]); Y=np.asarray([r[1] for r in rows],np.int64); T=np.asarray([r[2] for r in rows],np.float32)
    return {'lecture':lecture,'X':X,'Y':Y,'T':T,'n_slides':nslides}

def standardize(data):
    allx=np.concatenate([d['X'].reshape(-1,d['X'].shape[-1]) for d in data],0)
    mean=allx.mean(0).astype(np.float32); std=allx.std(0).astype(np.float32); std[std<1e-6]=1
    for d in data: d['X']=((d['X']-mean)/std).astype(np.float32)
    return mean,std

class Reranker(nn.Module):
    def __init__(self,f):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(f,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU(),nn.Dropout(.10),nn.Linear(64,1))
    def forward(self,x):
        b,s,f=x.shape; return self.net(x.reshape(b*s,f)).reshape(b,s)

def fit_model(train_data,epochs=35,batch=32):
    order=list(range(len(train_data))); random.Random(SEED).shuffle(order)
    dev_ids=order[:max(2,int(round(.25*len(order))))]; fit_ids=order[len(dev_ids):]
    f=train_data[0]['X'].shape[-1]; model=Reranker(f).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=-1; best_state=None
    for ep in range(1,epochs+1):
        model.train(); random.Random(SEED+ep).shuffle(fit_ids); losses=[]
        for idx in fit_ids:
            d=train_data[idx]; x=torch.from_numpy(d['X']).to(DEVICE); y=torch.from_numpy(d['Y']).to(DEVICE)
            for a in range(0,len(y),batch):
                xb=x[a:a+batch]; yb=y[a:a+batch]
                opt.zero_grad(); loss=F.cross_entropy(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); vals=[]
        with torch.no_grad():
            for idx in dev_ids:
                d=train_data[idx]; pred=model(torch.from_numpy(d['X']).to(DEVICE)).argmax(1).cpu().numpy(); vals.append(float((pred==d['Y']).mean()))
        dev=float(np.mean(vals))
        if dev>best: best=dev; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep==1 or ep%5==0 or ep==epochs: print(f'[TRAIN] epoch={ep:02d} loss={np.mean(losses):.4f} internal-dev={dev:.4f}')
    model.load_state_dict(best_state); return model,best

def grouped_matrix_predict(model, d):
    model.eval()
    with torch.no_grad():
        logits=model(torch.from_numpy(d['X']).to(DEVICE)).cpu().numpy()
    return logits

def greedy_viterbi(score,max_jump=3,jump_penalty=.05):
    n,s=score.shape; dp=np.full((n,s),-np.inf,np.float32); back=np.full((n,s),-1,np.int16)
    dp[0,:min(s,max_jump+1)]=score[0,:min(s,max_jump+1)]
    for t in range(1,n):
        for j in range(s):
            lo=max(0,j-max_jump); best=-np.inf; bp=-1
            for p in range(lo,j+1):
                if not np.isfinite(dp[t-1,p]): continue
                val=dp[t-1,p]+(0 if p==j else -jump_penalty*(j-p))
                if val>best: best=val; bp=p
            dp[t,j]=best+score[t,j]; back[t,j]=bp
    path=np.empty(n,np.int32); path[-1]=int(np.argmax(dp[-1]))
    for t in range(n-1,0,-1): path[t-1]=max(0,int(back[t,path[t]]))
    return path

def make_inference_data(v4,vis,boundary,mids,vt):
    X=[]
    for si,t in enumerate(mids):
        vi=int(np.argmin(np.abs(vt-float(t))))
        X.append(local_features(v4,vis,boundary,si,vi))
    return np.stack(X).astype(np.float32)

def eval_split(model,mean,std,lecture):
    v4,vis,boundary,mids,vt,gt_path=lecture_arrays(lecture,True)
    X=((make_inference_data(v4,vis,boundary,mids,vt)-mean)/std).astype(np.float32)
    d={'X':X,'Y':np.zeros(len(X),np.int64),'n_slides':X.shape[1]}
    learned=grouped_matrix_predict(model,d)
    path=greedy_viterbi(learned,max_jump=3,jump_penalty=.05)
    gt=load_gt(gt_path)
    pred=[]; truth=[]
    for t,g in zip(gt['time'],gt['gt']):
        si=max(0,min(len(mids)-1,int(np.searchsorted(mids,float(t),side='right')-1)))
        pred.append(int(path[si])); truth.append(int(g)-1)
    return float((np.asarray(pred)==np.asarray(truth)).mean()),len(truth)

def save_bundle(model,mean,std,internal):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'input_dim':len(mean),'mean':mean,'std':std,'internal_dev':float(internal)},MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({'model':'CandidateReranker','features':24,'max_jump':3,'jump_penalty':.05,'internal_dev':float(internal),'test_used_for_training':False},indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['train','validate','test'],default='train'); ap.add_argument('--only',default=None); args=ap.parse_args()
    print('='*78); print('LECTURESENSE V5 — CANDIDATE-LEVEL LEARNED FUSION V4'); print('='*78); print('Device:',DEVICE); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    if args.mode=='train':
        lectures=[args.only] if args.only else TRAIN
        data=[]
        for l in lectures:
            try: data.append(build_training_lecture(l))
            except Exception as e: print(f'[SKIP] {l}: {type(e).__name__}: {e}')
        if len(data)<5: raise RuntimeError(f'Only {len(data)} usable training lectures available; need at least 5.')
        mean,std=standardize(data); model,internal=fit_model(data); save_bundle(model,mean,std,internal)
        rows=[]
        for d in data:
            with torch.no_grad(): p=grouped_matrix_predict(model,d).argmax(1)
            rows.append({'lecture':d['lecture'],'points':len(d['Y']),'candidate_top1_accuracy':float((p==d['Y']).mean())})
        out=OUT; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'train_results.csv',index=False)
        print('\nTRAIN RESULTS'); print(pd.DataFrame(rows).to_string(index=False)); print(f'\nSaved model: {MODEL_PATH}'); print(f'Internal group-dev accuracy: {internal:.4f}')
        print('Validation/test were not used.')
    else:
        if not MODEL_PATH.exists(): raise FileNotFoundError(f'{MODEL_PATH} not found. Run --mode train first.')
        ck=torch.load(MODEL_PATH,map_location=DEVICE); mean=np.asarray(ck['mean'],np.float32); std=np.asarray(ck['std'],np.float32); model=Reranker(int(ck['input_dim'])).to(DEVICE); model.load_state_dict(ck['state_dict']); model.eval()
        lectures=VAL if args.mode=='validate' else TEST
        rows=[]
        for l in lectures:
            try:
                a,n=eval_split(model,mean,std,l); rows.append({'lecture':l,'points':n,'v5_accuracy':a})
            except Exception as e: print(f'[SKIP] {l}: {type(e).__name__}: {e}')
        if not rows: raise RuntimeError('No lectures evaluated.')
        df=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); df.to_csv(OUT/f'{args.mode}_results.csv',index=False)
        print('\n'+args.mode.upper()+' RESULTS'); print(df.to_string(index=False)); print(f'\nMacro accuracy: {df.v5_accuracy.mean():.4f}')
        if args.mode=='test': print('TEST MODEL SELECTION/TRAINING: NOT PERFORMED.')

if __name__=='__main__': main()
