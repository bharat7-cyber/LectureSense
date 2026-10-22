import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import StandardScaler

VAL=["physics_intro_02","psychology_MIT","reinforcement_learning_silver"]
F=["v4_stable","v4_boundary_proxy","visual_margin","visual_score","visual_stable","prediction_agree","prediction_distance"]

class Gate(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(len(F),32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); a=ap.parse_args(); root=Path(a.root).resolve()
    ck=torch.load(root/"checkpoints/v5_gate_fixed/v5_gate_best.pt",map_location="cpu")
    sc=StandardScaler(); sc.mean_=np.array(ck["mean"]); sc.scale_=np.array(ck["scale"]); sc.var_=sc.scale_**2; sc.n_features_in_=len(F)
    m=Gate(); m.load_state_dict(ck["state_dict"]); m.eval()
    parts=[]
    for n in VAL:
        p=root/"runs/v5_real_gate_analysis"/n/"timestamp_comparison.csv"
        if p.exists(): 
            d=pd.read_csv(p); d["lecture"]=n; parts.append(d)
    if not parts: raise RuntimeError("Run real V5 gate analysis first.")
    d=pd.concat(parts,ignore_index=True)
    X=sc.transform(d[F]).astype("float32")
    with torch.no_grad(): pr=torch.sigmoid(m(torch.tensor(X))).numpy()
    best=None
    for th in np.linspace(.05,.95,19):
        pred=np.where(pr>=th,d.visual_pred,d.v4_pred); acc=float(np.mean(pred==d.gt_slide))
        if best is None or acc>best[0]: best=(acc,float(th))
    d["gate_probability_visual"]=pr; d["gated_pred"]=np.where(pr>=best[1],d.visual_pred,d.v4_pred)
    out=root/"runs/v5_gate_validation_fixed"; out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"predictions.csv",index=False)
    res={"rows":len(d),"v4_accuracy":float(np.mean(d.v4_pred==d.gt_slide)),
         "visual_accuracy":float(np.mean(d.visual_pred==d.gt_slide)),
         "gated_accuracy":best[0],"threshold":best[1]}
    (out/"summary.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps(res,indent=2))
if __name__=="__main__": main()
