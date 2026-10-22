import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import StandardScaler

F=["v4_stable","v4_boundary_proxy","visual_margin","visual_score",
   "visual_stable","prediction_agree","prediction_distance"]

class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(len(F),32),nn.ReLU(),nn.Dropout(.15),
                               nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--epochs",type=int,default=80)
    a=ap.parse_args(); root=Path(a.root).resolve()
    p=root/"runs/v5_gate_dataset_fixed/gate_dataset.csv"
    if not p.exists(): raise FileNotFoundError(p)
    d=pd.read_csv(p).dropna(subset=F+["gate_target"]).copy()
    if d.gate_target.nunique()<2: raise RuntimeError("Need positive and negative gate examples.")
    lecs=sorted(d.lecture.unique()); rng=np.random.default_rng(42); rng.shuffle(lecs)
    nv=max(1,int(round(.25*len(lecs)))); held=set(lecs[:nv])
    tr=d[~d.lecture.isin(held)]; va=d[d.lecture.isin(held)]
    sc=StandardScaler(); X=sc.fit_transform(tr[F]).astype("float32"); Xv=sc.transform(va[F]).astype("float32")
    y=tr.gate_target.to_numpy("float32"); yv=va.gate_target.to_numpy("float32")
    dev="cuda" if torch.cuda.is_available() else "cpu"; m=Gate().to(dev)
    pos=max(float(y.sum()),1); neg=max(float(len(y)-y.sum()),1)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=dev))
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4)
    xt=torch.tensor(X,device=dev); yt=torch.tensor(y,device=dev); xv=torch.tensor(Xv,device=dev)
    best=1e9; state=None
    for ep in range(1,a.epochs+1):
        m.train(); opt.zero_grad(); loss=lossfn(m(xt),yt); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.functional.binary_cross_entropy_with_logits(m(xv),torch.tensor(yv,device=dev)).item()
        if vl<best: best=vl; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%10==0 or ep==1: print(f"Epoch {ep:03d} | train {loss.item():.4f} | val {vl:.4f}")
    m.load_state_dict(state)
    out=root/"checkpoints/v5_gate_fixed"; out.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":m.state_dict(),"features":F,"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist(),"heldout":sorted(held)},out/"v5_gate_best.pt")
    print("Saved:",out/"v5_gate_best.pt"); print("Held-out:",sorted(held))
if __name__=="__main__": main()
