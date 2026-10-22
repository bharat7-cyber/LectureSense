import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "v4_score", "v4_margin", "boundary_probability",
    "visual_score", "visual_margin",
    "score_gap_visual_v4", "margin_gap_visual_v4",
    "prediction_agree", "prediction_distance",
    "v4_pred_stable", "visual_pred_stable",
]

class Gate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = root / "runs" / "v5_gate_dataset" / "gate_dataset.csv"
    out = root / "checkpoints" / "v5_gate"
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}. Run run_v5_gate_dataset.py first.")

    df = pd.read_csv(src).dropna(subset=FEATURES + ["gate_target"]).copy()
    if df["gate_target"].nunique() < 2:
        raise RuntimeError("Gate dataset needs both positive and negative examples.")

    # Group split by lecture. This prevents rows from the same lecture from
    # appearing in both train and held-out gate evaluation.
    lectures = sorted(df["lecture"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(lectures)
    n_val = max(1, int(round(len(lectures) * 0.25)))
    val_lectures = set(lectures[:n_val])

    tr = df[~df["lecture"].isin(val_lectures)]
    va = df[df["lecture"].isin(val_lectures)]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[FEATURES].to_numpy(float)).astype("float32")
    Xva = scaler.transform(va[FEATURES].to_numpy(float)).astype("float32")
    ytr = tr["gate_target"].to_numpy("float32")
    yva = va["gate_target"].to_numpy("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gate(len(FEATURES)).to(device)

    pos = max(float(ytr.sum()), 1.0)
    neg = max(float(len(ytr) - ytr.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    Xv = torch.tensor(Xva, device=device)

    best = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                model(Xv), torch.tensor(yva, device=device)
            ).item()

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    torch.save({
        "state_dict": model.state_dict(),
        "features": FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "heldout_lectures": sorted(val_lectures),
        "best_val_loss": best,
    }, out / "v5_gate_best.pt")

    (out / "training_summary.json").write_text(json.dumps({
        "rows": len(df),
        "train_rows": len(tr),
        "heldout_rows": len(va),
        "train_lectures": sorted(set(tr["lecture"])),
        "heldout_lectures": sorted(val_lectures),
        "positive_train_examples": int(ytr.sum()),
        "best_val_loss": best,
        "device": device,
    }, indent=2), encoding="utf-8")

    print("V5 gate saved:", out / "v5_gate_best.pt")
    print("Held-out gate lectures:", sorted(val_lectures))
    print("Best validation loss:", best)

if __name__ == "__main__":
    main()
