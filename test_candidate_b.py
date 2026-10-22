from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train_candidate_b as exp


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate_split(split: str, prefix: str, verbose: bool = True):
    checkpoint_path = exp.OUT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Candidate B checkpoint not found: {checkpoint_path}. Run `python train_candidate_b.py` first."
        )

    ckpt = safe_torch_load(checkpoint_path, exp.DEVICE)
    v4_checkpoint = Path(ckpt.get("v4_checkpoint", str(exp.V4_CHECKPOINT)))
    if not v4_checkpoint.is_absolute():
        v4_checkpoint = ROOT / v4_checkpoint

    print(f"Device: {exp.DEVICE}")
    if exp.DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Split: {split.upper()}")
    print(f"Candidate checkpoint: {checkpoint_path}")
    print(f"Frozen V4 checkpoint: {v4_checkpoint}")

    v4_model, _, max_slides, temperature, _ = exp.v4.load_v4_checkpoint(v4_checkpoint)
    v4_model.eval()

    lectures = exp.load_lecture_items(
        split,
        prefix,
        score_batch=8,
        temporal_model=v4_model,
        max_slides=max_slides,
        temperature=temperature,
    )
    if not lectures:
        raise RuntimeError(f"No lectures loaded for split={split!r}. Check dataset split labels / feature caches.")

    model = exp.CandidateTemporalModel(
        context_dim=int(ckpt.get("context_dim", 15)),
        candidate_dim=int(ckpt.get("candidate_dim", 11)),
        hidden=int(ckpt.get("hidden", 96)),
        layers=int(ckpt.get("layers", 2)),
        heads=int(ckpt.get("heads", 4)),
    ).to(exp.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    cfg = dict(ckpt["decoder_config"])
    per = {}
    total_correct = 0
    total_rows = 0
    total_candidate_hits = 0
    total_candidate_rows = 0
    for lecture, item in lectures.items():
        path, candidate_recall = exp.decode_lecture(model, item, cfg)
        labels = item["labels"]
        correct = int((path == labels).sum())
        total = int(len(labels))
        acc = correct / max(1, total)
        per[lecture] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "candidate_recall": float(candidate_recall),
        }
        total_correct += correct
        total_rows += total
        total_candidate_hits += int(round(candidate_recall * total))
        total_candidate_rows += total
        if verbose:
            print(f"{lecture:35s} {correct:4d}/{total:4d}  {acc*100:6.2f}%  candidate recall={candidate_recall*100:6.2f}%")

    macro = sum(x["accuracy"] for x in per.values()) / max(1, len(per))
    weighted = total_correct / max(1, total_rows)
    result = {
        "split": split,
        "macro_accuracy": macro,
        "weighted_accuracy": weighted,
        "correct": total_correct,
        "total": total_rows,
        "candidate_recall_weighted": total_candidate_hits / max(1, total_candidate_rows),
        "per_lecture": per,
        "decoder_config": cfg,
        "test_used_for_model_selection": False,
    }

    out_dir = ROOT / "runs" / "temporal_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"{split.upper()} macro accuracy    : {macro*100:.2f}%")
    print(f"{split.upper()} weighted accuracy : {weighted*100:.2f}% ({total_correct}/{total_rows})")
    print(f"Candidate recall (weighted): {result['candidate_recall_weighted']*100:.2f}%")
    print(f"Saved evaluation           : {out_path}")
    return result


if __name__ == "__main__":
    print("FINAL HELD-OUT TEST. Do not tune model/decoder using these results.")
    evaluate_split("test", "test", verbose=True)
