from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.visual_first_alignment import load_ground_truth, run_visual_first, evaluate_timeline, save_timeline


VALIDATION = [
    {
        "id": "physics_intro_02",
        "srt": "physics_intro_02.srt",
        "pdf": "physics_intro_02.pdf",
        "gt": "ground_truth_physics.xlsx",
        "video_aliases": ["physics_intro_02", "physics_intro", "physics"],
    },
    {
        "id": "psychology_MIT",
        "srt": "psychology_MIT.srt",
        "pdf": "psychology.pdf",
        "gt": "ground_truth_psychology.xlsx",
        "video_aliases": ["psychology_MIT", "psychology"],
    },
    {
        "id": "reinforcement_learning_silver",
        "srt": "reinforcement_learning_silver.srt",
        "pdf": "reinforcement_learning.pdf",
        "gt": "ground_truth_reinforcement_learning.xlsx",
        "video_aliases": ["reinforcement_learning_silver", "reinforcement_learning", "RL_deepmind"],
    },
    {
        "id": "visual_system_MIT",
        "srt": "visual_system_MIT.srt",
        "pdf": "sensory_systems.pdf",
        "gt": "ground_truth_sensory_systems.xlsx",
        "video_aliases": ["visual_system_MIT", "visual_system", "sensory_systems"],
    },
]


def asset_dirs(root: Path, video_dir_override: str = ""):
    base = root / "data" / "MaViLS" / "data"
    if video_dir_override:
        video_dirs = [Path(video_dir_override).expanduser().resolve()]
    else:
        video_dirs = [
            root / "archive" / "video",
            root.resolve().parents[1] / "archive" / "video",
        ]
    unique = []
    seen = set()
    for d in video_dirs:
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return base / "audioscripts", base / "lectures", unique, base / "ground_truth_files"


def exact_asset(directory: Path, filename: str):
    p = directory / filename
    return p if p.exists() else None


def find_video(video_dirs, aliases):
    candidates = []
    normalized_aliases = ["".join(ch.lower() for ch in a if ch.isalnum()) for a in aliases]
    for d in video_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov"}:
                continue
            stem = "".join(ch.lower() for ch in p.stem if ch.isalnum())
            score = 0
            for alias in normalized_aliases:
                if not alias:
                    continue
                if stem == alias:
                    score = max(score, 1000)
                elif alias in stem:
                    score = max(score, 800 - abs(len(stem) - len(alias)))
            if score:
                candidates.append((score, p))
    candidates.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    if not candidates:
        return None, []
    return candidates[0][1], [p for _, p in candidates[:5]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--video-dir", default="")
    p.add_argument("--sample-seconds", type=float, default=1.0)
    p.add_argument("--max-jump", type=int, default=3)
    p.add_argument("--jump-penalty", type=float, default=0.015)
    p.add_argument("--out-dir", default="runs/visual_first_validation")
    p.add_argument("--only", default="", help="Run one validation lecture id as an asset smoke test")
    args = p.parse_args()

    root = Path(args.root).resolve()
    audio_dir, lecture_dir, video_dirs, gt_dir = asset_dirs(root, args.video_dir)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LECTURESENSE — VISUAL-FIRST VALIDATION (NO TEST DATA)")
    print("=" * 78)
    print(f"Project root: {root}")
    print("Video search directories:")
    for d in video_dirs:
        print(f"  {d} | {'FOUND' if d.exists() else 'MISSING'}")

    items = VALIDATION
    if args.only:
        items = [x for x in VALIDATION if x["id"] == args.only]
        if not items:
            raise SystemExit(f"Unknown validation lecture: {args.only}")

    total_correct = 0
    total_n = 0

    for k, item in enumerate(items, 1):
        print("\n" + "-" * 78)
        print(f"[{k}/{len(items)}] {item['id']}")
        srt = exact_asset(audio_dir, item["srt"])
        pdf = exact_asset(lecture_dir, item["pdf"])
        gt = exact_asset(gt_dir, item["gt"])
        video, top_candidates = find_video(video_dirs, item["video_aliases"])

        print(f"SRT:   {srt}")
        print(f"PDF:   {pdf}")
        print(f"GT:    {gt}")
        print(f"Video: {video}")
        if top_candidates:
            print("Video candidates considered:")
            for c in top_candidates:
                print(f"  {c}")

        missing = []
        if srt is None: missing.append("SRT")
        if pdf is None: missing.append("PDF")
        if gt is None: missing.append("GT")
        if video is None: missing.append("VIDEO")
        if missing:
            print("Missing required asset(s): " + ", ".join(missing))
            continue

        start = time.time()
        timeline, scores, times, duration = run_visual_first(
            video, pdf,
            sample_every=args.sample_seconds,
            max_jump=args.max_jump,
            jump_penalty=args.jump_penalty,
            progress=True,
        )
        gt_rows = load_ground_truth(gt)
        correct, total, acc, errors = evaluate_timeline(timeline, gt_rows)
        out = out_dir / f"{item['id']}_timeline.json"
        save_timeline(out, timeline, {
            "sample_seconds": args.sample_seconds,
            "max_jump": args.max_jump,
            "jump_penalty": args.jump_penalty,
            "video_dir": str(video.parent),
            "pdf": str(pdf),
            "ground_truth": str(gt),
        })
        print(f"Result: {correct}/{total} = {acc:.4%}")
        print(f"Timeline entries: {len(timeline)}")
        print(f"Runtime: {time.time() - start:.1f}s")
        if errors:
            print("First errors:")
            for e in errors[:10]:
                pred = "None" if e[2] is None else str(e[2])
                print(f"  {e[0]:.2f}s | GT {e[1]} | Pred {pred}")
        total_correct += correct
        total_n += total

    print("\n" + "=" * 78)
    print("VISUAL-FIRST VALIDATION SUMMARY")
    print("=" * 78)
    agg = total_correct / total_n if total_n else 0.0
    print(f"Aggregate: {total_correct}/{total_n} = {agg:.4%}")
    if args.only:
        print(f"Single-lecture smoke test: {args.only}")
    print("Test lectures were not loaded.")


if __name__ == "__main__":
    main()
