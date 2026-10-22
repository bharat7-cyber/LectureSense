"""
Clone the real MaViLS benchmark dataset and print an inventory of what it contains.

MaViLS: "MaViLS, a Benchmark Dataset for Video-to-Slide Alignment, Assessing
Baseline Accuracy with a Multimodal Alignment Algorithm Leveraging Speech,
OCR, and Visual Features" (Anderer, Reich, Wölfel, Interspeech 2024).
Repo: https://github.com/andererka/MaViLS

Run:
    python src/download_data.py
"""

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/andererka/MaViLS.git"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MaViLS"


def clone_repo():
    if DATA_DIR.exists():
        print(f"[skip] {DATA_DIR} already exists, pulling latest instead.")
        subprocess.run(["git", "-C", str(DATA_DIR), "pull"], check=True)
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clone] {REPO_URL} -> {DATA_DIR}")
    subprocess.run(["git", "clone", REPO_URL, str(DATA_DIR)], check=True)


def inventory():
    """Walk the cloned repo and report what's actually there.

    The MaViLS repo layout may change over time — this does not assume a
    fixed structure. It reports real file counts and extensions so you can
    see exactly what you're working with before writing parsing code.
    """
    if not DATA_DIR.exists():
        print("[error] data not found, run clone_repo() first", file=sys.stderr)
        return

    ext_counts = {}
    total_size = 0
    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower() or "(no extension)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += p.stat().st_size

    print("\n=== MaViLS dataset inventory ===")
    print(f"Location: {DATA_DIR}")
    print(f"Total size: {total_size / (1024**2):.1f} MB")
    print("File types found:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext:20s} {count}")

    # Surface likely candidates for ground-truth / transcript / slide files
    # so the next script (build_pairs.py) can be pointed at the right paths.
    print("\nLikely ground-truth / label files (json/csv containing 'gt' or 'label' or 'align'):")
    for p in DATA_DIR.rglob("*"):
        name = p.name.lower()
        if p.is_file() and any(k in name for k in ("gt", "ground_truth", "label", "align")):
            print(f"  {p.relative_to(DATA_DIR)}")

    print("\nLikely transcript files:")
    for p in DATA_DIR.rglob("*"):
        name = p.name.lower()
        if p.is_file() and any(k in name for k in ("transcript", "whisper", "asr")):
            print(f"  {p.relative_to(DATA_DIR)}")

    print("\nLikely slide files (pdf/pptx or slide image folders):")
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in (".pdf", ".pptx") or "slide" in p.name.lower():
            print(f"  {p.relative_to(DATA_DIR)}")

    print(
        "\nNote: paper reports 20 lectures / 22+ hours / 12,830 ground-truth "
        "points. Verify these counts against what's printed above — if they "
        "don't match, the repo structure has likely changed since this "
        "script was written and build_pairs.py's paths will need updating."
    )


if __name__ == "__main__":
    clone_repo()
    inventory()
