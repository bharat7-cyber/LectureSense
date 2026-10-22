from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SLIDE_DIR = ROOT / "data" / "MaViLS" / "slide_images"
OUT_DIR = ROOT / "data"

OCR_TEXT_FILE = OUT_DIR / "ocr_text.csv"
OCR_EMBED_FILE = OUT_DIR / "ocr_embeddings.npy"
OCR_INDEX_FILE = OUT_DIR / "ocr_embedding_index.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TESSERACT_CONFIG = "--oem 3 --psm 6"


# ============================================================
# TESSERACT
# ============================================================

# Explicit path so the script works even if Windows PATH
# has not been refreshed.
TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
else:
    print("ERROR: Tesseract executable not found:")
    print(TESSERACT_EXE)
    sys.exit(1)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare slide image for OCR.

    We keep preprocessing conservative because the original
    slide images are already high quality.
    """

    image = image.convert("RGB")

    # Convert to grayscale
    gray = ImageOps.grayscale(image)

    # Increase contrast
    gray = ImageOps.autocontrast(gray)

    return gray


# ============================================================
# OCR
# ============================================================

def extract_ocr(image_path):
    """
    Extract text from one slide image.
    """

    try:
        image = Image.open(image_path)

        processed = preprocess_image(image)

        text = pytesseract.image_to_string(
            processed,
            config=TESSERACT_CONFIG
        )

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    except Exception as e:
        print(f"\nOCR failed: {image_path}")
        print("Error:", e)
        return ""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LectureSense OCR Feature Extraction")
    print("=" * 70)

    print("\nRoot:", ROOT)
    print("Slide directory:", SLIDE_DIR)

    if not SLIDE_DIR.exists():
        print("\nERROR: Slide directory does not exist.")
        sys.exit(1)

    slide_files = sorted(SLIDE_DIR.rglob("*.png"))

    print("\nSlide images found:", len(slide_files))

    if len(slide_files) == 0:
        print("ERROR: No PNG slide images found.")
        sys.exit(1)

    # --------------------------------------------------------
    # OCR extraction
    # --------------------------------------------------------

    print("\nStarting OCR extraction...")

    records = []

    for image_path in tqdm(slide_files, desc="OCR"):

        # Lecture folder name
        lecture_id = image_path.parent.name

        # Slide filename
        slide_name = image_path.name

        # Extract slide number
        try:
            slide_id = int(
                image_path.stem.split("_")[-1]
            )
        except ValueError:
            slide_id = -1

        text = extract_ocr(image_path)

        records.append({
            "lecture_id": lecture_id,
            "slide_id": slide_id,
            "image_path": str(
                image_path.relative_to(ROOT)
            ),
            "ocr_text": text,
            "ocr_characters": len(text)
        })

    ocr_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save OCR text
    # --------------------------------------------------------

    ocr_df.to_csv(
        OCR_TEXT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nSaved OCR text:")
    print(OCR_TEXT_FILE)

    # --------------------------------------------------------
    # OCR statistics
    # --------------------------------------------------------

    non_empty = (
        ocr_df["ocr_text"].str.strip().str.len() > 0
    ).sum()

    empty = len(ocr_df) - non_empty

    print("\nOCR statistics:")
    print("Total slides:", len(ocr_df))
    print("Slides with text:", non_empty)
    print("Slides with no text:", empty)

    print(
        "Average characters:",
        round(
            ocr_df["ocr_characters"].mean(),
            2
        )
    )

    # --------------------------------------------------------
    # Load Sentence-BERT
    # --------------------------------------------------------

    print("\nLoading Sentence-BERT...")

    encoder = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Prepare OCR text
    # --------------------------------------------------------

    texts = []

    for text in ocr_df["ocr_text"]:

        text = str(text).strip()

        # Avoid completely empty input.
        if not text:
            text = "[NO OCR TEXT]"

        texts.append(text)

    print(
        f"\nEncoding {len(texts)} slide OCR texts..."
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print("\nEmbedding shape:", embeddings.shape)

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        OCR_EMBED_FILE,
        embeddings
    )

    print("Saved:")
    print(OCR_EMBED_FILE)

    # --------------------------------------------------------
    # Save embedding index
    # --------------------------------------------------------

    index_df = ocr_df[
        [
            "lecture_id",
            "slide_id",
            "image_path"
        ]
    ].copy()

    index_df["embedding_row"] = np.arange(
        len(index_df)
    )

    index_df.to_csv(
        OCR_INDEX_FILE,
        index=False,
        encoding="utf-8"
    )

    print("Saved:")
    print(OCR_INDEX_FILE)

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OCR EXTRACTION SUCCESSFUL")
    print("=" * 70)

    print("\nFiles created:")

    print("1.", OCR_TEXT_FILE)
    print("2.", OCR_EMBED_FILE)
    print("3.", OCR_INDEX_FILE)

    print("\nEmbedding dimensions:")
    print(embeddings.shape)

    print("\nFirst 5 OCR samples:")

    for _, row in ocr_df.head(5).iterrows():

        print("\n---")
        print("Lecture:", row["lecture_id"])
        print("Slide:", row["slide_id"])
        print("Text:")
        print(row["ocr_text"][:500])


if __name__ == "__main__":
    main()