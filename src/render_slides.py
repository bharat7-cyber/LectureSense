from pathlib import Path
import fitz

ROOT = Path("data/MaViLS")
PDF_DIR = ROOT / "data/lectures"
OUT_DIR = ROOT / "slide_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf in PDF_DIR.glob("*.pdf"):
    lecture = pdf.stem
    lecture_dir = OUT_DIR / lecture
    lecture_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf)

    for i, page in enumerate(doc):
        out = lecture_dir / f"slide_{i+1}.png"

        if out.exists():
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pix.save(out)

    print(f"[OK] {lecture}: {len(doc)} slides")

print("Finished rendering slides.")
