from importlib.util import find_spec

packages = ["cv2", "fitz", "numpy", "pandas", "torch", "sentence_transformers", "easyocr"]
print("="*70)
print("LECTURESENSE HYBRID DEPENDENCY CHECK")
print("="*70)
for p in packages:
    print(f"{p:22s} : {'FOUND' if find_spec(p) else 'MISSING'}")
print("="*70)
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check error:", e)

try:
    import easyocr
    print("EasyOCR import: OK")
except Exception as e:
    print("EasyOCR import error:", e)
