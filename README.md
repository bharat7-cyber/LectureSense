# LectureSense

**Deep-learning-based lecture video-to-slide alignment and study workspace**

LectureSense is a research prototype that aligns spoken lecture segments with the corresponding slides in a PDF deck. It combines language, slide text, slide images, temporal context, and optional video-frame evidence to produce a timestamped slide timeline that can be used for lecture navigation and revision.

## Problem

Recorded lectures often contain a video, an audio track, and a separate PDF slide deck. Finding the exact time when a particular slide or concept is discussed can require manually scrubbing through a long recording. LectureSense aims to automate that alignment and make lecture recordings easier to navigate and review.

## Key features

- Streamlit-based study workspace for uploading a lecture PDF and MP4 recording.
- Speech transcription using Faster-Whisper (transcription is run when a compatible installation is available).
- PDF slide rendering and OCR-based slide text extraction.
- Sentence-BERT text embeddings using `all-MiniLM-L6-v2`.
- ResNet-18 image embeddings for slide visuals.
- Cross-modal ranking that relates transcript text, slide OCR, and slide images.
- V4 temporal model based on a bidirectional GRU, attention, a boundary head, and a next-slide head.
- Optional lecture-video frame embeddings and video-to-slide visual matching.
- Boundary-aware dynamic-programming decoding with a monotonic slide path.
- JSON timeline output and rendered slide images for downstream UI use.
- Extractive study notes and keywords derived from aligned transcript/OCR content.

## Architecture

```text
Lecture MP4 ──> Faster-Whisper ──> Timestamped transcript
                                        │
PDF slide deck ──> Render slides ──> OCR text + slide images
                                        │
Transcript + OCR + images ──> Sentence-BERT / ResNet-18 features
                                        │
                               Cross-modal score matrix
                                        │
                         V4 temporal model + video evidence
                                        │
                       Boundary-aware dynamic programming
                                        │
                      timeline.json + rendered slide images
                                        │
                         Streamlit study workspace
```

## Technology stack

- **Python** for the inference and data-processing pipeline.
- **PyTorch** for the trainable temporal model.
- **Sentence-Transformers** for semantic text embeddings.
- **Torchvision / ResNet-18** for slide and video-frame image features.
- **EasyOCR** for extracting text from rendered slides.
- **PyMuPDF** for rendering PDF slides.
- **OpenCV** for video-frame processing.
- **Faster-Whisper** for audio transcription.
- **Streamlit** for the user interface.
- **NumPy and Pandas** for numerical processing and benchmark utilities.

## Repository layout

```text
.
├── app.py                         # Streamlit study workspace
├── src/
│   ├── inference.py                # Base cross-modal inference pipeline
│   ├── inference_temporal_v4.py    # V4 temporal inference entry point
│   ├── boundary_dp_v4.py           # Boundary-aware dynamic-programming decoder
│   ├── model.py                    # Core model definitions
│   ├── video_visual_alignment.py   # Video-frame / slide visual matching
│   └── ...                         # Training, evaluation, and experiment utilities
├── train_temporal_v4.py            # V4 training script
├── checkpoints/                    # Local model checkpoints (not committed by default)
├── data/                            # Local dataset files (not committed)
├── runs/                            # Inference outputs and cached matrices (not committed)
└── README.md
```

## Setup

Python 3.11 is recommended for the current development environment. A CUDA-capable NVIDIA GPU can accelerate inference, but a compatible CPU setup may also work more slowly. Install a PyTorch build compatible with your operating system and desired CUDA version before installing the remaining dependencies.

```powershell
# From the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install a compatible PyTorch/torchvision build using the official selector:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

The MaViLS benchmark assets and trained model checkpoints are intentionally not included in the repository by default. Place the required dataset files and checkpoint files in the expected local paths before running benchmark training/evaluation scripts. Follow the dataset's own access and usage terms.

## Run the Streamlit app

```powershell
streamlit run app.py
```

The app's inference target is configured near the top of `app.py` through `_INFERENCE_SCRIPT` and `_CHECKPOINT_PATH`. Check these settings before a demo and ensure that the selected inference script matches the checkpoint you have locally.

## Run V4 inference from the command line

With a transcript in SRT format, a PDF slide deck, an MP4 lecture video, and the V4 checkpoint available locally, run this from the repository root:

```powershell
python src\inference_temporal_v4.py `
  --transcript "path\to\lecture.srt" `
  --slides "path\to\slides.pdf" `
  --video "path\to\lecture.mp4" `
  --checkpoint "checkpoints\temporal_v4_best_model.pt" `
  --out "runs\demo\timeline.json" `
  --render-dir "runs\demo\rendered_slides" `
  --jump-penalty 0.05 `
  --max-duration 120 `
  --duration-penalty 0.003 `
  --video-weight 0.4 `
  --video-sample-seconds 1.0
```

The output JSON contains slide IDs and aligned transcript segments with start/end timestamps. The rendered slide images are saved in the render directory. Runtime depends on lecture length, video sampling rate, hardware, and whether model weights have already been cached.

## Training and evaluation

Training and evaluation utilities are located in the repository root and `src/`. Check each script's `--help` output and configuration paths before running it. Use the official train/validation/test split for reproducible evaluation. Tune models and decoder parameters on validation data only; do not use held-out test labels for training, feature selection, or hyperparameter tuning.

### V4 baseline benchmark result

A previous held-out test evaluation of the V4 pipeline reported the following per-lecture segment alignment accuracy:

| Held-out lecture | Correct / evaluated | Accuracy |
|---|---:|---:|
| Short Range MIT | 174 / 350 | 49.71% |
| Solar Resource | 345 / 518 | 66.60% |
| Team Dynamics / Game Design MIT | 34 / 55 | 61.82% |
| Theory of Computation | 390 / 554 | 70.40% |
| **Aggregate (micro average)** | **943 / 1,477** | **63.85%** |
| **Macro average across lectures** | — | **62.13%** |

These are historical baseline results for that evaluation setup, not a guarantee for every lecture or configuration. Decoder and model improvements are experimental; report new numbers only after running the prescribed evaluation cleanly.

## Output format

The pipeline saves a JSON result containing metadata and a `timeline` array. Each timeline entry typically includes a slide identifier/index, start and end time, transcript text, and display title. The UI groups these segments by slide to present slide-specific transcript content, keywords, and extractive key points.

## Current limitations

- Alignment quality varies between lectures and subject areas.
- Correct matching depends on the PDF deck corresponding to the recording and on readable slide content.
- Automatic transcription can introduce errors, particularly with noisy audio, accents, or technical vocabulary.
- Large video files, datasets, cached feature matrices, and model checkpoints are not included by default.
- The current codebase contains multiple research/experimental versions; check script names and checkpoint compatibility before running them.

## Reproducibility and responsible use

- Keep raw datasets, lecture videos, generated caches, and local checkpoints out of Git unless you have confirmed their size and redistribution permissions.
- Do not commit API keys, credentials, private files, or local environment settings.
- Keep the held-out test split untouched until final evaluation.
- Cite the MaViLS dataset and the source models/libraries in academic or public presentations, and follow their licenses and usage requirements.

## License

No project license has been specified yet. Unless a license is added, reuse and redistribution of this repository's code are not automatically granted. Dataset assets and third-party model/library licenses are separate.
