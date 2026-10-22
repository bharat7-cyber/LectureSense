"""Video-frame visual alignment utilities for LectureSense.

This module samples frames from the lecture video, extracts ResNet-18
features in the same 512-D feature space used for PDF slide images, and
builds a transcript-segment x slide visual similarity matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_DIM = 512


def _load_resnet18_feature_extractor(device: torch.device):
    """Load the same ImageNet ResNet-18 feature extractor used for slides."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model, weights.transforms()


def _sample_video_frames(
    video_path: str | Path,
    sample_every_seconds: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Read video sequentially and sample approximately every N seconds."""
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    sample_every_seconds = max(0.25, float(sample_every_seconds))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0

    print(f"Video FPS: {fps:.3f}")
    if duration > 0:
        print(f"Video duration: {duration:.1f}s")
    print(
        f"Sampling video every {sample_every_seconds:.2f}s..."
    )

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    next_sample_time = 0.0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if fps > 0:
            current_time = frame_index / fps
        else:
            current_time = float(
                capture.get(cv2.CAP_PROP_POS_MSEC)
            ) / 1000.0

        if current_time + 1e-6 >= next_sample_time:
            frames.append(frame)
            timestamps.append(float(current_time))
            next_sample_time += sample_every_seconds

        frame_index += 1

    capture.release()

    if not frames:
        raise RuntimeError(
            "No video frames were sampled. Check the video file."
        )

    print(f"Sampled video frames: {len(frames)}")

    return frames, np.asarray(timestamps, dtype=np.float32), duration


def create_video_frame_embeddings(
    video_path: str | Path,
    device: torch.device,
    sample_every_seconds: float = 1.0,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract 512-D ResNet-18 features from sampled video frames."""
    frames, timestamps, duration = _sample_video_frames(
        video_path,
        sample_every_seconds=sample_every_seconds,
    )

    model, preprocess = _load_resnet18_feature_extractor(device)

    features: list[torch.Tensor] = []
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        total = len(frames)

        for start in range(0, total, batch_size):
            batch_frames = frames[start : start + batch_size]
            tensors = []

            for frame in batch_frames:
                # OpenCV is BGR; PIL expects RGB.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                tensors.append(preprocess(image))

            batch = torch.stack(tensors).to(device)
            batch_features = model(batch)
            features.append(batch_features.cpu())

            processed = min(start + batch_size, total)
            if processed == total or processed % 64 == 0:
                print(
                    f"Video embeddings: {processed}/{total}"
                )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = (
        torch.cat(features, dim=0)
        .numpy()
        .astype(np.float32)
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != IMAGE_DIM:
        raise RuntimeError(
            f"Unexpected video embedding shape: {embeddings.shape}"
        )

    return embeddings, timestamps, duration


def build_video_slide_score_matrix(
    video_embeddings: np.ndarray,
    video_timestamps: np.ndarray,
    slide_embeddings: np.ndarray,
    segment_starts: Sequence[float],
    segment_ends: Sequence[float],
) -> np.ndarray:
    """Build transcript-segment x slide cosine similarity from video frames.

    For each transcript segment, all sampled video frames whose timestamps
    fall inside that segment are averaged. If a segment has no sampled frame,
    the nearest frame to its midpoint is used.
    """
    video_embeddings = np.asarray(video_embeddings, dtype=np.float32)
    video_timestamps = np.asarray(video_timestamps, dtype=np.float32)
    slide_embeddings = np.asarray(slide_embeddings, dtype=np.float32)

    if video_embeddings.ndim != 2:
        raise ValueError("video_embeddings must be 2-D")
    if slide_embeddings.ndim != 2:
        raise ValueError("slide_embeddings must be 2-D")
    if video_embeddings.shape[1] != slide_embeddings.shape[1]:
        raise ValueError(
            "Video and slide embeddings must have the same feature dimension"
        )

    if len(video_embeddings) != len(video_timestamps):
        raise ValueError(
            "video_embeddings and video_timestamps must have equal length"
        )

    frame_norm = video_embeddings / (
        np.linalg.norm(video_embeddings, axis=1, keepdims=True) + 1e-8
    )
    slide_norm = slide_embeddings / (
        np.linalg.norm(slide_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # One cosine score per sampled video frame and slide.
    frame_slide_scores = frame_norm @ slide_norm.T

    starts = np.asarray(segment_starts, dtype=np.float32)
    ends = np.asarray(segment_ends, dtype=np.float32)

    if len(starts) != len(ends):
        raise ValueError("segment_starts and segment_ends must have equal length")

    output = np.zeros(
        (len(starts), slide_embeddings.shape[0]),
        dtype=np.float32,
    )

    for i, (start, end) in enumerate(zip(starts, ends)):
        left = int(np.searchsorted(video_timestamps, start, side="left"))
        right = int(np.searchsorted(video_timestamps, end, side="right"))

        if right > left:
            output[i] = frame_slide_scores[left:right].mean(axis=0)
            continue

        midpoint = 0.5 * (float(start) + float(end))
        nearest = int(
            np.argmin(np.abs(video_timestamps - midpoint))
        )
        output[i] = frame_slide_scores[nearest]

    return output


def row_standardize(matrix: np.ndarray) -> np.ndarray:
    """Standardize each transcript row for safe score fusion."""
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / (std + 1e-6)


def fuse_model_and_video_scores(
    model_scores: np.ndarray,
    video_scores: np.ndarray,
    video_weight: float = 0.30,
) -> np.ndarray:
    """Fuse learned multimodal scores with actual video visual evidence."""
    model_scores = np.asarray(model_scores, dtype=np.float32)
    video_scores = np.asarray(video_scores, dtype=np.float32)

    if model_scores.shape != video_scores.shape:
        raise ValueError(
            "model_scores and video_scores must have the same shape"
        )

    weight = float(np.clip(video_weight, 0.0, 1.0))

    model_z = row_standardize(model_scores)
    video_z = row_standardize(video_scores)

    return (
        (1.0 - weight) * model_z
        + weight * video_z
    ).astype(np.float32)
