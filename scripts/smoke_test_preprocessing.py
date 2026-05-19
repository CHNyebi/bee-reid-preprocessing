#!/usr/bin/env python3
"""Smoke test the portable preprocessing package and bundled model files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bee_reid_preprocessing import BeeReIDCropPreprocessor
from bee_reid_preprocessing.preprocessor import (
    DEFAULT_FOREGROUND_CHECKPOINT,
    DEFAULT_ORIENTATION_MODEL,
    read_image,
)


def find_sample(repo_root: Path) -> Path:
    candidates = sorted((repo_root / "data" / "bee_foreground_v2" / "dataset" / "val" / "images").glob("*"))
    for path in candidates:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return path
    raise FileNotFoundError("no validation image found under data/bee_foreground_v2/dataset/val/images")


def assert_image(name: str, image: np.ndarray, expected_shape: tuple[int, int, int] | None = None) -> None:
    if image is None or image.size == 0:
        raise AssertionError(f"{name} returned an empty image")
    if image.dtype != np.uint8:
        raise AssertionError(f"{name} returned dtype {image.dtype}, expected uint8")
    if expected_shape is not None and image.shape != expected_shape:
        raise AssertionError(f"{name} returned shape {image.shape}, expected {expected_shape}")
    if expected_shape is None and (image.ndim != 3 or image.shape[2] != 3):
        raise AssertionError(f"{name} returned shape {image.shape}, expected HxWx3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--sample", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    foreground_checkpoint = DEFAULT_FOREGROUND_CHECKPOINT
    orientation_model = DEFAULT_ORIENTATION_MODEL
    if not foreground_checkpoint.is_file():
        raise FileNotFoundError(f"foreground checkpoint missing: {foreground_checkpoint}")
    if not orientation_model.is_file():
        raise FileNotFoundError(f"orientation model missing: {orientation_model}")

    sample_path = args.sample.resolve() if args.sample else find_sample(repo_root)
    crop = read_image(sample_path)
    if crop is None or crop.size == 0:
        raise RuntimeError(f"failed to read sample image: {sample_path}")
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    none_pre = BeeReIDCropPreprocessor(mode="none")
    none_out = none_pre.process(crop)
    assert_image("none", none_out, crop.shape)

    foreground_pre = BeeReIDCropPreprocessor(mode="foreground", foreground_device=args.device)
    foreground_out = foreground_pre.process(crop)
    assert_image("foreground", foreground_out, crop.shape)

    aligned_pre = BeeReIDCropPreprocessor(mode="raw_aligned", foreground_device=args.device)
    aligned_out = aligned_pre.process(crop)
    assert_image("raw_aligned", aligned_out)

    print(
        {
            "status": "ok",
            "sample": str(sample_path),
            "shape": crop.shape,
            "foreground_stats": foreground_pre.stats,
            "aligned_stats": aligned_pre.stats,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
