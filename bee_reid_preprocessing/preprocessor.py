"""Standalone crop preprocessing for bee ReID pipelines.

The API is intentionally independent of the tracker and ReID model. It accepts
OpenCV BGR crops and returns OpenCV BGR crops, so another codebase can insert it
between detection cropping and feature extraction.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from bee_orientation_normalizer.orientation import (
    align_long_axis,
    classify_head_side,
    infer_head_side,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOREGROUND_CHECKPOINT = (
    REPO_ROOT / "models" / "bee_foreground_unetpp_resnet18_v2" / "best_model.pt"
)
DEFAULT_ORIENTATION_MODEL = REPO_ROOT / "models" / "bee_direction_resnet18_triclass.joblib"

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PREPROCESS_MODES = {"none", "foreground", "foreground_aligned", "raw_aligned"}


@dataclass
class BeeCropPreprocessConfig:
    mode: str = "raw_aligned"
    foreground_checkpoint: Path | str = DEFAULT_FOREGROUND_CHECKPOINT
    foreground_device: str = "cuda"
    foreground_threshold: float = 0.50
    foreground_min_area: int = 8
    foreground_close_ratio: float = 0.012
    foreground_feather_ratio: float = 0.012
    background_value: int = 0
    orientation_model: Path | str | None = DEFAULT_ORIENTATION_MODEL
    orientation_target: str = "head-left"
    stats: dict[str, float | int] = field(default_factory=dict)


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.size == 0:
        return mask_u8.astype(bool)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return np.logical_or(mask_u8 > 0, holes)


def keep_near_main_components(
    mask: np.ndarray,
    radius_frac: float = 0.14,
    min_area: int = 2,
) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return mask_u8.astype(bool)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n_labels <= 1:
        return mask_u8.astype(bool)

    height, width = mask_u8.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    best_label = 1
    best_score = -1.0
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        x, y = centroids[label]
        dist = math.hypot(x - cx, y - cy) / max(1.0, min(height, width))
        score = area / (1.0 + 1.8 * dist)
        if score > best_score:
            best_label = label
            best_score = score

    main = labels == best_label
    dist_map = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = max(3, int(round(min(height, width) * radius_frac)))
    out = main.copy()
    for label in range(1, n_labels):
        if label == best_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        if float(dist_map[comp].min()) <= max_dist:
            out |= comp
    return out


def clean_bee_foreground_prediction(
    mask: np.ndarray,
    min_area: int = 8,
    close_ratio: float = 0.012,
) -> np.ndarray:
    if int(mask.sum()) < int(min_area):
        return mask.astype(bool)
    height, width = mask.shape[:2]
    close_size = max(3, int(round(min(height, width) * float(close_ratio))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    out = fill_mask_holes(out)
    out = keep_near_main_components(out, radius_frac=0.15, min_area=max(2, int(min_area) // 10))
    return out


def bee_foreground_soft_alpha(mask: np.ndarray, feather_ratio: float = 0.012) -> np.ndarray:
    height, width = mask.shape[:2]
    sigma = max(0.6, min(height, width) * float(feather_ratio))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    alpha[mask] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def apply_alpha(crop: np.ndarray, alpha: np.ndarray, background_value: int = 0) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    background = np.full_like(crop, int(background_value))
    out = crop.astype(np.float32) * alpha_3 + background.astype(np.float32) * (1.0 - alpha_3)
    return np.clip(out, 0, 255).astype(np.uint8)


class ForegroundSegmenter:
    """Unet++ foreground model wrapper for BGR crop batches."""

    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_FOREGROUND_CHECKPOINT,
        device: str = "cuda",
        threshold: float = 0.50,
        min_area: int = 8,
        close_ratio: float = 0.012,
        feather_ratio: float = 0.012,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device_request = str(device)
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.close_ratio = float(close_ratio)
        self.feather_ratio = float(feather_ratio)
        self.model = None
        self.torch = None
        self.device = None
        self.image_size = None
        self.mean = None
        self.std = None
        self.encoder = None
        self.stats = {
            "foreground_model_batches": 0,
            "foreground_model_calls": 0,
            "foreground_model_seconds": 0.0,
            "foreground_empty_masks": 0,
        }

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "torch and segmentation-models-pytorch are required for bee foreground preprocessing"
            ) from exc

        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"bee foreground checkpoint not found: {self.checkpoint}")

        requested = self.device_request
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        device = torch.device(requested)
        checkpoint = torch.load(str(self.checkpoint), map_location=device)
        self.encoder = checkpoint.get("encoder", "resnet18")
        self.image_size = int(checkpoint.get("image_size", 256))
        self.mean = np.array(checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(checkpoint.get("imagenet_std", [0.229, 0.224, 0.225]), dtype=np.float32)

        model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        self.torch = torch
        self.device = device
        self.model = model

    def _preprocess(self, crop: np.ndarray):
        rgb = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return self.torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _mask_alpha_from_probability(self, prob_small: np.ndarray, crop: np.ndarray):
        height, width = crop.shape[:2]
        confidence = cv2.resize(prob_small, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = confidence >= self.threshold
        mask = clean_bee_foreground_prediction(
            mask,
            min_area=self.min_area,
            close_ratio=self.close_ratio,
        )
        if int(mask.sum()) == 0:
            self.stats["foreground_empty_masks"] += 1
        alpha = bee_foreground_soft_alpha(mask, self.feather_ratio)
        return mask.astype(bool), alpha

    def infer_masks_and_alphas(self, crops: Sequence[np.ndarray]):
        if not crops:
            return []

        self._load_model()
        valid_pairs = []
        outputs = [None] * len(crops)
        for idx, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                if crop is None or crop.size == 0:
                    outputs[idx] = (np.zeros((0, 0), dtype=bool), np.zeros((0, 0), dtype=np.float32))
                else:
                    outputs[idx] = (
                        np.ones(crop.shape[:2], dtype=bool),
                        np.ones(crop.shape[:2], dtype=np.float32),
                    )
                continue
            valid_pairs.append((idx, crop))

        if valid_pairs:
            tensors = [self._preprocess(crop) for _, crop in valid_pairs]
            tensor = self.torch.cat(tensors, dim=0)
            t0 = time.time()
            with self.torch.no_grad():
                logits = self.model(tensor)
                probs = self.torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
            self.stats["foreground_model_batches"] += 1
            self.stats["foreground_model_calls"] += len(valid_pairs)
            self.stats["foreground_model_seconds"] += time.time() - t0
            for prob_small, (idx, crop) in zip(probs, valid_pairs):
                outputs[idx] = self._mask_alpha_from_probability(prob_small, crop)

        return outputs


class BeeReIDCropPreprocessor:
    """Reusable crop preprocessor for foreground removal and orientation."""

    def __init__(self, config: BeeCropPreprocessConfig | None = None, **kwargs) -> None:
        if config is None:
            config = BeeCropPreprocessConfig(**kwargs)
        elif kwargs:
            raise ValueError("pass either config or keyword arguments, not both")
        if config.mode not in PREPROCESS_MODES:
            raise ValueError(f"unsupported mode {config.mode!r}; expected one of {sorted(PREPROCESS_MODES)}")
        if config.orientation_target not in {"head-left", "head-right"}:
            raise ValueError("orientation_target must be 'head-left' or 'head-right'")

        self.config = config
        self.foreground = None
        self.direction_model = None
        self.stats = {
            "processed": 0,
            "failures": 0,
            "orientation_processed": 0,
            "orientation_flipped_180": 0,
            "orientation_left": 0,
            "orientation_unclear": 0,
            "orientation_failures": 0,
        }

    def _foreground_segmenter(self) -> ForegroundSegmenter:
        if self.foreground is None:
            cfg = self.config
            self.foreground = ForegroundSegmenter(
                checkpoint=cfg.foreground_checkpoint,
                device=cfg.foreground_device,
                threshold=cfg.foreground_threshold,
                min_area=cfg.foreground_min_area,
                close_ratio=cfg.foreground_close_ratio,
                feather_ratio=cfg.foreground_feather_ratio,
            )
        return self.foreground

    def _load_direction_model(self):
        if self.direction_model is not None:
            return self.direction_model
        model_path = self.config.orientation_model
        if not model_path:
            return None
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"bee orientation model not found: {model_path}")
        import joblib

        self.direction_model = joblib.load(model_path)
        return self.direction_model

    def _orient_with_mask(
        self,
        image: np.ndarray,
        label_mask: np.ndarray,
        analysis_image: np.ndarray | None = None,
    ) -> np.ndarray:
        self.stats["orientation_processed"] += 1
        try:
            mask = (label_mask > 0).astype(np.uint8) * 255
            aligned, aligned_mask, _ = align_long_axis(image, mask, crop_output=False)
            if analysis_image is None:
                classifier_image = apply_alpha(aligned, (aligned_mask > 0).astype(np.float32), 0)
                classifier_mask = aligned_mask
            else:
                classifier_image, classifier_mask, _ = align_long_axis(
                    analysis_image,
                    mask,
                    crop_output=False,
                )

            direction_model = self._load_direction_model()
            if direction_model is None:
                head_side, _, _ = infer_head_side(classifier_image, classifier_mask)
            else:
                head_side, _, _ = classify_head_side(
                    direction_model,
                    classifier_image,
                    classifier_mask,
                )

            target_side = "left" if self.config.orientation_target == "head-left" else "right"
            rotated_180 = False if head_side in {"unclear", "unknown"} else head_side != target_side
            if rotated_180:
                aligned = cv2.rotate(aligned, cv2.ROTATE_180)
            self.stats["orientation_flipped_180"] += int(rotated_180)
            self.stats["orientation_left"] += int(head_side == "left")
            self.stats["orientation_unclear"] += int(head_side == "unclear")
            return aligned
        except Exception:
            self.stats["orientation_failures"] += 1
            return image

    def process(self, crop: np.ndarray) -> np.ndarray:
        return self.process_batch([crop])[0]

    def process_batch(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        mode = self.config.mode
        outputs: list[np.ndarray] = []
        if mode == "none":
            self.stats["processed"] += len(crops)
            return [crop for crop in crops]

        masks_alphas = self._foreground_segmenter().infer_masks_and_alphas(crops)
        for crop, mask_alpha in zip(crops, masks_alphas):
            self.stats["processed"] += 1
            if crop is None or crop.size == 0:
                self.stats["failures"] += 1
                outputs.append(crop)
                continue

            label_mask, alpha = mask_alpha
            if label_mask.size == 0 or alpha.size == 0:
                self.stats["failures"] += 1
                outputs.append(crop)
                continue

            foreground = apply_alpha(crop, alpha, self.config.background_value)
            if mode == "foreground":
                outputs.append(foreground)
            elif mode == "foreground_aligned":
                outputs.append(self._orient_with_mask(foreground, label_mask, analysis_image=foreground))
            elif mode == "raw_aligned":
                outputs.append(self._orient_with_mask(crop, label_mask, analysis_image=foreground))
            else:
                raise ValueError(f"unsupported mode: {mode}")

        if self.foreground is not None:
            self.stats.update(self.foreground.stats)
        self.config.stats.update(self.stats)
        return outputs
