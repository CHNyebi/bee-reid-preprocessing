"""Bee orientation normalization utilities.

The pipeline mirrors the thrips preprocessing paper at a practical level:
build an analysis mask, align the long body axis horizontally, then decide
which horizontal end is more head-like and rotate the original image to a
target side. The analysis mask is never saved over the input pixels unless the
caller explicitly asks for background cleaning or cropping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TORCH_RESNET_DIRECTION_KIND = "torch_resnet18_direction"
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


class OrientationError(RuntimeError):
    """Raised when an image cannot be orientation-normalized."""


@dataclass(frozen=True)
class OrientationResult:
    image: np.ndarray
    mask: np.ndarray
    head_side_before: str
    target: str
    rotated_180: bool
    angle_degrees: float
    confidence: float
    features: Dict[str, float]


def normalize_bee_orientation(
    image: np.ndarray,
    target: str = "head-left",
    mask_mode: str = "masked",
    output_mode: str = "rotate",
    direction_model=None,
    end_fraction: float = 0.30,
    min_bg_distance: float = 8.0,
    crop_padding_ratio: float = 0.18,
    clean_background: bool = False,
) -> OrientationResult:
    """Rotate a bee crop so its head points to the requested side.

    Args:
        image: OpenCV image array, BGR/BGRA/RGB-like channel order is accepted.
        target: Either ``head-left`` or ``head-right``.
        mask_mode: ``masked`` for images that already have a mask applied, or
            ``auto`` for raw crops where foreground must be estimated.
        output_mode: ``rotate`` keeps the full rotated original masked image;
            ``crop`` additionally crops around the analysis mask.
        direction_model: Optional model bundle trained from corrected direction
            labels. Three-class bundles may return ``unclear``; those images
            are only long-axis aligned and are not rotated 180 degrees.
        end_fraction: Fraction of the aligned body length used for head/abdomen
            side features. The paper used 10%-50% search and adopted 30%.
        min_bg_distance: Minimum color distance from border background used
            when alpha is unavailable.
        crop_padding_ratio: Extra crop padding around the aligned foreground.
        clean_background: Fill non-foreground pixels after rotation.

    Returns:
        OrientationResult containing the normalized image and diagnostics.
    """

    if target not in {"head-left", "head-right"}:
        raise ValueError("target must be 'head-left' or 'head-right'")
    if mask_mode not in {"masked", "auto"}:
        raise ValueError("mask_mode must be 'masked' or 'auto'")
    if output_mode not in {"rotate", "crop"}:
        raise ValueError("output_mode must be 'rotate' or 'crop'")
    if image is None or image.size == 0:
        raise OrientationError("empty image")
    if image.ndim not in {2, 3}:
        raise OrientationError(f"unsupported image shape: {image.shape}")

    mask = build_foreground_mask(
        image, min_bg_distance=min_bg_distance, mode=mask_mode
    )
    aligned, aligned_mask, angle_degrees = align_long_axis(
        image,
        mask,
        crop_padding_ratio=crop_padding_ratio,
        crop_output=output_mode == "crop",
    )
    if direction_model is None:
        head_side, confidence, features = infer_head_side(
            aligned, aligned_mask, end_fraction=end_fraction
        )
    else:
        head_side, confidence, features = classify_head_side(
            direction_model, aligned, aligned_mask, end_fraction=end_fraction
        )

    target_side = "left" if target == "head-left" else "right"
    rotated_180 = False if head_side in {"unclear", "unknown"} else head_side != target_side
    if rotated_180:
        aligned = cv2.rotate(aligned, cv2.ROTATE_180)
        aligned_mask = cv2.rotate(aligned_mask, cv2.ROTATE_180)

    if clean_background:
        aligned = _fill_background(aligned, aligned_mask, _border_value(image))

    return OrientationResult(
        image=aligned,
        mask=aligned_mask,
        head_side_before=head_side,
        target=target,
        rotated_180=rotated_180,
        angle_degrees=angle_degrees,
        confidence=confidence,
        features=features,
    )


def build_foreground_mask(
    image: np.ndarray,
    min_bg_distance: float = 8.0,
    mode: str = "masked",
) -> np.ndarray:
    """Build a binary foreground mask for background-removed insect crops."""

    if mode not in {"masked", "auto"}:
        raise ValueError("mode must be 'masked' or 'auto'")

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if np.any(alpha < 250) and np.any(alpha > 8):
            mask = (alpha > 8).astype(np.uint8) * 255
            return clean_mask(mask)

    if mode == "masked":
        return clean_mask(_masked_axis_mask(image, min_bg_distance=min_bg_distance))

    candidates = []
    for builder in (_dark_body_mask, _distance_from_border_mask):
        try:
            candidate = builder(image, min_bg_distance=min_bg_distance)
            candidates.append((_score_mask(candidate), candidate))
        except OrientationError:
            continue

    if not candidates:
        raise OrientationError("could not build foreground mask")
    _, best = max(candidates, key=lambda item: item[0])
    return clean_mask(best)


def _masked_axis_mask(
    image: np.ndarray,
    min_bg_distance: float = 8.0,
) -> np.ndarray:
    """Build an analysis-only mask for images that already have a soft mask."""

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    border = _border_pixels(gray[..., None]).reshape(-1)
    bg = float(np.median(border))
    otsu_threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if bg <= 12.0:
        threshold = max(float(otsu_threshold) + 8.0, float(min_bg_distance), 12.0)
        threshold = min(threshold, max(float(gray.max()) - 4.0, 1.0))
        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255
    else:
        dist = np.abs(gray.astype(np.float32) - bg)
        threshold = max(float(otsu_threshold) * 0.35, float(min_bg_distance))
        mask = (dist > threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _distance_from_border_mask(
    image: np.ndarray,
    min_bg_distance: float = 8.0,
) -> np.ndarray:
    if image.ndim == 2:
        work = image
    else:
        work = image[:, :, :3]

    if work.ndim == 2:
        gray = work
        border = _border_pixels(gray[..., None])
        bg = float(np.median(border))
        dist = np.abs(gray.astype(np.float32) - bg)
    else:
        border = _border_pixels(work)
        bg = np.median(border, axis=0).astype(np.float32)
        dist = np.linalg.norm(work.astype(np.float32) - bg, axis=2)

    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(float(otsu_threshold) * 0.7, float(min_bg_distance))
    mask = (dist > threshold).astype(np.uint8) * 255
    return mask


def _dark_body_mask(
    image: np.ndarray,
    min_bg_distance: float = 8.0,
) -> np.ndarray:
    del min_bg_distance
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    margin = max(3, int(round(min(h, w) * 0.06)))
    if h <= margin * 2 or w <= margin * 2:
        raise OrientationError("image is too small for dark-body mask")

    roi = gray[margin : h - margin, margin : w - margin]
    median = float(np.median(roi))
    threshold = min(float(np.percentile(roi, 42)), median - 2.0, 125.0)
    if threshold <= 0:
        raise OrientationError("dark-body threshold is invalid")

    mask = (gray <= threshold).astype(np.uint8) * 255
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    mask = _erase_border_bands(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return _select_body_components(mask)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    kernel_size = max(3, int(round(min(h, w) * 0.04)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contour = largest_contour(cleaned)
    out = np.zeros_like(cleaned)
    cv2.drawContours(out, [contour], -1, 255, cv2.FILLED)
    return out


def _erase_border_bands(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape[:2]
    row_counts = (out > 0).sum(axis=1)
    col_counts = (out > 0).sum(axis=0)

    for y, count in enumerate(row_counts):
        near_border = y < 0.28 * h or y > 0.72 * h
        if near_border and count > 0.62 * w:
            out[max(0, y - 1) : min(h, y + 2), :] = 0

    for x, count in enumerate(col_counts):
        near_border = x < 0.22 * w or x > 0.78 * w
        if near_border and count > 0.62 * h:
            out[:, max(0, x - 1) : min(w, x + 2)] = 0

    return out


def _select_body_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    min_area = max(8, int(round(h * w * 0.002)))
    components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        raise OrientationError("dark-body mask has no components")

    cx, cy = w / 2.0, h / 2.0
    scored = []
    for idx in range(1, components):
        x, y, bw, bh, area = stats[idx]
        if area < min_area:
            continue
        if bw > 0.85 * w and bh < 0.25 * h:
            continue
        ccx, ccy = centroids[idx]
        dist = ((ccx - cx) / max(w, 1)) ** 2 + ((ccy - cy) / max(h, 1)) ** 2
        score = float(area) / (1.0 + 2.0 * dist)
        scored.append((score, idx))

    if not scored:
        raise OrientationError("dark-body mask has no usable components")

    scored.sort(reverse=True)
    keep = np.zeros_like(mask)
    for _, idx in scored[:4]:
        keep[labels == idx] = 255

    kernel = np.ones((3, 3), np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=2)
    return keep


def _score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    area_ratio = float((mask > 0).mean())
    if area_ratio <= 0.0:
        return -1.0

    border_pixels = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]], axis=0
    )
    border_ratio = float((border_pixels > 0).mean())
    ideal_area = 0.22
    area_score = 1.0 - min(abs(area_ratio - ideal_area) / ideal_area, 1.0)
    border_score = 1.0 - min(border_ratio * 2.0, 1.0)
    return 0.75 * area_score + 0.25 * border_score


def align_long_axis(
    image: np.ndarray,
    mask: np.ndarray,
    crop_padding_ratio: float = 0.18,
    crop_output: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotate the whole crop so the foreground body axis is horizontal."""

    largest_contour(mask)
    angle = _principal_axis_angle(mask)
    rotated, rotated_mask = _rotate_bound(image, mask, angle)

    rotated_mask = (rotated_mask > 0).astype(np.uint8) * 255
    if crop_output:
        rotated, rotated_mask = _crop_to_mask(
            rotated, rotated_mask, crop_padding_ratio=crop_padding_ratio
        )
    rotated_mask = clean_mask(rotated_mask)

    axis_angle = _principal_axis_angle(rotated_mask)
    horizontal_error = min(abs(axis_angle), abs(abs(axis_angle) - 180.0))
    vertical_error = abs(abs(axis_angle) - 90.0)
    if horizontal_error > vertical_error:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_CLOCKWISE)
        angle = (angle + 90.0) % 180.0

    rotated_mask = clean_mask(rotated_mask)
    return rotated, rotated_mask, angle


def infer_head_side(
    aligned_image: np.ndarray,
    aligned_mask: np.ndarray,
    end_fraction: float = 0.30,
) -> Tuple[str, float, Dict[str, float]]:
    """Infer whether the head-like end is on the left or right."""

    if not 0.05 <= end_fraction <= 0.50:
        raise ValueError("end_fraction must be between 0.05 and 0.50")

    mask = (aligned_mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise OrientationError("foreground mask is empty after alignment")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    profile = mask[:, x1:x2].sum(axis=0).astype(np.float32)
    length = len(profile)
    end = max(3, int(round(length * end_fraction)))
    if length < end * 2:
        end = max(2, length // 2)
    if end < 2:
        raise OrientationError("foreground body is too short for side inference")

    left_profile = profile[:end]
    right_profile = profile[-end:]

    left_area = float(left_profile.sum())
    right_area = float(right_profile.sum())
    left_mse = _quadratic_mse(left_profile)
    right_mse = _quadratic_mse(right_profile)
    left_width = float(np.median(left_profile))
    right_width = float(np.median(right_profile))
    left_brightness, right_brightness, left_texture, right_texture = _end_intensity_stats(
        aligned_image, mask, x1, x2, end
    )

    area_score = _smaller_score(left_area, right_area)
    width_score = _smaller_score(left_width, right_width)
    centroid_x = float(xs.mean())
    center_x = x1 + (x2 - x1) / 2.0
    centroid_delta = (centroid_x - center_x) / max((x2 - x1) / 2.0, 1.0)
    centroid_score = float(np.clip(0.5 + 0.5 * centroid_delta, 0.0, 1.0))
    brightness_score = _smaller_score(left_brightness, right_brightness)
    texture_score = _smaller_score(left_texture, right_texture)
    mse_score = _larger_score(left_mse, right_mse)

    # Bees usually have a larger abdomen than head. For masked crops, the most
    # stable cues are end area/width and the whole-body centroid skew. Brightness
    # and texture help when abdominal bands are visible, but they are weaker.
    left_probability = (
        0.30 * area_score
        + 0.25 * width_score
        + 0.25 * centroid_score
        + 0.10 * brightness_score
        + 0.10 * texture_score
    )
    head_side = "left" if left_probability >= 0.5 else "right"
    confidence = float(max(left_probability, 1.0 - left_probability))

    features = {
        "left_area": left_area,
        "right_area": right_area,
        "left_mse": left_mse,
        "right_mse": right_mse,
        "left_width": left_width,
        "right_width": right_width,
        "left_brightness": left_brightness,
        "right_brightness": right_brightness,
        "left_texture": left_texture,
        "right_texture": right_texture,
        "centroid_score": centroid_score,
        "mse_score": mse_score,
        "left_probability": float(left_probability),
        "end_fraction": float(end_fraction),
    }
    return head_side, confidence, features


def classify_head_side(
    model_bundle,
    aligned_image: np.ndarray,
    aligned_mask: np.ndarray,
    end_fraction: float = 0.30,
) -> Tuple[str, float, Dict[str, float]]:
    """Predict head side with a trained classifier bundle."""

    if isinstance(model_bundle, dict) and model_bundle.get("model_kind") == TORCH_RESNET_DIRECTION_KIND:
        return _classify_head_side_torch(model_bundle, aligned_image, aligned_mask)

    features = extract_direction_features(
        aligned_image, aligned_mask, end_fraction=end_fraction
    )
    estimator = model_bundle.get("estimator", model_bundle)
    feature_names = model_bundle.get("feature_names")
    if not feature_names:
        feature_names = sorted(features.keys())

    x = np.array([[features[name] for name in feature_names]], dtype=np.float32)
    label = estimator.predict(x)[0]
    label = str(label)
    if label not in {"head_left", "head_right", "left", "right", "unclear", "unknown"}:
        raise OrientationError(f"unsupported direction label from model: {label}")

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)[0]
        confidence = float(np.max(proba))
    elif hasattr(estimator, "decision_function"):
        score = np.ravel(estimator.decision_function(x))[0]
        confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    if label in {"unclear", "unknown"}:
        head_side = "unclear"
    else:
        head_side = "left" if label in {"head_left", "left"} else "right"
    threshold = float(model_bundle.get("confidence_threshold", 0.0) or 0.0)
    if threshold > 0.0 and confidence < threshold:
        head_side = "unclear"
    features["model_label"] = 1.0 if head_side == "left" else 0.0 if head_side == "right" else -1.0
    features["confidence_threshold"] = threshold
    return head_side, confidence, features


def _classify_head_side_torch(
    model_bundle,
    aligned_image: np.ndarray,
    aligned_mask: np.ndarray,
) -> Tuple[str, float, Dict[str, float]]:
    try:
        import torch
        from torch import nn
        from torchvision.models import resnet18
    except Exception as exc:  # noqa: BLE001
        raise OrientationError(
            "torch/torchvision are required for the ResNet direction model"
        ) from exc

    model = model_bundle.get("_runtime_model")
    device = model_bundle.get("_runtime_device")
    if model is None or device is None:
        state_dict = model_bundle.get("state_dict")
        if not state_dict:
            raise OrientationError("ResNet direction model bundle is missing state_dict")

        class_names = model_bundle.get("class_names", ["head_left", "head_right"])
        num_classes = len(class_names)
        try:
            model = resnet18(weights=None)
        except TypeError:
            model = resnet18(pretrained=False)
        in_features = model.fc.in_features
        dropout = float(model_bundle.get("training_args", {}).get("dropout", 0.0))
        if "fc.1.weight" in state_dict:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        model_bundle["_runtime_model"] = model
        model_bundle["_runtime_device"] = device

    tensor = _torch_direction_tensor(model_bundle, aligned_image, aligned_mask)
    tensor = tensor.to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()

    class_names = model_bundle.get("class_names", ["head_left", "head_right"])
    pred_idx = int(np.argmax(probabilities))
    label = str(class_names[pred_idx])
    if label not in {"head_left", "head_right", "left", "right", "unclear", "unknown"}:
        raise OrientationError(f"unsupported direction label from ResNet model: {label}")
    if label in {"unclear", "unknown"}:
        head_side = "unclear"
    else:
        head_side = "left" if label in {"head_left", "left"} else "right"
    confidence = float(np.max(probabilities))
    threshold = float(model_bundle.get("confidence_threshold", 0.0) or 0.0)
    if threshold > 0.0 and confidence < threshold:
        head_side = "unclear"
    probability_by_label = {
        str(name): float(probabilities[idx])
        for idx, name in enumerate(class_names)
    }
    left_probability = probability_by_label.get("head_left", probability_by_label.get("left", 0.0))
    right_probability = probability_by_label.get("head_right", probability_by_label.get("right", 0.0))
    unclear_probability = probability_by_label.get("unclear", probability_by_label.get("unknown", 0.0))
    features = {
        "model_label": 1.0 if head_side == "left" else 0.0 if head_side == "right" else -1.0,
        "torch_left_probability": left_probability,
        "torch_right_probability": right_probability,
        "torch_unclear_probability": unclear_probability,
        "left_probability": left_probability,
        "confidence_threshold": threshold,
    }
    return head_side, confidence, features


def _torch_direction_tensor(model_bundle, image: np.ndarray, mask: np.ndarray):
    import torch

    image_size = int(model_bundle.get("image_size", 224))
    padding_ratio = float(model_bundle.get("crop_padding_ratio", 0.25))
    rgb_mean = np.array(
        model_bundle.get("rgb_mean", IMAGENET_RGB_MEAN),
        dtype=np.float32,
    )
    rgb_std = np.array(
        model_bundle.get("rgb_std", IMAGENET_RGB_STD),
        dtype=np.float32,
    )

    rgb = _crop_to_mask_square_rgb(image, mask, padding_ratio=padding_ratio)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - rgb_mean.reshape(1, 1, 3)) / rgb_std.reshape(1, 1, 3)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def _crop_to_mask_square_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.25,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise OrientationError("empty aligned mask")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = x1 - x0
    height = y1 - y0
    pad = int(round(max(width, height) * padding_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.shape[1], x1 + pad)
    y1 = min(image.shape[0], y1 + pad)

    crop = image[y0:y1, x0:x1]
    if crop.ndim == 2:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    crop_h, crop_w = rgb.shape[:2]
    side = max(crop_h, crop_w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - crop_h) // 2
    x_off = (side - crop_w) // 2
    canvas[y_off : y_off + crop_h, x_off : x_off + crop_w] = rgb
    return canvas


def extract_direction_features(
    aligned_image: np.ndarray,
    aligned_mask: np.ndarray,
    end_fraction: float = 0.30,
) -> Dict[str, float]:
    """Extract side-aware features from a horizontally aligned masked crop."""

    if not 0.05 <= end_fraction <= 0.50:
        raise ValueError("end_fraction must be between 0.05 and 0.50")

    mask = (aligned_mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise OrientationError("foreground mask is empty after alignment")

    if aligned_image.ndim == 2:
        gray = aligned_image
    else:
        gray = cv2.cvtColor(aligned_image[:, :, :3], cv2.COLOR_BGR2GRAY)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    body = mask[:, x1:x2]
    profile = body.sum(axis=0).astype(np.float32)
    length = len(profile)
    features: Dict[str, float] = {}

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    features["bbox_aspect"] = float(bbox_w / bbox_h)
    features["mask_area_ratio"] = float(mask.sum() / max(mask.size, 1))
    features["centroid_x_norm"] = float(((xs.mean() - x1) / bbox_w) - 0.5)
    features["centroid_y_norm"] = float(((ys.mean() - y1) / bbox_h) - 0.5)

    fractions = sorted({0.15, 0.25, float(end_fraction), 0.35, 0.45})
    for fraction in fractions:
        end = max(3, int(round(length * fraction)))
        if length < end * 2:
            end = max(2, length // 2)
        if end < 2:
            continue

        left_profile = profile[:end]
        right_profile = profile[-end:]
        left_area = float(left_profile.sum())
        right_area = float(right_profile.sum())
        left_width = float(np.median(left_profile))
        right_width = float(np.median(right_profile))
        left_mse = _quadratic_mse(left_profile)
        right_mse = _quadratic_mse(right_profile)
        left_mean, right_mean, left_std, right_std = _end_intensity_stats(
            aligned_image, mask, x1, x2, end
        )

        prefix = f"f{int(round(fraction * 100)):02d}"
        features[f"{prefix}_area_delta"] = _pair_delta(left_area, right_area)
        features[f"{prefix}_width_delta"] = _pair_delta(left_width, right_width)
        features[f"{prefix}_mse_delta"] = _pair_delta(left_mse, right_mse)
        features[f"{prefix}_brightness_delta"] = _pair_delta(left_mean, right_mean)
        features[f"{prefix}_texture_delta"] = _pair_delta(left_std, right_std)
        features[f"{prefix}_left_area"] = left_area
        features[f"{prefix}_right_area"] = right_area
        features[f"{prefix}_left_width"] = left_width
        features[f"{prefix}_right_width"] = right_width

    resized_mask = cv2.resize(
        mask[y1:y2, x1:x2].astype(np.float32), (24, 12), interpolation=cv2.INTER_AREA
    )
    resized_gray = cv2.resize(
        gray[y1:y2, x1:x2].astype(np.float32) / 255.0,
        (24, 12),
        interpolation=cv2.INTER_AREA,
    )
    for idx, value in enumerate(resized_mask.reshape(-1) / 255.0):
        features[f"mask_px_{idx:03d}"] = float(value)
    for idx, value in enumerate(resized_gray.reshape(-1)):
        features[f"gray_px_{idx:03d}"] = float(value)

    return features


def largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise OrientationError("no foreground contour found")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        raise OrientationError("foreground contour is too small")
    return contour


def _border_pixels(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    edge = max(2, int(round(min(h, w) * 0.08)))
    return np.concatenate(
        [
            image[:edge, :].reshape(-1, image.shape[-1]),
            image[-edge:, :].reshape(-1, image.shape[-1]),
            image[:, :edge].reshape(-1, image.shape[-1]),
            image[:, -edge:].reshape(-1, image.shape[-1]),
        ],
        axis=0,
    )


def _border_value(image: np.ndarray):
    if image.ndim == 2:
        border = _border_pixels(image[..., None])
        return float(np.median(border))
    channels = image.shape[2]
    if channels == 4:
        return (0, 0, 0, 0)
    border = _border_pixels(image[:, :, :3])
    values = np.median(border, axis=0).astype(float).tolist()
    return tuple(values)


def _rotate_bound(
    image: np.ndarray,
    mask: np.ndarray,
    angle_degrees: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(transform[0, 0])
    sin = abs(transform[0, 1])
    new_w = max(1, int(round((h * sin) + (w * cos))))
    new_h = max(1, int(round((h * cos) + (w * sin))))

    transform[0, 2] += (new_w / 2.0) - center[0]
    transform[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        transform,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=_border_value(image),
    )
    rotated_mask = cv2.warpAffine(
        mask,
        transform,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated, rotated_mask


def _crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    crop_padding_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise OrientationError("foreground mask is empty after rotation")

    h, w = mask.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    span = max(x2 - x1, y2 - y1)
    padding = max(2, int(round(span * crop_padding_ratio)))

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _fill_background(image: np.ndarray, mask: np.ndarray, fill_value) -> np.ndarray:
    protected = (mask > 0).astype(np.uint8) * 255
    kernel_size = max(3, int(round(min(mask.shape[:2]) * 0.06)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    protected = cv2.dilate(protected, kernel, iterations=1)

    out = image.copy()
    background = protected == 0
    if image.ndim == 2:
        out[background] = fill_value
    else:
        out[background] = np.array(fill_value, dtype=out.dtype)
    return out


def _mask_bbox_width_height(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise OrientationError("foreground mask is empty")
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _order_box_points(points: np.ndarray) -> np.ndarray:
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    top_left, top_right = top
    bottom_left, bottom_right = bottom
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _principal_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    _, vectors, _ = cv2.PCACompute2(coords, mean=None)
    vx, vy = vectors[0]
    return float(np.degrees(np.arctan2(vy, vx)))


def _quadratic_mse(values: np.ndarray) -> float:
    y = values.astype(np.float64)
    if len(y) < 3 or np.allclose(y, y[0]):
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    degree = min(2, len(y) - 1)
    coeffs = np.polyfit(x, y, degree)
    pred = np.polyval(coeffs, x)
    return float(np.mean((y - pred) ** 2))


def _end_intensity_stats(
    image: np.ndarray,
    mask: np.ndarray,
    x1: int,
    x2: int,
    end: int,
) -> Tuple[float, float, float, float]:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    left_cols = slice(x1, min(x2, x1 + end))
    right_cols = slice(max(x1, x2 - end), x2)

    def stats_for(cols: slice) -> Tuple[float, float]:
        section_mask = mask[:, cols] > 0
        if not np.any(section_mask):
            return 255.0, 0.0
        values = gray[:, cols][section_mask].astype(np.float32)
        return float(values.mean()), float(values.std())

    left_mean, left_std = stats_for(left_cols)
    right_mean, right_std = stats_for(right_cols)
    return left_mean, right_mean, left_std, right_std


def _pair_delta(left: float, right: float) -> float:
    denom = abs(left) + abs(right) + 1e-6
    return float(np.clip((right - left) / denom, -1.0, 1.0))


def _smaller_score(left: float, right: float) -> float:
    return 0.5 + 0.5 * _pair_delta(left, right)


def _larger_score(left: float, right: float) -> float:
    return 0.5 - 0.5 * _pair_delta(left, right)
