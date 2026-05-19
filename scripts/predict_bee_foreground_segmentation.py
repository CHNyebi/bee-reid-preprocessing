"""Run a trained binary bee foreground model over an ImageFolder-style crop root."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

import segmentation_models_pytorch as smp


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def imread(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def iter_images(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = checkpoint.get("encoder", "resnet18")
    image_size = int(checkpoint.get("image_size", 256))
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    mean = np.array(checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return model, image_size, mean, std, checkpoint


def preprocess(image_bgr: np.ndarray, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.size == 0:
        return mask_u8.astype(bool)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return np.logical_or(mask_u8 > 0, holes)


def keep_near_main_components(mask: np.ndarray, radius_frac: float = 0.14, min_area: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    if int(m.sum()) == 0:
        return m.astype(bool)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
    if n_labels <= 1:
        return m.astype(bool)
    h, w = m.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    best_label = 1
    best_score = -1.0
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        x, y = centroids[label]
        dist = np.hypot(x - cx, y - cy) / max(1.0, min(h, w))
        score = area / (1.0 + 1.8 * dist)
        if score > best_score:
            best_label = label
            best_score = score
    main = labels == best_label
    dist_map = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = max(3, int(round(min(h, w) * radius_frac)))
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


def clean_prediction(mask: np.ndarray, min_area: int, close_ratio: float) -> np.ndarray:
    if int(mask.sum()) < min_area:
        return mask.astype(bool)
    h, w = mask.shape[:2]
    close_size = max(3, int(round(min(h, w) * close_ratio)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    out = fill_mask_holes(out)
    out = keep_near_main_components(out, radius_frac=0.15, min_area=max(2, min_area // 10))
    return out


def soft_alpha(mask: np.ndarray, feather_ratio: float) -> np.ndarray:
    h, w = mask.shape[:2]
    sigma = max(0.6, min(h, w) * feather_ratio)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    alpha[mask] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray, background_value: int, feather_ratio: float) -> np.ndarray:
    alpha = soft_alpha(mask, feather_ratio)
    background = np.full_like(image_bgr, int(background_value))
    out = image_bgr.astype(np.float32) * alpha[:, :, None] + background.astype(np.float32) * (1.0 - alpha[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :] = (255, 96, 0)
    blended = cv2.addWeighted(image_bgr, 0.62, color, 0.38, 0.0)
    overlay[mask] = blended[mask]
    heat = cv2.applyColorMap(np.clip(confidence * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    gap = np.full((image_bgr.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([image_bgr, gap, overlay, gap, heat], axis=1)


def mask_stats(mask: np.ndarray, confidence: np.ndarray) -> dict[str, float | int]:
    h, w = mask.shape[:2]
    area = int(mask.sum())
    if area <= 0:
        return {
            "foreground_pixels": 0,
            "foreground_ratio": 0.0,
            "bbox_ratio": 0.0,
            "border_touch_ratio": 0.0,
            "mean_confidence": float(confidence.mean()) if confidence.size else 0.0,
            "foreground_confidence": 0.0,
        }
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    edge = max(2, int(round(min(h, w) * 0.06)))
    border = np.zeros_like(mask, dtype=bool)
    border[:edge, :] = True
    border[-edge:, :] = True
    border[:, :edge] = True
    border[:, -edge:] = True
    return {
        "foreground_pixels": area,
        "foreground_ratio": area / float(h * w),
        "bbox_ratio": ((x1 - x0) * (y1 - y0)) / float(h * w),
        "border_touch_ratio": float((mask & border).sum()) / float(area),
        "mean_confidence": float(confidence.mean()),
        "foreground_confidence": float(confidence[mask].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--close-ratio", type=float, default=0.012)
    parser.add_argument("--feather-ratio", type=float, default=0.012)
    parser.add_argument("--background-value", type=int, default=0)
    parser.add_argument("--overlay-limit", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=5000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    label_dir = output_dir / "labels"
    masked_dir = output_dir / "masked"
    overlay_dir = output_dir / "overlays"
    report_path = output_dir / "prediction_report.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, mean, std, checkpoint = load_model(Path(args.checkpoint), device)

    paths = list(iter_images(input_dir))
    if args.max_images:
        paths = paths[: args.max_images]

    start_time = time.time()
    total = 0
    ok = 0
    failed = 0
    batch_tensors: list[np.ndarray] = []
    batch_meta: list[tuple[Path, np.ndarray]] = []

    def flush_batch(writer: csv.DictWriter):
        nonlocal ok, failed
        if not batch_tensors:
            return
        tensor = torch.from_numpy(np.stack(batch_tensors, axis=0)).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

        for prob_small, (src, image_bgr) in zip(probs, batch_meta):
            rel = src.relative_to(input_dir)
            row = {
                "relative_path": str(rel),
                "source": str(src),
                "label_path": str(label_dir / rel.with_suffix(".png")),
                "masked_path": str(masked_dir / rel),
                "overlay_path": "",
                "status": "ok",
                "error": "",
            }
            try:
                confidence = cv2.resize(prob_small, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = confidence >= args.threshold
                mask = clean_prediction(mask, min_area=args.min_area, close_ratio=args.close_ratio)
                label = mask.astype(np.uint8)
                masked = apply_mask(image_bgr, mask, args.background_value, args.feather_ratio)

                label_path = label_dir / rel.with_suffix(".png")
                masked_path = masked_dir / rel
                if not imwrite(label_path, label):
                    raise RuntimeError("failed to write label")
                if not imwrite(masked_path, masked):
                    raise RuntimeError("failed to write masked crop")
                if args.overlay_limit <= 0 or ok < args.overlay_limit:
                    overlay_path = overlay_dir / rel.with_suffix(".jpg")
                    if not imwrite(overlay_path, make_overlay(image_bgr, mask, confidence)):
                        raise RuntimeError("failed to write overlay")
                    row["overlay_path"] = str(overlay_path)
                row.update({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in mask_stats(mask, confidence).items()})
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                row.update({"status": "failed", "error": str(exc)})
            writer.writerow(row)

        batch_tensors.clear()
        batch_meta.clear()

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "relative_path",
            "source",
            "label_path",
            "masked_path",
            "overlay_path",
            "status",
            "error",
            "foreground_pixels",
            "foreground_ratio",
            "bbox_ratio",
            "border_touch_ratio",
            "mean_confidence",
            "foreground_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            total += 1
            image_bgr = imread(path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                failed += 1
                writer.writerow(
                    {
                        "relative_path": str(path.relative_to(input_dir)),
                        "source": str(path),
                        "status": "failed",
                        "error": "failed_to_read_image",
                    }
                )
                continue
            batch_tensors.append(preprocess(image_bgr, image_size, mean, std))
            batch_meta.append((path, image_bgr))
            if len(batch_tensors) >= args.batch_size:
                flush_batch(writer)
            if args.log_interval and total % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"processed={total}/{len(paths)} ok={ok} failed={failed} elapsed={elapsed:.1f}s")
        flush_batch(writer)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "encoder": checkpoint.get("encoder", "resnet18"),
        "image_size": image_size,
        "threshold": args.threshold,
        "min_area": args.min_area,
        "close_ratio": args.close_ratio,
        "feather_ratio": args.feather_ratio,
        "background_value": args.background_value,
        "total": total,
        "ok": ok,
        "failed": failed,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "label_dir": str(label_dir),
        "masked_dir": str(masked_dir),
        "overlay_dir": str(overlay_dir),
        "report_path": str(report_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
