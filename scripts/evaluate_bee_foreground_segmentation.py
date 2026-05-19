"""Evaluate the bee foreground segmentation model on a dataset split.

This restores predicted probabilities to each original image size before
thresholding, matching the deployment path more closely than fixed-size
training validation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

import segmentation_models_pytorch as smp


EPS = 1e-7


def imread(path: Path, flags=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def list_pairs(dataset_dir: Path, split: str) -> list[tuple[Path, Path]]:
    split_dir = dataset_dir / split
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_dir / image_path.with_suffix(".png").name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


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
        dist = np.hypot(x - cx, y - cy) / max(1.0, min(height, width))
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


def clean_prediction(
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
    return keep_near_main_components(
        out,
        radius_frac=0.15,
        min_area=max(2, int(min_area) // 10),
    )


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    return {
        "iou": tp / max(tp + fp + fn, EPS),
        "dice": 2.0 * tp / max(2.0 * tp + fp + fn, EPS),
        "precision": tp / max(tp + fp, EPS),
        "recall": tp / max(tp + fn, EPS),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, EPS),
    }


def update_counts(pred: np.ndarray, target: np.ndarray, counts: dict[str, float]) -> None:
    counts["tp"] += float((pred & target).sum())
    counts["fp"] += float((pred & ~target).sum())
    counts["fn"] += float((~pred & target).sum())
    counts["tn"] += float((~pred & ~target).sum())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = smp.UnetPlusPlus(
        encoder_name=checkpoint.get("encoder", "resnet18"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", 256))
    mean = np.array(checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return model, image_size, mean, std, checkpoint


def preprocess(image_bgr: np.ndarray, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset_dir)
    checkpoint_path = Path(args.checkpoint)
    pairs = list_pairs(dataset_dir, args.split)
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found under {dataset_dir / args.split}")

    device = resolve_device(args.device)
    model, image_size, mean, std, checkpoint = load_model(checkpoint_path, device)

    micro_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    per_image_metrics = []
    batch_tensors = []
    batch_meta = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        tensor = torch.from_numpy(np.stack(batch_tensors, axis=0)).to(device)
        probs = torch.sigmoid(model(tensor))[:, 0].detach().cpu().numpy()
        for prob_small, (image_bgr, target) in zip(probs, batch_meta):
            prob = cv2.resize(
                prob_small,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            pred = prob >= args.threshold
            if args.postprocess:
                pred = clean_prediction(
                    pred,
                    min_area=args.min_area,
                    close_ratio=args.close_ratio,
                )
            counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
            update_counts(pred, target, counts)
            update_counts(pred, target, micro_counts)
            per_image_metrics.append(metrics_from_counts(counts))
        batch_tensors.clear()
        batch_meta.clear()

    for image_path, mask_path in pairs:
        image_bgr = imread(image_path, cv2.IMREAD_COLOR)
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise RuntimeError(f"failed to read sample: {image_path}")
        batch_tensors.append(preprocess(image_bgr, image_size, mean, std))
        batch_meta.append((image_bgr, mask > 0))
        if len(batch_tensors) >= args.batch_size:
            flush_batch()
    flush_batch()

    pixel_micro = metrics_from_counts(micro_counts)
    image_macro = {
        key: float(np.mean([metrics[key] for metrics in per_image_metrics]))
        for key in pixel_micro
    }
    return {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "samples": len(pairs),
        "device": str(device),
        "encoder": checkpoint.get("encoder", "resnet18"),
        "image_size": image_size,
        "threshold": args.threshold,
        "postprocess": bool(args.postprocess),
        "pixel_micro": pixel_micro,
        "image_macro": image_macro,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/bee_foreground_v2/dataset")
    parser.add_argument(
        "--checkpoint",
        default="models/bee_foreground_unetpp_resnet18_v2/best_model.pt",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--close-ratio", type=float, default=0.012)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
