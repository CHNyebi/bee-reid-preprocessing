"""Train a binary bee foreground segmentation model from corrected CVAT masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imread(path: Path, flags=cv2.IMREAD_UNCHANGED):
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_dir / image_path.with_suffix(".png").name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def resize_pair(image: np.ndarray, mask: np.ndarray, image_size: int):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def augment_pair(image: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    if random.random() < 0.35:
        k = random.randint(1, 3)
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
    if random.random() < 0.7:
        alpha = random.uniform(0.78, 1.22)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.45:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.78, 1.25)
        hsv[:, :, 2] *= random.uniform(0.86, 1.16)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if random.random() < 0.20:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
    if random.random() < 0.25:
        noise = np.random.normal(0.0, random.uniform(2.0, 8.0), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


class BeeForegroundDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, train: bool):
        self.pairs = pairs
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image_bgr = imread(image_path, cv2.IMREAD_COLOR)
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise RuntimeError(f"failed to read sample: {image_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8)
        image, mask = resize_pair(image, mask, self.image_size)
        if self.train:
            image, mask = augment_pair(image, mask)

        arr = image.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
        image_tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, str(image_path)


def compute_pos_weight(pairs: list[tuple[Path, Path]]) -> torch.Tensor:
    pos = 0
    total = 0
    for _, mask_path in pairs:
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        fg = int((mask > 0).sum())
        pos += fg
        total += int(mask.size)
    neg = max(total - pos, 1)
    pos = max(pos, 1)
    return torch.tensor([min(max(neg / float(pos), 0.5), 6.0)], dtype=torch.float32)


def update_binary_counts(logits: torch.Tensor, target: torch.Tensor, counts: dict[str, float]) -> None:
    with torch.no_grad():
        pred = torch.sigmoid(logits.float()) >= 0.5
        tgt = target >= 0.5
        counts["tp"] += float((pred & tgt).sum().item())
        counts["fp"] += float((pred & ~tgt).sum().item())
        counts["fn"] += float((~pred & tgt).sum().item())
        counts["tn"] += float((~pred & ~tgt).sum().item())


def metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1e-7
    return {
        "iou": tp / max(tp + fp + fn, eps),
        "dice": 2.0 * tp / max(2.0 * tp + fp + fn, eps),
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, eps),
    }


def make_model(encoder: str):
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def make_losses(pos_weight: torch.Tensor, device: torch.device):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    return bce, dice


def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device, use_amp: bool):
    model.train()
    total_loss = 0.0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        update_binary_counts(logits.detach(), masks.detach(), counts)
    metrics = metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, bce_loss, dice_loss, device):
    model.eval()
    total_loss = 0.0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        total_loss += float(loss.item()) * images.size(0)
        update_binary_counts(logits.detach(), masks.detach(), counts)
    metrics = metrics_from_counts(counts)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :] = color_bgr
    blended = cv2.addWeighted(image_bgr, 0.62, color, 0.38, 0.0)
    out[mask > 0] = blended[mask > 0]
    return out


@torch.no_grad()
def write_previews(model, pairs: list[tuple[Path, Path]], output_dir: Path, image_size: int, device, limit: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = BeeForegroundDataset(pairs[:limit], image_size=image_size, train=False)
    model.eval()
    for image_tensor, target, image_path_text in dataset:
        image_path = Path(image_path_text)
        logits = model(image_tensor.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)
        image_bgr = imread(image_path, cv2.IMREAD_COLOR)
        target_full = imread(image_path.with_name(image_path.name).parent.parent / "masks" / image_path.with_suffix(".png").name, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            continue
        pred_full = cv2.resize(pred, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        if target_full is None:
            target_full = cv2.resize(target[0].numpy().astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        target_full = (target_full > 0).astype(np.uint8)
        gt_overlay = overlay_mask(image_bgr, target_full, (0, 180, 0))
        pred_overlay = overlay_mask(image_bgr, pred_full, (255, 96, 0))
        gap = np.full((image_bgr.shape[0], 8, 3), 255, dtype=np.uint8)
        combined = np.concatenate([image_bgr, gap, gt_overlay, gap, pred_overlay], axis=1)
        imwrite(output_dir / image_path.name, combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-limit", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = list_pairs(dataset_dir / "train")
    val_pairs = list_pairs(dataset_dir / "val")
    if not train_pairs:
        raise RuntimeError(f"no training pairs found in {dataset_dir / 'train'}")
    if not val_pairs:
        raise RuntimeError(f"no validation pairs found in {dataset_dir / 'val'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    use_amp = device.type == "cuda" and not args.no_amp
    model = make_model(args.encoder).to(device)
    resume_epoch = 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        resume_epoch = int(checkpoint.get("epoch") or 0)
    pos_weight = compute_pos_weight(train_pairs)
    bce_loss, dice_loss = make_losses(pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = DataLoader(
        BeeForegroundDataset(train_pairs, args.image_size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        BeeForegroundDataset(val_pairs, args.image_size, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    best_iou = -1.0
    start_time = time.time()

    with history_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "train_iou",
            "val_iou",
            "train_dice",
            "val_dice",
            "val_precision",
            "val_recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(resume_epoch + 1, resume_epoch + args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device, use_amp)
            val_metrics = evaluate(model, val_loader, bce_loss, dice_loss, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "val_iou": val_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_dice": val_metrics["dice"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()})
            f.flush()

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "encoder": args.encoder,
                        "image_size": args.image_size,
                        "classes": ["background", "bee"],
                        "imagenet_mean": IMAGENET_MEAN.tolist(),
                        "imagenet_std": IMAGENET_STD.tolist(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch:03d} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_iou={val_metrics['iou']:.4f} "
                    f"val_dice={val_metrics['dice']:.4f} "
                    f"precision={val_metrics['precision']:.4f} "
                    f"recall={val_metrics['recall']:.4f}"
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder": args.encoder,
            "image_size": args.image_size,
            "classes": ["background", "bee"],
            "imagenet_mean": IMAGENET_MEAN.tolist(),
            "imagenet_std": IMAGENET_STD.tolist(),
            "epoch": args.epochs,
        },
        last_path,
    )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    write_previews(model, val_pairs, output_dir / "val_previews", args.image_size, device, args.preview_limit)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "device": str(device),
        "encoder": args.encoder,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pos_weight": pos_weight.tolist(),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_metrics": checkpoint.get("val_metrics"),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
