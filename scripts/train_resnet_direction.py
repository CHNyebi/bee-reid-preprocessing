"""Train a ResNet bee head-direction classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bee_orientation_normalizer.orientation import (  # noqa: E402
    align_long_axis,
    build_foreground_mask,
)


VALID_LABELS = {"head_left", "head_right", "unclear"}
CLASS_NAMES = ["head_left", "head_right", "unclear"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
OPPOSITE = {"head_left": "head_right", "head_right": "head_left", "unclear": "unclear"}
LABEL_ALIASES = {
    "": "unclear",
    "blank": "unclear",
    "unknown": "unclear",
    "unrecognizable": "unclear",
    "uncertain": "unclear",
    "无法辨认": "unclear",
    "不可辨认": "unclear",
    "不确定": "unclear",
    "head-left": "head_left",
    "head-right": "head_right",
    "left": "head_left",
    "right": "head_right",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def load_labeled_rows(labels_csv: Path):
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    normalized_rows = []
    for row in rows:
        row = dict(row)
        row["label"] = normalize_label(row.get("label", ""))
        image_path, already_aligned = resolve_row_image(row, labels_csv.parent)
        row["_image_path"] = str(image_path)
        row["_already_aligned"] = int(already_aligned)
        normalized_rows.append(row)
    labeled = [row for row in normalized_rows if row.get("label") in VALID_LABELS]
    skipped = [row for row in normalized_rows if row.get("label") not in VALID_LABELS]
    return labeled, skipped


def normalize_label(label: str) -> str:
    value = (label or "").strip().lower()
    return LABEL_ALIASES.get(value, value)


def resolve_row_image(row, base_dir: Path):
    for key, already_aligned in (
        ("image_path", True),
        ("aligned_path", True),
        ("source", False),
    ):
        value = (row.get(key) or "").strip()
        if not value:
            continue
        path = Path(value.replace("\\", "/"))
        if not path.is_absolute():
            path = base_dir / path
        return path, already_aligned
    raise RuntimeError("row is missing image_path, aligned_path, or source")


def crop_to_mask_square(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.25,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("empty aligned mask")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w = x1 - x0
    h = y1 - y0
    pad = int(round(max(w, h) * padding_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.shape[1], x1 + pad)
    y1 = min(image.shape[0], y1 + pad)

    crop = image[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
    else:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    height, width = crop.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_off = (side - height) // 2
    x_off = (side - width) // 2
    canvas[y_off : y_off + height, x_off : x_off + width] = crop
    return canvas


def prepare_samples(
    rows,
    mask_mode: str,
    crop_padding_ratio: float,
):
    samples = []
    failures = []
    for row in rows:
        try:
            image = imread(Path(row["_image_path"]))
            if image is None:
                raise RuntimeError("failed to read image")
            mask = build_foreground_mask(image, mode=mask_mode)
            if int(row.get("_already_aligned", 0)):
                aligned, aligned_mask, angle = image, mask, float(row.get("angle_degrees") or 0.0)
            else:
                aligned, aligned_mask, angle = align_long_axis(image, mask, crop_output=False)
            crop = crop_to_mask_square(
                aligned,
                aligned_mask,
                padding_ratio=crop_padding_ratio,
            )
            samples.append(
                {
                    "row": row,
                    "image": crop,
                    "label": row["label"],
                    "angle_degrees": float(angle),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch training robust.
            failures.append(
                {
                    "index": row_identifier(row),
                    "source": row.get("source") or row.get("image_path") or row.get("aligned_path"),
                    "label": row.get("label"),
                    "error": str(exc),
                }
            )
    return samples, failures


class BeeDirectionDataset(Dataset):
    def __init__(self, samples, transform, augment_rot180: bool = False):
        self.items = []
        for sample in samples:
            self.items.append((sample["image"], sample["label"], sample["row"]))
            if augment_rot180:
                flipped = np.ascontiguousarray(np.rot90(sample["image"], 2))
                self.items.append((flipped, OPPOSITE[sample["label"]], sample["row"]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, label, row = self.items[index]
        pil = Image.fromarray(image)
        tensor = self.transform(pil)
        return tensor, CLASS_TO_IDX[label], row_identifier(row)


def row_identifier(row) -> str:
    for key in ("index", "global_index", "source_index"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def make_transforms(image_size: int, train: bool):
    base = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        base.extend(
            [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.03, 0.03),
                    scale=(0.92, 1.08),
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.12,
                    hue=0.02,
                ),
            ]
        )
    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(base)


def make_model(pretrained: bool, dropout: float, num_classes: int):
    model = resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable(model: nn.Module, stage: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    if stage == "finetune":
        for param in model.layer4.parameters():
            param.requires_grad = True


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    labels_out = []
    pred_out = []
    prob_out = []
    index_out = []
    for images, labels, indexes in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total += int(labels.size(0))
        labels_out.extend(labels.cpu().numpy().tolist())
        pred_out.extend(preds.cpu().numpy().tolist())
        prob_out.extend(probs.cpu().numpy().tolist())
        index_out.extend(list(indexes))

    y_true = np.array(labels_out, dtype=np.int64)
    y_pred = np.array(pred_out, dtype=np.int64)
    if y_true.size == 0:
        acc = 0.0
        bal = 0.0
    else:
        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "y_true": labels_out,
        "y_pred": pred_out,
        "probabilities": prob_out,
        "indexes": index_out,
    }


def train_one_model(
    train_samples,
    val_samples,
    args,
    device,
    seed: int,
):
    set_seed(seed)
    train_ds = BeeDirectionDataset(
        train_samples,
        transform=make_transforms(args.image_size, train=True),
        augment_rot180=args.augment_rot180,
    )
    val_ds = BeeDirectionDataset(
        val_samples,
        transform=make_transforms(args.image_size, train=False),
        augment_rot180=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(
        pretrained=args.pretrained,
        dropout=args.dropout,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    class_counts = Counter(sample["label"] for sample in train_samples)
    weights = torch.tensor(
        [1.0 / max(class_counts[name], 1) for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.sum() * float(len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_state = None
    best_score = -1.0
    best_eval = None

    phases = [("head", args.epochs_head, args.lr_head)]
    if args.epochs_finetune > 0:
        phases.append(("finetune", args.epochs_finetune, args.lr_finetune))

    for phase, epochs, lr in phases:
        set_trainable(model, phase)
        optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            record = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
            history.append(record)
            score = val_metrics["balanced_accuracy"]
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_eval = val_metrics

    if best_state is not None:
        model.load_state_dict(best_state)
        best_eval = evaluate(model, val_loader, criterion, device)
    return model, history, best_eval


def cross_validate(samples, args, device):
    labels = np.array([sample["label"] for sample in samples])
    counts = Counter(labels)
    n_splits = min(args.n_splits, min(counts.values()))
    if n_splits < 2:
        raise RuntimeError(f"not enough labels per class for CV: {dict(counts)}")
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    all_true = []
    all_pred = []
    rows = []
    fold_summaries = []
    histories = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels), start=1):
        train_samples = [samples[idx] for idx in train_idx]
        val_samples = [samples[idx] for idx in val_idx]
        _, history, best_eval = train_one_model(
            train_samples,
            val_samples,
            args,
            device,
            seed=args.random_state + fold,
        )
        y_true = best_eval["y_true"]
        y_pred = best_eval["y_pred"]
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        for idx_text, true_idx, pred_idx, probs in zip(
            best_eval["indexes"],
            y_true,
            y_pred,
            best_eval["probabilities"],
        ):
            rows.append(
                {
                    "fold": fold,
                    "index": idx_text,
                    "label": CLASS_NAMES[int(true_idx)],
                    "cv_prediction": CLASS_NAMES[int(pred_idx)],
                    "correct": int(true_idx == pred_idx),
                    **{
                        f"{name}_probability": f"{float(probs[class_idx]):.6f}"
                        for class_idx, name in enumerate(CLASS_NAMES)
                    },
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "val_count": len(y_true),
            }
        )
        for record in history:
            record = dict(record)
            record["fold"] = fold
            histories.append(record)
        print(
            f"fold={fold} acc={fold_summaries[-1]['accuracy']:.3f} "
            f"bal_acc={fold_summaries[-1]['balanced_accuracy']:.3f}"
        )

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CLASS_NAMES))))
    summary = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "confusion_matrix": cm.tolist(),
        "folds": fold_summaries,
    }
    return summary, rows, histories


def train_final_model(samples, args, device):
    model, history, _ = train_one_model(
        samples,
        samples,
        args,
        device,
        seed=args.random_state + 999,
    )
    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--output-model",
        default="output/bee_orientation_models/bee_direction_resnet18.joblib",
    )
    parser.add_argument(
        "--report-dir",
        default="output/bee_orientation_models/resnet18",
    )
    parser.add_argument("--mask-mode", choices=["masked", "auto"], default="masked")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.25)
    parser.add_argument(
        "--label-mode",
        choices=["triclass", "binary"],
        default="triclass",
        help="triclass keeps unclear as a class; binary trains only head_left/head_right labels.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Saved inference threshold; predictions below it are treated as unclear.",
    )
    parser.add_argument("--random-state", type=int, default=20260516)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, or a torch device string.",
    )
    add_bool_arg(
        parser,
        "pretrained",
        default=True,
        help_text="Use ImageNet pretrained ResNet18 weights.",
    )
    add_bool_arg(
        parser,
        "augment-rot180",
        default=True,
        help_text="Add 180-degree rotated samples with opposite labels to training folds.",
    )
    return parser.parse_args()


def add_bool_arg(parser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{dest: default})


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    global CLASS_NAMES, CLASS_TO_IDX
    args = parse_args()
    set_seed(args.random_state)

    if args.label_mode == "binary":
        CLASS_NAMES = ["head_left", "head_right"]
        CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    labels_csv = Path(args.labels_csv)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    rows, skipped = load_labeled_rows(labels_csv)
    if args.label_mode == "binary":
        binary_labels = set(CLASS_NAMES)
        skipped.extend([row for row in rows if row.get("label") not in binary_labels])
        rows = [row for row in rows if row.get("label") in binary_labels]
    if not rows:
        raise RuntimeError("no usable direction labels found")

    samples, failures = prepare_samples(
        rows,
        mask_mode=args.mask_mode,
        crop_padding_ratio=args.crop_padding_ratio,
    )
    if not samples:
        raise RuntimeError("no samples could be prepared")

    device = resolve_device(args.device)
    print("device", device)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0))
    print("usable_labels", dict(Counter(sample["label"] for sample in samples)))
    print("skipped_labels", len(skipped))
    print("preprocess_failures", len(failures))

    cv_summary, cv_rows, histories = cross_validate(samples, args, device)
    print(
        f"cv acc={cv_summary['accuracy']:.3f} "
        f"bal_acc={cv_summary['balanced_accuracy']:.3f} "
        f"cm={cv_summary['confusion_matrix']}"
    )

    final_model, final_history = train_final_model(samples, args, device)
    state_dict = {
        key: value.detach().cpu()
        for key, value in final_model.state_dict().items()
    }
    bundle = {
        "model_kind": "torch_resnet18_direction",
        "architecture": "resnet18",
        "state_dict": state_dict,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "image_size": args.image_size,
        "crop_padding_ratio": args.crop_padding_ratio,
        "rgb_mean": IMAGENET_MEAN,
        "rgb_std": IMAGENET_STD,
        "pretrained": args.pretrained,
        "labels_csv": str(labels_csv),
        "mask_mode": args.mask_mode,
        "augment_rot180": args.augment_rot180,
        "label_mode": args.label_mode,
        "confidence_threshold": float(args.confidence_threshold),
        "unclear_policy": "do_not_rotate_180",
        "cv_summary": cv_summary,
        "training_args": vars(args),
    }
    joblib.dump(bundle, output_model)

    pred_report = report_dir / "resnet18_cv_predictions.csv"
    with pred_report.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fold",
            "index",
            "label",
            "cv_prediction",
            "head_left_probability",
            "head_right_probability",
            "unclear_probability",
            "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cv_rows)

    history_path = report_dir / "resnet18_training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fold",
            "phase",
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_balanced_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(histories)

    metrics_path = report_dir / "resnet18_metrics.json"
    metrics = {
        "usable_labels": dict(Counter(sample["label"] for sample in samples)),
        "skipped_labels": len(skipped),
        "preprocess_failures": failures,
        "device": str(device),
        "cv_summary": cv_summary,
        "model_path": str(output_model),
        "prediction_report": str(pred_report),
        "history": str(history_path),
        "final_history": final_history,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("model", output_model)
    print("metrics", metrics_path)
    print("predictions", pred_report)
    print("history", history_path)


if __name__ == "__main__":
    main()
