"""Convert a CVAT Segmentation Mask export into a binary bee foreground dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


FOREGROUND_LABELS = {"bee", "head", "body"}


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def parse_labelmap(path: Path) -> dict[tuple[int, int, int], str]:
    color_to_name: dict[tuple[int, int, int], str] = {}
    if not path.exists():
        return color_to_name
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        color_text = rest.split(":", 1)[0]
        parts = [part.strip() for part in color_text.split(",") if part.strip()]
        if len(parts) < 3:
            continue
        rgb = tuple(int(float(part)) for part in parts[:3])
        color_to_name[rgb] = name.strip()
    return color_to_name


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filename = row.get("flat_filename") or row.get("filename")
            base = row.get("base_name") or (Path(filename).stem if filename else "")
            if filename:
                rows[filename] = row
            if base:
                rows[base] = row
    return rows


def decode_foreground_mask(
    mask: np.ndarray,
    color_to_name: dict[tuple[int, int, int], str],
    foreground_labels: set[str],
) -> np.ndarray:
    if mask is None:
        raise ValueError("empty mask")
    if mask.ndim == 2:
        return (mask > 0).astype(np.uint8)

    rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
    decoded = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if color_to_name:
        for color_rgb, name in color_to_name.items():
            if name not in foreground_labels:
                continue
            color = np.array(color_rgb, dtype=np.uint8).reshape(1, 1, 3)
            decoded[np.all(rgb == color, axis=2)] = 1
        return decoded

    decoded[np.any(rgb != 0, axis=2)] = 1
    return decoded


def find_mask_files(extract_dir: Path) -> list[Path]:
    class_masks = sorted((extract_dir / "SegmentationClass").glob("*.png"))
    if class_masks:
        return class_masks
    candidates = []
    for path in extract_dir.rglob("*.png"):
        parts = {part.lower() for part in path.parts}
        if "segmentationclass" in parts:
            candidates.append(path)
    return sorted(candidates or extract_dir.rglob("*.png"))


def split_rows(rows: list[dict[str, str]], val_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = sorted(rows, key=lambda row: row["filename"])
    if len(rows) <= 1:
        return rows, []
    val_count = max(1, int(round(len(rows) * val_fraction)))
    val_every = max(2, len(rows) // val_count)
    val = [row for idx, row in enumerate(rows) if idx % val_every == 0][:val_count]
    val_names = {row["filename"] for row in val}
    train = [row for row in rows if row["filename"] not in val_names]
    return train, val


def copy_split(rows: list[dict[str, str]], dataset_dir: Path, split: str) -> None:
    image_dir = dataset_dir / split / "images"
    mask_dir = dataset_dir / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        shutil.copy2(row["image_path"], image_dir / row["filename"])
        shutil.copy2(row["mask_path"], mask_dir / Path(row["filename"]).with_suffix(".png"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", required=True)
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--foreground-label", action="append", default=[])
    parser.add_argument("--min-foreground-pixels", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    annotations_zip = Path(args.annotations_zip).resolve()
    subset_dir = Path(args.subset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    extract_dir = output_dir / "cvat_export_unzipped"
    masks_all_dir = output_dir / "masks_all"
    dataset_dir = output_dir / "dataset"

    for path in (extract_dir, masks_all_dir, dataset_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(annotations_zip, "r") as zf:
        zf.extractall(extract_dir)

    manifest = load_manifest(subset_dir / "manifest.csv")
    image_dir = subset_dir / "images"
    color_to_name = parse_labelmap(next(iter(extract_dir.rglob("labelmap.txt")), Path()))
    foreground_labels = set(args.foreground_label) if args.foreground_label else set(FOREGROUND_LABELS)

    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for mask_path in find_mask_files(extract_dir):
        base_name = mask_path.stem
        filename = f"{base_name}.jpg"
        manifest_row = manifest.get(filename) or manifest.get(base_name)
        if manifest_row is None:
            skipped.append({"mask": str(mask_path), "reason": "not_in_manifest"})
            continue

        filename = manifest_row.get("flat_filename") or filename
        image_path = image_dir / filename
        if not image_path.exists():
            skipped.append({"mask": str(mask_path), "filename": filename, "reason": "missing_image"})
            continue

        decoded = decode_foreground_mask(imread(mask_path), color_to_name, foreground_labels)
        foreground_pixels = int(decoded.sum())
        if foreground_pixels < args.min_foreground_pixels:
            skipped.append(
                {
                    "mask": str(mask_path),
                    "filename": filename,
                    "reason": f"too_few_foreground_pixels_{foreground_pixels}",
                }
            )
            continue

        image = imread(image_path)
        if image is None:
            skipped.append({"mask": str(mask_path), "filename": filename, "reason": "failed_to_read_image"})
            continue
        if decoded.shape[:2] != image.shape[:2]:
            skipped.append(
                {
                    "mask": str(mask_path),
                    "filename": filename,
                    "reason": f"shape_mismatch_mask_{decoded.shape[:2]}_image_{image.shape[:2]}",
                }
            )
            continue

        out_mask = masks_all_dir / Path(filename).with_suffix(".png")
        if not imwrite(out_mask, decoded.astype(np.uint8)):
            skipped.append({"mask": str(mask_path), "filename": filename, "reason": "failed_to_write_mask"})
            continue

        rows.append(
            {
                "filename": filename,
                "base_name": base_name,
                "relative_path": manifest_row.get("relative_path", ""),
                "image_path": str(image_path),
                "mask_path": str(out_mask),
                "foreground_pixels": str(foreground_pixels),
                "foreground_ratio": f"{foreground_pixels / float(decoded.size):.6f}",
            }
        )

    train_rows, val_rows = split_rows(rows, args.val_fraction)
    copy_split(train_rows, dataset_dir, "train")
    copy_split(val_rows, dataset_dir, "val")

    for name, data in (
        ("usable_annotations.csv", rows),
        ("train.csv", train_rows),
        ("val.csv", val_rows),
        ("skipped_annotations.csv", skipped),
    ):
        keys = sorted({key for row in data for key in row}) if data else ["filename"]
        with (output_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    summary = {
        "annotations_zip": str(annotations_zip),
        "subset_dir": str(subset_dir),
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "foreground_labels": sorted(foreground_labels),
        "total_masks": len(find_mask_files(extract_dir)),
        "usable_annotations": len(rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "skipped_annotations": len(skipped),
        "labelmap": {",".join(map(str, color)): name for color, name in color_to_name.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
