"""Build a registry of labeled bee foreground samples.

The registry is the source of truth for de-duplication. Future annotation
subsets should be sampled only after excluding both `sample_id` and `sha256`
values already present here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/bee_foreground_v2/dataset")
    parser.add_argument("--metadata-csv", default="data/bee_foreground_v2/usable_annotations.csv")
    parser.add_argument("--output", default="data/bee_foreground_v2/annotation_registry.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    metadata_path = (repo_root / args.metadata_csv).resolve()
    output_path = (repo_root / args.output).resolve()

    metadata: dict[str, dict[str, str]] = {}
    if metadata_path.exists():
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                filename = row.get("combined_filename") or row.get("filename")
                if filename:
                    metadata[Path(filename).name] = row

    rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        image_dir = dataset_dir / split / "images"
        mask_dir = dataset_dir / split / "masks"
        for image_path in sorted(image_dir.glob("*")):
            mask_path = mask_dir / image_path.with_suffix(".png").name
            if not mask_path.exists():
                continue
            meta = metadata.get(image_path.name, {})
            source_image = meta.get("image_path", "")
            rel_path = meta.get("relative_path", "")
            rows.append(
                {
                    "sample_id": sample_id_from_filename(image_path.name),
                    "filename": image_path.name,
                    "split": split,
                    "relative_path": rel_path,
                    "source_image_path": source_image,
                    "image_sha256": sha256_file(image_path),
                    "mask_sha256": sha256_file(mask_path),
                    "source_dataset": meta.get("source_dataset", ""),
                    "foreground_pixels": meta.get("foreground_pixels", ""),
                    "foreground_ratio": meta.get("foreground_ratio", ""),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "filename",
        "split",
        "relative_path",
        "source_image_path",
        "image_sha256",
        "mask_sha256",
        "source_dataset",
        "foreground_pixels",
        "foreground_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
