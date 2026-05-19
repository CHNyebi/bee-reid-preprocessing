"""Select new annotation candidates while avoiding already labeled samples.

This scans an ImageFolder-style crop root, computes stable IDs and image
hashes, removes anything already present in `annotation_registry.csv`, and
writes a reproducible candidate CSV. Use this before creating future CVAT
annotation batches from the same original crop pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return f"{rel.parent.name}__{rel.stem}"


def load_registry(path: Path) -> tuple[set[str], set[str]]:
    sample_ids: set[str] = set()
    hashes: set[str] = set()
    if not path.exists():
        return sample_ids, hashes
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("sample_id"):
                sample_ids.add(row["sample_id"])
            if row.get("image_sha256"):
                hashes.add(row["image_sha256"])
    return sample_ids, hashes


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Original crop root, e.g. train_20260501")
    parser.add_argument("--registry", default="data/bee_foreground_v2/annotation_registry.csv")
    parser.add_argument("--output", default="data/new_candidates.csv")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = Path(args.source_root).resolve()
    registry_path = (repo_root / args.registry).resolve()
    output_path = (repo_root / args.output).resolve()

    used_ids, used_hashes = load_registry(registry_path)
    candidates: list[dict[str, str]] = []
    skipped_id = 0
    skipped_hash = 0
    for path in iter_images(source_root):
        sid = sample_id_for(path, source_root)
        if sid in used_ids:
            skipped_id += 1
            continue
        digest = sha256_file(path)
        if digest in used_hashes:
            skipped_hash += 1
            continue
        candidates.append(
            {
                "sample_id": sid,
                "relative_path": str(path.relative_to(source_root)),
                "source_image_path": str(path),
                "image_sha256": digest,
            }
        )

    if args.sample_size and len(candidates) > args.sample_size:
        rng = random.Random(args.seed)
        candidates = rng.sample(candidates, args.sample_size)
        candidates.sort(key=lambda row: row["sample_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "relative_path", "source_image_path", "image_sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(
        f"wrote {len(candidates)} candidates to {output_path}; "
        f"skipped sample_id={skipped_id}, sha256={skipped_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
