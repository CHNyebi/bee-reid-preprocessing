#!/usr/bin/env python3
"""Extract packed ReID datasets and optionally verify SHA256 checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    checksums: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("archive")
            checksum = row.get("sha256")
            if name and checksum:
                checksums[name] = checksum
    return checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("data/reid_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reid_extracted"))
    parser.add_argument("--manifest", type=Path, default=Path("data/reid_datasets/manifest.csv"))
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"archive directory not found: {archive_dir}")

    expected = load_manifest(manifest_path)
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        raise RuntimeError(f"no .tar.gz archives found in {archive_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        if not args.skip_checksum and archive.name in expected:
            actual = sha256_file(archive)
            if actual != expected[archive.name]:
                raise RuntimeError(
                    f"checksum mismatch for {archive.name}: expected {expected[archive.name]}, got {actual}"
                )

        dataset_name = archive.name[:-7] if archive.name.endswith(".tar.gz") else archive.stem
        target = output_dir / dataset_name
        if target.exists() and not args.overwrite:
            print(f"skip existing {target}")
            continue

        print(f"extract {archive.name} -> {output_dir}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
