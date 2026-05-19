#!/usr/bin/env python3
"""Build a manifest for packed ReID dataset archives."""

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


def archive_file_count(path: Path) -> int:
    count = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("data/reid_datasets"))
    parser.add_argument("--output", type=Path, default=Path("data/reid_datasets/manifest.csv"))
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    output = args.output.resolve()
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        raise RuntimeError(f"no .tar.gz archives found in {archive_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "archive", "bytes", "file_count", "sha256"],
        )
        writer.writeheader()
        for archive in archives:
            writer.writerow(
                {
                    "dataset": archive.name[:-7] if archive.name.endswith(".tar.gz") else archive.stem,
                    "archive": archive.name,
                    "bytes": archive.stat().st_size,
                    "file_count": archive_file_count(archive),
                    "sha256": sha256_file(archive),
                }
            )
            print(f"indexed {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
