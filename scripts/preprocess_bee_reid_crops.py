#!/usr/bin/env python3
"""Preprocess ImageFolder-style bee crops for another ReID pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bee_reid_preprocessing import BeeCropPreprocessConfig, BeeReIDCropPreprocessor
from bee_reid_preprocessing.preprocessor import (
    DEFAULT_FOREGROUND_CHECKPOINT,
    DEFAULT_ORIENTATION_MODEL,
    PREPROCESS_MODES,
    iter_image_paths,
    read_image,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        default="raw_aligned",
        choices=sorted(PREPROCESS_MODES),
        help="none, foreground, foreground_aligned, or raw_aligned",
    )
    parser.add_argument("--foreground-checkpoint", default=DEFAULT_FOREGROUND_CHECKPOINT, type=Path)
    parser.add_argument("--foreground-device", default="cuda")
    parser.add_argument("--foreground-threshold", default=0.50, type=float)
    parser.add_argument("--foreground-min-area", default=8, type=int)
    parser.add_argument("--foreground-close-ratio", default=0.012, type=float)
    parser.add_argument("--foreground-feather-ratio", default=0.012, type=float)
    parser.add_argument("--background-value", default=0, type=int)
    parser.add_argument("--orientation-model", default=DEFAULT_ORIENTATION_MODEL, type=Path)
    parser.add_argument("--orientation-target", default="head-left", choices=["head-left", "head-right"])
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", default=1000, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    config = BeeCropPreprocessConfig(
        mode=args.mode,
        foreground_checkpoint=args.foreground_checkpoint,
        foreground_device=args.foreground_device,
        foreground_threshold=args.foreground_threshold,
        foreground_min_area=args.foreground_min_area,
        foreground_close_ratio=args.foreground_close_ratio,
        foreground_feather_ratio=args.foreground_feather_ratio,
        background_value=args.background_value,
        orientation_model=args.orientation_model if args.orientation_model else None,
        orientation_target=args.orientation_target,
    )
    preprocessor = BeeReIDCropPreprocessor(config)

    paths = list(iter_image_paths(input_dir))
    if args.max_images > 0:
        paths = paths[: args.max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preprocess_report.csv"
    started = time.time()
    total = 0
    written = 0
    skipped = 0
    failed = 0
    pending_paths = []
    pending_images = []

    def flush_batch(writer):
        nonlocal written, failed
        if not pending_paths:
            return
        outputs = preprocessor.process_batch(pending_images)
        for src, image in zip(pending_paths, outputs):
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            row = {
                "relative_path": str(rel),
                "source": str(src),
                "output": str(dst),
                "status": "ok",
                "error": "",
            }
            try:
                if image is None or image.size == 0:
                    raise RuntimeError("empty_preprocessed_image")
                if not write_image(dst, image):
                    raise RuntimeError("failed_to_write_image")
                written += 1
            except Exception as exc:
                failed += 1
                row.update({"status": "failed", "error": str(exc)})
            writer.writerow(row)
        pending_paths.clear()
        pending_images.clear()

    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "source", "output", "status", "error"],
        )
        writer.writeheader()
        for src in paths:
            total += 1
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            if dst.exists() and not args.overwrite:
                skipped += 1
                writer.writerow(
                    {
                        "relative_path": str(rel),
                        "source": str(src),
                        "output": str(dst),
                        "status": "skipped_exists",
                        "error": "",
                    }
                )
                continue
            image = read_image(src)
            if image is None or image.size == 0:
                failed += 1
                writer.writerow(
                    {
                        "relative_path": str(rel),
                        "source": str(src),
                        "output": str(dst),
                        "status": "failed",
                        "error": "failed_to_read_image",
                    }
                )
                continue
            pending_paths.append(src)
            pending_images.append(image)
            if len(pending_paths) >= max(1, args.batch_size):
                flush_batch(writer)
            if args.log_interval and total % args.log_interval == 0:
                elapsed = time.time() - started
                print(
                    f"processed={total}/{len(paths)} written={written} "
                    f"skipped={skipped} failed={failed} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        flush_batch(writer)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "foreground_checkpoint": str(args.foreground_checkpoint),
        "foreground_device": args.foreground_device,
        "foreground_threshold": args.foreground_threshold,
        "foreground_min_area": args.foreground_min_area,
        "foreground_close_ratio": args.foreground_close_ratio,
        "foreground_feather_ratio": args.foreground_feather_ratio,
        "background_value": args.background_value,
        "orientation_model": str(args.orientation_model) if args.orientation_model else None,
        "orientation_target": args.orientation_target,
        "total": total,
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": round(time.time() - started, 2),
        "preprocessor_stats": dict(preprocessor.stats),
        "report_path": str(report_path),
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
