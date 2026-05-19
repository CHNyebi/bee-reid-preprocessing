"""Batch-normalize already-masked bee crops to a common horizontal direction."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bee_orientation_normalizer.orientation import (  # noqa: E402
    IMAGE_SUFFIXES,
    OrientationError,
    normalize_bee_orientation,
)


REPORT_FIELDS = [
    "source",
    "output",
    "status",
    "head_side_before",
    "rotated_180",
    "confidence",
    "angle_degrees",
    "torch_left_probability",
    "torch_right_probability",
    "torch_unclear_probability",
    "error",
]


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


def iter_images(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def process_directory(args) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file does not exist: {model_path}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("output-dir must be different from input-dir")

    direction_model = joblib.load(model_path)
    report_path = Path(args.report) if args.report else output_dir / "orientation_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ok_count = 0
    failed_count = 0
    copied_failures = 0
    unclear_count = 0
    flipped_count = 0

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for src in iter_images(input_dir):
            if args.max_images and total >= args.max_images:
                break
            total += 1
            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            if dst.exists() and not args.overwrite:
                writer.writerow({"source": str(src), "output": str(dst), "status": "skipped_exists"})
                continue

            row = {"source": str(src), "output": str(dst)}
            try:
                image = imread(src)
                if image is None:
                    raise OrientationError("failed to read image")
                result = normalize_bee_orientation(
                    image,
                    target=args.target,
                    mask_mode="masked",
                    output_mode="rotate",
                    direction_model=direction_model,
                    clean_background=False,
                )
                if not imwrite(dst, result.image):
                    raise OrientationError("failed to write output image")

                row.update(
                    {
                        "status": "ok",
                        "head_side_before": result.head_side_before,
                        "rotated_180": int(result.rotated_180),
                        "confidence": f"{result.confidence:.6f}",
                        "angle_degrees": f"{result.angle_degrees:.6f}",
                    }
                )
                row.update(
                    {
                        key: f"{value:.6f}"
                        for key, value in result.features.items()
                        if key.startswith("torch_") and isinstance(value, (int, float, np.floating))
                    }
                )
                ok_count += 1
                unclear_count += int(result.head_side_before == "unclear")
                flipped_count += int(result.rotated_180)
            except Exception as exc:  # noqa: BLE001 - report and continue batch jobs.
                failed_count += 1
                row.update({"status": "failed", "error": str(exc)})
                if args.copy_failures:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_failures += 1
                    row["status"] = "failed_copied_original"
            writer.writerow(row)

            if args.log_interval and total % args.log_interval == 0:
                print(
                    f"processed={total} ok={ok_count} failed={failed_count} "
                    f"unclear={unclear_count} flipped_180={flipped_count}"
                )

    print(
        f"done: processed={total} ok={ok_count} failed={failed_count} "
        f"unclear={unclear_count} flipped_180={flipped_count} "
        f"copied_failures={copied_failures}"
    )
    print(f"output_dir: {output_dir}")
    print(f"report: {report_path}")
    return 0 if failed_count == 0 else 1


def add_bool_arg(parser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Root containing already-masked bee images.")
    parser.add_argument("--output-dir", required=True, help="Output root; subfolders are preserved.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "bee_direction_resnet18_triclass.joblib"),
        help="Path to the trained three-class ResNet18 joblib model.",
    )
    parser.add_argument(
        "--target",
        choices=["head-left", "head-right"],
        default="head-left",
        help="Desired final direction for recognizable bees.",
    )
    parser.add_argument("--report", default=None, help="CSV diagnostics path.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional preview limit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    add_bool_arg(parser, "copy-failures", default=True, help_text="Copy original image on failures.")
    parser.add_argument("--log-interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(process_directory(parse_args()))
