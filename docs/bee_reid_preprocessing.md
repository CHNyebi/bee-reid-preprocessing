# Bee ReID Crop Preprocessing

This package is the portable deployment surface for bee ReID preprocessing. It
is independent of the detector, tracker, and ReID network.

## What It Does

- `foreground`: remove crop background with the bee Unet++ foreground model.
- `foreground_aligned`: remove background, align the long body axis, then use
  the direction classifier to make the head point to the configured side.
- `raw_aligned`: use the Unet++ mask only for analysis, but return rotated raw
  crop pixels. This is usually the safest mode when deploying with a new ReID
  model that was trained on raw-looking crops.

SAM is retired. The current deployment path uses only:

- `models/bee_foreground_unetpp_resnet18_v2/best_model.pt`
- `models/bee_direction_resnet18_triclass.joblib`

## Online Use

Insert the preprocessor between detection cropping and ReID feature extraction:

```python
from bee_reid_preprocessing import BeeReIDCropPreprocessor

preprocessor = BeeReIDCropPreprocessor(mode="raw_aligned")

for bbox in detections:
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_bgr = preprocessor.process(crop_bgr)
    feature = reid_model.extract(crop_bgr)
```

For better throughput, call `process_batch()` with all crops from one frame:

```python
processed_crops = preprocessor.process_batch(crops_bgr)
features = reid_model.extract_batch(processed_crops)
```

## Offline Use

For an ImageFolder-style dataset:

```bash
python scripts/preprocess_bee_reid_crops.py \
  --input-dir path/to/reid_crops \
  --output-dir path/to/reid_crops_raw_aligned \
  --mode raw_aligned \
  --foreground-checkpoint models/bee_foreground_unetpp_resnet18_v2/best_model.pt \
  --orientation-model models/bee_direction_resnet18_triclass.joblib \
  --batch-size 64 \
  --overwrite
```

The script preserves subdirectories and writes:

- processed images under `--output-dir`
- `preprocess_report.csv`
- `preprocess_summary.json`

## Retraining

Foreground segmentation:

```bash
python scripts/train_bee_foreground_segmentation.py \
  --dataset-dir data/bee_foreground_v2/dataset \
  --output-dir outputs/bee_foreground_retrain \
  --encoder resnet18 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 120
```

Direction classifier:

```bash
python scripts/train_resnet_direction.py \
  --labels-csv data/bee_direction_labeled_batch001_004/labels.csv \
  --output-model models/bee_direction_resnet18_triclass_retrained.joblib \
  --report-dir outputs/bee_direction_retrain \
  --device cuda
```

Both model artifacts should be stored through Git LFS if they are committed.
