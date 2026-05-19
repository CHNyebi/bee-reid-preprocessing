# Codex Handoff

This repository is the portable source for bee ReID crop preprocessing. The target server may use another tracker and another ReID model, so do not assume any DeepSORT-specific code exists there.

## What This Repo Provides

- Runtime crop preprocessor: `bee_reid_preprocessing`
- Orientation utilities and head-side classifier wrapper: `bee_orientation_normalizer`
- Foreground model training and inference scripts
- Direction classifier training and batch-normalization scripts
- Current trained model weights
- Training datasets for both preprocessing sub-models
- Packed ReID image datasets from the source server

SAM is deprecated. Some packed ReID dataset archives still contain `sam_post` in their names because that was the historical experiment name. Do not add a SAM dependency for new deployment.

## First Setup On A New Server

```bash
git clone https://github.com/CHNyebi/bee-reid-preprocessing.git
cd bee-reid-preprocessing
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python scripts/smoke_test_preprocessing.py --device cpu
```

Use `--device cuda` after CUDA/PyTorch is confirmed.

## Deployment API

Insert preprocessing after detection crop extraction and before the ReID model transform.

```python
from bee_reid_preprocessing import BeeReIDCropPreprocessor

preprocessor = BeeReIDCropPreprocessor(mode="foreground_aligned", foreground_device="cuda")

for crop_bgr in detection_crops:
    crop_for_reid = preprocessor.process(crop_bgr)
    # Continue with the target ReID model resize/normalize/tensor conversion.
```

Input: OpenCV `BGR uint8` crop, shape `(H, W, 3)`.

Output: OpenCV `BGR uint8` crop. Aligned modes may change height/width after rotation. This repo intentionally does not do ReID-model-specific resize or normalization.

Recommended modes:

- `foreground_aligned`: clean background and normalize orientation
- `raw_aligned`: normalize orientation while preserving raw crop pixels
- `foreground`: only remove background
- `none`: no-op smoke/debug mode

## Bundled Model Weights

- Foreground model: `models/bee_foreground_unetpp_resnet18_v2/best_model.pt`
- Direction classifier: `models/bee_direction_resnet18_triclass.joblib`

Both are Git LFS files. If they look tiny after cloning, run `git lfs pull`.

## Sub-Model Training Data

Direction classifier data:

- Labels: `data/bee_direction_labeled_batch001_004/labels.csv`
- Images: `data/bee_direction_labeled_batch001_003/images/` and `data/bee_direction_labeled_batch004_200/images/`
- Current count: 500 images, 500 label rows

Foreground segmentation data:

- Dataset root: `data/bee_foreground_v2/dataset`
- Train: 211 images and 211 masks
- Val: 53 images and 53 masks
- Metadata: `data/bee_foreground_v2/*.csv` and `data/bee_foreground_v2/dataset_summary.json`

## Re-train The Sub-Models

Direction classifier:

```bash
python scripts/train_resnet_direction.py \
  --labels-csv data/bee_direction_labeled_batch001_004/labels.csv \
  --output-model models/bee_direction_resnet18_triclass.joblib
```

Foreground segmentation:

```bash
python scripts/train_bee_foreground_segmentation.py \
  --dataset-dir data/bee_foreground_v2/dataset \
  --output-dir models/bee_foreground_unetpp_resnet18_v2
```

After retraining, rerun:

```bash
python scripts/smoke_test_preprocessing.py --device cpu
```

## Packed ReID Datasets

Archives live in `data/reid_datasets/`; checksums and file counts live in `data/reid_datasets/manifest.csv`.

Extract:

```bash
python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir data/reid_extracted
```

The packed archives are:

- `eval.tar.gz`
- `eval_bee_unetpp_v2_sam_post.tar.gz`
- `eval_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `eval_raw_unet_aligned.tar.gz`
- `train_20260501.tar.gz`
- `train_20260501_bee_unetpp_v2_sam_post.tar.gz`
- `train_20260501_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `train_20260501_raw_unet_aligned.tar.gz`
- `val_20260501.tar.gz`
- `val_20260501_bee_unetpp_v2_sam_post.tar.gz`
- `val_20260501_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `val_20260501_raw_unet_aligned.tar.gz`

## Validation Checklist

Run these before integrating into another tracking codebase:

```bash
python -m py_compile \
  bee_reid_preprocessing/__init__.py \
  bee_reid_preprocessing/preprocessor.py \
  bee_orientation_normalizer/__init__.py \
  bee_orientation_normalizer/orientation.py \
  scripts/*.py

python scripts/smoke_test_preprocessing.py --device cpu

python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir /tmp/bee_reid_dataset_check \
  --overwrite
```

For online integration, confirm the tracker passes a valid BGR crop and the downstream ReID model still performs its own expected resize/normalization.
