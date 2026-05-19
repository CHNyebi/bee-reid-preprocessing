# Bee ReID Preprocessing

[中文说明](README.zh-CN.md)

Portable bee crop preprocessing for deployment with another tracker and another ReID model.

This repository contains:

- foreground/background removal with the trained Unet++ model
- long-axis rotation and head-side normalization with the trained direction classifier
- training scripts and small training datasets for both preprocessing models
- packed ReID image datasets from the current server

The runtime API is intentionally tracker-agnostic: it accepts OpenCV crops and returns OpenCV crops.

```python
from bee_reid_preprocessing import BeeReIDCropPreprocessor

preprocessor = BeeReIDCropPreprocessor(mode="foreground_aligned")
crop_for_reid = preprocessor.process(crop_bgr)
```

Input and output are both `numpy.ndarray` images in OpenCV `BGR uint8` format. Aligned modes may change height/width after rotation; the downstream ReID model should still do its own expected resize and normalization.

## Modes

- `none`: return the crop unchanged
- `foreground`: remove background only
- `foreground_aligned`: remove background, align the long body axis, and normalize head side
- `raw_aligned`: use the foreground mask for alignment/head-side analysis, but return aligned raw pixels

For a ReID model that should see a clean, direction-normalized crop, start with `foreground_aligned`. For a model that should preserve original pixel texture but use consistent orientation, start with `raw_aligned`.

## Install

```bash
git clone https://github.com/CHNyebi/bee-reid-preprocessing.git
cd bee-reid-preprocessing
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

Run the smoke test before wiring this into another tracker:

```bash
python scripts/smoke_test_preprocessing.py --device cpu
```

Use `--device cuda` on a GPU server.

For a Codex instance on another server with no project memory, start with
`docs/CODEX_HANDOFF.md`.

## Offline Dataset Preprocessing

```bash
python scripts/preprocess_bee_reid_crops.py \
  --input-dir data/reid_extracted/train_20260501 \
  --output-dir outputs/train_20260501_foreground_aligned \
  --mode foreground_aligned \
  --foreground-device cuda \
  --batch-size 64 \
  --overwrite
```

## ReID Datasets

The packed ReID datasets are stored under `data/reid_datasets/`. Extract them with:

```bash
python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir data/reid_extracted
```

The names containing `sam_post` are legacy dataset names from earlier experiments. New deployment code in this repository does not depend on SAM.

## Direction Classifier Data And Metrics

The bundled direction classifier is a ResNet18 three-class model:

- `head_left`
- `head_right`
- `unclear`

It corresponds to `data/bee_direction_newmask_train400_20260517.zip`, also extracted in `data/bee_direction_newmask_train400_20260517/` for immediate retraining. The training set has 400 already horizontally aligned masked bee crops: `head_left=170`, `head_right=150`, `unclear=80`.

Known validation for the current model:

- 5-fold CV accuracy: `77.00%`
- 5-fold CV balanced accuracy: `75.93%`
- CV confusion matrix in class order `head_left, head_right, unclear`: `[[130, 17, 23], [12, 122, 16], [8, 16, 56]]`
- Full-flow 113-sample test accuracy: `88/113 = 77.88%`
- Full-flow per-class recall: `head_left=36/42`, `head_right=30/39`, `unclear=22/32`

The old `bee_direction_labeled_batch001_003` direction dataset is deprecated and is not included for new training.

## Train Preprocessing Models

Direction classifier:

```bash
python scripts/train_resnet_direction.py \
  --labels-csv data/bee_direction_newmask_train400_20260517/labels.csv \
  --output-model models/bee_direction_resnet18_triclass.joblib
```

Retraining is optional. The default model already points to the new-mask 400-sample training source in its joblib metadata.

Foreground segmentation:

```bash
python scripts/train_bee_foreground_segmentation.py \
  --dataset-dir data/bee_foreground_v2/dataset \
  --output-dir models/bee_foreground_unetpp_resnet18_v2
```

## Model Defaults

The default runtime paths are:

- `models/bee_foreground_unetpp_resnet18_v2/best_model.pt`
- `models/bee_direction_resnet18_triclass.joblib`

Both are tracked with Git LFS.
