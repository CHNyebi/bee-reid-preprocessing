# Data Layout

This directory contains all data needed to reproduce or deploy the preprocessing models.

- `bee_direction_newmask_train400_20260517.zip`: original packed direction classifier training dataset from the new masked-bee workflow
- `bee_direction_newmask_train400_20260517/`: extracted direction classifier images and labels for immediate retraining
- `bee_foreground_v2/`: foreground segmentation images, masks, metadata, and split files
- `reid_datasets/`: packed ReID image datasets from the source server

The ReID datasets are stored as archives to keep the repository usable. Extract them with `scripts/extract_reid_datasets.py`.
