# Model Artifacts

These artifacts are used by the portable bee ReID crop preprocessing path.

| Path | Purpose |
| --- | --- |
| `bee_foreground_unetpp_resnet18_v2/best_model.pt` | Unet++ foreground segmentation checkpoint |
| `bee_direction_resnet18_triclass.joblib` | ResNet18 head-direction classifier |

Large model files are tracked with Git LFS. After cloning on a new server, run:

```bash
git lfs install
git lfs pull
```

If you retrain either model, keep the new artifact under `models/` and commit it
through Git LFS.
