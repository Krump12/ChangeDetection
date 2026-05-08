# CLI Contract: Remote Sensing Change Detection Workflows

This project exposes local command-line workflows. Commands should be runnable from the repository root.

## Common Inputs

- `--config PATH`: Optional path to a YAML configuration file. Default: `configs/default.yaml`.
- CLI arguments may override configuration values when explicitly supported.
- Commands must fail with actionable messages when required data paths, pairs, labels, or checkpoints are missing.

## Train

```bash
python -m src.train --config configs/default.yaml
```

**Required data**

- `data/train/A`
- `data/train/B`
- `data/train/label`
- `data/val/A`
- `data/val/B`
- `data/val/label`

**Expected behavior**

- Pair train and validation samples by filename.
- Apply spatially aligned augmentation to A, B, and label during training.
- Save the best model artifact under the configured checkpoint directory.
- Print simple progress logs including training loss and validation loss.

**Outputs**

- Best checkpoint, default path pattern: `outputs/checkpoints/best_model.pth`.
- Training log text to console and optionally to the output directory.

## Validate

```bash
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
```

**Required data**

- `data/val/A`
- `data/val/B`
- `data/val/label`

**Expected behavior**

- Load the checkpoint.
- Pair validation samples by filename.
- Report simple validation loss and optional qualitative summary.
- Do not require test labels.

## Predict

```bash
python -m src.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5
```

**Required data**

- `data/test/A`
- `data/test/B`

**Expected behavior**

- Pair test samples by filename.
- Load the checkpoint.
- Produce one binary PNG mask for every valid test pair.
- Preserve the 128x128 spatial size.
- Use white pixels for change and black pixels for non-change.

**Outputs**

- Prediction masks under `outputs/predict_masks/` by default.
- Mask filenames correspond to input filenames and use `.png` extension.

## Error Conditions

- Missing A/B counterpart: command exits before processing with the unmatched filename listed.
- Missing train or validation label: train or validate exits before processing with the unmatched filename listed.
- Invalid image or mask size: command exits or skips according to configured strictness; default is fail-fast.
- Threshold outside `[0.0, 1.0]`: predict exits with an actionable validation error.
- Missing checkpoint for validation or prediction: command exits before loading data.
