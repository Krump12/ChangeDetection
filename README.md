# Remote Sensing Change Detection

This project trains a local paired-image change detection model. Input A is the earlier remote sensing image, input B is the later image, and output is a binary change mask with the configured image size.

The implementation is intentionally practical: it provides train, validate, and predict commands, saves the best checkpoint, and writes black/white PNG masks for test pairs. Detailed metric dashboards are not the core workflow.

## Data Layout

```text
data/
|-- train/
|   |-- A/
|   |-- B/
|   `-- label/
|-- val/
|   |-- A/
|   |-- B/
|   `-- label/
`-- test/
    |-- A/
    `-- B/
```

Rules:

- Files are paired by identical filenames across A, B, and label folders.
- `train` and `val` require labels; `test` does not.
- Images and masks must match `data.image_size` in `configs/default.yaml`. The default is `[128, 128]` as `[width, height]`.
- A and B may be PNG or JPG.
- Prediction masks are saved as single-channel PNG files with white change pixels and black non-change pixels.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use the activation command for your shell if not using Windows PowerShell.

## Configure

Edit `configs/default.yaml` to change image size, data paths, output paths, batch size, epochs, learning rate, device, threshold, and augmentation settings.

Set the expected input/output size with:

```yaml
data:
  image_size: [128, 128]  # [width, height]
```

The loader validates this size. It does not resize images automatically, so A, B, and label files must already have the configured dimensions.

Default outputs:

```text
outputs/checkpoints/best_model.pth
outputs/predict_masks/
```

## Train

```bash
python -m src.train --config configs/default.yaml
```

The training command validates data pairing, applies aligned spatial augmentation, logs train/validation loss, and saves the best checkpoint.

## Validate

```bash
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Optionally export a small set of validation preview masks:

```bash
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --export-samples
```

## Predict

```bash
python -m src.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5
```

The command pairs `data/test/A` and `data/test/B` by filename and writes corresponding `.png` masks to `outputs/predict_masks/`.

## Practical Review

Review predictions conservatively. Seasonal wetland variation, boats on rivers, cloud or haze, lighting changes, color shifts, mild sensor differences, and slight registration errors should not be treated as persistent real change. Increase the threshold or `predict.min_region_size` in `configs/default.yaml` when false positives are too frequent.

## Quick Validation

After preparing data and installing dependencies:

```bash
pytest
python -m src.train --config configs/default.yaml
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
python -m src.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5
```

Expected results are a checkpoint under `outputs/checkpoints/` and one binary PNG mask per valid test pair under `outputs/predict_masks/`, using the configured image size.
