# Quickstart: Remote Sensing Change Detection Model

## 1. Prepare Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use the platform-appropriate activation command if not on Windows PowerShell.

## 2. Prepare Data

Place images in the required layout:

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

- A and B images in each split must use matching filenames.
- Train and validation labels must match the image filenames.
- Test has no labels.
- Images and masks are expected to be 128x128.
- Input images may be PNG or JPG.

## 3. Review Configuration

Default configuration path:

```text
configs/default.yaml
```

Expected configurable values include data paths, output paths, batch size, epoch count, learning rate, device, seed, and prediction threshold.

## 4. Train

```bash
python -m src.train --config configs/default.yaml
```

Expected result:

```text
outputs/checkpoints/best_model.pth
```

## 5. Validate

```bash
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Validation should report simple loss-oriented feedback. Detailed metric dashboards are not required for this feature.

## 6. Predict Test Masks

```bash
python -m src.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5
```

Expected result:

```text
outputs/predict_masks/
```

Each output mask should:

- Correspond to one test pair filename.
- Be saved as PNG.
- Be 128x128.
- Use white for changed pixels and black for unchanged pixels.

## 7. Practical Review

Inspect predictions on cases with seasonal wetlands, boats, cloud or haze, lighting/color shifts, and minor registration differences. The desired behavior is conservative: suppress nuisance differences while keeping coherent real change regions.
