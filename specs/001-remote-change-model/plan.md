# Implementation Plan: Remote Sensing Change Detection Model

**Branch**: `001-remote-change-model` | **Date**: 2026-05-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-remote-change-model/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a local remote sensing change detection training and inference workflow for paired 128x128 before/after images. The implementation will use a configurable Python/PyTorch CLI project with file-based datasets, a compact Siamese U-Net style baseline, BCE+Dice supervision, paired spatial augmentation, appearance robustness augmentation, checkpoint selection, and PNG binary mask prediction under `outputs/predict_masks/`.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: PyTorch, torchvision, Pillow, NumPy, PyYAML, tqdm; optional OpenCV or Albumentations for image transforms if available  
**Storage**: Local filesystem for input images, label masks, checkpoints, logs, and prediction masks  
**Testing**: pytest for dataset pairing, mask conversion, config loading, model forward shape, and prediction output contract  
**Target Platform**: Local workstation with CPU support and optional CUDA GPU acceleration  
**Project Type**: Single-project machine learning CLI application  
**Performance Goals**: Train on 128x128 imagery with default settings suitable for a typical single GPU; predict one 128x128 mask per valid test pair without manual intervention  
**Constraints**: Preserve A/B/label spatial alignment; output single-channel binary PNG masks at 128x128; do not require test labels; keep metric reporting secondary to practical prediction output  
**Scale/Scope**: One supervised binary change detection model, one dataset layout, local train/validate/predict workflows, and swappable model module boundaries

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The current constitution file still contains placeholder principles and does not define enforceable project gates. This plan therefore applies the feature specification as the controlling requirement source. No constitution violations are identified.

Post-design check: Phase 0 and Phase 1 artifacts keep the same scope, remain local and file-based, and do not introduce additional services or external interfaces beyond documented CLI commands. No constitution violations are identified.

## Project Structure

### Documentation (this feature)

```text
specs/001-remote-change-model/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- cli.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- default.yaml

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

src/
|-- datasets/
|   `-- change_dataset.py
|-- losses/
|   `-- losses.py
|-- models/
|   `-- change_model.py
|-- utils/
|   |-- image_utils.py
|   `-- mask_utils.py
|-- train.py
|-- validate.py
`-- predict.py

tests/
|-- contract/
|   `-- test_cli_contract.py
|-- integration/
|   `-- test_train_predict_smoke.py
`-- unit/
    |-- test_change_dataset.py
    |-- test_mask_utils.py
    `-- test_model_shapes.py

outputs/
|-- checkpoints/
`-- predict_masks/

requirements.txt
README.md
```

**Structure Decision**: Use a single Python ML CLI project matching the requested repository shape. Keep model code isolated under `src/models/` so future Siamese U-Net, UNet++, BIT, ChangeFormer, or lightweight Transformer variants can preserve the same input/output contract.

## Complexity Tracking

No constitution violations or extra project complexity require justification.
