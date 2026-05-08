# Feature Specification: Remote Sensing Change Detection Model

**Feature Branch**: `001-remote-change-model`  
**Created**: 2026-05-08  
**Status**: Draft  
**Input**: User description: "Develop a remote sensing change detection model that takes two temporal images A and B and predicts a binary change mask, with emphasis on stable real-world predictions and low false positives rather than leaderboard-style metric optimization."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train on Paired Change Data (Priority: P1)

A practitioner can prepare paired before/after remote sensing images and supervised labels in the expected training and validation folders, then run a training workflow that learns to predict real change masks from paired inputs.

**Why this priority**: Training is the foundation for producing application-specific predictions and must correctly use the paired temporal data and labels.

**Independent Test**: Can be fully tested by placing valid paired images and labels under the training and validation folders, starting the training workflow, and confirming that training completes and produces a reusable best model artifact.

**Acceptance Scenarios**:

1. **Given** paired 128x128 images with matching names exist under `data/train/A`, `data/train/B`, and `data/train/label`, **When** training starts, **Then** the system loads each before image, after image, and label as one supervised sample.
2. **Given** paired validation images and labels exist under `data/val/A`, `data/val/B`, and `data/val/label`, **When** training runs validation, **Then** the system evaluates the current model on validation samples without requiring test labels.
3. **Given** training completes successfully, **When** model selection is performed, **Then** the best available model artifact is saved for later prediction.

---

### User Story 2 - Predict Test Change Masks (Priority: P2)

A practitioner can run prediction on unlabeled test image pairs and receive one binary mask per test pair, with filenames that correspond to the source imagery.

**Why this priority**: The primary real-world value is producing usable change masks for new before/after imagery where labels are not available.

**Independent Test**: Can be fully tested by placing same-named images under `data/test/A` and `data/test/B`, running prediction with a trained model artifact, and checking the generated mask files.

**Acceptance Scenarios**:

1. **Given** same-named 128x128 images exist under `data/test/A` and `data/test/B`, **When** prediction runs, **Then** the system pairs them by filename and processes each pair exactly once.
2. **Given** prediction finishes, **When** output files are inspected, **Then** each test pair has a corresponding single-channel PNG mask in `outputs/predict_masks/`.
3. **Given** a generated mask is opened, **When** its dimensions and values are checked, **Then** it is 128x128 pixels and contains only black non-change pixels and white change pixels.

---

### User Story 3 - Control Practical False Positives (Priority: P3)

A practitioner can train and use a model whose predictions are biased toward stable, plausible real-world change regions and avoid common non-change differences such as seasonal wetland variation, boats, cloud or haze occlusion, lighting shifts, color differences, minor registration error, and sensor imaging differences.

**Why this priority**: The project goal is practical prediction quality, especially fewer false alarms, rather than maximizing complex evaluation reports.

**Independent Test**: Can be tested by reviewing predictions on representative validation or test pairs containing common nuisance differences and confirming that outputs suppress non-real changes while preserving true change regions.

**Acceptance Scenarios**:

1. **Given** image pairs contain seasonal wetland appearance differences without real land-cover change, **When** prediction runs, **Then** the generated mask should avoid marking broad seasonal texture or color differences as change.
2. **Given** image pairs contain boats on rivers or water surfaces, **When** prediction runs, **Then** the generated mask should avoid marking transient boats as persistent change.
3. **Given** image pairs contain cloud, haze, lighting, color, or minor alignment differences, **When** prediction runs, **Then** the generated mask should suppress those nuisance differences where no real change is present.

---

### Edge Cases

- Training or validation folders contain unmatched filenames across A, B, and label.
- Test folders contain unmatched filenames across A and B.
- Input files use supported image extensions but include invalid, unreadable, or non-128x128 content.
- Label masks contain grayscale or color values instead of strict binary values.
- A validation set is very small or contains no positive change pixels.
- Prediction output directory already contains masks from a previous run.
- Threshold values at prediction time are outside the accepted probability range.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support the data layout `data/train/A`, `data/train/B`, `data/train/label`, `data/val/A`, `data/val/B`, `data/val/label`, `data/test/A`, and `data/test/B`.
- **FR-002**: The system MUST create supervised training samples by pairing before image A, after image B, and label mask by matching filenames.
- **FR-003**: The system MUST create prediction samples by pairing test before image A and after image B by matching filenames.
- **FR-004**: The system MUST reject or clearly report missing counterpart files instead of silently training or predicting on incorrect pairs.
- **FR-005**: The system MUST support PNG and JPG image inputs for A and B.
- **FR-006**: The system MUST support supervised label masks for training and validation and normalize them into binary change/non-change targets.
- **FR-007**: The system MUST keep the spatial correspondence among A, B, and label during training-time augmentation.
- **FR-008**: The system MUST support paired geometric augmentation, including flips and rotations, without breaking A/B/label alignment.
- **FR-009**: The system MUST support appearance augmentation that improves robustness to brightness, contrast, color, lighting, and sensor differences without changing the label geometry.
- **FR-010**: The system MUST provide configurable training settings for data paths, output paths, batch size, epoch count, learning rate, and prediction threshold.
- **FR-011**: The system MUST provide default settings suitable for training on a typical single GPU workstation.
- **FR-012**: The system MUST provide separate runnable workflows for training, validation, and test prediction.
- **FR-013**: The training workflow MUST save the best model artifact to a clear checkpoint output location.
- **FR-014**: The validation workflow MUST summarize model behavior with simple training or validation logs without making complex metrics the central product output.
- **FR-015**: The prediction workflow MUST load a saved model artifact and generate masks for all valid test pairs.
- **FR-016**: The prediction workflow MUST allow users to set the binary mask threshold, with a default threshold of 0.5.
- **FR-017**: Generated prediction masks MUST be saved as PNG files under a clear output directory such as `outputs/predict_masks/`.
- **FR-018**: Generated prediction masks MUST keep the input spatial size of 128x128 pixels.
- **FR-019**: Generated prediction masks MUST be single-channel binary images where change is white and non-change is black.
- **FR-020**: The model behavior MUST prioritize coherent real change regions, complete boundaries, and false-positive control over extensive metric reporting.
- **FR-021**: The system MUST be organized so future model replacements can reuse the same data loading, training, validation, prediction, configuration, and mask output contracts.
- **FR-022**: The project documentation MUST explain dataset preparation, training, validation, prediction, configuration, checkpoint output, and prediction mask output.

### Key Entities

- **Temporal Image Pair**: A matched before image and after image representing the same scene at two times; key attributes include filename, split, before image, after image, image size, and image format.
- **Change Label Mask**: A supervised training or validation target aligned with a temporal image pair; key attributes include filename, binary change pixels, binary non-change pixels, and spatial size.
- **Prediction Mask**: A generated binary output for a test temporal image pair; key attributes include source filename, threshold used, output path, spatial size, and binary pixel values.
- **Model Artifact**: The saved best-performing trained model selected during training; key attributes include checkpoint path, training configuration reference, and intended prediction use.
- **Configuration**: User-adjustable project settings; key attributes include data paths, output paths, training schedule, batch size, learning rate, and prediction threshold.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With a correctly prepared dataset, a user can start training and produce a reusable best model artifact without modifying source files.
- **SC-002**: For every valid same-named test A/B pair, prediction produces exactly one corresponding PNG mask in the configured output directory.
- **SC-003**: 100% of generated prediction masks preserve the required 128x128 dimensions.
- **SC-004**: 100% of generated prediction masks contain only binary non-change and change values.
- **SC-005**: Invalid or unmatched dataset samples are reported with actionable messages before they can corrupt training or prediction results.
- **SC-006**: On representative review cases containing seasonal wetlands, boats, cloud or haze, lighting/color shifts, and minor registration differences, predictions visibly suppress nuisance differences while retaining plausible true change regions.
- **SC-007**: A new practitioner can prepare data, train, validate, and predict by following the README in under 30 minutes after dependencies and data are available.
- **SC-008**: The project remains usable when replacing the underlying change detection model, as long as the same input and output contracts are preserved.

## Assumptions

- The initial scope is a local research or application workflow rather than a hosted service.
- Input image pairs are expected to be pre-cropped or prepared at 128x128 pixels before use.
- Test data does not include labels, so prediction output is the primary deliverable for the test split.
- The validation process may use simple loss or qualitative logging; detailed IoU, F1, precision, and recall reporting is intentionally not a core requirement.
- False-positive control for nuisance changes will be addressed through training data handling, augmentation, model behavior, and thresholding, but the system cannot guarantee perfect semantic rejection of every ambiguous real-world case.
- Existing data filenames are the source of truth for pairing corresponding A, B, and label files.
