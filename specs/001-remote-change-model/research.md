# Research: Remote Sensing Change Detection Model

## Decision: Use a compact Siamese U-Net style baseline first

**Rationale**: The input size is fixed at 128x128, so a compact encoder-decoder with shared or symmetric feature extraction is easier to train, cheaper to run, and more suitable as a reliable baseline than a heavier Transformer-first design. A Siamese design encourages the model to compare semantic features from A and B instead of relying only on raw pixel differences.

**Alternatives considered**: UNet++ can improve dense segmentation detail but adds complexity. BIT and ChangeFormer are stronger research architectures but are heavier and require more careful tuning for small datasets. A plain concatenated U-Net is simple but less explicit about temporal comparison.

## Decision: Use BCE + Dice loss as the default objective

**Rationale**: Binary change masks are often imbalanced, with many more non-change pixels than change pixels. BCE provides stable pixel-wise supervision while Dice encourages region overlap and helps preserve connected change areas.

**Alternatives considered**: BCE alone is simple but can under-emphasize small changed regions. Dice alone can be unstable early in training. Focal loss may reduce easy negatives but adds another hyperparameter and is better introduced after the baseline is working.

## Decision: Keep evaluation lightweight and operational

**Rationale**: The specification explicitly prioritizes real-world visual quality and false-positive control over complex metric reporting. Validation should track loss and optionally produce sample masks for visual review, while avoiding a metrics-heavy workflow as the primary product.

**Alternatives considered**: Full IoU/F1/precision/recall dashboards were rejected because they shift the project focus away from practical mask output and are not required for acceptance.

## Decision: Pair samples by filename and fail early on mismatches

**Rationale**: The dataset contract depends on same-named files across A, B, and label directories. Early validation prevents silent A/B/label misalignment, which would corrupt training and make predictions unreliable.

**Alternatives considered**: Sorting all files and zipping by order is fragile. Metadata manifests add operational overhead not requested for the initial workflow.

## Decision: Apply geometric transforms jointly and appearance transforms carefully

**Rationale**: Flips and rotations must be applied consistently to A, B, and label to preserve spatial alignment. Brightness, contrast, color, haze-like, and mild noise perturbations should improve robustness to lighting, sensor, and atmospheric differences without changing label geometry.

**Alternatives considered**: Independent augmentation for A and B can model acquisition differences but risks teaching artificial false changes if too strong. No augmentation would make the model more sensitive to nuisance differences.

## Decision: Save binary masks as single-channel PNG files

**Rationale**: PNG is lossless, common for masks, and preserves exact binary values. Single-channel black/white output is easy to inspect and consume downstream.

**Alternatives considered**: JPG masks are lossy and can introduce non-binary values. Multi-channel RGB masks are visually convenient but less clean as a data contract.

## Decision: Use YAML configuration with CLI overrides where needed

**Rationale**: The user needs to change batch size, epochs, learning rate, data paths, output paths, and threshold without modifying source files. YAML is readable and sufficient for a local training workflow.

**Alternatives considered**: Hard-coded constants are too rigid. A database or experiment tracker is unnecessary for the requested scope.
