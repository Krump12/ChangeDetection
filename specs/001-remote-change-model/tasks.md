# Tasks: Remote Sensing Change Detection Model

**Input**: Design documents from `/specs/001-remote-change-model/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Included because the implementation plan requires pytest coverage for dataset pairing, mask conversion, configuration loading, model shape, CLI contract, and train/predict smoke behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files and does not depend on incomplete tasks
- **[Story]**: Maps task to a specific user story, for example `[US1]`
- Every task includes exact file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the project skeleton, dependency list, package markers, and default configuration.

- [x] T001 Create repository directories `configs/`, `src/datasets/`, `src/models/`, `src/losses/`, `src/utils/`, `tests/unit/`, `tests/contract/`, `tests/integration/`, `outputs/checkpoints/`, and `outputs/predict_masks/`
- [x] T002 Create Python package marker files `src/__init__.py`, `src/datasets/__init__.py`, `src/models/__init__.py`, `src/losses/__init__.py`, and `src/utils/__init__.py`
- [x] T003 Create dependency list in `requirements.txt` with PyTorch, torchvision, Pillow, NumPy, PyYAML, tqdm, and pytest
- [x] T004 Create default training and prediction settings in `configs/default.yaml`
- [x] T005 [P] Create `.gitignore` entries for caches, virtual environments, checkpoints, generated masks, and local data artifacts in `.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared configuration, image, mask, pairing, and model utilities required by all workflows.

**CRITICAL**: No user story work can begin until this phase is complete.

- [x] T006 Implement YAML config loading, CLI override merging, seed setup, output directory creation, and device selection helpers in `src/utils/config_utils.py`
- [x] T007 Implement image loading, RGB conversion, 128x128 validation, tensor conversion, and supported extension discovery in `src/utils/image_utils.py`
- [x] T008 Implement label normalization, probability thresholding, binary PNG writing, and binary value validation in `src/utils/mask_utils.py`
- [x] T009 Implement filename-based A/B/label pair discovery with fail-fast mismatch reporting in `src/datasets/change_dataset.py`
- [x] T010 Implement paired geometric augmentation and cautious appearance augmentation utilities in `src/datasets/change_dataset.py`
- [x] T011 Implement compact Siamese U-Net style model with 2-image input and 1-channel logit output in `src/models/change_model.py`
- [x] T012 Implement Dice loss and BCE+Dice combined loss in `src/losses/losses.py`
- [x] T013 [P] Add unit tests for mask normalization, thresholding, and PNG binary value behavior in `tests/unit/test_mask_utils.py`
- [x] T014 [P] Add unit tests for model forward output shape `[N, 1, 128, 128]` in `tests/unit/test_model_shapes.py`
- [x] T015 Add unit tests for config loading defaults and threshold validation in `tests/unit/test_config_utils.py`
- [x] T016 Add unit tests for filename pairing, missing counterpart errors, label loading, and 128x128 validation in `tests/unit/test_change_dataset.py`

**Checkpoint**: Shared dataset, config, model, loss, and utility code is ready for user story implementation.

---

## Phase 3: User Story 1 - Train on Paired Change Data (Priority: P1) MVP

**Goal**: A practitioner can train on paired `data/train` and `data/val` samples and save a reusable best model artifact.

**Independent Test**: Place a tiny valid paired dataset under train and val folders, run training for one epoch, and confirm that `outputs/checkpoints/best_model.pth` is produced with simple train/validation loss logs.

### Tests for User Story 1

- [x] T017 [P] [US1] Add CLI contract test for `python -m src.train --config configs/default.yaml` argument parsing and required path validation in `tests/contract/test_cli_contract.py`
- [x] T018 [P] [US1] Add integration smoke test that builds a tiny synthetic train/val dataset and verifies one-epoch checkpoint creation in `tests/integration/test_train_predict_smoke.py`

### Implementation for User Story 1

- [x] T019 [US1] Implement supervised train and validation dataset modes returning A tensor, B tensor, label tensor, and filename in `src/datasets/change_dataset.py`
- [x] T020 [US1] Implement training loop, optimizer setup, loss computation, validation loss pass, best checkpoint saving, and console logging in `src/train.py`
- [x] T021 [US1] Implement standalone validation workflow that loads a checkpoint and reports validation loss in `src/validate.py`
- [x] T022 [US1] Wire train and validate CLI arguments `--config` and `--checkpoint` with actionable error messages in `src/train.py` and `src/validate.py`
- [x] T023 [US1] Add README sections for dataset preparation, configuration, training, validation, and checkpoint output in `README.md`

**Checkpoint**: User Story 1 is fully functional and testable independently as the MVP.

---

## Phase 4: User Story 2 - Predict Test Change Masks (Priority: P2)

**Goal**: A practitioner can run prediction on unlabeled same-named test A/B pairs and receive one binary PNG mask per pair.

**Independent Test**: Place same-named images under `data/test/A` and `data/test/B`, run prediction with a valid checkpoint, and confirm each output mask is 128x128, PNG, binary, and filename-aligned.

### Tests for User Story 2

- [x] T024 [P] [US2] Add CLI contract test for `python -m src.predict --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pth --threshold 0.5` threshold and checkpoint validation in `tests/contract/test_cli_contract.py`
- [x] T025 [P] [US2] Extend integration smoke test to run prediction on synthetic test pairs and verify binary PNG masks in `tests/integration/test_train_predict_smoke.py`

### Implementation for User Story 2

- [x] T026 [US2] Implement test dataset mode without labels returning A tensor, B tensor, and filename in `src/datasets/change_dataset.py`
- [x] T027 [US2] Implement checkpoint loading, inference loop, sigmoid thresholding, and mask writing in `src/predict.py`
- [x] T028 [US2] Preserve input filename correspondence and force `.png` mask extension in `src/predict.py`
- [x] T029 [US2] Validate prediction output dimensions and binary values before saving in `src/utils/mask_utils.py`
- [x] T030 [US2] Add README prediction instructions, threshold usage, and output mask contract in `README.md`

**Checkpoint**: User Stories 1 and 2 both work independently with the documented CLI commands.

---

## Phase 5: User Story 3 - Control Practical False Positives (Priority: P3)

**Goal**: Improve practical prediction stability by reducing nuisance false positives while preserving coherent true change regions.

**Independent Test**: Review predictions or synthetic validation cases containing seasonal/brightness/color/cloud-like/water-object nuisance differences and confirm conservative masks with fewer broad false positives.

### Tests for User Story 3

- [x] T031 [P] [US3] Add unit tests that paired geometric augmentation preserves A/B/label spatial alignment in `tests/unit/test_change_dataset.py`
- [x] T032 [P] [US3] Add unit tests for conservative threshold and minimum-region post-processing behavior in `tests/unit/test_mask_utils.py`

### Implementation for User Story 3

- [x] T033 [US3] Add configurable appearance augmentation strength for brightness, contrast, color, haze-like, and mild noise perturbations in `src/datasets/change_dataset.py` and `configs/default.yaml`
- [x] T034 [US3] Add optional conservative prediction post-processing settings for minimum changed region size and small-noise removal in `src/utils/mask_utils.py` and `configs/default.yaml`
- [x] T035 [US3] Expose false-positive-control settings through prediction configuration and document threshold tradeoffs in `src/predict.py`
- [x] T036 [US3] Add optional validation sample mask export for qualitative review in `src/validate.py`
- [x] T037 [US3] Add README guidance for practical review of wetlands, boats, cloud or haze, lighting/color shifts, and minor registration differences in `README.md`

**Checkpoint**: User Stories 1, 2, and 3 are independently functional and aligned with the practical false-positive-control goal.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete documentation, quality checks, quickstart validation, and implementation cleanup across all stories.

- [x] T038 [P] Add quickstart command validation notes and expected outputs to `README.md`
- [x] T039 [P] Add sample minimal config comments and safe defaults in `configs/default.yaml`
- [x] T040 Run unit, contract, and integration tests with `pytest` and fix failures in affected files under `src/` and `tests/`
- [x] T041 Run the quickstart workflow manually on a tiny synthetic dataset and document any caveats in `README.md`
- [x] T042 Review all CLI error messages for missing pairs, invalid image sizes, missing checkpoints, and invalid thresholds in `src/train.py`, `src/validate.py`, `src/predict.py`, and `src/datasets/change_dataset.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; this is the MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational and needs a checkpoint produced by US1 for end-to-end manual prediction.
- **User Story 3 (Phase 5)**: Depends on Foundational and can be implemented after or alongside US1/US2 once the dataset and mask utilities exist.
- **Polish (Phase 6)**: Depends on the desired stories being complete.

### User Story Dependencies

- **US1 Train on Paired Change Data**: Starts after Phase 2 and has no dependency on US2 or US3.
- **US2 Predict Test Change Masks**: Starts after Phase 2; end-to-end use depends on a checkpoint from US1, but its dataset and CLI contract can be built independently.
- **US3 Control Practical False Positives**: Starts after Phase 2; improves augmentation, thresholding, post-processing, and review guidance used by US1 and US2.

### Within Each User Story

- Write story tests before implementation tasks when possible.
- Dataset behavior before train/predict workflow code.
- Model and loss code before training loop.
- Mask utilities before prediction output validation.
- Documentation after the workflow behavior is implemented.

### Parallel Opportunities

- T005 can run in parallel with T002-T004 after directory creation.
- T013 and T014 can run in parallel once T008 and T011 are sketched.
- T017 and T018 can run in parallel before US1 implementation.
- T024 and T025 can run in parallel before US2 implementation.
- T031 and T032 can run in parallel before US3 implementation.
- README/config polish tasks T038 and T039 can run in parallel.

---

## Parallel Example: User Story 1

```bash
Task: "T017 [P] [US1] Add CLI contract test for train in tests/contract/test_cli_contract.py"
Task: "T018 [P] [US1] Add integration smoke test for one-epoch checkpoint creation in tests/integration/test_train_predict_smoke.py"
```

## Parallel Example: User Story 2

```bash
Task: "T024 [P] [US2] Add CLI contract test for predict in tests/contract/test_cli_contract.py"
Task: "T025 [P] [US2] Extend integration smoke test for binary PNG prediction in tests/integration/test_train_predict_smoke.py"
```

## Parallel Example: User Story 3

```bash
Task: "T031 [P] [US3] Add paired augmentation alignment tests in tests/unit/test_change_dataset.py"
Task: "T032 [P] [US3] Add conservative post-processing tests in tests/unit/test_mask_utils.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational utilities, dataset pairing, model, and loss.
3. Complete Phase 3 training and validation.
4. Validate with the tiny dataset smoke test and confirm `outputs/checkpoints/best_model.pth`.

### Incremental Delivery

1. Deliver US1 so the project can train and save a model.
2. Add US2 so the trained model can produce test masks.
3. Add US3 to improve practical robustness and false-positive control.
4. Finish polish tasks and validate README quickstart.

### Parallel Team Strategy

1. One developer completes setup and shared config/image/mask utilities.
2. Another developer works on model/loss and shape tests after the package skeleton exists.
3. After Phase 2, separate developers can implement US1 train, US2 predict, and US3 robustness tasks with limited file overlap.

## Notes

- `[P]` tasks must touch different files or avoid dependency on incomplete work.
- Each user story has its own independent test criteria.
- Keep implementation aligned with the CLI contract in `specs/001-remote-change-model/contracts/cli.md`.
- Keep detailed metric dashboards out of scope unless requested later.
