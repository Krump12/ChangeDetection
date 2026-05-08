from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.mask_utils import (
    normalize_mask,
    remove_small_regions,
    save_binary_mask,
    threshold_probabilities,
    validate_binary_mask_file,
)


def test_normalize_mask_converts_grayscale_to_binary():
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[10:20, 10:20] = 255
    mask = normalize_mask(Image.fromarray(arr))
    assert mask.shape == (1, 128, 128)
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    assert mask[:, 10:20, 10:20].sum() == 100


def test_threshold_probabilities_validates_range():
    probs = torch.tensor([[[0.2, 0.7]]])
    out = threshold_probabilities(probs, 0.5)
    assert out.tolist() == [[[0, 1]]]

    try:
        threshold_probabilities(probs, 1.5)
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("Expected invalid threshold to raise")


def test_save_binary_mask_writes_binary_png(tmp_path: Path):
    mask = torch.zeros((1, 128, 128), dtype=torch.uint8)
    mask[:, 5:10, 5:10] = 1
    out = tmp_path / "mask.png"
    save_binary_mask(mask, out)
    validate_binary_mask_file(out)
    values = set(np.asarray(Image.open(out).convert("L")).reshape(-1).tolist())
    assert values == {0, 255}


def test_remove_small_regions_drops_tiny_components():
    mask = torch.zeros((1, 128, 128), dtype=torch.uint8)
    mask[:, 1, 1] = 1
    mask[:, 10:15, 10:15] = 1
    cleaned = remove_small_regions(mask, min_region_size=4)
    assert cleaned[:, 1, 1].item() == 0
    assert cleaned[:, 12, 12].item() == 1
