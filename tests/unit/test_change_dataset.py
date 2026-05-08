from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.change_dataset import ChangeDetectionDataset, discover_pairs


def _write_rgb(path: Path, value: int = 0, size: tuple[int, int] = (128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, size: tuple[int, int] = (128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[size[1] // 4 : size[1] // 2, size[0] // 4 : size[0] // 2] = 255
    Image.fromarray(arr).save(path)


def test_discover_pairs_by_filename(tmp_path: Path):
    _write_rgb(tmp_path / "A" / "x.png", 10)
    _write_rgb(tmp_path / "B" / "x.png", 20)
    _write_mask(tmp_path / "label" / "x.png")
    samples = discover_pairs(tmp_path / "A", tmp_path / "B", tmp_path / "label", require_label=True)
    assert len(samples) == 1
    assert samples[0].filename == "x.png"


def test_discover_pairs_reports_missing_counterpart(tmp_path: Path):
    _write_rgb(tmp_path / "A" / "x.png", 10)
    (tmp_path / "B").mkdir()
    try:
        discover_pairs(tmp_path / "A", tmp_path / "B")
    except ValueError as exc:
        assert "missing B" in str(exc)
    else:
        raise AssertionError("Expected missing counterpart error")


def test_dataset_loads_supervised_sample(tmp_path: Path):
    _write_rgb(tmp_path / "A" / "x.png", 10)
    _write_rgb(tmp_path / "B" / "x.png", 20)
    _write_mask(tmp_path / "label" / "x.png")
    ds = ChangeDetectionDataset(tmp_path / "A", tmp_path / "B", tmp_path / "label", split="train")
    item = ds[0]
    assert item["a"].shape == (3, 128, 128)
    assert item["b"].shape == (3, 128, 128)
    assert item["mask"].shape == (1, 128, 128)
    assert item["filename"] == "x.png"


def test_dataset_rejects_invalid_size(tmp_path: Path):
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(tmp_path / "bad.png")
    _write_rgb(tmp_path / "B" / "bad.png", 20)
    _write_mask(tmp_path / "label" / "bad.png")
    ds = ChangeDetectionDataset(tmp_path, tmp_path / "B", tmp_path / "label", split="train")
    try:
        ds[0]
    except ValueError as exc:
        assert "128x128" in str(exc)
    else:
        raise AssertionError("Expected invalid size error")


def test_dataset_uses_configurable_image_size(tmp_path: Path):
    size = (96, 64)
    _write_rgb(tmp_path / "A" / "x.png", 10, size=size)
    _write_rgb(tmp_path / "B" / "x.png", 20, size=size)
    _write_mask(tmp_path / "label" / "x.png", size=size)
    ds = ChangeDetectionDataset(tmp_path / "A", tmp_path / "B", tmp_path / "label", split="train", image_size=size)
    item = ds[0]
    assert item["a"].shape == (3, 64, 96)
    assert item["b"].shape == (3, 64, 96)
    assert item["mask"].shape == (1, 64, 96)


def test_paired_geometric_augmentation_preserves_label_shape(tmp_path: Path):
    _write_rgb(tmp_path / "A" / "x.png", 10)
    _write_rgb(tmp_path / "B" / "x.png", 20)
    _write_mask(tmp_path / "label" / "x.png")
    ds = ChangeDetectionDataset(
        tmp_path / "A",
        tmp_path / "B",
        tmp_path / "label",
        split="train",
        augment=True,
        augmentation_config={"hflip_prob": 1.0, "vflip_prob": 1.0, "rotate90": False},
    )
    item = ds[0]
    assert item["a"].shape[-2:] == item["mask"].shape[-2:] == (128, 128)
    assert torch.isin(item["mask"], torch.tensor([0.0, 1.0])).all()
