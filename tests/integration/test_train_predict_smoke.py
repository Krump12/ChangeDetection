from pathlib import Path

import numpy as np
from PIL import Image

from src.predict import predict
from src.train import train
from src.utils.mask_utils import validate_binary_mask_file


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((128, 128, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[48:80, 48:80] = 255
    Image.fromarray(arr).save(path)


def _config(tmp_path: Path) -> dict:
    return {
        "data": {
            "image_size": [128, 128],
            "train_a_dir": str(tmp_path / "data/train/A"),
            "train_b_dir": str(tmp_path / "data/train/B"),
            "train_label_dir": str(tmp_path / "data/train/label"),
            "val_a_dir": str(tmp_path / "data/val/A"),
            "val_b_dir": str(tmp_path / "data/val/B"),
            "val_label_dir": str(tmp_path / "data/val/label"),
            "test_a_dir": str(tmp_path / "data/test/A"),
            "test_b_dir": str(tmp_path / "data/test/B"),
        },
        "outputs": {
            "checkpoint_dir": str(tmp_path / "outputs/checkpoints"),
            "best_checkpoint": str(tmp_path / "outputs/checkpoints/best_model.pth"),
            "predict_output_dir": str(tmp_path / "outputs/predict_masks"),
            "val_preview_dir": str(tmp_path / "outputs/val_preview_masks"),
        },
        "train": {"batch_size": 1, "epochs": 1, "learning_rate": 0.001, "num_workers": 0, "seed": 1, "device": "cpu"},
        "model": {"in_channels": 3, "base_channels": 4},
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
        "predict": {"threshold": 0.5, "min_region_size": 0},
        "augmentation": {"enabled": False},
    }


def test_train_and_predict_smoke(tmp_path: Path):
    cfg = _config(tmp_path)
    for split in ("train", "val"):
        _write_rgb(tmp_path / f"data/{split}/A/sample.png", 20)
        _write_rgb(tmp_path / f"data/{split}/B/sample.png", 40)
        _write_mask(tmp_path / f"data/{split}/label/sample.png")
    _write_rgb(tmp_path / "data/test/A/sample.png", 20)
    _write_rgb(tmp_path / "data/test/B/sample.png", 40)

    checkpoint = train(cfg)
    assert checkpoint.exists()

    outputs = predict(cfg, str(checkpoint), threshold=0.5)
    assert len(outputs) == 1
    assert outputs[0].name == "sample.png"
    validate_binary_mask_file(outputs[0])
