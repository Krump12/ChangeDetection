from pathlib import Path

import pytest

from src.predict import build_argparser as predict_parser
from src.predict import predict
from src.train import build_argparser as train_parser
from src.train import make_datasets


def test_train_cli_accepts_config_argument():
    args = train_parser().parse_args(["--config", "configs/default.yaml"])
    assert args.config == "configs/default.yaml"


def test_predict_cli_accepts_checkpoint_and_threshold():
    args = predict_parser().parse_args(
        ["--config", "configs/default.yaml", "--checkpoint", "outputs/checkpoints/best_model.pth", "--threshold", "0.5"]
    )
    assert args.checkpoint == "outputs/checkpoints/best_model.pth"
    assert args.threshold == 0.5


def test_train_required_paths_are_actionable(tmp_path: Path):
    config = {
        "data": {
            "train_a_dir": str(tmp_path / "missing_train_a"),
            "train_b_dir": str(tmp_path / "missing_train_b"),
            "train_label_dir": str(tmp_path / "missing_train_label"),
            "val_a_dir": str(tmp_path / "missing_val_a"),
            "val_b_dir": str(tmp_path / "missing_val_b"),
            "val_label_dir": str(tmp_path / "missing_val_label"),
        },
        "augmentation": {},
    }
    with pytest.raises(FileNotFoundError, match="training data"):
        make_datasets(config)


def test_predict_rejects_invalid_threshold_before_data_loading(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    config = {"predict": {"threshold": 0.5}, "data": {"test_a_dir": "x", "test_b_dir": "y"}, "train": {"device": "cpu"}}
    with pytest.raises(ValueError, match="threshold"):
        predict(config, str(checkpoint), threshold=1.5)
