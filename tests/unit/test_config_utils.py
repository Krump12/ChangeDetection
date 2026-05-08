from pathlib import Path

import yaml

from src.utils.config_utils import ensure_output_dirs, load_config, parse_image_size


def test_load_config_and_create_output_dirs(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config = {
        "outputs": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "predict_output_dir": str(tmp_path / "pred"),
            "val_preview_dir": str(tmp_path / "preview"),
        },
        "train": {"batch_size": 1, "epochs": 1, "learning_rate": 0.001},
        "predict": {"threshold": 0.5, "min_region_size": 0},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = load_config(config_path)
    ensure_output_dirs(loaded)
    assert (tmp_path / "ckpt").is_dir()
    assert (tmp_path / "pred").is_dir()


def test_invalid_threshold_raises(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"predict": {"threshold": 1.2}}), encoding="utf-8")
    try:
        load_config(config_path)
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("Expected invalid threshold to raise")


def test_parse_configurable_image_size(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": {"image_size": [96, 64]}}), encoding="utf-8")
    loaded = load_config(config_path)
    assert parse_image_size(loaded) == (96, 64)
