from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    validate_config(data)
    return data


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: Mapping[str, Any]) -> None:
    parse_image_size(config)

    threshold = float(config.get("predict", {}).get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Prediction threshold must be between 0.0 and 1.0, got {threshold}")

    train = config.get("train", {})
    for key in ("batch_size", "epochs", "learning_rate"):
        value = train.get(key)
        if value is not None and float(value) <= 0:
            raise ValueError(f"train.{key} must be positive, got {value}")

    min_region_size = int(config.get("predict", {}).get("min_region_size", 0))
    if min_region_size < 0:
        raise ValueError(f"predict.min_region_size must be >= 0, got {min_region_size}")


def ensure_output_dirs(config: Mapping[str, Any]) -> None:
    outputs = config.get("outputs", {})
    for key in ("checkpoint_dir", "predict_output_dir", "val_preview_dir"):
        path = outputs.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def parse_image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    raw = config.get("data", {}).get("image_size", (128, 128))
    if isinstance(raw, int):
        size = (raw, raw)
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        size = (int(raw[0]), int(raw[1]))
    else:
        raise ValueError("data.image_size must be an integer or a [width, height] pair")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"data.image_size values must be positive, got {size}")
    return size


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_paths(paths: Mapping[str, str | Path], *, kind: str) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required {kind} path(s): {', '.join(missing)}")
