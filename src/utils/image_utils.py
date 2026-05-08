from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif"}
DEFAULT_IMAGE_SIZE = (128, 128)


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files = [p for p in root.iterdir() if p.is_file() and is_supported_image(p)]
    return sorted(files, key=lambda p: p.name.lower())


def load_rgb_image(path: str | Path, *, expected_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Image.Image:
    image_path = Path(path)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc
    if image.size != expected_size:
        raise ValueError(f"Image must be {expected_size[0]}x{expected_size[1]}, got {image.size} for {image_path}")
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_image_tensor(path: str | Path) -> torch.Tensor:
    return image_to_tensor(load_rgb_image(path))
