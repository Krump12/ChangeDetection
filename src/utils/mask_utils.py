from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.image_utils import DEFAULT_IMAGE_SIZE


def load_mask(path: str | Path, *, expected_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    mask_path = Path(path)
    try:
        image = Image.open(mask_path)
    except Exception as exc:
        raise ValueError(f"Could not read mask: {mask_path}") from exc
    if image.size != expected_size:
        raise ValueError(f"Mask must be {expected_size[0]}x{expected_size[1]}, got {image.size} for {mask_path}")
    return normalize_mask(image)


def normalize_mask(mask: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().clone().float()
        if tensor.ndim == 3 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return (tensor > 0.5).float() if tensor.max() <= 1.0 else (tensor > 127).float()

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    tensor = torch.from_numpy(array.astype(np.float32))
    return (tensor > 127).float().unsqueeze(0)


def threshold_probabilities(probabilities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"Prediction threshold must be between 0.0 and 1.0, got {threshold}")
    return (probabilities >= float(threshold)).to(torch.uint8)


def remove_small_regions(mask: torch.Tensor, min_region_size: int = 0) -> torch.Tensor:
    if min_region_size <= 1:
        return mask
    arr = _to_2d_uint8(mask).copy()
    height, width = arr.shape
    visited = np.zeros_like(arr, dtype=bool)

    for y in range(height):
        for x in range(width):
            if arr[y, x] == 0 or visited[y, x]:
                continue
            pixels = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and arr[ny, nx] == 1:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) < min_region_size:
                for py, px in pixels:
                    arr[py, px] = 0
    return torch.from_numpy(arr).unsqueeze(0).to(mask.device, dtype=torch.uint8)


def save_binary_mask(
    mask: torch.Tensor,
    path: str | Path,
    *,
    min_region_size: int = 0,
    expected_size: tuple[int, int] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    binary = remove_small_regions(mask, min_region_size=min_region_size)
    arr = _to_2d_uint8(binary) * 255
    image = Image.fromarray(arr.astype(np.uint8), mode="L")
    if expected_size is not None and image.size != expected_size:
        raise ValueError(f"Output mask must be {expected_size[0]}x{expected_size[1]}, got {image.size}")
    image.save(output.with_suffix(".png"))


def validate_binary_mask_file(path: str | Path, *, expected_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE) -> None:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        raise ValueError(f"Output mask must be {expected_size[0]}x{expected_size[1]}, got {image.size} for {path}")
    values = set(np.asarray(image).reshape(-1).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"Output mask must be binary 0/255, got values {sorted(values)[:8]} for {path}")


def _to_2d_uint8(mask: torch.Tensor) -> np.ndarray:
    tensor = mask.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    elif tensor.ndim != 2:
        raise ValueError(f"Expected 2D, 3D, or 4D mask tensor, got shape {tuple(tensor.shape)}")
    return (tensor > 0).to(torch.uint8).numpy()
