from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import ImageEnhance
from torch.utils.data import Dataset

from src.utils.image_utils import image_to_tensor, list_images, load_rgb_image
from src.utils.mask_utils import load_mask


@dataclass(frozen=True)
class ChangeSample:
    filename: str
    image_a_path: Path
    image_b_path: Path
    label_path: Path | None = None


def discover_pairs(
    a_dir: str | Path,
    b_dir: str | Path,
    label_dir: str | Path | None = None,
    *,
    require_label: bool = False,
) -> list[ChangeSample]:
    a_files = {p.name: p for p in list_images(a_dir)}
    b_files = {p.name: p for p in list_images(b_dir)}
    labels = {p.name: p for p in list_images(label_dir)} if label_dir else {}

    missing_b = sorted(set(a_files) - set(b_files))
    missing_a = sorted(set(b_files) - set(a_files))
    if missing_a or missing_b:
        parts = []
        if missing_b:
            parts.append(f"missing B for: {', '.join(missing_b)}")
        if missing_a:
            parts.append(f"missing A for: {', '.join(missing_a)}")
        raise ValueError("Unmatched A/B image pairs: " + "; ".join(parts))

    if require_label:
        missing_label = sorted(set(a_files) - set(labels))
        extra_label = sorted(set(labels) - set(a_files))
        if missing_label or extra_label:
            parts = []
            if missing_label:
                parts.append(f"missing label for: {', '.join(missing_label)}")
            if extra_label:
                parts.append(f"label without image pair: {', '.join(extra_label)}")
            raise ValueError("Unmatched labels: " + "; ".join(parts))

    samples = []
    for name in sorted(a_files, key=str.lower):
        samples.append(ChangeSample(name, a_files[name], b_files[name], labels.get(name)))
    return samples


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        a_dir: str | Path,
        b_dir: str | Path,
        label_dir: str | Path | None = None,
        *,
        split: str,
        augment: bool = False,
        augmentation_config: dict[str, Any] | None = None,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.split = split
        self.require_label = split in {"train", "val"}
        if self.require_label and label_dir is None:
            raise ValueError(f"{split} dataset requires a label directory")
        self.samples = discover_pairs(a_dir, b_dir, label_dir, require_label=self.require_label)
        self.augment = augment
        self.aug = augmentation_config or {}
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_a = load_rgb_image(sample.image_a_path, expected_size=self.image_size)
        image_b = load_rgb_image(sample.image_b_path, expected_size=self.image_size)
        label = load_mask(sample.label_path, expected_size=self.image_size) if sample.label_path else None

        if self.augment and self.split == "train":
            image_a, image_b, label = self._augment(image_a, image_b, label)

        tensor_a = image_to_tensor(image_a)
        tensor_b = image_to_tensor(image_b)
        if self.augment and self.split == "train":
            tensor_a = self.add_tensor_appearance_noise(tensor_a, self.aug)
            tensor_b = self.add_tensor_appearance_noise(tensor_b, self.aug)

        item: dict[str, torch.Tensor | str] = {"a": tensor_a, "b": tensor_b, "filename": sample.filename}
        if label is not None:
            item["mask"] = label.float()
        return item

    def _augment(self, image_a, image_b, label):
        if random.random() < float(self.aug.get("hflip_prob", 0.0)):
            image_a = image_a.transpose(0)
            image_b = image_b.transpose(0)
            label = torch.flip(label, dims=[2]) if label is not None else None
        if random.random() < float(self.aug.get("vflip_prob", 0.0)):
            image_a = image_a.transpose(1)
            image_b = image_b.transpose(1)
            label = torch.flip(label, dims=[1]) if label is not None else None
        if self.aug.get("rotate90", False):
            k = random.randint(0, 3)
            if k:
                image_a = image_a.rotate(90 * k)
                image_b = image_b.rotate(90 * k)
                label = torch.rot90(label, k=k, dims=[1, 2]) if label is not None else None
        image_a = self._appearance(image_a)
        image_b = self._appearance(image_b)
        return image_a, image_b, label

    def _appearance(self, image):
        brightness = float(self.aug.get("brightness", 0.0))
        contrast = float(self.aug.get("contrast", 0.0))
        color = float(self.aug.get("color", 0.0))
        if brightness > 0:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - brightness, 1 + brightness))
        if contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - contrast, 1 + contrast))
        if color > 0:
            image = ImageEnhance.Color(image).enhance(random.uniform(1 - color, 1 + color))
        return image

    @staticmethod
    def add_tensor_appearance_noise(tensor: torch.Tensor, aug: dict[str, Any]) -> torch.Tensor:
        haze = float(aug.get("haze", 0.0))
        noise_std = float(aug.get("noise_std", 0.0))
        out = tensor
        if haze > 0:
            out = out * (1 - haze) + haze
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out.clamp(0.0, 1.0)
