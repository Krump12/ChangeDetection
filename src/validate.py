from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.change_dataset import ChangeDetectionDataset
from src.losses.losses import build_loss
from src.models.change_model import build_model
from src.train import run_epoch
from src.utils.config_utils import ensure_output_dirs, load_config, parse_image_size, require_paths, resolve_device
from src.utils.mask_utils import save_binary_mask


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a remote sensing change detection model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--export-samples", action="store_true", help="Save validation preview masks for qualitative review.")
    return parser


def validate(config: dict, checkpoint: str, *, export_samples: bool = False) -> float:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data = config["data"]
    require_paths(
        {"val_a_dir": data["val_a_dir"], "val_b_dir": data["val_b_dir"], "val_label_dir": data["val_label_dir"]},
        kind="validation data",
    )
    ensure_output_dirs(config)
    image_size = parse_image_size(config)
    device = resolve_device(config)
    dataset = ChangeDetectionDataset(
        data["val_a_dir"], data["val_b_dir"], data["val_label_dir"], split="val", image_size=image_size
    )
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")
    loader = DataLoader(dataset, batch_size=int(config.get("train", {}).get("batch_size", 16)), shuffle=False)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    criterion = build_loss(config)
    val_loss = run_epoch(model, loader, criterion, device)
    print(f"val_loss={val_loss:.4f}")

    if export_samples:
        preview_dir = Path(config.get("outputs", {}).get("val_preview_dir", "outputs/val_preview_masks"))
        threshold = float(config.get("predict", {}).get("threshold", 0.5))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                probs = torch.sigmoid(model(batch["a"].to(device), batch["b"].to(device))).cpu()
                for prob, name in zip(probs, batch["filename"]):
                    save_binary_mask(
                        (prob >= threshold).to(torch.uint8),
                        preview_dir / Path(name).with_suffix(".png").name,
                        expected_size=image_size,
                    )
                break
    return val_loss


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    validate(config, args.checkpoint, export_samples=args.export_samples)


if __name__ == "__main__":
    main()
