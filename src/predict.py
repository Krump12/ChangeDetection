from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.change_dataset import ChangeDetectionDataset
from src.models.change_model import build_model
from src.utils.config_utils import ensure_output_dirs, load_config, parse_image_size, require_paths, resolve_device
from src.utils.mask_utils import save_binary_mask, threshold_probabilities, validate_binary_mask_file


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict binary change masks for test image pairs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold for binary masks.")
    return parser


def predict(config: dict, checkpoint: str, *, threshold: float | None = None) -> list[Path]:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    threshold_value = float(config.get("predict", {}).get("threshold", 0.5) if threshold is None else threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError(f"Prediction threshold must be between 0.0 and 1.0, got {threshold_value}")

    data = config["data"]
    require_paths({"test_a_dir": data["test_a_dir"], "test_b_dir": data["test_b_dir"]}, kind="test data")
    ensure_output_dirs(config)
    image_size = parse_image_size(config)
    output_dir = Path(config.get("outputs", {}).get("predict_output_dir", "outputs/predict_masks"))
    min_region_size = int(config.get("predict", {}).get("min_region_size", 0))
    device = resolve_device(config)
    dataset = ChangeDetectionDataset(data["test_a_dir"], data["test_b_dir"], split="test", image_size=image_size)
    if len(dataset) == 0:
        raise ValueError("Test dataset is empty")
    loader = DataLoader(dataset, batch_size=int(config.get("train", {}).get("batch_size", 16)), shuffle=False)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    saved: list[Path] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            probs = torch.sigmoid(model(batch["a"].to(device), batch["b"].to(device))).cpu()
            masks = threshold_probabilities(probs, threshold_value)
            for mask, name in zip(masks, batch["filename"]):
                out_path = output_dir / Path(name).with_suffix(".png").name
                save_binary_mask(mask, out_path, min_region_size=min_region_size, expected_size=image_size)
                validate_binary_mask_file(out_path, expected_size=image_size)
                saved.append(out_path)
    print(f"saved {len(saved)} mask(s) to {output_dir}")
    return saved


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    predict(config, args.checkpoint, threshold=args.threshold)


if __name__ == "__main__":
    main()
