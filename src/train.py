from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.change_dataset import ChangeDetectionDataset
from src.losses.losses import build_loss
from src.models.change_model import build_model
from src.utils.config_utils import ensure_output_dirs, load_config, parse_image_size, require_paths, resolve_device, set_seed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a remote sensing change detection model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for resume/fine-tuning.")
    return parser


def make_datasets(config: dict) -> tuple[ChangeDetectionDataset, ChangeDetectionDataset]:
    data = config["data"]
    require_paths(
        {
            "train_a_dir": data["train_a_dir"],
            "train_b_dir": data["train_b_dir"],
            "train_label_dir": data["train_label_dir"],
            "val_a_dir": data["val_a_dir"],
            "val_b_dir": data["val_b_dir"],
            "val_label_dir": data["val_label_dir"],
        },
        kind="training data",
    )
    aug = config.get("augmentation", {})
    image_size = parse_image_size(config)
    train_ds = ChangeDetectionDataset(
        data["train_a_dir"],
        data["train_b_dir"],
        data["train_label_dir"],
        split="train",
        augment=bool(aug.get("enabled", True)),
        augmentation_config=aug,
        image_size=image_size,
    )
    val_ds = ChangeDetectionDataset(
        data["val_a_dir"], data["val_b_dir"], data["val_label_dir"], split="val", image_size=image_size
    )
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty")
    return train_ds, val_ds


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
            a = batch["a"].to(device)
            b = batch["b"].to(device)
            mask = batch["mask"].to(device)
            logits = model(a, b)
            loss = criterion(logits, mask)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * a.shape[0]
            count += a.shape[0]
    return total / max(count, 1)


def train(config: dict, checkpoint: str | None = None) -> Path:
    ensure_output_dirs(config)
    set_seed(int(config.get("train", {}).get("seed", 42)))
    device = resolve_device(config)
    train_ds, val_ds = make_datasets(config)
    train_cfg = config["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = build_model(config).to(device)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 3e-4)))

    best_loss = float("inf")
    best_path = Path(config.get("outputs", {}).get("best_checkpoint", "outputs/checkpoints/best_model.pth"))
    best_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(train_cfg.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": config}, best_path)
            print(f"saved best checkpoint: {best_path}")
    return best_path


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    train(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
