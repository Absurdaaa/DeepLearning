#!/usr/bin/env python3
"""Project entrypoint for CIFAR-10 experiments."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from src.utils.runtime import setup_matplotlib

setup_matplotlib(PROJECT_ROOT)

import torch

from src.config import parse_config
from src.data import build_dataloaders
from src.engine import run_training
from src.models import build_model
from src.utils.io import ensure_dir, save_model_summary
from src.utils.plotting import save_image_grid
from src.utils.runtime import set_seed


def main() -> None:
    config = parse_config(PROJECT_ROOT)
    set_seed(config.seed)

    device = torch.device(config.device)
    output_dir = config.output_dir / config.model
    ensure_dir(output_dir)

    train_loader, val_loader, test_loader = build_dataloaders(config)
    model = build_model(config.model).to(device)

    print(f"Using device: {device}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print("\nModel structure:\n")
    print(model)

    save_model_summary(model, str(device), output_dir / "model_structure.txt")
    save_image_grid(train_loader, output_dir / "train_samples.png", "CIFAR-10 Train Samples")
    run_training(model, train_loader, val_loader, test_loader, config, output_dir)

    print(f"\nSaved outputs to: {output_dir}")
    print(f"- model structure: {output_dir / 'model_structure.txt'}")
    print(f"- sample images: {output_dir / 'train_samples.png'}")
    print(f"- curves: {output_dir / 'training_curves.png'}")
    print(f"- predictions: {output_dir / 'val_predictions.png'}")
    print(f"- checkpoint: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
