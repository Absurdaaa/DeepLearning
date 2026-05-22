"""训练曲线与生成样例绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


def save_training_curves(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return

    epochs = [int(item["epoch"]) for item in history]
    train_g = [float(item["train_generator_loss"]) for item in history]
    val_g = [float(item["val_generator_loss"]) for item in history]
    train_d = [float(item["train_discriminator_loss"]) for item in history]
    val_d = [float(item["val_discriminator_loss"]) for item in history]
    train_real = [float(item["train_d_real_mean"]) for item in history]
    train_fake = [float(item["train_d_fake_mean"]) for item in history]
    val_real = [float(item["val_d_real_mean"]) for item in history]
    val_fake = [float(item["val_d_fake_mean"]) for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, train_g, label="Train G Loss")
    axes[0].plot(epochs, val_g, label="Val G Loss")
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_d, label="Train D Loss")
    axes[1].plot(epochs, val_d, label="Val D Loss")
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_real, label="Train D(x)")
    axes[2].plot(epochs, train_fake, label="Train D(G(z))")
    axes[2].plot(epochs, val_real, label="Val D(x)")
    axes[2].plot(epochs, val_fake, label="Val D(G(z))")
    axes[2].set_title("Discriminator Scores")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    grid = vutils.make_grid(images.detach().cpu(), nrow=nrow, normalize=True, pad_value=0.3)
    figure = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray" if images.size(1) == 1 else None)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
