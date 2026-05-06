"""Training and evaluation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .constants import CLASSES
from .utils.io import save_csv_rows, save_lines, save_metrics
from .utils.plotting import plot_training_curves, save_prediction_grid
from .utils.wandb_logger import WandbLogger


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    if config.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct_per_class = {class_name: 0 for class_name in CLASSES}
    total_per_class = {class_name: 0 for class_name in CLASSES}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for label, pred in zip(labels, preds):
            class_name = CLASSES[label.item()]
            total_per_class[class_name] += 1
            if label == pred:
                correct_per_class[class_name] += 1

    return {
        class_name: correct_per_class[class_name] / total_per_class[class_name]
        for class_name in CLASSES
    }


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    device = torch.device(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    wandb_logger = None
    if config.use_wandb:
        try:
            wandb_logger = WandbLogger(config, output_dir)
        except Exception as exc:
            print(f"W&B initialization failed, fallback to local logging only: {exc}")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    epoch_rows: List[Dict[str, object]] = []
    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
        )
        if wandb_logger is not None:
            wandb_logger.log_epoch(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                }
            )

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc * 100:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    save_csv_rows(
        output_dir / "epoch_metrics.csv",
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        epoch_rows,
    )

    if config.save_plots:
        plot_training_curves(history, output_dir / "training_curves.png")
        save_prediction_grid(model, val_loader, device, output_dir / "val_predictions.png")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    class_acc = evaluate_per_class(model, test_loader, device)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("\nPer-class accuracy:")
    class_acc_lines = []
    class_acc_rows = []
    for class_name, accuracy in class_acc.items():
        line = f"{class_name:5s}: {accuracy * 100:.2f}%"
        class_acc_lines.append(line)
        class_acc_rows.append(
            {
                "class_name": class_name,
                "accuracy": f"{accuracy:.6f}",
            }
        )
        print(line)

    save_metrics(
        output_dir / "metrics.txt",
        {
            "model": config.model,
            "optimizer": config.optimizer,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "final_train_loss": f"{history['train_loss'][-1]:.6f}",
            "final_train_acc": f"{history['train_acc'][-1]:.6f}",
            "best_val_acc": f"{best_val_acc:.6f}",
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
        },
    )
    save_lines(output_dir / "class_accuracy.txt", class_acc_lines)
    save_csv_rows(
        output_dir / "class_accuracy.csv",
        ["class_name", "accuracy"],
        class_acc_rows,
    )
    if wandb_logger is not None:
        wandb_logger.log_summary(
            {
                "best_val_acc": best_val_acc,
                "final_train_loss": history["train_loss"][-1],
                "final_train_acc": history["train_acc"][-1],
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        wandb_logger.log_class_accuracy(class_acc)
        wandb_logger.finish()
