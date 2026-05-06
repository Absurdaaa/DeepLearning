"""Training and evaluation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .utils.io import save_metrics
from .utils.plotting import plot_training_curves, save_prediction_grid


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


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
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    plot_training_curves(history, output_dir / "training_curves.png")
    save_prediction_grid(model, val_loader, device, output_dir / "val_predictions.png")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    save_metrics(
        output_dir / "metrics.txt",
        {
            "model": config.model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "best_val_acc": f"{best_val_acc:.6f}",
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
        },
    )
