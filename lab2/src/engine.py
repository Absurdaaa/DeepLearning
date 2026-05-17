"""Training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable
import time

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .utils.io import (
    save_class_accuracy,
    save_confusion_csv,
    save_epoch_metrics,
    save_summary_metrics,
)
from .utils.profiling import count_parameters, estimate_flops_per_sample, measure_inference_time


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr)
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "sequences": batch["sequences"].to(device),
        "lengths": batch["lengths"].to(device),
        "labels": batch["labels"].to(device),
        "names": batch["names"],
        "label_names": batch["label_names"],
    }


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    num_classes: int | None = None,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion = None if num_classes is None else torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        sequences = batch["sequences"]
        lengths = batch["lengths"]
        labels = batch["labels"]

        if is_training:
            optimizer.zero_grad()

        logits = model(sequences, lengths)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        if confusion is not None:
            for target, pred in zip(labels.cpu(), predictions.cpu()):
                confusion[target.item(), pred.item()] += 1

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc, confusion


def evaluate(model, loader, criterion, device, num_classes):
    with torch.no_grad():
        return run_epoch(model, loader, criterion, device, optimizer=None, num_classes=num_classes)


def build_class_accuracy(confusion: torch.Tensor, class_names: list[str]) -> list[dict[str, float | int | str]]:
    rows = []
    for index, class_name in enumerate(class_names):
        correct = int(confusion[index, index].item())
        total = int(confusion[index].sum().item())
        accuracy = correct / total if total > 0 else 0.0
        rows.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )
    return rows


def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_names: list[str],
    config: ExperimentConfig,
    output_dir,
):
    criterion = nn.NLLLoss()
    optimizer = build_optimizer(model, config)
    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_confusion = None
    start_time = time.time()
    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, config.device, optimizer=optimizer)
        val_loss, val_acc, val_confusion = evaluate(model, val_loader, criterion, config.device, len(class_names))
        epoch_time = time.time() - epoch_start
        elapsed_train_time = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_val_acc_gap": train_acc - val_acc,
                "epoch_time_sec": epoch_time,
                "elapsed_train_time_sec": elapsed_train_time,
            }
        )

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_val_confusion = val_confusion.clone() if val_confusion is not None else None
            torch.save(best_state, output_dir / "best_model.pth")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_confusion = evaluate(model, test_loader, criterion, config.device, len(class_names))
    inference_metrics = measure_inference_time(model, test_loader, config.device)
    param_count, trainable_param_count = count_parameters(model)
    avg_length = sum(len(name) for batch in test_loader for name in batch["names"]) / len(test_loader.dataset)

    total_time = time.time() - start_time
    final_metrics = history[-1]
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(config.device) / (1024 ** 2) if config.device.type == "cuda" else 0.0
    )
    summary = {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_metrics["train_loss"],
        "final_train_acc": final_metrics["train_acc"],
        "final_val_loss": final_metrics["val_loss"],
        "final_val_acc": final_metrics["val_acc"],
        "final_train_val_acc_gap": final_metrics["train_val_acc_gap"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_train_time_sec": total_time,
        "avg_epoch_time_sec": total_time / max(config.epochs, 1),
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "peak_memory_mb": peak_memory_mb,
        "avg_test_sequence_length": avg_length,
        "flops_per_sample": estimate_flops_per_sample(
            config.model,
            config.input_size,
            config.hidden_size,
            len(class_names),
            avg_length,
        ),
    }
    summary.update(inference_metrics)

    save_epoch_metrics(history, output_dir / "epoch_metrics.csv")
    save_summary_metrics(summary, output_dir / "summary_metrics.csv")
    save_confusion_csv(best_val_confusion, class_names, output_dir / "val_confusion_matrix.csv")
    save_confusion_csv(test_confusion, class_names, output_dir / "test_confusion_matrix.csv")
    save_class_accuracy(build_class_accuracy(test_confusion, class_names), output_dir / "class_accuracy.csv")
    return history, best_val_confusion, test_confusion, summary
