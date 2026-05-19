"""Training and sampling helpers for conditional name generation."""

from __future__ import annotations

import random
import time

import torch
import torch.nn as nn

from .generation_config import GenerationConfig
from .generation_data import (
    ALLOWED_CHARACTERS,
    EOS_INDEX,
    NameGenerationDataset,
    category_tensor,
    input_tensor,
    target_tensor,
)
from .utils.io import save_epoch_metrics, save_summary_metrics


def build_optimizer(model: nn.Module, config: GenerationConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr)
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def train_one_sample(
    model: nn.Module,
    sample,
    num_categories: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: GenerationConfig,
) -> float:
    category = category_tensor(sample.category_index, num_categories).to(config.device)
    input_line = input_tensor(sample.name).to(config.device)
    target_line = target_tensor(sample.name).to(config.device)
    hidden = model.init_hidden(config.device)

    optimizer.zero_grad()
    total_loss = 0.0

    for step_index in range(input_line.size(0)):
        output, hidden = model(category, input_line[step_index], hidden)
        step_loss = criterion(output, target_line[step_index].unsqueeze(0))
        total_loss = total_loss + step_loss

    total_loss.backward()
    # 这个任务本质上也是序列模型，必要时也可以开梯度裁剪稳一下
    if config.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
    optimizer.step()
    return float(total_loss.item() / input_line.size(0))


def sample_name(
    model: nn.Module,
    category_index: int,
    num_categories: int,
    start_letter: str,
    config: GenerationConfig,
) -> str:
    model.eval()
    with torch.no_grad():
        category = category_tensor(category_index, num_categories).to(config.device)
        current_input = input_tensor(start_letter).to(config.device)
        hidden = model.init_hidden(config.device)
        output_name = start_letter

        for _ in range(config.sample_max_length):
            output, hidden = model(category, current_input[0], hidden)
            top_index = int(output.topk(1).indices[0, 0].item())
            if top_index == EOS_INDEX:
                break
            letter = ALLOWED_CHARACTERS[top_index]
            output_name += letter
            current_input = input_tensor(letter).to(config.device)

    model.train()
    return output_name


def save_generated_samples(path, generated_samples: dict[str, list[str]]) -> None:
    lines: list[str] = []
    for category_name, samples in generated_samples.items():
        lines.append(f"[{category_name}]")
        lines.extend(samples)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_generation_training(
    model: nn.Module,
    dataset: NameGenerationDataset,
    config: GenerationConfig,
    output_dir,
):
    criterion = nn.NLLLoss()
    optimizer = build_optimizer(model, config)
    rng = random.Random(config.seed)
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_loss = float("inf")
    best_state = None
    start_time = time.time()

    samples = list(dataset.samples)
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(samples)
        epoch_samples = samples
        if config.max_samples_per_epoch > 0:
            epoch_samples = samples[: config.max_samples_per_epoch]
        epoch_start = time.time()
        total_loss = 0.0

        for sample in epoch_samples:
            total_loss += train_one_sample(
                model=model,
                sample=sample,
                num_categories=len(dataset.class_names),
                criterion=criterion,
                optimizer=optimizer,
                config=config,
            )

        avg_loss = total_loss / max(len(epoch_samples), 1)
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "epoch_time_sec": epoch_time,
                "elapsed_train_time_sec": elapsed_time,
            }
        )
        print(f"Epoch [{epoch}/{config.epochs}] train_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_model.pth")

    if best_state is None:
        raise RuntimeError("Generation training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    total_time = time.time() - start_time
    summary = {
        "best_train_loss": best_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history[-1]["train_loss"],
        "total_train_time_sec": total_time,
        "avg_epoch_time_sec": total_time / max(config.epochs, 1),
        "sample_count": len(dataset.samples),
        "max_samples_per_epoch": config.max_samples_per_epoch if config.max_samples_per_epoch > 0 else len(dataset.samples),
        "class_count": len(dataset.class_names),
    }

    generated_samples: dict[str, list[str]] = {}
    for category_name in ("Russian", "German", "Spanish", "Chinese"):
        if category_name not in dataset.class_names:
            continue
        category_index = dataset.class_names.index(category_name)
        start_letters = category_name[:3].upper()
        generated_samples[category_name] = [
            sample_name(model, category_index, len(dataset.class_names), start_letter, config)
            for start_letter in start_letters
        ]

    save_epoch_metrics(history, output_dir / "epoch_metrics.csv")
    save_summary_metrics(summary, output_dir / "summary_metrics.csv")
    save_generated_samples(output_dir / "generated_samples.txt", generated_samples)
    return history, summary, generated_samples
