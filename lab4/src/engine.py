"""GAN / DCGAN 训练引擎。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import TrainConfig
from .data import DataBundle
from .utils.io import save_epoch_metrics, save_summary_metrics
from .utils.plotting import save_image_grid, save_training_curves
from .utils.profiling import count_parameters


def compute_validation_score(
    *,
    generator_loss: float,
    discriminator_loss: float,
    d_real_mean: float,
    d_fake_mean: float,
) -> float:
    """Use a balanced adversarial score to avoid selecting collapsed checkpoints."""

    equilibrium_generator_loss = math.log(2.0)
    equilibrium_discriminator_loss = 2.0 * math.log(2.0)
    return (
        abs(generator_loss - equilibrium_generator_loss)
        + abs(discriminator_loss - equilibrium_discriminator_loss)
        + abs(d_real_mean - 0.5)
        + abs(d_fake_mean - 0.5)
    )


def build_optimizer(parameters, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=config.lr, momentum=0.9)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, betas=(config.beta1, 0.999))
    return torch.optim.Adam(parameters, lr=config.lr, betas=(config.beta1, 0.999))


def build_noise(config: TrainConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    if config.model == "dcgan":
        return torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
    return torch.randn(batch_size, config.latent_dim, device=device)


def evaluate_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            real_scores = discriminator(images)
            if real_scores.dim() > 2:
                real_labels = torch.ones_like(real_scores, device=device)
                fake_labels = torch.zeros_like(real_scores, device=device)
            d_real_loss = criterion(real_scores, real_labels)

            noise = build_noise(config, batch_size, device)
            fake_images = generator(noise)
            fake_scores = discriminator(fake_images)
            d_fake_loss = criterion(fake_scores, fake_labels)
            g_loss = criterion(fake_scores, real_labels)

            total_g_loss += g_loss.item() * batch_size
            total_d_loss += (d_real_loss.item() + d_fake_loss.item()) * batch_size
            total_d_real += real_scores.mean().item() * batch_size
            total_d_fake += fake_scores.mean().item() * batch_size
            total_examples += batch_size

    return {
        "generator_loss": total_g_loss / max(total_examples, 1),
        "discriminator_loss": total_d_loss / max(total_examples, 1),
        "d_real_mean": total_d_real / max(total_examples, 1),
        "d_fake_mean": total_d_fake / max(total_examples, 1),
    }


def run_training(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloaders: DataBundle,
    config: TrainConfig,
    output_dir: Path,
) -> dict[str, float]:
    device = config.device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator_optimizer = build_optimizer(generator.parameters(), config)
    discriminator_optimizer = build_optimizer(discriminator.parameters(), config)
    criterion = nn.BCELoss()
    fixed_noise = build_noise(config, config.fixed_noise_count, device)

    history: list[dict[str, float | int]] = []
    best_validation_score = float("inf")
    best_val_generator_loss = float("inf")
    best_val_discriminator_loss = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        generator.train()
        discriminator.train()
        train_g_loss = 0.0
        train_d_loss = 0.0
        train_d_real = 0.0
        train_d_fake = 0.0
        total_examples = 0

        for batch in dataloaders.train_loader:
            images = batch["images"].to(device)
            batch_size = images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad()
            real_scores = discriminator(images)
            if real_scores.dim() > 2:
                real_labels = torch.ones_like(real_scores, device=device)
                fake_labels = torch.zeros_like(real_scores, device=device)
            d_real_loss = criterion(real_scores, real_labels)

            noise = build_noise(config, batch_size, device)
            fake_images = generator(noise)
            fake_scores = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_scores, fake_labels)
            discriminator_loss = d_real_loss + d_fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            noise = build_noise(config, batch_size, device)
            generated_images = generator(noise)
            generator_scores = discriminator(generated_images)
            generator_loss = criterion(generator_scores, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            train_g_loss += generator_loss.item() * batch_size
            train_d_loss += discriminator_loss.item() * batch_size
            train_d_real += real_scores.mean().item() * batch_size
            train_d_fake += fake_scores.mean().item() * batch_size
            total_examples += batch_size

        val_metrics = evaluate_epoch(generator, discriminator, dataloaders.val_loader, criterion, config, device)
        validation_score = compute_validation_score(
            generator_loss=val_metrics["generator_loss"],
            discriminator_loss=val_metrics["discriminator_loss"],
            d_real_mean=val_metrics["d_real_mean"],
            d_fake_mean=val_metrics["d_fake_mean"],
        )
        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_generator_loss": train_g_loss / max(total_examples, 1),
                "train_discriminator_loss": train_d_loss / max(total_examples, 1),
                "val_generator_loss": val_metrics["generator_loss"],
                "val_discriminator_loss": val_metrics["discriminator_loss"],
                "train_d_real_mean": train_d_real / max(total_examples, 1),
                "train_d_fake_mean": train_d_fake / max(total_examples, 1),
                "val_d_real_mean": val_metrics["d_real_mean"],
                "val_d_fake_mean": val_metrics["d_fake_mean"],
                "val_selection_score": validation_score,
                "epoch_time_sec": epoch_time,
                "elapsed_train_time_sec": time.perf_counter() - start_time,
            }
        )

        epoch_record = history[-1]
        print(
            f"[{config.model}][epoch {epoch}/{config.epochs}] "
            f"train_g={epoch_record['train_generator_loss']:.6f} "
            f"train_d={epoch_record['train_discriminator_loss']:.6f} "
            f"val_g={epoch_record['val_generator_loss']:.6f} "
            f"val_d={epoch_record['val_discriminator_loss']:.6f} "
            f"D(x)={epoch_record['train_d_real_mean']:.4f} "
            f"D(G(z))={epoch_record['train_d_fake_mean']:.4f} "
            f"time={epoch_record['epoch_time_sec']:.2f}s",
            flush=True,
        )

        # 仅记录“最接近均衡”的 epoch 供报告参考——不要用它来挑选交付模型：
        # GAN 的 loss 不是样本质量指标，且均衡分数在随机初始化时本就最低，
        # 用它选 checkpoint 会锁死在没训练过的生成器上。
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_val_generator_loss = val_metrics["generator_loss"]
            best_val_discriminator_loss = val_metrics["discriminator_loss"]
            best_epoch = epoch

    # 交付物（checkpoint / 样例图 / 测试指标）一律取最终 epoch 的生成器，即训练最充分的模型。
    torch.save(
        {
            "epoch": config.epochs,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        },
        output_dir / "best_model.pth",
    )
    generator.eval()
    with torch.no_grad():
        final_images = generator(fixed_noise)
    save_image_grid(final_images, output_dir / "generated_samples.png", nrow=8)
    print(
        f"[{config.model}] saved final-epoch model; balanced-best epoch was "
        f"{best_epoch} (score={best_validation_score:.6f})",
        flush=True,
    )

    test_metrics = evaluate_epoch(generator, discriminator, dataloaders.test_loader, criterion, config, device)
    final_record = history[-1]
    total_train_time = time.perf_counter() - start_time
    generator_params, generator_trainable = count_parameters(generator)
    discriminator_params, discriminator_trainable = count_parameters(discriminator)

    save_epoch_metrics(history, output_dir / "epoch_metrics.csv")
    save_training_curves(history, output_dir / "training_curves.png")

    summary = {
        "best_validation_score": best_validation_score,
        "best_val_generator_loss": best_val_generator_loss,
        "best_val_discriminator_loss": best_val_discriminator_loss,
        "best_epoch": best_epoch,
        "final_epoch": config.epochs,
        "final_val_generator_loss": final_record["val_generator_loss"],
        "final_val_discriminator_loss": final_record["val_discriminator_loss"],
        "test_generator_loss": test_metrics["generator_loss"],
        "test_discriminator_loss": test_metrics["discriminator_loss"],
        "test_d_real_mean": test_metrics["d_real_mean"],
        "test_d_fake_mean": test_metrics["d_fake_mean"],
        "generator_param_count": generator_params,
        "generator_trainable_param_count": generator_trainable,
        "discriminator_param_count": discriminator_params,
        "discriminator_trainable_param_count": discriminator_trainable,
        "total_train_time_sec": total_train_time,
        "avg_epoch_time_sec": total_train_time / max(config.epochs, 1),
    }
    save_summary_metrics(summary, output_dir / "summary_metrics.csv")
    return summary
