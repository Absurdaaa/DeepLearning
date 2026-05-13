#!/usr/bin/env python3
"""Run a learning-rate sweep and summarize the best run."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning-rate sweep helper")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, for example simple_cnn / resnet18 / vgg11_bn.",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=("sgd", "adam", "adamw"),
        help="Optimizer name.",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        required=True,
        help="Learning rates to sweep.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--device", type=str, default=None, help="Training device override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Output root directory.",
    )
    parser.add_argument("--download", action="store_true", help="Download dataset if needed.")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for each run.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="cifar10-lab1", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity.")
    parser.add_argument("--sweep-name", type=str, default=None, help="Optional name for this sweep.")
    return parser.parse_args()


def format_float_tag(value: float) -> str:
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p")


def run_training(args: argparse.Namespace, lr: float) -> Path:
    run_name = f"{args.sweep_name or 'sweep'}_{args.optimizer}_lr{format_float_tag(lr)}_bs{args.batch_size}"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        args.model,
        "--run-name",
        run_name,
        "--optimizer",
        args.optimizer,
        "--lr",
        str(lr),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--weight-decay",
        str(args.weight_decay),
        "--momentum",
        str(args.momentum),
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.download:
        cmd.append("--download")
    if args.save_plots:
        cmd.append("--save-plots")
    if args.use_wandb:
        cmd.extend(
            [
                "--use-wandb",
                "--wandb-project",
                args.wandb_project,
            ]
        )
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return Path(args.output_dir) / args.model / run_name / "summary_metrics.csv"


def load_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return next(reader)


def summarize_runs(model: str, rows: Iterable[dict[str, str]], output_dir: Path, sweep_name: str) -> None:
    rows = list(rows)
    summary_path = output_dir / model / f"{sweep_name}_lr_sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "run_name",
            "optimizer",
            "learning_rate",
            "best_val_acc",
            "best_val_epoch",
            "time_to_best_val_sec",
            "total_train_time_sec",
            "test_acc",
            "test_loss",
            "param_count",
            "flops_per_image",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_row = max(rows, key=lambda row: float(row["best_val_acc"]))
    best_path = output_dir / model / f"{sweep_name}_best_lr.txt"
    best_path.write_text(
        "\n".join(
            [
                f"model={model}",
                f"optimizer={best_row['optimizer']}",
                f"best_learning_rate={best_row['learning_rate']}",
                f"best_val_acc={best_row['best_val_acc']}",
                f"best_val_epoch={best_row['best_val_epoch']}",
                f"test_acc={best_row['test_acc']}",
                f"run_name={best_row['run_name']}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nSweep summary saved to: {summary_path}")
    print(f"Best lr: {best_row['learning_rate']} | best_val_acc: {best_row['best_val_acc']} | run: {best_row['run_name']}")


def main() -> None:
    args = parse_args()
    sweep_name = args.sweep_name or f"{args.model}_{args.optimizer}"
    rows: list[dict[str, str]] = []

    for lr in args.lrs:
        summary_csv = run_training(args, lr)
        summary = load_single_row_csv(summary_csv)
        rows.append(
            {
                "run_name": summary_csv.parent.name,
                "optimizer": summary["optimizer"],
                "learning_rate": summary["learning_rate"],
                "best_val_acc": summary["best_val_acc"],
                "best_val_epoch": summary["best_val_epoch"],
                "time_to_best_val_sec": summary["time_to_best_val_sec"],
                "total_train_time_sec": summary["total_train_time_sec"],
                "test_acc": summary["test_acc"],
                "test_loss": summary["test_loss"],
                "param_count": summary["param_count"],
                "flops_per_image": summary["flops_per_image"],
            }
        )

    summarize_runs(args.model, rows, Path(args.output_dir), sweep_name)


if __name__ == "__main__":
    main()
