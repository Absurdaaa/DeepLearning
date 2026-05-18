"""CLI config for sequence-to-sequence translation experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .models import AVAILABLE_MODELS


@dataclass
class ExperimentConfig:
    model: str
    run_name: str | None
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    hidden_size: int
    num_layers: int
    dropout: float
    teacher_forcing_ratio: float
    seed: int
    num_workers: int
    train_ratio: float
    val_ratio: float
    max_length: int
    max_samples: int
    reverse_translation: bool
    filter_english_prefixes: bool
    data_root: Path
    output_dir: Path
    device: torch.device


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab3 Seq2Seq translation framework")
    parser.add_argument("--model", type=str, default="seq2seq_rnn", choices=AVAILABLE_MODELS, help="Model name.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "sgd", "adamw"),
        help="Optimizer name.",
    )
    parser.add_argument("--hidden-size", type=int, default=128, help="Shared embedding/hidden size.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of encoder/decoder recurrent layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in encoder/decoder.")
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5, help="Teacher forcing ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max-length", type=int, default=10, help="Maximum token count before EOS.")
    parser.add_argument("--max-samples", type=int, default=12000, help="Maximum number of filtered pairs to load.")
    parser.add_argument(
        "--reverse-translation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fra->eng translation when enabled; otherwise eng->fra.",
    )
    parser.add_argument(
        "--filter-english-prefixes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the tutorial-style simple English target sentences only.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(project_root / "data" / "eng-fra.txt"),
        help="Parallel corpus file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    return parser


def parse_config(project_root: Path) -> ExperimentConfig:
    args = build_parser(project_root).parse_args()
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0, 1).")
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be in (0, 1).")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")
    if not (0.0 <= args.teacher_forcing_ratio <= 1.0):
        raise ValueError("--teacher-forcing-ratio must be in [0, 1].")
    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2.")

    return ExperimentConfig(
        model=args.model,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_length=args.max_length,
        max_samples=args.max_samples,
        reverse_translation=args.reverse_translation,
        filter_english_prefixes=args.filter_english_prefixes,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        device=torch.device(args.device),
    )
