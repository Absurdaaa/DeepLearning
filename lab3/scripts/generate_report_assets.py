#!/usr/bin/env python3
"""Generate report figures and tables from training outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIG_ROOT = PROJECT_ROOT / "实验模板" / "fig" / "generated"
TABLE_ROOT = PROJECT_ROOT / "实验模板" / "tables"
MANIFEST_PATH = PROJECT_ROOT / "实验模板" / "generated_assets_manifest.txt"

os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

import matplotlib.pyplot as plt


def read_summary_metrics(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["metric"]: row["value"] for row in reader}


def read_epoch_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_best_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_dir()):
        best_row: dict[str, object] | None = None
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            summary_path = run_dir / "summary_metrics.csv"
            metadata_path = run_dir / "run_metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
                continue
            summary = read_summary_metrics(summary_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row = {
                "model": model_dir.name,
                "run_name": run_dir.name,
                "lr": metadata["lr"],
                "hidden_size": metadata["hidden_size"],
                "scheduled_sampling": bool(metadata.get("scheduled_sampling", False)),
                "scheduled_sampling_strategy": metadata.get("scheduled_sampling_strategy", "fixed"),
                "best_val_acc": float(summary["best_val_acc"]),
                "best_val_exact_match": float(summary["best_val_exact_match"]),
                "test_acc": float(summary["test_acc"]),
                "test_exact_match": float(summary["test_exact_match"]),
                "test_bleu": float(summary.get("test_bleu", 0.0)),
                "best_val_loss": float(summary["best_val_loss"]),
                "summary_path": str(summary_path),
                "epoch_metrics_path": str(run_dir / "epoch_metrics.csv"),
                "curves_path": str(run_dir / "training_curves.png"),
                "length_bucket_path": str(run_dir / "length_bucket_metrics.csv"),
                "all_test_path": str(run_dir / "all_test_translations.csv"),
            }
            if best_row is None or row["best_val_acc"] > best_row["best_val_acc"]:
                best_row = row
        if best_row is not None:
            rows.append(best_row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex_table(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & LR & Hidden & SS & Best Val Acc & Test Exact & Test BLEU \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['lr']} & {row['hidden_size']} & "
            f"{'Y' if row['scheduled_sampling'] else 'N'} & {row['best_val_acc']:.4f} & "
            f"{row['test_exact_match']:.4f} & {row['test_bleu']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_best_run_curve_plot(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in rows:
        epoch_rows = read_epoch_metrics(Path(str(row["epoch_metrics_path"])))
        epochs = [int(item["epoch"]) for item in epoch_rows]
        val_loss = [float(item["val_loss"]) for item in epoch_rows]
        val_acc = [float(item["val_acc"]) for item in epoch_rows]
        axes[0].plot(epochs, val_loss, label=str(row["model"]))
        axes[1].plot(epochs, val_acc, label=str(row["model"]))

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Token Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_best_run_bar_plot(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    models = [str(row["model"]) for row in rows]
    test_acc = [float(row["test_acc"]) for row in rows]
    test_exact = [float(row["test_exact_match"]) for row in rows]
    test_bleu = [float(row["test_bleu"]) for row in rows]

    x_positions = range(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([x - width for x in x_positions], test_acc, width=width, label="Test Token Acc")
    ax.bar(list(x_positions), test_exact, width=width, label="Test Exact Match")
    ax.bar([x + width for x in x_positions], test_bleu, width=width, label="Test BLEU")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Best-Run Test Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_length_bucket_table(rows: list[dict[str, object]], csv_path: Path, tex_path: Path) -> None:
    table_rows: list[dict[str, object]] = []
    tex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Bucket & Count & Exact Match & BLEU \\",
        r"\midrule",
    ]
    for row in rows:
        bucket_rows = read_csv_rows(Path(str(row["length_bucket_path"])))
        for bucket_row in bucket_rows:
            table_rows.append(
                {
                    "model": row["model"],
                    "bucket": bucket_row["bucket"],
                    "sample_count": bucket_row["sample_count"],
                    "exact_match": bucket_row["exact_match"],
                    "bleu": bucket_row["bleu"],
                }
            )
            tex_lines.append(
                f"{row['model']} & {bucket_row['bucket']} & {bucket_row['sample_count']} & "
                f"{float(bucket_row['exact_match']):.4f} & {float(bucket_row['bleu']):.4f} \\\\"
            )
    write_csv(table_rows, csv_path)
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def save_length_bucket_plot(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    bucket_order = ["1-3", "4-6", "7-9"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x_positions = range(len(bucket_order))
    width = 0.22
    for model_index, row in enumerate(rows):
        bucket_rows = {item["bucket"]: item for item in read_csv_rows(Path(str(row["length_bucket_path"])))}
        exact_values = [float(bucket_rows.get(bucket, {}).get("exact_match", 0.0)) for bucket in bucket_order]
        bleu_values = [float(bucket_rows.get(bucket, {}).get("bleu", 0.0)) for bucket in bucket_order]
        offsets = [x + (model_index - (len(rows) - 1) / 2) * width for x in x_positions]
        axes[0].bar(offsets, exact_values, width=width, label=str(row["model"]))
        axes[1].bar(offsets, bleu_values, width=width, label=str(row["model"]))

    for axis, title in zip(axes, ("Exact Match by Source Length", "BLEU by Source Length"), strict=True):
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(bucket_order)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_qualitative_examples(rows: list[dict[str, object]], path: Path) -> None:
    picked_rows: list[dict[str, object]] = []
    for row in rows:
        test_rows = read_csv_rows(Path(str(row["all_test_path"])))
        wrong_cases = [item for item in test_rows if item.get("exact_match") == "0"]
        for item in wrong_cases[:3]:
            picked_rows.append(
                {
                    "model": row["model"],
                    "source_text": item["source_text"],
                    "target_text": item["target_text"],
                    "prediction_text": item["prediction_text"],
                    "source_length": item["source_length"],
                }
            )
    write_csv(picked_rows, path)


def save_lr_sweep_plots() -> list[str]:
    generated: list[str] = []
    for model_dir in sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_dir()):
        sweep_files = sorted(model_dir.glob(f"{model_dir.name}_*_lr_sweep_summary.csv"))
        for sweep_file in sweep_files:
            with sweep_file.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            lrs = [float(row["learning_rate"]) for row in rows]
            val_acc = [float(row["best_val_acc"]) for row in rows]
            val_exact = [float(row["best_val_exact_match"]) for row in rows]

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(lrs, val_acc, marker="o", label="Best Val Token Acc")
            ax.plot(lrs, val_exact, marker="s", label="Best Val Exact Match")
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Score")
            ax.set_title(f"LR Sweep - {model_dir.name}")
            ax.grid(alpha=0.3)
            ax.legend()
            out_path = FIG_ROOT / f"{sweep_file.stem}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(out_path.relative_to(PROJECT_ROOT)))
    return generated


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    best_runs = collect_best_runs()
    comparison_csv = TABLE_ROOT / "model_comparison.csv"
    comparison_tex = TABLE_ROOT / "model_comparison.tex"
    length_bucket_csv = TABLE_ROOT / "length_bucket_comparison.csv"
    length_bucket_tex = TABLE_ROOT / "length_bucket_comparison.tex"
    qualitative_csv = TABLE_ROOT / "qualitative_examples.csv"
    curves_path = FIG_ROOT / "best_run_val_curves.png"
    metrics_bar_path = FIG_ROOT / "best_run_test_metrics.png"
    length_bucket_path = FIG_ROOT / "length_bucket_comparison.png"

    generated_items: list[str] = []
    if best_runs:
        write_csv(best_runs, comparison_csv)
        write_tex_table(best_runs, comparison_tex)
        write_length_bucket_table(best_runs, length_bucket_csv, length_bucket_tex)
        write_qualitative_examples(best_runs, qualitative_csv)
        save_best_run_curve_plot(best_runs, curves_path)
        save_best_run_bar_plot(best_runs, metrics_bar_path)
        save_length_bucket_plot(best_runs, length_bucket_path)
        generated_items.extend(
            [
                str(comparison_csv.relative_to(PROJECT_ROOT)),
                str(comparison_tex.relative_to(PROJECT_ROOT)),
                str(length_bucket_csv.relative_to(PROJECT_ROOT)),
                str(length_bucket_tex.relative_to(PROJECT_ROOT)),
                str(qualitative_csv.relative_to(PROJECT_ROOT)),
                str(curves_path.relative_to(PROJECT_ROOT)),
                str(metrics_bar_path.relative_to(PROJECT_ROOT)),
                str(length_bucket_path.relative_to(PROJECT_ROOT)),
            ]
        )

    generated_items.extend(save_lr_sweep_plots())
    MANIFEST_PATH.write_text("\n".join(generated_items) + ("\n" if generated_items else ""), encoding="utf-8")

    print(f"Generated {len(generated_items)} report assets.")
    print(f"Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
