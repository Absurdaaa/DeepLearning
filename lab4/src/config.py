"""CLI 参数与实验配置定义。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    """Lab4 训练配置占位。"""

    project_root: Path
    data_root: Path
    output_dir: Path
    model: str = "gan"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 2e-4
    latent_dim: int = 100
    seed: int = 42


def parse_config(project_root: Path) -> TrainConfig:
    """返回占位配置，下一步接入完整 CLI。"""

    return TrainConfig(
        project_root=project_root,
        data_root=project_root / "data",
        output_dir=project_root / "outputs",
    )
