"""FashionMNIST 数据读取占位。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DataBundle:
    """后续用于承载 train/val/test dataloader。"""

    train_loader: object | None = None
    val_loader: object | None = None
    test_loader: object | None = None


def build_dataloaders(*args: object, **kwargs: object) -> DataBundle:
    """下一步在这里实现 FashionMNIST dataloader 构建。"""

    raise NotImplementedError("lab4/src/data.py 尚未实现数据加载逻辑。")
