"""统一维护模型名与构建入口。"""

from __future__ import annotations

from src.constants import AVAILABLE_MODELS


def build_model(model_name: str, *args: object, **kwargs: object) -> tuple[object, object]:
    """后续按模型名返回 generator / discriminator。"""

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    raise NotImplementedError("lab4/src/models/registry.py 尚未实现模型构建逻辑。")
