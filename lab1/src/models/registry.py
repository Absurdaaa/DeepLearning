"""Model registry for experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .simple_cnn import SimpleCNN


AVAILABLE_MODELS = ("simple_cnn",)


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
