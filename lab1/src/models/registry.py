"""Model registry for experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .simple_cnn import SimpleCNN


AVAILABLE_MODELS = ("simple_cnn", "resnet18", "vgg11_bn")


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "vgg11_bn":
        model = models.vgg11_bn(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")
