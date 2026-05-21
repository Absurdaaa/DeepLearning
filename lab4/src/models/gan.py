"""基础全连接 GAN 模型。"""

from __future__ import annotations

import torch
from torch import nn


class GANGenerator(nn.Module):
    """基于 MLP 的 FashionMNIST 生成器。"""

    def __init__(self, latent_dim: int = 100, hidden_dim: int = 128, image_size: int = 28) -> None:
        super().__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim, image_size * image_size)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        hidden = self.nonlin1(self.fc1(noise))
        output = torch.tanh(self.fc2(hidden))
        return output.view(output.size(0), 1, self.image_size, self.image_size)


class GANDiscriminator(nn.Module):
    """基于 MLP 的 FashionMNIST 判别器。"""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.view(images.size(0), -1)
        hidden = self.nonlin1(self.fc1(flat))
        return torch.sigmoid(self.fc2(hidden))
