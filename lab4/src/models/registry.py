"""统一维护模型名与构建入口。"""

from __future__ import annotations

from src.models.dcgan import DCGANDiscriminator, DCGANGenerator
from src.models.gan import GANDiscriminator, GANGenerator
from src.constants import AVAILABLE_MODELS


def build_model(model_name: str, *args: object, **kwargs: object) -> tuple[object, object]:
    """按模型名构建 generator / discriminator。"""

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "gan":
        latent_dim = int(kwargs.get("latent_dim", 100))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        image_size = int(kwargs.get("image_size", 28))
        return (
            GANGenerator(latent_dim=latent_dim, hidden_dim=hidden_dim, image_size=image_size),
            GANDiscriminator(input_dim=image_size * image_size, hidden_dim=hidden_dim),
        )

    latent_dim = int(kwargs.get("latent_dim", 100))
    image_channels = int(kwargs.get("image_channels", 1))
    generator_base_channels = int(kwargs.get("generator_base_channels", 64))
    discriminator_base_channels = int(kwargs.get("discriminator_base_channels", 64))
    return (
        DCGANGenerator(
            latent_dim=latent_dim,
            image_channels=image_channels,
            base_channels=generator_base_channels,
        ),
        DCGANDiscriminator(
            image_channels=image_channels,
            base_channels=discriminator_base_channels,
        ),
    )
