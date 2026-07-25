"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aether.config.schemas import (
    AetherConfig,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    NoiseScheduleConfig,
)

_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def load_config(overrides: list[str] | None = None) -> AetherConfig:
    """Compose the Hydra config and validate it against the typed schema.

    Args:
        overrides: Hydra-style dotlist overrides, e.g. ``["model.d_model=256"]``.

    Returns:
        A fully typed, validated :class:`AetherConfig` instance.
    """
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        composed = compose(config_name="config", overrides=overrides or [])
    schema = OmegaConf.structured(AetherConfig)
    merged = OmegaConf.merge(schema, composed)
    return cast(AetherConfig, OmegaConf.to_object(merged))


__all__ = [
    "AetherConfig",
    "DataConfig",
    "DiffusionConfig",
    "ModelConfig",
    "NoiseScheduleConfig",
    "load_config",
]
