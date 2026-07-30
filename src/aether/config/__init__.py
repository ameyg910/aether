"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from aether.config.schemas import (
    AetherConfig,
    DataConfig,
    DiffusionConfig,
    EvalConfig,
    ModelConfig,
    NoiseScheduleConfig,
    ServeConfig,
    TrackingConfig,
    TrainConfig,
)


def _resolve_config_dir() -> str:
    """Locate the ``configs/`` directory.

    ``AETHER_CONFIG_DIR`` wins when set. This is what a packaged install needs:
    once ``aether`` is installed into site-packages, walking up from ``__file__``
    lands inside the environment, not at a repo root, so the container sets the
    variable to where it copied the configs. The source-tree path
    (repo_root/configs) remains the fallback for editable installs and tests.
    """
    override = os.environ.get("AETHER_CONFIG_DIR")
    if override:
        return override
    return str(Path(__file__).resolve().parents[3] / "configs")


_CONFIG_DIR = _resolve_config_dir()


def _register_schema() -> None:
    """Register the typed schema as a Hydra base config.

    This makes every dataclass field present in the composed config, so any field
    is overridable from the CLI even when a group file (e.g. ``train=debug``) only
    specifies a subset.
    """
    cs = ConfigStore.instance()
    cs.store(name="_aether_schema", node=AetherConfig)


def cli_overrides(argv: list[str], usage: str) -> list[str]:
    """Turn argv into Hydra overrides, handling ``--help`` ourselves.

    ``load_config`` calls Hydra's ``compose`` API directly rather than through the
    ``@hydra.main`` decorator, so the usual ``--help`` handling does not exist:
    Hydra would try to parse ``--help`` as a ``key=value`` override and raise a
    lexer error. We intercept the help flags, print usage, and exit cleanly --
    which is also what makes a bare ``docker run <image>`` behave sensibly.
    """
    import sys

    if any(a in ("--help", "-h") for a in argv):
        print(usage)
        sys.exit(0)
    return argv


def load_config(overrides: list[str] | None = None) -> AetherConfig:
    """Compose the Hydra config and validate it against the typed schema.

    Args:
        overrides: Hydra-style dotlist overrides, e.g. ``["model.d_model=256"]``.

    Returns:
        A fully typed, validated :class:`AetherConfig` instance.
    """
    _register_schema()
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        composed = compose(config_name="config", overrides=overrides or [])
    schema = OmegaConf.structured(AetherConfig)
    merged = OmegaConf.merge(schema, composed)
    return cast(AetherConfig, OmegaConf.to_object(merged))


__all__ = [
    "AetherConfig",
    "DataConfig",
    "DiffusionConfig",
    "EvalConfig",
    "ModelConfig",
    "NoiseScheduleConfig",
    "ServeConfig",
    "TrackingConfig",
    "TrainConfig",
    "cli_overrides",
    "load_config",
]
