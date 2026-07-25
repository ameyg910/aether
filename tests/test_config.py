"""The Hydra config composes and validates against the typed schema."""

from __future__ import annotations

from aether.config import AetherConfig, load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, AetherConfig)
    assert cfg.model.d_model == 128
    assert cfg.diffusion.mask_token_id == 0
    assert cfg.diffusion.schedule.kind == "linear"


def test_override_applies() -> None:
    cfg = load_config(overrides=["model.d_model=256"])
    assert cfg.model.d_model == 256
