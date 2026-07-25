"""Typed configuration schemas (Hydra structured configs).

These dataclasses are the single source of truth for run configuration. YAML in
``configs/`` overrides their defaults; Hydra/OmegaConf validates the merge, so a
typo or wrong type in a config fails loudly instead of silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NoiseScheduleConfig:
    """Noise schedule for the absorbing forward process."""

    kind: str = "linear"  # "linear" | "cosine"
    sigma_min: float = 1e-4  # reserved for future (geometric/D3PM) schedules
    sigma_max: float = 20.0


@dataclass
class DiffusionConfig:
    """Absorbing-state discrete diffusion configuration."""

    mask_token_id: int = 0
    num_timesteps: int = 1000  # sampling discretization (used from Week 6)
    schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)


@dataclass
class ModelConfig:
    """Denoiser backbone configuration (the model itself lands in Week 3)."""

    vocab_size: int = 256  # toy default; real vocab is set in Week 2
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 128
    dropout: float = 0.0


@dataclass
class AetherConfig:
    """Top-level run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    seed: int = 42
