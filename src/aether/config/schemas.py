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
class DataConfig:
    """Dataset preparation and loading configuration.

    ``source`` uses a scheme prefix: ``"hf:<name>:<config>"`` for a Hugging Face
    dataset (real runs) or ``"local:<path>"`` for a plain-text file (offline
    debug/tests). ``tokenizer`` selects ``"gpt2"`` (BPE, needs the ``data`` extra)
    or ``"byte"`` (offline, dependency-free).
    """

    source: str = "hf:Salesforce/wikitext:wikitext-103-raw-v1"
    split: str = "train"
    tokenizer: str = "gpt2"
    block_size: int = 1024
    val_blocks: int = 256  # absolute, not a fraction, so it works while streaming
    blocks_per_shard: int = 4096
    max_documents: int | None = None  # cap document count for quick runs
    seed: int = 42
    output_dir: str = "data/wikitext103"


@dataclass
class AetherConfig:
    """Top-level run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
