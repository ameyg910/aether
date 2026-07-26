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

    source: str = "hf:wikitext:wikitext-103-raw-v1"
    split: str = "train"
    tokenizer: str = "gpt2"
    block_size: int = 1024
    val_blocks: int = 256  # absolute, not a fraction, so it works while streaming
    blocks_per_shard: int = 4096
    max_documents: int | None = None  # cap document count for quick runs
    seed: int = 42
    output_dir: str = "data/wikitext103"


@dataclass
class TrainConfig:
    """Training loop configuration (Week 4)."""

    max_steps: int = 5000
    batch_size: int = 8  # micro-batch per forward
    grad_accum: int = 1  # optimizer steps every ``grad_accum`` micro-batches
    lr: float = 3e-4
    min_lr_ratio: float = 0.1  # cosine floor as a fraction of ``lr``
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"  # "bf16" | "fp16" | "fp32"
    ema_decay: float = 0.999
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compile: bool = False  # torch.compile the model (stretch goal)
    log_every: int = 10
    sample_every: int = 500
    ckpt_every: int = 1000
    keep_last: int = 3  # rolling checkpoints to retain (besides latest.pt)
    sample_length: int = 64
    sample_steps: int = 64
    out_dir: str = "runs"
    run_name: str | None = None
    resume: str | None = None  # path to a checkpoint to resume from


@dataclass
class TrackingConfig:
    """Experiment-tracking backend configuration."""

    backend: str = "jsonl"  # "none" | "jsonl" | "wandb"
    project: str = "aether"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class AetherConfig:
    """Top-level run configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 42
