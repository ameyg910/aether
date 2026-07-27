"""Save and restore complete training state for resumable runs.

A checkpoint bundles everything needed to continue after preemption: model
weights, EMA shadow, optimizer and scheduler state, the step counter, and the RNG
states of every library. ``save_checkpoint`` writes atomically (temp file +
rename) so a crash mid-write cannot corrupt the latest checkpoint.

Checkpoints are **world-size independent**. Model and optimizer state are pulled
through ``torch.distributed.checkpoint.state_dict``, which consolidates FSDP
shards and strips DDP's ``module.`` prefix, so a checkpoint written by a 3-GPU
FSDP run restores into a single-process run and vice versa. Gathering that full
state dict is a *collective*: every rank must call ``save_checkpoint``, even
though only rank 0 writes the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aether.seed import get_rng_state, set_rng_state
from aether.train.ema import EMA

# ``cpu_offload`` keeps the consolidated copy off the GPU that is already holding
# the live model, which matters when the model only just fits.
_FULL = StateDictOptions(full_state_dict=True, cpu_offload=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    extra: dict[str, Any] | None = None,
    is_main: bool = True,
) -> None:
    """Write a full checkpoint. Must be called on every rank; writes on rank 0."""
    model_sd = get_model_state_dict(model, options=_FULL)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=_FULL)
    if not is_main:
        return
    payload: dict[str, Any] = {
        "model": model_sd,
        "ema": ema.state_dict(),
        "optimizer": optim_sd,
        "scheduler": scheduler.state_dict(),
        "step": step,
        "rng": get_rng_state(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    ckpt: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt


def restore_training_state(
    ckpt: dict[str, Any],
    model: nn.Module,
    ema: EMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    restore_rng: bool = True,
) -> int:
    """Load all component states in place; return the step to resume from."""
    set_model_state_dict(model, ckpt["model"], options=_FULL)
    set_optimizer_state_dict(model, optimizer, ckpt["optimizer"], options=_FULL)
    ema.load_state_dict(ckpt["ema"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng" in ckpt:
        set_rng_state(ckpt["rng"])
    step: int = ckpt["step"]
    return step
