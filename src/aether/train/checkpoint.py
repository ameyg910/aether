"""Save and restore complete training state for resumable runs.

A checkpoint bundles everything needed to continue a run byte-for-byte after
preemption: model weights, EMA shadow, optimizer and scheduler state, the step
counter, and the RNG states of every library. ``save_checkpoint`` writes atomically
(temp file + rename) so a crash mid-write cannot corrupt the latest checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aether.seed import get_rng_state, set_rng_state
from aether.train.ema import EMA


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
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
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng" in ckpt:
        set_rng_state(ckpt["rng"])
    step: int = ckpt["step"]
    return step
