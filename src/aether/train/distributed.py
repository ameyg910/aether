"""Distributed training: process groups, DDP/FSDP wrapping, and collectives.

Aether supports three execution strategies, selected by ``train.strategy``:

- ``none`` -- single process, no collectives.
- ``ddp``  -- every rank holds a full model replica and an all-reduce averages
  gradients after each backward pass. The right default whenever the model,
  its gradients, and its optimizer state fit comfortably on one device.
- ``fsdp`` -- parameters, gradients, and optimizer state are *sharded* across
  ranks; each is all-gathered just in time for the layers that need it. This
  trades extra communication for a roughly ``world_size``-fold reduction in
  per-device memory, which is what lets a model larger than one GPU train at all.

See ADR-0003 for the decision record. Rank/world-size come from the environment
variables ``torchrun`` sets, so nothing here needs to know how it was launched --
the same code runs under ``torchrun`` locally and under ``srun`` on SLURM.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from aether.log import get_logger

logger = get_logger("distributed")

STRATEGIES = ("none", "ddp", "fsdp")


@dataclass(frozen=True)
class DistInfo:
    """Identity of this process within the job."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on exactly one rank; guards logging, checkpointing, sampling."""
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else default


def init_distributed(timeout_minutes: int = 30) -> DistInfo:
    """Initialize the process group from ``torchrun``/SLURM environment variables.

    Falls back cleanly to a single-process ``DistInfo`` when the job was not
    launched distributed, so the same entry point works either way.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size <= 1:
        return DistInfo(rank=0, local_rank=0, world_size=1, backend="")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        from datetime import timedelta

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, backend=backend)


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def resolve_strategy(spec: str, world_size: int) -> str:
    """Map ``train.strategy`` to a concrete strategy.

    ``auto`` means "DDP when there is more than one rank, otherwise nothing",
    which is the behaviour that makes a single command work everywhere.
    """
    if spec == "auto":
        return "ddp" if world_size > 1 else "none"
    if spec not in STRATEGIES:
        raise ValueError(f"Unknown strategy {spec!r}; expected 'auto' or one of {STRATEGIES}")
    if spec != "none" and world_size <= 1:
        # Wrapping a single process in DDP/FSDP adds overhead and no benefit.
        return "none"
    return spec


def broadcast_object(obj: object, src: int = 0) -> object:
    """Broadcast a picklable object from ``src`` to every rank.

    Used for values that must agree across the job but are computed on one rank --
    a timestamp-derived run name, for instance, would otherwise differ per rank
    and scatter one job across several directories.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across ranks so logged metrics describe the whole job."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP wrappers to reach the underlying module.

    Checkpoints, EMA, and sampling all operate on the unwrapped module so their
    parameter keys are identical regardless of how the run was launched -- a
    checkpoint written by a 3-GPU DDP run loads into a single-process run.
    """
    seen = 0
    while isinstance(model, DDP | FSDP) and seen < 8:
        model = model.module
        seen += 1
    return model


def enable_activation_checkpointing(
    model: nn.Module, layer_cls: type[nn.Module] | None = None
) -> int:
    """Recompute layer activations during backward instead of storing them.

    Trades roughly 30% extra compute for a large activation-memory saving, which
    is what allows a bigger batch (or a bigger model) to fit. Returns the number
    of wrapped layers. ``use_reentrant=False`` is required for compatibility with
    DDP/FSDP gradient hooks.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    if layer_cls is None:
        # Convention: transformer blocks are the recompute unit.
        def check_fn(m: nn.Module) -> bool:
            return type(m).__name__.endswith("Block")

    else:

        def check_fn(m: nn.Module) -> bool:
            return isinstance(m, layer_cls)

    wrapped = sum(1 for m in model.modules() if check_fn(m))
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn,
    )
    return wrapped


def wrap_model(
    model: nn.Module,
    strategy: str,
    dist_info: DistInfo,
    device: torch.device,
    min_num_params: int = 1_000_000,
    precision: str = "fp32",
) -> nn.Module:
    """Wrap ``model`` for the chosen distributed strategy.

    The model must already be on ``device``.
    """
    if strategy == "none":
        return model

    if strategy == "ddp":
        device_ids = [device.index] if device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, find_unused_parameters=False)

    if strategy == "fsdp":
        # Size-based auto-wrap: any submodule holding at least ``min_num_params``
        # parameters becomes its own FSDP unit, so all-gathers are scoped to one
        # block at a time rather than the whole model at once.
        auto_wrap = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        mixed = None
        if device.type == "cuda" and precision in ("bf16", "fp16"):
            param_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            # Reduce gradients in fp32: sharded reductions accumulate error, and
            # the bandwidth saving is not worth the precision loss.
            mixed = MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=param_dtype,
            )
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed,
            device_id=device.index if device.type == "cuda" else None,
            use_orig_params=True,
        )

    raise ValueError(f"Unknown strategy {strategy!r}")
