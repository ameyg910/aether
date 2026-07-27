"""Distributed helpers that are testable in a single process."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aether.train.distributed import (
    DistInfo,
    all_reduce_mean,
    broadcast_object,
    enable_activation_checkpointing,
    resolve_strategy,
    unwrap_model,
)


def test_dist_info_defaults_to_single_process() -> None:
    info = DistInfo()
    assert info.world_size == 1
    assert info.is_distributed is False
    assert info.is_main is True


def test_only_rank_zero_is_main() -> None:
    assert DistInfo(rank=0, world_size=4).is_main is True
    assert DistInfo(rank=3, world_size=4).is_main is False


def test_auto_strategy_follows_world_size() -> None:
    assert resolve_strategy("auto", 1) == "none"
    assert resolve_strategy("auto", 3) == "ddp"


def test_explicit_strategy_is_ignored_for_one_process() -> None:
    # Wrapping a lone process in DDP/FSDP costs overhead and buys nothing.
    assert resolve_strategy("ddp", 1) == "none"
    assert resolve_strategy("fsdp", 1) == "none"
    assert resolve_strategy("fsdp", 2) == "fsdp"


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        resolve_strategy("horovod", 2)


def test_unwrap_returns_plain_module_untouched() -> None:
    model = nn.Linear(4, 4)
    assert unwrap_model(model) is model


def test_collectives_are_noops_without_a_process_group() -> None:
    # The same code path must work when launched as a single process.
    assert all_reduce_mean(2.5, torch.device("cpu")) == 2.5
    assert broadcast_object("run-name") == "run-name"


def test_activation_checkpointing_wraps_blocks() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.lin(x)
            return out

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for b in self.blocks:
                x = b(x)
            return x

    net = Net()
    assert enable_activation_checkpointing(net) == 2
    # Still trains: gradients flow through the recomputed graph.
    out = net(torch.randn(2, 4)).sum()
    out.backward()
    assert net.blocks[0].lin.weight.grad is not None  # type: ignore[index,union-attr]
