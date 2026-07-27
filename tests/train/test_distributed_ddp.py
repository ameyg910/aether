"""End-to-end DDP verification with real worker processes.

These tests launch actual subprocesses over the ``gloo`` backend on CPU, so they
exercise the same code path a multi-GPU run takes -- process group setup, model
wrapping, gradient all-reduce, rank-0-only side effects, and checkpoint
consolidation -- without needing a GPU. They are the CI proxy for the 3xA6000 run.

The load-bearing assertion is that ranks fed *different* data end with *identical*
parameters. That can only happen if gradients were actually synchronized; if the
all-reduce silently no-op'd, the replicas would drift apart immediately.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from aether.config.schemas import ModelConfig, TrainConfig
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.models.aether_model import AetherModel
from aether.train.trainer import Trainer

_VOCAB, _MASK, _LEN = 40, 39, 16
_WORLD = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _cfg(**overrides: object) -> TrainConfig:
    base = {
        "max_steps": 4,
        "batch_size": 4,
        "grad_accum": 2,
        "lr": 1e-3,
        "warmup_steps": 1,
        "precision": "fp32",
        "ema_decay": 0.9,
        "device": "cpu",
        "log_every": 100,
        "sample_every": 0,
        "ckpt_every": 4,
        "keep_last": 1,
    }
    base.update(overrides)
    return TrainConfig(**base)  # type: ignore[arg-type]


def _build_model() -> AetherModel:
    torch.manual_seed(0)
    return AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    )


def _ddp_worker(rank: int, world: int, port: int, run_dir: str, queue: mp.Queue) -> None:  # type: ignore[type-arg]
    """Train a few steps under DDP; report the resulting parameter checksum."""
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from aether.train.distributed import init_distributed, shutdown_distributed

    info = init_distributed()
    try:
        trainer = Trainer(
            _build_model(),
            MaskedDiffusionLoss(_MASK, "linear"),
            _cfg(),
            Path(run_dir),
            dist_info=info,
        )
        # Each rank sees a different data stream, exactly as real sharding gives.
        gen = torch.Generator().manual_seed(1000 + rank)

        def batches() -> object:
            while True:
                yield torch.randint(0, _MASK, (4, _LEN), generator=gen)

        trainer.fit(batches())  # type: ignore[arg-type]
        checksum = float(sum(p.detach().float().sum() for p in trainer.core.parameters()))
        queue.put((rank, trainer.strategy, round(checksum, 4), trainer.step))
    finally:
        shutdown_distributed()


def _spawn(run_dir: str) -> list[tuple[int, str, float, int]]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    mp.spawn(  # type: ignore[no-untyped-call]
        _ddp_worker, args=(_WORLD, _free_port(), run_dir, queue), nprocs=_WORLD, join=True
    )
    return sorted(queue.get() for _ in range(_WORLD))


@pytest.mark.timeout(300)
def test_ddp_synchronizes_gradients_across_ranks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn(tmp)

    assert len(results) == _WORLD
    assert all(strategy == "ddp" for _, strategy, _, _ in results)
    assert all(step == 4 for *_, step in results)
    # Different data per rank, identical parameters after training => grads synced.
    checksums = {checksum for _, _, checksum, _ in results}
    assert len(checksums) == 1, f"replicas diverged: {checksums}"


@pytest.mark.timeout(300)
def test_only_rank_zero_writes_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _spawn(tmp)
        written = sorted(p.name for p in (Path(tmp) / "checkpoints").glob("*.pt"))
    # One set of files, not one per rank.
    assert written == ["latest.pt", "step_4.pt"]


@pytest.mark.timeout(300)
def test_ddp_checkpoint_restores_into_a_single_process_run() -> None:
    """A checkpoint written by a 2-rank DDP job must load into a lone process.

    This is what makes checkpoints portable across cluster allocations: the saved
    state dict carries no ``module.`` prefix and no sharding.
    """
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn(tmp)
        distributed_checksum = results[0][2]

        solo = Trainer(
            _build_model(),
            MaskedDiffusionLoss(_MASK, "linear"),
            _cfg(strategy="none", ckpt_every=0),
            Path(tempfile.mkdtemp()),
        )
        solo.load(Path(tmp) / "checkpoints" / "latest.pt")

    restored = round(float(sum(p.detach().float().sum() for p in solo.core.parameters())), 4)
    assert restored == pytest.approx(distributed_checksum, abs=1e-3)
    assert solo.step == 4
