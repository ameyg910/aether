"""End-to-end trainer smoke: runs, logs, and checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from aether.config.schemas import ModelConfig, TrainConfig
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.models.aether_model import AetherModel
from aether.train.tracking import JSONLTracker
from aether.train.trainer import Trainer

_VOCAB, _MASK, _LEN = 40, 39, 16


def _synth() -> Iterator[torch.Tensor]:
    while True:
        yield torch.randint(0, _MASK, (4, _LEN))


def test_trainer_runs_and_logs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = TrainConfig(
        max_steps=6,
        batch_size=4,
        grad_accum=2,
        lr=1e-3,
        warmup_steps=2,
        precision="bf16",
        ema_decay=0.9,
        device="cpu",
        log_every=2,
        sample_every=0,
        ckpt_every=3,
        sample_length=_LEN,
        sample_steps=4,
    )
    model = AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    )
    run = tmp_path / "run"
    tr = Trainer(model, MaskedDiffusionLoss(_MASK, "linear"), cfg, run, tracker=JSONLTracker(run))
    tr.fit(_synth())

    assert tr.step == 6
    assert (run / "checkpoints" / "latest.pt").exists()
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) == 3  # steps 2, 4, 6
    assert all(v == v for v in losses)  # finite (no NaN)


def test_sample_callback_produces_text(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = TrainConfig(
        max_steps=1,
        batch_size=4,
        grad_accum=1,
        precision="fp32",
        device="cpu",
        warmup_steps=1,
        log_every=100,
        sample_every=0,
        ckpt_every=0,
        sample_length=_LEN,
        sample_steps=4,
    )
    model = AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    )
    tr = Trainer(model, MaskedDiffusionLoss(_MASK, "linear"), cfg, tmp_path / "run")
    text = tr.sample_text()
    assert isinstance(text, str)
    assert len(text) > 0
