"""Checkpoint save/restore round-trip and resume determinism."""

from __future__ import annotations

from pathlib import Path

import torch

from aether.config.schemas import ModelConfig, TrainConfig
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.models.aether_model import AetherModel
from aether.train.trainer import Trainer

_VOCAB, _MASK, _LEN = 40, 39, 16


def _trainer(run_dir: Path, seed: int = 0) -> Trainer:
    torch.manual_seed(seed)
    cfg = TrainConfig(
        max_steps=5,
        batch_size=4,
        grad_accum=1,
        lr=1e-3,
        warmup_steps=2,
        precision="fp32",
        ema_decay=0.9,
        device="cpu",
        log_every=100,
        sample_every=0,
        ckpt_every=0,
    )
    model = AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    )
    return Trainer(model, MaskedDiffusionLoss(_MASK, "linear"), cfg, run_dir)


def _fixed_data() -> list[torch.Tensor]:
    torch.manual_seed(123)
    return [torch.randint(0, _MASK, (4, _LEN)) for _ in range(20)]


def test_checkpoint_restores_step_and_states(tmp_path: Path) -> None:
    tr = _trainer(tmp_path / "a")
    tr.fit(iter(_fixed_data()))
    path = tr.save("ck")

    tr2 = _trainer(tmp_path / "b")
    assert tr2.step == 0
    tr2.load(path)
    assert tr2.step == tr.step
    # model params match exactly after restore
    for p1, p2 in zip(tr.model.parameters(), tr2.model.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert tr2.scheduler.get_last_lr()[0] == tr.scheduler.get_last_lr()[0]


def test_resume_is_bit_for_bit(tmp_path: Path) -> None:
    data = _fixed_data()

    tr = _trainer(tmp_path / "a")
    tr.cfg.max_steps = 3
    tr.fit(iter(data))
    ckpt = tr.save("mid")
    tr.cfg.max_steps = 6
    tr.fit(iter(data[3:]))
    ref = torch.cat([p.flatten() for p in tr.model.parameters()])

    tr2 = _trainer(tmp_path / "b")
    tr2.load(ckpt)
    tr2.cfg.max_steps = 6
    tr2.fit(iter(data[3:]))
    got = torch.cat([p.flatten() for p in tr2.model.parameters()])

    assert torch.equal(ref, got)


def test_pruning_only_touches_this_run_s_checkpoints(tmp_path: Path) -> None:
    """Retention must not be computed from a directory glob.

    A glob sorted by step number deletes the *lowest* numbered files. Start a
    fresh run in a directory that already holds high-numbered checkpoints from a
    previous run and it deletes the checkpoint it just wrote, keeping the stale
    ones. This is not hypothetical: it destroyed real checkpoints.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for step in (26000, 28000, 30000):
        (run / "checkpoints" / f"step_{step}.pt").write_bytes(b"previous run")

    trainer = _trainer(run)
    trainer.cfg.max_steps = 6
    trainer.cfg.ckpt_every = 2
    trainer.cfg.keep_last = 2
    trainer.fit(iter(_fixed_data()))

    present = {p.name for p in (run / "checkpoints").glob("*.pt")}
    # The previous run's files are untouched.
    assert {"step_26000.pt", "step_28000.pt", "step_30000.pt"} <= present
    # This run kept its own most recent two...
    assert {"step_4.pt", "step_6.pt"} <= present
    # ...and pruned its own oldest.
    assert "step_2.pt" not in present


def test_keep_last_zero_disables_pruning(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trainer = _trainer(run)
    trainer.cfg.max_steps = 6
    trainer.cfg.ckpt_every = 2
    trainer.cfg.keep_last = 0
    trainer.fit(iter(_fixed_data()))

    present = {p.name for p in (run / "checkpoints").glob("step_*.pt")}
    assert present == {"step_2.pt", "step_4.pt", "step_6.pt"}
