"""The Aether training loop.

A typed, config-driven ``Trainer`` that turns the tested model + loss into a real
training system: bf16 AMP, gradient accumulation, cosine+warmup LR, gradient
clipping, EMA, structured logging, pluggable experiment tracking, a periodic
sample callback, and resumable checkpointing. The loop is decoupled from the data
source -- it consumes an iterator of clean-token batches -- so it can be exercised
on CPU with synthetic data in tests and on real shards in production.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn

from aether.config.schemas import TrainConfig
from aether.data.tokenizer import Tokenizer
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.diffusion.sampler import ancestral_sample
from aether.log import get_logger
from aether.train.checkpoint import restore_training_state, save_checkpoint
from aether.train.ema import EMA
from aether.train.lr_schedule import build_scheduler
from aether.train.tracking import NoOpTracker, Tracker

logger = get_logger("trainer")

_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Trainer:
    """Owns the optimizer/scheduler/EMA and runs the training loop."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MaskedDiffusionLoss,
        cfg: TrainConfig,
        run_dir: Path,
        tracker: Tracker | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.tracker: Tracker = tracker or NoOpTracker()
        self.tokenizer = tokenizer
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.amp_dtype = _AMP_DTYPE[cfg.precision]
        self.amp_enabled = cfg.precision in ("bf16", "fp16")
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16"))

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
        )
        self.ema = EMA(self.model, cfg.ema_decay)
        self.step = 0

    # -- checkpointing -------------------------------------------------------
    def save(self, tag: str = "latest") -> Path:
        path = self.run_dir / "checkpoints" / f"{tag}.pt"
        save_checkpoint(path, self.model, self.ema, self.optimizer, self.scheduler, self.step)
        return path

    def load(self, path: Path, restore_rng: bool = True) -> None:
        from aether.train.checkpoint import load_checkpoint

        ckpt = load_checkpoint(path, map_location=self.device.type)
        self.step = restore_training_state(
            ckpt, self.model, self.ema, self.optimizer, self.scheduler, restore_rng
        )
        logger.info("resumed", step=self.step, path=str(path))

    def _prune_checkpoints(self) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        step_ckpts = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        for stale in step_ckpts[: -self.cfg.keep_last] if self.cfg.keep_last > 0 else []:
            stale.unlink(missing_ok=True)

    # -- sampling callback ---------------------------------------------------
    @torch.no_grad()
    def sample_text(self) -> str:
        self.ema.store(self.model)
        self.ema.copy_to(self.model)
        self.model.eval()
        try:
            ids = ancestral_sample(
                self.model,
                batch=1,
                length=self.cfg.sample_length,
                mask_token_id=self.loss_fn.mask_token_id,
                steps=self.cfg.sample_steps,
                schedule=self.loss_fn.schedule,
                device=self.device,
            )
        finally:
            self.model.train()
            self.ema.restore(self.model)
        row = ids[0].tolist()
        if self.tokenizer is not None:
            return self.tokenizer.decode(row)
        return " ".join(map(str, row))

    # -- training ------------------------------------------------------------
    def _micro_step(self, batch: Tensor) -> float:
        x0 = batch.to(self.device)
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            out = self.loss_fn(self.model, x0)
            loss = out.loss / self.cfg.grad_accum
        self.scaler.scale(loss).backward()
        return float(out.loss.detach())

    def fit(self, batches: Iterator[Tensor]) -> None:
        self.model.train()
        cfg = self.cfg
        seqs_per_step = cfg.batch_size * cfg.grad_accum
        t_last = time.perf_counter()

        while self.step < cfg.max_steps:
            running = 0.0
            for _ in range(cfg.grad_accum):
                running += self._micro_step(next(batches))

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if self.step % cfg.log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                steps_per_sec = cfg.log_every / dt if dt > 0 else 0.0
                lr = float(self.scheduler.get_last_lr()[0])
                metrics = {
                    "loss": running / cfg.grad_accum,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "steps_per_sec": steps_per_sec,
                    "seqs_per_sec": steps_per_sec * seqs_per_step,
                }
                self.tracker.log(metrics, self.step)
                logger.info("train", step=self.step, **{k: round(v, 5) for k, v in metrics.items()})

            if cfg.sample_every and self.step % cfg.sample_every == 0:
                text = self.sample_text()
                self.tracker.log_text("sample", text, self.step)
                logger.info("sample", step=self.step, text=text[:120])

            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save(f"step_{self.step}")
                self.save("latest")
                self._prune_checkpoints()

        self.save("latest")
        self.tracker.finish()
