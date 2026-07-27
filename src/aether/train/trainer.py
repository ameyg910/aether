"""The Aether training loop.

A typed, config-driven ``Trainer``: bf16 AMP, gradient accumulation, cosine+warmup
LR, gradient clipping, EMA, structured logging, pluggable experiment tracking, a
periodic sample callback, and resumable checkpointing. Week 5 makes it
distributed-aware -- it wraps the model in DDP or FSDP, reduces metrics across
ranks, restricts side effects to rank 0, and reports throughput and MFU.

The loop stays decoupled from the data source: it consumes an iterator of clean
token batches, so it can be exercised on CPU with synthetic data in tests and on
real shards across many GPUs in production.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from aether.config.schemas import TrainConfig
from aether.data.tokenizer import Tokenizer
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.diffusion.sampler import ancestral_sample
from aether.log import get_logger
from aether.train.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from aether.train.distributed import (
    DistInfo,
    all_reduce_mean,
    enable_activation_checkpointing,
    resolve_strategy,
    wrap_model,
)
from aether.train.ema import EMA
from aether.train.lr_schedule import build_scheduler
from aether.train.mfu import ThroughputMeter, lookup_peak_tflops, training_flops_per_token
from aether.train.precision import PrecisionPlan, resolve_device
from aether.train.tracking import NoOpTracker, Tracker

logger = get_logger("trainer")


class Trainer:
    """Owns the optimizer/scheduler/EMA and runs the (optionally distributed) loop."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: MaskedDiffusionLoss,
        cfg: TrainConfig,
        run_dir: Path,
        tracker: Tracker | None = None,
        tokenizer: Tokenizer | None = None,
        dist_info: DistInfo | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.dist = dist_info or DistInfo()
        # Only rank 0 writes: N ranks appending to one metrics file on a shared
        # filesystem is corruption waiting to happen.
        self.tracker: Tracker = (tracker or NoOpTracker()) if self.dist.is_main else NoOpTracker()
        self.tokenizer = tokenizer
        self.device = resolve_device(cfg.device, self.dist.local_rank)
        self.precision = PrecisionPlan.from_spec(cfg.precision)
        self.loss_fn = loss_fn

        self.strategy = resolve_strategy(cfg.strategy, self.dist.world_size)
        core = model.to(self.device)
        if cfg.activation_checkpointing:
            n_wrapped = enable_activation_checkpointing(core)
            logger.info("activation_checkpointing", layers=n_wrapped)
        # ``core`` is the unwrapped module: EMA, checkpoints, and sampling all use
        # it so their keys never depend on how the job was launched.
        self.core = core
        self.model = wrap_model(
            core,
            self.strategy,
            self.dist,
            self.device,
            min_num_params=cfg.fsdp_min_params,
            precision=cfg.precision,
        )

        self.scaler = self.precision.scaler(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
        )
        self.ema = EMA(self.core, cfg.ema_decay)
        self.step = 0
        self._meter: ThroughputMeter | None = None

    # -- throughput ----------------------------------------------------------
    def _build_meter(self, seq_len: int) -> ThroughputMeter:
        """Build the MFU meter once the sequence length is known from the data."""
        model_cfg = getattr(self.core, "cfg", None)
        n_layers = int(getattr(model_cfg, "n_layers", 0))
        d_model = int(getattr(model_cfg, "d_model", 0))
        num_params = int(getattr(self.core, "num_params", 0))
        flops = training_flops_per_token(num_params, n_layers, d_model, seq_len)
        peak = self.cfg.device_peak_tflops
        if peak is None and self.device.type == "cuda":
            peak = lookup_peak_tflops(torch.cuda.get_device_name(self.device))
        if peak is None and self.dist.is_main:
            logger.info("mfu_disabled", reason="unknown device; set train.device_peak_tflops")
        return ThroughputMeter(flops, peak, self.dist.world_size)

    # -- checkpointing -------------------------------------------------------
    def save(self, tag: str = "latest") -> Path | None:
        """Write a full, world-size-independent checkpoint from rank 0."""
        path = self.run_dir / "checkpoints" / f"{tag}.pt"
        save_checkpoint(
            path,
            self.model,
            self.ema,
            self.optimizer,
            self.scheduler,
            self.step,
            is_main=self.dist.is_main,
        )
        return path if self.dist.is_main else None

    def load(self, path: Path, restore_rng: bool = True) -> None:
        ckpt = load_checkpoint(path, map_location="cpu")
        self.step = restore_training_state(
            ckpt, self.model, self.ema, self.optimizer, self.scheduler, restore_rng
        )
        logger.info("resumed", step=self.step, path=str(path), rank=self.dist.rank)

    def _prune_checkpoints(self) -> None:
        if not self.dist.is_main:
            return
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
        """Generate a sample from the EMA weights.

        Under FSDP the parameters are sharded, so no single rank holds the full
        model and in-loop sampling is skipped (returns ""); sample from a saved
        full checkpoint instead.
        """
        if self.strategy == "fsdp":
            return ""
        self.ema.store(self.core)
        self.ema.copy_to(self.core)
        self.core.eval()
        try:
            ids = ancestral_sample(
                self.core,
                batch=1,
                length=self.cfg.sample_length,
                mask_token_id=self.loss_fn.mask_token_id,
                steps=self.cfg.sample_steps,
                schedule=self.loss_fn.schedule,
                device=self.device,
            )
        finally:
            self.core.train()
            self.ema.restore(self.core)
        row = ids[0].tolist()
        if self.tokenizer is not None:
            return self.tokenizer.decode(row)
        return " ".join(map(str, row))

    # -- training ------------------------------------------------------------
    def _micro_step(self, batch: Tensor, sync_grads: bool) -> float:
        """One forward/backward on a micro-batch.

        With gradient accumulation under DDP only the *final* micro-batch should
        trigger the gradient all-reduce; ``no_sync()`` suppresses it for the rest,
        turning ``grad_accum`` all-reduces per optimizer step into one.
        """
        x0 = batch.to(self.device, non_blocking=True)
        ctx: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
        if isinstance(self.model, DDP) and not sync_grads:
            ctx = self.model.no_sync()
        with ctx:
            with self.precision.autocast(self.device):
                out = self.loss_fn(self.model, x0)
                loss = out.loss / self.cfg.grad_accum
            self.scaler.scale(loss).backward()
        return float(out.loss.detach())

    def fit(self, batches: Iterator[Tensor]) -> None:
        self.model.train()
        cfg = self.cfg
        t_last = time.perf_counter()
        tokens_since_log = 0

        while self.step < cfg.max_steps:
            running = 0.0
            for micro in range(cfg.grad_accum):
                batch = next(batches)
                if self._meter is None:
                    self._meter = self._build_meter(int(batch.shape[1]))
                tokens_since_log += batch.numel()
                running += self._micro_step(batch, sync_grads=(micro == cfg.grad_accum - 1))

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self.core)
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if self.step % cfg.log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                steps_per_sec = cfg.log_every / dt if dt > 0 else 0.0
                # Reduce so the logged loss describes the global batch rather than
                # rank 0's slice; throughput sums because every rank contributes.
                loss = all_reduce_mean(running / cfg.grad_accum, self.device)
                gnorm = all_reduce_mean(float(grad_norm), self.device)
                metrics = {
                    "loss": loss,
                    "lr": float(self.scheduler.get_last_lr()[0]),
                    "grad_norm": gnorm,
                    "steps_per_sec": steps_per_sec,
                    "seqs_per_sec": steps_per_sec
                    * cfg.batch_size
                    * cfg.grad_accum
                    * self.dist.world_size,
                }
                if self._meter is not None:
                    metrics.update(self._meter.metrics(tokens_since_log * self.dist.world_size, dt))
                tokens_since_log = 0
                self.tracker.log(metrics, self.step)
                if self.dist.is_main:
                    logger.info(
                        "train", step=self.step, **{k: round(v, 5) for k, v in metrics.items()}
                    )

            if cfg.sample_every and self.step % cfg.sample_every == 0 and self.dist.is_main:
                text = self.sample_text()
                if text:
                    self.tracker.log_text("sample", text, self.step)
                    logger.info("sample", step=self.step, text=text[:120])

            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save(f"step_{self.step}")
                self.save("latest")
                self._prune_checkpoints()

        self.save("latest")
        self.tracker.finish()
