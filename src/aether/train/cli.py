"""Command-line entry point for training: ``aether-train [hydra overrides]``.

Wires the prepared data shards, tokenizer, model, and loss into the ``Trainer``.
Example::

    aether-train train=debug data=local_debug tracking.backend=none
    aether-train train.resume=runs/<name>/checkpoints/latest.pt

The model's vocab and sequence length are taken from the dataset manifest so the
model and data can never silently disagree.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from aether.config import load_config
from aether.data.datamodule import DiffusionDataModule
from aether.data.tokenizer import build_tokenizer
from aether.diffusion.loss import MaskedDiffusionLoss
from aether.log import configure_logging, get_logger
from aether.models.aether_model import AetherModel
from aether.seed import seed_everything
from aether.train.manifest import write_manifest
from aether.train.tracking import build_tracker
from aether.train.trainer import Trainer

logger = get_logger("cli")


def torch_batches(dm: DiffusionDataModule) -> Iterator[Tensor]:
    """Infinite iterator of ``int64`` token batches across epochs."""
    epoch = 0
    while True:
        for batch in dm.epoch_batches(epoch):
            yield torch.from_numpy(np.ascontiguousarray(batch))
        epoch += 1


def main() -> None:
    cfg = load_config(sys.argv[1:])
    configure_logging()
    seed_everything(cfg.seed)

    dm = DiffusionDataModule(
        cfg.data.output_dir,
        split="train",
        batch_size=cfg.train.batch_size,
        seed=cfg.seed,
    )
    # Model and loss inherit vocab/mask/length from the data manifest.
    model_cfg = dataclasses.replace(cfg.model, vocab_size=dm.vocab_size, max_seq_len=dm.block_size)
    model = AetherModel(model_cfg)
    loss_fn = MaskedDiffusionLoss(
        mask_token_id=dm.mask_token_id, schedule=cfg.diffusion.schedule.kind
    )

    run_name = cfg.train.run_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.train.out_dir) / run_name
    write_manifest(
        run_dir,
        OmegaConf.to_yaml(OmegaConf.structured(cfg)),
        seed=cfg.seed,
        data_dir=cfg.data.output_dir,
        extra={"params": model.num_params, "vocab_size": dm.vocab_size},
    )

    tracker = build_tracker(
        cfg.tracking.backend,
        run_dir,
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        tags=list(cfg.tracking.tags),
        config=OmegaConf.to_container(OmegaConf.structured(cfg)),  # type: ignore[arg-type]
    )

    tokenizer = build_tokenizer(cfg.data.tokenizer)
    trainer = Trainer(model, loss_fn, cfg.train, run_dir, tracker, tokenizer)

    if cfg.train.resume:
        resume = Path(cfg.train.resume)
        if resume.is_dir():
            resume = resume / "checkpoints" / "latest.pt"
        trainer.load(resume)

    logger.info(
        "train_start", run_dir=str(run_dir), params=model.num_params, device=str(trainer.device)
    )
    trainer.fit(torch_batches(dm))
    logger.info("train_done", run_dir=str(run_dir), step=trainer.step)


if __name__ == "__main__":
    main()
