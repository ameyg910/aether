"""DiffusionDataModule: shape, determinism, and split isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aether.config.schemas import DataConfig
from aether.data.datamodule import DiffusionDataModule
from aether.data.prepare import build_dataset

CORPUS = "\n\n".join(f"line {i} with a handful of words here" for i in range(80))


def _prepare(tmp: Path) -> str:
    src = tmp / "corpus.txt"
    src.write_text(CORPUS, encoding="utf-8")
    cfg = DataConfig(
        source=f"local:{src}",
        tokenizer="byte",
        block_size=16,
        val_blocks=4,
        blocks_per_shard=8,
        output_dir=str(tmp / "out"),
    )
    build_dataset(cfg)
    return cfg.output_dir


def test_batch_shape_and_dtype(tmp_path: Path) -> None:
    dm = DiffusionDataModule(_prepare(tmp_path), split="train", batch_size=4, seed=0)
    batch = next(dm.epoch_batches())
    assert batch.shape == (4, dm.block_size)
    assert batch.dtype == np.int64


def test_same_seed_same_first_batch(tmp_path: Path) -> None:
    out = _prepare(tmp_path)
    a = next(DiffusionDataModule(out, batch_size=4, seed=0).epoch_batches(0))
    b = next(DiffusionDataModule(out, batch_size=4, seed=0).epoch_batches(0))
    assert np.array_equal(a, b)


def test_different_epoch_reorders(tmp_path: Path) -> None:
    dm = DiffusionDataModule(_prepare(tmp_path), batch_size=4, seed=0)
    assert not np.array_equal(next(dm.epoch_batches(0)), next(dm.epoch_batches(1)))


def test_split_isolation(tmp_path: Path) -> None:
    out = _prepare(tmp_path)
    train = DiffusionDataModule(out, split="train")
    val = DiffusionDataModule(out, split="val")
    assert val.num_blocks == 4
    assert train.num_blocks > 0
    assert train.mask_token_id == 257
