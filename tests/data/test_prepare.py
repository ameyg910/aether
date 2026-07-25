"""Dataset build: shape/dtype, manifest correctness, and reproducibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aether.config.schemas import DataConfig
from aether.data.prepare import (
    build_dataset,
    load_manifest,
    storage_dtype,
    tokenize_and_pack,
)
from aether.data.tokenizer import ByteTokenizer

CORPUS = "\n\n".join(f"document number {i} has some words in it" for i in range(60))


def _local_config(tmp: Path) -> DataConfig:
    tmp.mkdir(parents=True, exist_ok=True)
    src = tmp / "corpus.txt"
    src.write_text(CORPUS, encoding="utf-8")
    return DataConfig(
        source=f"local:{src}",
        tokenizer="byte",
        block_size=16,
        val_blocks=2,
        blocks_per_shard=8,
        output_dir=str(tmp / "out"),
    )


def test_storage_dtype_picks_uint16_for_small_vocab() -> None:
    assert storage_dtype(258) is np.uint16
    assert storage_dtype(70000) is np.uint32


def test_packing_yields_exact_blocks() -> None:
    blocks = list(tokenize_and_pack(["abc def ghi jkl mno pqr"], ByteTokenizer(), 8))
    assert blocks
    assert all(b.shape == (8,) for b in blocks)


def test_build_shapes_and_manifest(tmp_path: Path) -> None:
    cfg = _local_config(tmp_path)
    manifest = build_dataset(cfg)

    assert manifest.tokenizer == "byte"
    assert manifest.mask_token_id == 257
    assert manifest.storage_dtype == "uint16"
    assert manifest.num_val_blocks == 2
    assert manifest.num_train_blocks > 0

    for shard in manifest.shards:
        arr = np.load(Path(cfg.output_dir) / shard.filename)
        assert arr.shape[1] == cfg.block_size
        assert arr.dtype == np.uint16
        assert arr.shape[0] == shard.num_blocks


def test_shard_filenames_are_content_addressed(tmp_path: Path) -> None:
    import hashlib

    cfg = _local_config(tmp_path)
    manifest = build_dataset(cfg)
    for shard in manifest.shards:
        arr = np.load(Path(cfg.output_dir) / shard.filename)
        assert hashlib.sha256(arr.tobytes()).hexdigest() == shard.sha256
        assert shard.sha256[:12] in shard.filename


def test_build_is_reproducible(tmp_path: Path) -> None:
    cfg_a = _local_config(tmp_path / "a")
    cfg_b = _local_config(tmp_path / "b")
    a = build_dataset(cfg_a)
    b = build_dataset(cfg_b)
    assert a.dataset_hash == b.dataset_hash
    assert [s.sha256 for s in a.shards] == [s.sha256 for s in b.shards]


def test_manifest_roundtrip(tmp_path: Path) -> None:
    cfg = _local_config(tmp_path)
    written = build_dataset(cfg)
    loaded = load_manifest(Path(cfg.output_dir))
    assert loaded.dataset_hash == written.dataset_hash
    assert loaded.shards == written.shards
