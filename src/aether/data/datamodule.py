"""Deterministic, memory-mapped loading of prepared shards.

The data module reads the manifest, memory-maps the shards for a split, and yields
fixed-length ``int64`` batches. Shuffling is seeded by ``seed + epoch``, so the
first batch of a given epoch is byte-for-byte reproducible. Masking is applied
later by the diffusion loss, not here.

Batches are returned as NumPy arrays; the torch-tensor conversion is a one-line
adapter added in Week 3 when the model lands, keeping the data layer framework-
agnostic and its tests hermetic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from aether.data.prepare import Manifest, load_manifest

IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]


@dataclass
class DiffusionDataModule:
    """Loads clean token blocks for one split with deterministic shuffling."""

    data_dir: str
    split: str = "train"
    batch_size: int = 8
    seed: int = 42
    drop_last: bool = True

    _manifest: Manifest = field(init=False, repr=False)
    _shards: list[IntArray] = field(init=False, repr=False)
    _index: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.data_dir)
        self._manifest = load_manifest(root)
        self._shards = [
            np.load(root / s.filename, mmap_mode="r")
            for s in self._manifest.shards
            if s.split == self.split
        ]
        self._index = [
            (shard_idx, block_idx)
            for shard_idx, shard in enumerate(self._shards)
            for block_idx in range(shard.shape[0])
        ]

    @property
    def block_size(self) -> int:
        return self._manifest.block_size

    @property
    def vocab_size(self) -> int:
        return self._manifest.vocab_size

    @property
    def mask_token_id(self) -> int:
        return self._manifest.mask_token_id

    @property
    def num_blocks(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        n = self.num_blocks
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _gather(self, block_ids: IntArray) -> IntArray:
        rows = [
            np.asarray(self._shards[self._index[i][0]][self._index[i][1]], dtype=np.int64)
            for i in block_ids
        ]
        out: IntArray = np.stack(rows)
        return out

    def epoch_batches(self, epoch: int = 0) -> Iterator[IntArray]:
        """Yield ``int64`` batches of shape ``(batch_size, block_size)``."""
        rng = np.random.default_rng(self.seed + epoch)
        order: IntArray = rng.permutation(self.num_blocks)
        limit = len(order) - (len(order) % self.batch_size) if self.drop_last else len(order)
        for start in range(0, limit, self.batch_size):
            yield self._gather(order[start : start + self.batch_size])
