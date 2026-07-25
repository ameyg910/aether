#!/usr/bin/env bash
# apply_week2.sh - apply Week 2 (data pipeline) + numpy-typing fixes to Aether.
# Safe to run on a clean Week-1 repo or on a repo where an older copy was applied.
# Usage (from repo root):  bash apply_week2.sh  &&  make all
set -euo pipefail
cd "$(dirname "$0")"

echo ">> creating directories"
mkdir -p src/aether/data src/aether/diffusion configs/data examples docs tests/data scripts

echo ">> writing files (new + type-fixed)"
cat > src/aether/diffusion/schedule.py << '__AETHER_EOF__'
"""Noise schedules for absorbing-state discrete diffusion.

A schedule maps diffusion time ``t in [0, 1]`` to the *survival probability*
``alpha(t)`` = P(a token is NOT masked at time ``t``). It decreases from 1 at
``t=0`` (clean) to 0 at ``t=1`` (fully masked). The mask rate ``1 - alpha(t)``
is therefore monotonically increasing on ``[0, 1]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np

FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]


class NoiseSchedule(ABC):
    """Base class for monotone absorbing-diffusion noise schedules."""

    @abstractmethod
    def alpha(self, t: FloatArray) -> FloatArray:
        """Survival probability at time ``t`` (decreasing from 1 to 0)."""

    def mask_rate(self, t: FloatArray) -> FloatArray:
        """Masking probability at time ``t`` (increasing from 0 to 1)."""
        out: FloatArray = np.asarray(1.0 - self.alpha(t), dtype=np.float64)
        return out


class LinearSchedule(NoiseSchedule):
    """``alpha(t) = 1 - t``. Mask rate rises linearly with time."""

    def alpha(self, t: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(1.0 - t, dtype=np.float64)
        return out


class CosineSchedule(NoiseSchedule):
    """``alpha(t) = cos(pi/2 * t)^2``. Slower masking early, faster late."""

    def alpha(self, t: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(np.cos(0.5 * np.pi * t) ** 2, dtype=np.float64)
        return out


_SCHEDULES: dict[str, type[NoiseSchedule]] = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
}


def build_schedule(kind: str) -> NoiseSchedule:
    """Construct a schedule by name.

    Raises:
        ValueError: if ``kind`` is not a registered schedule.
    """
    try:
        return _SCHEDULES[kind]()
    except KeyError as exc:
        raise ValueError(f"Unknown schedule {kind!r}. Options: {sorted(_SCHEDULES)}") from exc
__AETHER_EOF__

cat > src/aether/diffusion/forward.py << '__AETHER_EOF__'
"""Absorbing-state (masked) discrete diffusion forward process.

Toy/reference implementation over integer token-id arrays, used for tests and
visualization. Each position is independently replaced by ``mask_token_id`` with
probability ``mask_rate(t) = 1 - alpha(t)``. Week 3 adds a batched torch-tensor
path for training; the math here is the reference that path is checked against.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from aether.diffusion.schedule import NoiseSchedule, build_schedule

IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]


@dataclass(frozen=True)
class AbsorbingForwardProcess:
    """Samples ``x_t ~ q(x_t | x_0)`` for absorbing-state diffusion."""

    schedule: NoiseSchedule
    mask_token_id: int = 0

    def mask_rate_at(self, t: float) -> float:
        """Scalar mask rate at time ``t``."""
        return float(self.schedule.mask_rate(np.asarray([t], dtype=np.float64))[0])

    def sample(self, x0: IntArray, t: float, rng: np.random.Generator) -> IntArray:
        """Sample the noised sequence at time ``t``.

        Args:
            x0: Clean token ids, shape ``(seq_len,)``.
            t: Diffusion time in ``[0, 1]``.
            rng: NumPy random generator (pass a seeded one for determinism).

        Returns:
            Token ids with masked positions set to ``mask_token_id``.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t must be in [0, 1], got {t}")
        rate = self.mask_rate_at(t)
        keep = rng.random(size=x0.shape) >= rate
        return np.asarray(np.where(keep, x0, np.int64(self.mask_token_id)), dtype=np.int64)


def _toy_encode(sentence: str, mask_token_id: int) -> tuple[IntArray, dict[int, str]]:
    """Whitespace toy tokenizer (a real BPE tokenizer arrives in Week 2)."""
    vocab: dict[str, int] = {}
    ids: list[int] = []
    next_id = mask_token_id + 1
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1
        ids.append(vocab[word])
    id_to_word = {i: w for w, i in vocab.items()}
    return np.asarray(ids, dtype=np.int64), id_to_word


def _render(x: IntArray, id_to_word: dict[int, str], mask_token_id: int) -> str:
    return " ".join(
        "\u2591\u2591\u2591" if int(tok) == mask_token_id else id_to_word[int(tok)]
        for tok in x.tolist()
    )


def main() -> None:
    """CLI: show a sentence being progressively masked as ``t`` grows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default="the cat sat on the mat")
    parser.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    mask_id = 0
    process = AbsorbingForwardProcess(build_schedule(args.schedule), mask_token_id=mask_id)
    x0, id_to_word = _toy_encode(args.sentence, mask_id)
    rng = np.random.default_rng(args.seed)

    print(f"schedule={args.schedule}  seed={args.seed}")
    print(f"t=0.00 | {_render(x0, id_to_word, mask_id)}  (mask 0%)")
    for t in args.steps:
        xt = process.sample(x0, float(t), rng)
        frac = float((xt == mask_id).mean())
        print(f"t={t:.2f} | {_render(xt, id_to_word, mask_id)}  (mask {frac * 100:.0f}%)")


if __name__ == "__main__":
    main()
__AETHER_EOF__

cat > src/aether/data/tokenizer.py << '__AETHER_EOF__'
"""Tokenizers for Aether.

Absorbing-state diffusion needs a dedicated ``[MASK]`` token that never appears
in real text, plus an end-of-sequence token to separate packed documents. Both
tokenizers here expose ``mask_token_id`` and ``eos_token_id`` and a ``vocab_size``
that already accounts for the mask token, so downstream embeddings and output
heads size themselves correctly.

- :class:`GPT2Tokenizer` wraps the GPT-2 BPE via ``tiktoken`` (install the
  ``data`` extra) and appends ``[MASK]`` as a new highest id. Used for real runs.
- :class:`ByteTokenizer` is a dependency-free byte-level tokenizer used for tests
  and offline debugging, so CI never needs a network download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface the data pipeline depends on."""

    name: str
    version: str

    @property
    def vocab_size(self) -> int: ...

    @property
    def mask_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 byte-level tokenizer with appended EOS and MASK tokens.

    Ids 0-255 are raw bytes; 256 is EOS; 257 is MASK; vocab size is 258.
    Deterministic and offline, which keeps tests and CI hermetic.
    """

    name = "byte"
    version = "1"

    def __init__(self) -> None:
        self._eos = 256
        self._mask = 257
        self._vocab = 258

    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def mask_token_id(self) -> int:
        return self._mask

    @property
    def eos_token_id(self) -> int:
        return self._eos

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    """GPT-2 BPE with an appended ``[MASK]`` token (requires ``tiktoken``)."""

    name = "gpt2"

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "GPT2Tokenizer needs tiktoken. Install the data extra: "
                "pip install 'aether-dlm[data]'"
            ) from exc

        base = tiktoken.get_encoding("gpt2")
        mask_id = int(base.n_vocab)  # 50257 (one past <|endoftext|>=50256)
        self._enc = tiktoken.Encoding(
            name="gpt2-aether",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, "[MASK]": mask_id},
        )
        self._mask: int = mask_id
        self._eos: int = int(self._enc.encode_single_token("<|endoftext|>"))
        self._vocab: int = int(self._enc.n_vocab)
        self.version = getattr(tiktoken, "__version__", "unknown")

    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def mask_token_id(self) -> int:
        return self._mask

    @property
    def eos_token_id(self) -> int:
        return self._eos

    def encode(self, text: str) -> list[int]:
        # encode_ordinary ignores special tokens found in raw text.
        return [int(t) for t in self._enc.encode_ordinary(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._enc.decode(list(ids)))


_TOKENIZERS: dict[str, type[Tokenizer]] = {
    "byte": ByteTokenizer,
    "gpt2": GPT2Tokenizer,
}


def build_tokenizer(name: str) -> Tokenizer:
    """Construct a tokenizer by name.

    Raises:
        ValueError: if ``name`` is not a registered tokenizer.
    """
    try:
        return _TOKENIZERS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown tokenizer {name!r}. Options: {sorted(_TOKENIZERS)}") from exc
__AETHER_EOF__

cat > src/aether/data/prepare.py << '__AETHER_EOF__'
"""Dataset preparation: stream text -> tokenize -> pack -> content-addressed shards.

The build is deterministic: identical source, tokenizer, and block size produce
identical shard bytes, identical per-shard SHA-256 hashes, and a single
``dataset_hash`` fingerprint. Rerunning is a no-op that reproduces the same hash,
which is the reproducibility guarantee a training run cites.

Storage is memory-mapped ``.npy`` shards of shape ``(num_blocks, block_size)`` in
the smallest integer dtype that fits the vocabulary (uint16 for GPT-2). Masking is
*not* applied here; it happens in the diffusion loss so the same clean blocks can
be renoised differently every step.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from aether.config import load_config
from aether.config.schemas import DataConfig
from aether.data.tokenizer import Tokenizer, build_tokenizer

IntArray: TypeAlias = np.ndarray[Any, np.dtype[np.int64]]


@dataclass(frozen=True)
class ShardInfo:
    """One content-addressed shard file."""

    filename: str
    sha256: str
    num_blocks: int
    split: str


@dataclass(frozen=True)
class Manifest:
    """Reproducibility record for a prepared dataset."""

    tokenizer: str
    tokenizer_version: str
    vocab_size: int
    mask_token_id: int
    eos_token_id: int
    block_size: int
    storage_dtype: str
    source: str
    split: str
    val_blocks: int
    seed: int
    num_train_blocks: int
    num_val_blocks: int
    shards: list[ShardInfo]
    dataset_hash: str


def _split_documents(text: str) -> Iterator[str]:
    """Split a plain-text file into documents on blank lines."""
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            yield chunk


def iter_documents(source: str, split: str, max_documents: int | None) -> Iterator[str]:
    """Yield document strings from a ``local:`` file or an ``hf:`` dataset."""
    scheme, _, rest = source.partition(":")
    if scheme == "local":
        text = Path(rest).read_text(encoding="utf-8")
        docs: Iterable[str] = _split_documents(text)
    elif scheme == "hf":
        try:
            import datasets
        except ImportError as exc:  # pragma: no cover - needs the data extra
            raise ImportError(
                "Hugging Face sources need the data extra: pip install 'aether-dlm[data]'"
            ) from exc
        name, _, config = rest.partition(":")
        stream = datasets.load_dataset(name, config or None, split=split, streaming=True)
        docs = (row["text"] for row in stream if row.get("text", "").strip())
    else:
        raise ValueError(f"Unknown source scheme {scheme!r}; use 'local:...' or 'hf:...'")

    for i, doc in enumerate(docs):
        if max_documents is not None and i >= max_documents:
            return
        yield doc


def tokenize_and_pack(
    documents: Iterable[str], tokenizer: Tokenizer, block_size: int
) -> Iterator[IntArray]:
    """Concatenate tokenized documents (EOS-separated) into fixed-length blocks."""
    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    for doc in documents:
        buffer.extend(tokenizer.encode(doc))
        buffer.append(eos)
        while len(buffer) >= block_size:
            yield np.asarray(buffer[:block_size], dtype=np.int64)
            del buffer[:block_size]
    # The trailing partial block (< block_size) is intentionally dropped.


def storage_dtype(vocab_size: int) -> type[np.unsignedinteger[Any]]:
    """Smallest unsigned dtype that can hold every token id (uint16 fits GPT-2)."""
    if vocab_size <= 65536:
        return np.uint16
    return np.uint32


def _write_shard(
    blocks: list[IntArray],
    split: str,
    index: int,
    out_dir: Path,
    dtype: type[np.unsignedinteger[Any]],
) -> ShardInfo:
    stacked: IntArray = np.stack(blocks)
    array = stacked.astype(dtype)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    filename = f"{split}-{index:04d}-{digest[:12]}.npy"
    np.save(out_dir / filename, array)
    return ShardInfo(filename=filename, sha256=digest, num_blocks=array.shape[0], split=split)


def _dataset_hash(cfg: DataConfig, tokenizer: Tokenizer, shards: list[ShardInfo]) -> str:
    # Fingerprints content + preprocessing only. The source path is recorded in the
    # manifest for provenance but deliberately excluded here, so an identical build
    # from a different directory yields the same hash.
    payload = json.dumps(
        {
            "tokenizer": tokenizer.name,
            "tokenizer_version": tokenizer.version,
            "block_size": cfg.block_size,
            "val_blocks": cfg.val_blocks,
            "shard_hashes": sorted(s.sha256 for s in shards),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dataset(cfg: DataConfig, blocks: Iterable[IntArray] | None = None) -> Manifest:
    """Build shards + manifest for ``cfg``.

    Args:
        cfg: Dataset configuration.
        blocks: Optional pre-packed blocks (used by tests to bypass I/O); when
            ``None`` the blocks are produced from ``cfg.source``.

    Returns:
        The :class:`Manifest` describing the prepared dataset (also written to
        ``<output_dir>/manifest.json``).
    """
    tokenizer = build_tokenizer(cfg.tokenizer)
    dtype = storage_dtype(tokenizer.vocab_size)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if blocks is None:
        blocks = tokenize_and_pack(
            iter_documents(cfg.source, cfg.split, cfg.max_documents),
            tokenizer,
            cfg.block_size,
        )

    shards: list[ShardInfo] = []
    val_buf: list[IntArray] = []
    train_buf: list[IntArray] = []
    n_train = 0
    n_val = 0

    for seen, block in enumerate(blocks):
        if seen < cfg.val_blocks:
            val_buf.append(block)
            n_val += 1
            if len(val_buf) >= cfg.blocks_per_shard:
                shards.append(_write_shard(val_buf, "val", len(shards), out_dir, dtype))
                val_buf = []
        else:
            train_buf.append(block)
            n_train += 1
            if len(train_buf) >= cfg.blocks_per_shard:
                shards.append(_write_shard(train_buf, "train", len(shards), out_dir, dtype))
                train_buf = []

    if val_buf:
        shards.append(_write_shard(val_buf, "val", len(shards), out_dir, dtype))
    if train_buf:
        shards.append(_write_shard(train_buf, "train", len(shards), out_dir, dtype))

    manifest = Manifest(
        tokenizer=tokenizer.name,
        tokenizer_version=tokenizer.version,
        vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        block_size=cfg.block_size,
        storage_dtype=np.dtype(dtype).name,
        source=cfg.source,
        split=cfg.split,
        val_blocks=cfg.val_blocks,
        seed=cfg.seed,
        num_train_blocks=n_train,
        num_val_blocks=n_val,
        shards=shards,
        dataset_hash=_dataset_hash(cfg, tokenizer, shards),
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return manifest


def load_manifest(data_dir: Path) -> Manifest:
    """Read a :class:`Manifest` back from ``<data_dir>/manifest.json``."""
    raw = json.loads((data_dir / "manifest.json").read_text())
    shards = [ShardInfo(**s) for s in raw.pop("shards")]
    return Manifest(shards=shards, **raw)


def main() -> None:
    """CLI entry point: ``python -m aether.data.prepare data=wikitext103``."""
    cfg = load_config(sys.argv[1:])
    manifest = build_dataset(cfg.data)
    print(
        f"tokenizer={manifest.tokenizer} vocab={manifest.vocab_size} "
        f"mask_id={manifest.mask_token_id} block={manifest.block_size}"
    )
    print(
        f"train_blocks={manifest.num_train_blocks} val_blocks={manifest.num_val_blocks} "
        f"shards={len(manifest.shards)}"
    )
    print(f"dataset_hash={manifest.dataset_hash}")


if __name__ == "__main__":
    main()
__AETHER_EOF__

cat > src/aether/data/datamodule.py << '__AETHER_EOF__'
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
__AETHER_EOF__

cat > src/aether/data/__init__.py << '__AETHER_EOF__'
"""Data pipeline: tokenization, packing, sharding, and loading."""

from aether.data.datamodule import DiffusionDataModule
from aether.data.prepare import Manifest, ShardInfo, build_dataset, load_manifest
from aether.data.tokenizer import ByteTokenizer, GPT2Tokenizer, Tokenizer, build_tokenizer

__all__ = [
    "ByteTokenizer",
    "DiffusionDataModule",
    "GPT2Tokenizer",
    "Manifest",
    "ShardInfo",
    "Tokenizer",
    "build_dataset",
    "build_tokenizer",
    "load_manifest",
]
__AETHER_EOF__

cat > scripts/prepare_data.py << '__AETHER_EOF__'
"""Thin CLI wrapper: `python scripts/prepare_data.py data=wikitext103`."""

from aether.data.prepare import main

if __name__ == "__main__":
    main()
__AETHER_EOF__

cat > configs/data/wikitext103.yaml << '__AETHER_EOF__'
source: "hf:wikitext:wikitext-103-raw-v1"
split: train
tokenizer: gpt2
block_size: 1024
val_blocks: 256
blocks_per_shard: 4096
max_documents: null
seed: 42
output_dir: data/wikitext103
__AETHER_EOF__

cat > configs/data/local_debug.yaml << '__AETHER_EOF__'
# Offline, dependency-free dataset for `make data-debug` and quick local tries.
source: "local:examples/sample_corpus.txt"
split: train
tokenizer: byte
block_size: 64
val_blocks: 2
blocks_per_shard: 16
max_documents: null
seed: 42
output_dir: data/debug
__AETHER_EOF__

cat > examples/sample_corpus.txt << '__AETHER_EOF__'
The forward process of an absorbing diffusion model replaces tokens with a mask.
Each position is corrupted independently, at a rate that grows with diffusion time.
By the final step every token has become the absorbing mask state.

The reverse process learns to unmask, predicting clean tokens from partial context.
Because the model reads the whole sequence at once, it is bidirectional, not causal.
Training reduces to a weighted sum of masked language modeling losses.

Packing concatenates many short documents into fixed length blocks.
This removes padding waste and keeps every training token useful.
A single end of sequence token separates one document from the next.

Reproducibility comes from content addressing and a manifest.
Two identical builds produce identical shard hashes and one dataset hash.
That hash is the fingerprint you cite when you say a run is reproducible.
__AETHER_EOF__

cat > docs/data.md << '__AETHER_EOF__'
# Data card

## Source

- **Real runs:** [WikiText-103](https://huggingface.co/datasets/wikitext)
  (`wikitext-103-raw-v1`), ~103M tokens of verified Wikipedia articles.
  License: **CC BY-SA 3.0**. Loaded via `datasets` streaming (install `[data]`).
- **Offline debug/tests:** `examples/sample_corpus.txt`, a tiny hand-written corpus
  tokenized with the dependency-free byte tokenizer.

## Preprocessing

1. **Tokenize** with GPT-2 BPE plus an appended `[MASK]` absorbing token
   (`vocab_size = 50258`, `mask_token_id = 50257`, `eos_token_id = 50256`).
2. **Pack**: concatenate documents into a single token stream, one EOS token
   between documents, then chunk into fixed-length blocks of `block_size` (1024).
   The trailing partial block is dropped.
3. **Split**: the first `val_blocks` (256) blocks are held out for validation; the
   rest are training. An absolute count keeps the split deterministic while
   streaming, without buffering the whole corpus.
4. **Store**: `.npy` shards of shape `(num_blocks, block_size)` in the smallest
   unsigned dtype that fits the vocab (`uint16` for GPT-2), memory-mapped at load.

Masking is **not** baked into the data; the diffusion loss renoises clean blocks
freshly each step, so a block is seen at many noise levels across training.

## Versioning

Each shard is content-addressed: its filename embeds the SHA-256 of its raw bytes,
and `manifest.json` records every shard hash plus a single `dataset_hash`
fingerprint over the tokenizer, block size, val split, and shard hashes (not the
source path, so it is location-independent). Two
identical builds produce an identical `dataset_hash` — the value a training run
cites for reproducibility.

## Manifest fields

`tokenizer`, `tokenizer_version`, `vocab_size`, `mask_token_id`, `eos_token_id`,
`block_size`, `storage_dtype`, `source`, `split`, `val_blocks`, `seed`,
`num_train_blocks`, `num_val_blocks`, `shards[]` (`filename`, `sha256`,
`num_blocks`, `split`), `dataset_hash`.
__AETHER_EOF__

cat > docs/reproducibility.md << '__AETHER_EOF__'
# Reproducibility

## Rebuild the dataset

Offline (no downloads, uses the shipped sample corpus):

```bash
make data-debug            # aether-prepare data=local_debug
```

Real WikiText-103 (needs the data extra and network access):

```bash
pip install -e ".[data]"
make data                  # python -m aether.data.prepare data=wikitext103
```

Override any field on the command line (Hydra):

```bash
python -m aether.data.prepare data=wikitext103 data.block_size=512 data.max_documents=1000
```

## Verify a build is reproducible

`prepare` prints a `dataset_hash`. Run it twice and confirm the hash is identical;
it is computed from the tokenizer, block size, val split, and the SHA-256 of
every shard's bytes, so any change to the data changes the hash.

```bash
python -m aether.data.prepare data=local_debug | grep dataset_hash
python -m aether.data.prepare data=local_debug | grep dataset_hash   # same value
```

## What determinism relies on

- Fixed source document order + fixed tokenizer + fixed `block_size` -> identical
  packed blocks -> identical shard bytes -> identical hashes.
- Data-loading shuffle is seeded by `seed + epoch` in `DiffusionDataModule`, so the
  first batch of any epoch is byte-for-byte reproducible.
__AETHER_EOF__

cat > tests/data/__init__.py << '__AETHER_EOF__'

__AETHER_EOF__

cat > tests/data/test_tokenizer.py << '__AETHER_EOF__'
"""Tokenizer invariants (byte tokenizer is hermetic; gpt2 is optional)."""

from __future__ import annotations

import pytest

from aether.data.tokenizer import ByteTokenizer, build_tokenizer


def test_byte_roundtrip() -> None:
    tok = ByteTokenizer()
    text = "the cat sat on the mat"
    assert tok.decode(tok.encode(text)) == text


def test_byte_special_tokens_distinct_and_in_vocab() -> None:
    tok = ByteTokenizer()
    assert tok.mask_token_id != tok.eos_token_id
    assert tok.mask_token_id < tok.vocab_size
    assert tok.eos_token_id < tok.vocab_size
    # A real byte can never collide with the mask id.
    assert all(b < tok.mask_token_id for b in tok.encode("hello"))


def test_build_unknown_tokenizer_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_tokenizer("nope")


def test_gpt2_tokenizer_if_available() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        tok = build_tokenizer("gpt2")
    except Exception as exc:  # offline: BPE ranks can't be fetched
        pytest.skip(f"gpt2 tokenizer unavailable offline: {exc}")
    assert tok.mask_token_id == tok.vocab_size - 1
    assert tok.vocab_size > 50000
    assert tok.decode(tok.encode("hello world")) == "hello world"
    del tiktoken
__AETHER_EOF__

cat > tests/data/test_prepare.py << '__AETHER_EOF__'
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
__AETHER_EOF__

cat > tests/data/test_datamodule.py << '__AETHER_EOF__'
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
__AETHER_EOF__

echo ">> patching existing files (idempotent)"
python - << 'PYPATCH'
import pathlib

p = pathlib.Path("src/aether/config/schemas.py"); s = p.read_text()
if "class DataConfig" not in s:
    s = s.replace(
        '@dataclass\nclass AetherConfig:\n    """Top-level run configuration."""\n\n'
        '    model: ModelConfig = field(default_factory=ModelConfig)\n'
        '    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)\n'
        '    seed: int = 42',
        '@dataclass\nclass DataConfig:\n'
        '    """Dataset preparation and loading configuration.\n\n'
        '    ``source`` uses a scheme prefix: ``"hf:<name>:<config>"`` for a Hugging Face\n'
        '    dataset (real runs) or ``"local:<path>"`` for a plain-text file (offline\n'
        '    debug/tests). ``tokenizer`` selects ``"gpt2"`` (BPE, needs the ``data`` extra)\n'
        '    or ``"byte"`` (offline, dependency-free).\n'
        '    """\n\n'
        '    source: str = "hf:wikitext:wikitext-103-raw-v1"\n'
        '    split: str = "train"\n'
        '    tokenizer: str = "gpt2"\n'
        '    block_size: int = 1024\n'
        '    val_blocks: int = 256  # absolute, not a fraction, so it works while streaming\n'
        '    blocks_per_shard: int = 4096\n'
        '    max_documents: int | None = None  # cap document count for quick runs\n'
        '    seed: int = 42\n'
        '    output_dir: str = "data/wikitext103"\n\n\n'
        '@dataclass\nclass AetherConfig:\n    """Top-level run configuration."""\n\n'
        '    model: ModelConfig = field(default_factory=ModelConfig)\n'
        '    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)\n'
        '    data: DataConfig = field(default_factory=DataConfig)\n'
        '    seed: int = 42',
    )
    p.write_text(s); print("patched schemas.py")
else: print("schemas.py already has DataConfig")

p = pathlib.Path("src/aether/config/__init__.py"); s = p.read_text()
if "DataConfig" not in s:
    s = s.replace("    AetherConfig,\n    DiffusionConfig,",
                  "    AetherConfig,\n    DataConfig,\n    DiffusionConfig,")
    s = s.replace('    "AetherConfig",\n    "DiffusionConfig",',
                  '    "AetherConfig",\n    "DataConfig",\n    "DiffusionConfig",')
    p.write_text(s); print("patched config/__init__.py")
else: print("config/__init__.py already exports DataConfig")

p = pathlib.Path("configs/config.yaml"); s = p.read_text()
if "data: wikitext103" not in s:
    s = s.replace("  - diffusion: absorbing\n", "  - diffusion: absorbing\n  - data: wikitext103\n")
    p.write_text(s); print("patched configs/config.yaml")
else: print("configs/config.yaml already has data default")

p = pathlib.Path("pyproject.toml"); s = p.read_text()
if 'data = ["tiktoken' not in s:
    s = s.replace('viz = ["matplotlib>=3.8"]\n',
                  'viz = ["matplotlib>=3.8"]\ndata = ["tiktoken>=0.7", "datasets>=2.19"]\n')
if '"tiktoken.*"' not in s:
    s = s.replace('module = ["hydra.*", "matplotlib.*"]',
                  'module = ["hydra.*", "matplotlib.*", "tiktoken.*", "datasets.*"]')
if "aether-prepare" not in s:
    s = s.replace('aether-forward = "aether.diffusion.forward:main"\n',
                  'aether-forward = "aether.diffusion.forward:main"\naether-prepare = "aether.data.prepare:main"\n')
if "UP040" not in s:
    s = s.replace('select = ["E", "F", "I", "UP", "B", "SIM", "N", "C4", "PT", "RUF"]',
                  'select = ["E", "F", "I", "UP", "B", "SIM", "N", "C4", "PT", "RUF"]\n'
                  '# UP040 wants PEP 695 `type` aliases, which the pinned pre-commit mypy 1.11.2\n'
                  '# cannot parse; we use typing.TypeAlias for broad compatibility.\n'
                  'ignore = ["UP040"]')
p.write_text(s); print("patched pyproject.toml")

p = pathlib.Path("Makefile"); s = p.read_text()
if "data-debug:" not in s:
    s = s.replace(".PHONY: help install lint format type test all demo plot config",
                  ".PHONY: help install lint format type test all demo plot config data data-debug")
    s = s.replace("config:\n> python scripts/show_config.py\n",
                  "config:\n> python scripts/show_config.py\n\ndata:\n> aether-prepare data=wikitext103\n\n"
                  "data-debug:\n> aether-prepare data=local_debug\n")
    p.write_text(s); print("patched Makefile")
else: print("Makefile already has data targets")

p = pathlib.Path("README.md"); s = p.read_text()
if "make data-debug" not in s:
    s = s.replace("make all                # lint + type-check + test\n```",
        "make all                # lint + type-check + test\n"
        "make data-debug         # build a tiny offline dataset (shards + manifest)\n```\n\n"
        '> For real WikiText-103: `pip install -e ".[data]"` then `make data`.', 1)
    s = s.replace("  data/        # tokenizer & dataset pipeline (Week 2)",
                  "  data/        # tokenizer, packing, sharding, datamodule")
    p.write_text(s); print("patched README.md")
else: print("README.md already has data step")

p = pathlib.Path(".gitignore"); s = p.read_text()
if "\ndata/\n" not in ("\n"+s+"\n"):
    p.write_text(s.rstrip("\n") + "\ndata/\n"); print("patched .gitignore")
else: print(".gitignore already ignores data/")
PYPATCH
echo ">> done. Now run:  make all"