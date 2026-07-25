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
