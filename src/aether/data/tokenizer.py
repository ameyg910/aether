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
