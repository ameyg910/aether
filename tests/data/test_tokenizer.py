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
