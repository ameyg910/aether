"""Versioned model loading, swapping, and rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from aether.serve.registry import ModelRegistry, resolve_version


def test_resolves_a_bare_path(checkpoint: Path) -> None:
    assert resolve_version(str(checkpoint)) == checkpoint


def test_resolves_a_local_prefixed_tag(checkpoint: Path) -> None:
    assert resolve_version(f"local:{checkpoint}") == checkpoint


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        resolve_version(str(tmp_path / "nope.pt"))


def test_registry_starts_unready() -> None:
    registry = ModelRegistry(tokenizer_name="byte", device="cpu")
    assert registry.is_ready is False
    assert registry.current is None
    with pytest.raises(RuntimeError, match="no model loaded"):
        registry.require()


def test_load_recovers_geometry_from_the_checkpoint(checkpoint: Path) -> None:
    registry = ModelRegistry(tokenizer_name="byte", device="cpu")
    loaded = registry.swap(f"local:{checkpoint}")
    assert registry.is_ready
    assert loaded.config.n_heads == 4  # recorded, not inferred from d_model
    assert loaded.config.max_seq_len == 64
    assert loaded.mask_token_id == loaded.config.vocab_size - 1
    assert loaded.metadata["step"] == 4242


def test_rollback_restores_the_previous_version(checkpoint: Path, tmp_path: Path) -> None:
    import shutil

    second = tmp_path / "v2.pt"
    shutil.copy(checkpoint, second)

    registry = ModelRegistry(tokenizer_name="byte", device="cpu")
    registry.swap(f"local:{checkpoint}")
    registry.swap(f"local:{second}")
    assert registry.require().version == f"local:{second}"

    restored = registry.rollback()
    assert restored.version == f"local:{checkpoint}"


def test_rollback_without_history_raises(checkpoint: Path) -> None:
    registry = ModelRegistry(tokenizer_name="byte", device="cpu")
    registry.swap(f"local:{checkpoint}")
    with pytest.raises(RuntimeError, match="no previous version"):
        registry.rollback()


def test_failed_swap_leaves_the_live_model_untouched(checkpoint: Path) -> None:
    # A bad version tag must not take down a running service.
    registry = ModelRegistry(tokenizer_name="byte", device="cpu")
    registry.swap(f"local:{checkpoint}")
    with pytest.raises(FileNotFoundError):
        registry.swap("local:/nonexistent/model.pt")
    assert registry.require().version == f"local:{checkpoint}"
