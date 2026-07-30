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


class TestHubVersionTags:
    """The ``hf:`` resolution path.

    Every other registry test uses ``local:`` paths, so until now this branch had
    never executed -- it was the one code path in the registry with no coverage,
    and the one that matters most for reproducible deploys. ``hf_hub_download`` is
    patched out: the parsing and the arguments passed to the Hub are what this
    module is responsible for, not the download itself.
    """

    @staticmethod
    def _capture(monkeypatch, checkpoint: Path) -> dict:  # type: ignore[type-arg]
        seen: dict = {}  # type: ignore[type-arg]

        def fake_download(**kwargs: object) -> str:
            seen.update(kwargs)
            return str(checkpoint)

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        return seen

    def test_repo_and_revision_are_parsed(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        seen = self._capture(monkeypatch, checkpoint)
        resolve_version("hf:ameyg910/aether-55m@v1.0.0")
        assert seen["repo_id"] == "ameyg910/aether-55m"
        assert seen["revision"] == "v1.0.0"
        # Default filename when the tag does not name one.
        assert seen["filename"] == "latest.pt"

    def test_revision_is_optional(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        seen = self._capture(monkeypatch, checkpoint)
        resolve_version("hf:ameyg910/aether-55m")
        assert seen["repo_id"] == "ameyg910/aether-55m"
        # None, not "": the Hub treats an empty revision as invalid.
        assert seen["revision"] is None

    def test_explicit_filename_is_parsed(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        seen = self._capture(monkeypatch, checkpoint)
        resolve_version("hf:ameyg910/aether-55m:aether-55m-30k.pt@v1.0.0")
        assert seen["repo_id"] == "ameyg910/aether-55m"
        assert seen["filename"] == "aether-55m-30k.pt"
        assert seen["revision"] == "v1.0.0"

    def test_cache_dir_is_forwarded(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        seen = self._capture(monkeypatch, checkpoint)
        resolve_version("hf:owner/repo@v1", cache_dir="/tmp/hfcache")
        assert seen["cache_dir"] == "/tmp/hfcache"

    def test_registry_loads_a_model_from_a_hub_tag(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        # End to end through the registry, so the version tag is what is recorded
        # and reported by /model -- a deploy must be identifiable by tag.
        self._capture(monkeypatch, checkpoint)
        registry = ModelRegistry(tokenizer_name="byte", device="cpu")
        loaded = registry.swap("hf:ameyg910/aether-55m@v1.0.0")
        assert loaded.version == "hf:ameyg910/aether-55m@v1.0.0"
        assert registry.is_ready

    def test_rollback_between_hub_revisions(self, monkeypatch, checkpoint: Path) -> None:  # type: ignore[no-untyped-def]
        # The reason immutable revisions matter: rolling back to a pinned tag
        # gets you the same weights it did the first time.
        self._capture(monkeypatch, checkpoint)
        registry = ModelRegistry(tokenizer_name="byte", device="cpu")
        registry.swap("hf:owner/repo@v1.0.0")
        registry.swap("hf:owner/repo@v1.1.0")
        assert registry.require().version == "hf:owner/repo@v1.1.0"
        assert registry.rollback().version == "hf:owner/repo@v1.0.0"
