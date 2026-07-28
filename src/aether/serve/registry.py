"""Versioned model loading, so a deploy is reproducible and a rollback is one call.

A served model is identified by a **version tag**, not a file path, and the tag
records where the weights came from:

``local:runs/my-run/checkpoints/latest.pt``
    A checkpoint on disk. Convenient for development; not reproducible, because
    the file behind the path can change under you.

``hf:owner/repo@revision``
    A checkpoint pulled from the Hugging Face Hub at a pinned git revision. This
    is the reproducible form -- a revision is immutable, so the same tag always
    yields the same weights, which is what makes a rollback meaningful.

Loading is atomic. ``swap()`` resolves and fully constructs the new model *before*
touching the served one, so a bad version fails without taking the service down,
and the previously-served version is retained so ``rollback()`` is instant and
cannot itself fail by re-downloading.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from aether.config.schemas import ModelConfig
from aether.data.tokenizer import Tokenizer, build_tokenizer
from aether.log import get_logger
from aether.models.aether_model import AetherModel
from aether.models.loading import build_model_from_checkpoint

logger = get_logger("registry")

DEFAULT_CHECKPOINT_FILE = "latest.pt"


@dataclass
class LoadedModel:
    """A model ready to serve, plus the provenance of the weights inside it."""

    version: str
    model: AetherModel
    config: ModelConfig
    tokenizer: Tokenizer
    mask_token_id: int
    device: torch.device
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_length(self) -> int:
        return self.config.max_seq_len


def resolve_version(version: str, cache_dir: str | None = None) -> Path:
    """Resolve a version tag to a local checkpoint file, downloading if needed."""
    if version.startswith("hf:"):
        spec = version[3:]
        repo_id, _, revision = spec.partition("@")
        repo_id, _, filename = repo_id.partition(":")
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename or DEFAULT_CHECKPOINT_FILE,
            revision=revision or None,
            cache_dir=cache_dir,
        )
        return Path(downloaded)

    raw = version[6:] if version.startswith("local:") else version
    path = Path(raw)
    if path.is_dir():
        path = path / "checkpoints" / DEFAULT_CHECKPOINT_FILE
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path} (version tag {version!r})")
    return path


class ModelRegistry:
    """Loads and serves models by version tag, with rollback."""

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self._current: LoadedModel | None = None
        self._previous: LoadedModel | None = None
        # Guards the current/previous pointers; loading happens outside the lock
        # so a slow download never blocks in-flight requests.
        self._lock = threading.Lock()

    @property
    def current(self) -> LoadedModel | None:
        with self._lock:
            return self._current

    @property
    def is_ready(self) -> bool:
        """True once a model is loaded and able to serve -- drives ``/ready``."""
        return self.current is not None

    def load(self, version: str) -> LoadedModel:
        """Construct a model from a version tag without installing it."""
        path = resolve_version(version, self.cache_dir)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model, config = build_model_from_checkpoint(ckpt)
        model = model.to(self.device).eval()

        tokenizer = build_tokenizer(self.tokenizer_name)
        loaded = LoadedModel(
            version=version,
            model=model,
            config=config,
            tokenizer=tokenizer,
            # Aether appends [MASK] as the final vocabulary entry.
            mask_token_id=config.vocab_size - 1,
            device=self.device,
            metadata={
                "path": str(path),
                "step": ckpt.get("step"),
                "params": sum(p.numel() for p in model.parameters()),
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "max_seq_len": config.max_seq_len,
            },
        )
        logger.info(
            "model_loaded",
            version=version,
            **{k: v for k, v in loaded.metadata.items() if k in ("step", "params")},
        )
        return loaded

    def swap(self, version: str) -> LoadedModel:
        """Load ``version`` and make it live, keeping the old one for rollback.

        The new model is fully constructed before anything is swapped, so a bad
        version raises and leaves the running service untouched.
        """
        loaded = self.load(version)
        with self._lock:
            if self._current is not None and self._current.version != version:
                self._previous = self._current
            self._current = loaded
        return loaded

    def rollback(self) -> LoadedModel:
        """Restore the previously-served version. Already in memory, so instant."""
        with self._lock:
            if self._previous is None:
                raise RuntimeError("no previous version to roll back to")
            self._current, self._previous = self._previous, self._current
            logger.info("rolled_back", version=self._current.version)
            return self._current

    def require(self) -> LoadedModel:
        """Return the live model or raise -- used by request handlers."""
        model = self.current
        if model is None:
            raise RuntimeError("no model loaded")
        return model
