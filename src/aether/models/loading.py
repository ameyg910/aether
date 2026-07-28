"""Reconstruct a model from a checkpoint without needing the original config.

A checkpoint's ``state_dict`` already encodes the geometry: embedding shapes give
vocabulary and width, the positional table gives the maximum sequence length, and
the block keys give the depth. Recovering the config from those means a served
checkpoint needs nothing beside it -- no matching YAML, no risk of loading weights
into a mismatched architecture.

The one value that cannot be recovered is ``n_heads``: attention reshapes into
heads inside the forward pass, so head count leaves no trace in any parameter
shape. It is taken from the checkpoint's saved config when present, and otherwise
inferred with the convention used throughout Aether (64 channels per head).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from aether.config.schemas import ModelConfig
from aether.models.aether_model import AetherModel

_HEAD_DIM = 64


def infer_model_config(state: dict[str, Tensor], n_heads: int | None = None) -> ModelConfig:
    """Recover a :class:`ModelConfig` from checkpoint tensors."""
    try:
        vocab_size, d_model = state["tok_emb.weight"].shape
        max_seq_len = int(state["pos_emb"].shape[1])
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"checkpoint is missing expected key: {exc}") from exc

    block_indices = {
        int(key.split(".")[1]) for key in state if key.startswith("blocks.") and key.count(".") > 1
    }
    if not block_indices:
        raise ValueError("checkpoint contains no transformer blocks")

    heads = n_heads or max(1, int(d_model) // _HEAD_DIM)
    if int(d_model) % heads != 0:
        heads = max(1, int(d_model) // _HEAD_DIM)

    return ModelConfig(
        vocab_size=int(vocab_size),
        d_model=int(d_model),
        n_layers=max(block_indices) + 1,
        n_heads=heads,
        max_seq_len=max_seq_len,
    )


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[AetherModel, ModelConfig]:
    """Instantiate and load a model from a full training checkpoint."""
    state = ckpt["model"]
    extra = ckpt.get("extra") or {}
    saved = extra.get("model_config") if isinstance(extra, dict) else None
    if isinstance(saved, dict):
        # Recorded at save time -- authoritative, no inference needed.
        fields = ModelConfig.__dataclass_fields__
        cfg = ModelConfig(**{k: v for k, v in saved.items() if k in fields})
    else:
        cfg = infer_model_config(state)
    model = AetherModel(cfg)
    model.load_state_dict(state)
    return model, cfg
