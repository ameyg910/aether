"""Ancestral sampler stub: produces a fully-unmasked sequence."""

from __future__ import annotations

import torch

from aether.config.schemas import ModelConfig
from aether.diffusion.sampler import ancestral_sample
from aether.models.aether_model import AetherModel


def test_sample_shape_and_no_mask_left() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = AetherModel(cfg)
    out = ancestral_sample(
        model,
        batch=2,
        length=10,
        mask_token_id=31,
        steps=16,
        generator=torch.Generator().manual_seed(0),
    )
    assert out.shape == (2, 10)
    assert not bool((out == 31).any())
