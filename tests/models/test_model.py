"""Aether model: shape, param count, and finiteness."""

from __future__ import annotations

import torch

from aether.config.schemas import ModelConfig
from aether.models.aether_model import AetherModel


def _model() -> AetherModel:
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    return AetherModel(cfg)


def test_forward_shape() -> None:
    model = _model()
    x_t = torch.randint(0, 32, (3, 16))
    logits = model(x_t, torch.rand(3))
    assert logits.shape == (3, 16, 32)
    assert torch.isfinite(logits).all()


def test_param_count_positive() -> None:
    model = _model()
    assert model.num_params > 0
    assert model.flops_per_token == 2 * model.num_params


def test_handles_variable_length() -> None:
    model = _model()
    for length in (1, 8, 32):
        out = model(torch.randint(0, 32, (2, length)), torch.rand(2))
        assert out.shape == (2, length, 32)


def test_adaln_zero_init_is_identity_modulation() -> None:
    # AdaLN-Zero: the last layer of every block's modulation MLP starts at zero.
    model = _model()
    for block in model.blocks:
        assert torch.count_nonzero(block.adaln[-1].weight) == 0
        assert torch.count_nonzero(block.adaln[-1].bias) == 0
