"""Sampler invariants: no mask leakage, valid ids, NFE accounting."""

from __future__ import annotations

import pytest
import torch

from aether.config.schemas import ModelConfig
from aether.diffusion.samplers import SAMPLERS, sample
from aether.models.aether_model import AetherModel

_VOCAB, _MASK, _LEN = 40, 39, 16


def _model() -> AetherModel:
    torch.manual_seed(0)
    return AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    ).eval()


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_output_shape_and_valid_ids(sampler: str) -> None:
    out = sample(_model(), batch=3, length=_LEN, mask_token_id=_MASK, sampler=sampler, steps=8)
    assert out.tokens.shape == (3, _LEN)
    assert out.tokens.dtype == torch.long
    assert int(out.tokens.min()) >= 0
    assert int(out.tokens.max()) < _VOCAB


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_never_emits_the_mask_token(sampler: str) -> None:
    # SUBS zero-masking: [MASK] is an input-only symbol and must never be generated.
    out = sample(_model(), batch=4, length=_LEN, mask_token_id=_MASK, sampler=sampler, steps=6)
    assert not bool((out.tokens == _MASK).any())


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_nfe_is_bounded_by_steps(sampler: str) -> None:
    # At most one forward per step, plus at most one cleanup pass.
    steps = 10
    out = sample(_model(), batch=2, length=_LEN, mask_token_id=_MASK, sampler=sampler, steps=steps)
    assert 0 < out.nfe <= steps + 1


def test_more_steps_costs_more_nfe() -> None:
    model = _model()
    few = sample(model, 2, _LEN, _MASK, sampler="ancestral", steps=4)
    many = sample(model, 2, _LEN, _MASK, sampler="ancestral", steps=32)
    assert many.nfe > few.nfe


def test_confidence_sampler_terminates_early_when_fully_unmasked() -> None:
    # Given far more steps than positions, it should stop once nothing is masked
    # rather than burning the full budget.
    out = sample(
        _model(), batch=2, length=_LEN, mask_token_id=_MASK, sampler="confidence", steps=200
    )
    assert out.nfe <= _LEN + 1


def test_unknown_sampler_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sampler"):
        sample(_model(), 1, _LEN, _MASK, sampler="beam", steps=2)


def test_ancestral_is_stochastic_across_seeds() -> None:
    model = _model()
    a = sample(
        model,
        4,
        _LEN,
        _MASK,
        sampler="ancestral",
        steps=8,
        generator=torch.Generator().manual_seed(1),
    ).tokens
    b = sample(
        model,
        4,
        _LEN,
        _MASK,
        sampler="ancestral",
        steps=8,
        generator=torch.Generator().manual_seed(2),
    ).tokens
    assert not torch.equal(a, b)


def test_backcompat_wrapper_returns_tensor() -> None:
    from aether.diffusion.sampler import ancestral_sample

    tokens = ancestral_sample(_model(), 2, _LEN, _MASK, steps=4)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (2, _LEN)
