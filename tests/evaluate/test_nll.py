"""NELBO / bits-per-dim estimation."""

from __future__ import annotations

import math

import pytest
import torch

from aether.config.schemas import ModelConfig
from aether.evaluate.nll import evaluate_nll, sequence_nelbo, stratified_times
from aether.models.aether_model import AetherModel

_VOCAB, _MASK, _LEN = 40, 39, 16


def _model() -> AetherModel:
    torch.manual_seed(0)
    return AetherModel(
        ModelConfig(vocab_size=_VOCAB, d_model=48, n_layers=2, n_heads=4, max_seq_len=_LEN)
    ).eval()


def test_stratified_times_cover_the_interval() -> None:
    n, t_min = 8, 1e-3
    times = list(stratified_times(n, 4, t_min, torch.device("cpu"), None))
    assert len(times) == n
    # One draw per equal sub-interval, so the i-th must land inside stratum i.
    for i, t in enumerate(times):
        lo = t_min + (1 - t_min) * i / n
        hi = t_min + (1 - t_min) * (i + 1) / n
        assert bool(((t >= lo) & (t <= hi)).all())


def test_untrained_model_scores_near_uniform_entropy() -> None:
    # A randomly-initialised model is ~uniform over the non-mask vocabulary, so
    # nats/token should sit close to ln(vocab - 1). This is the load-bearing
    # sanity check that the bound is scaled correctly.
    batches = [torch.randint(0, _MASK, (4, _LEN)) for _ in range(3)]
    result = evaluate_nll(_model(), batches, _MASK, mc_samples=8)
    assert result.nats_per_token == pytest.approx(math.log(_VOCAB - 1), abs=0.5)


def test_metric_relationships_hold() -> None:
    batches = [torch.randint(0, _MASK, (4, _LEN)) for _ in range(2)]
    r = evaluate_nll(_model(), batches, _MASK, mc_samples=4)
    assert r.bits_per_dim == pytest.approx(r.nats_per_token / math.log(2), rel=1e-9)
    assert r.perplexity == pytest.approx(math.exp(r.nats_per_token), rel=1e-6)
    assert r.n_tokens == 2 * 4 * _LEN
    assert r.n_sequences == 8


def test_nelbo_is_per_sequence_and_positive() -> None:
    x0 = torch.randint(0, _MASK, (5, _LEN))
    out = sequence_nelbo(_model(), x0, _MASK, mc_samples=4)
    assert out.shape == (5,)
    assert bool((out > 0).all())


def test_more_mc_samples_reduces_variance() -> None:
    # The estimator is Monte Carlo; averaging more draws must tighten the spread.
    batches = [torch.randint(0, _MASK, (4, _LEN))]
    model = _model()

    def spread(mc: int) -> float:
        vals = [
            evaluate_nll(
                model,
                batches,
                _MASK,
                mc_samples=mc,
                generator=torch.Generator().manual_seed(s),
            ).nats_per_token
            for s in range(5)
        ]
        return max(vals) - min(vals)

    assert spread(16) < spread(1)


def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError, match="no batches"):
        evaluate_nll(_model(), [], _MASK)


def test_model_left_in_original_mode() -> None:
    model = _model()
    model.train()
    evaluate_nll(model, [torch.randint(0, _MASK, (2, _LEN))], _MASK, mc_samples=1)
    assert model.training is True
