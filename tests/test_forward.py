"""Invariants for the absorbing forward process."""

from __future__ import annotations

import numpy as np
import pytest

from aether.diffusion.forward import AbsorbingForwardProcess
from aether.diffusion.schedule import build_schedule


def _process(kind: str = "linear", mask_id: int = 0) -> AbsorbingForwardProcess:
    return AbsorbingForwardProcess(build_schedule(kind), mask_token_id=mask_id)


def test_shape_and_dtype_preserved() -> None:
    x0 = np.arange(1, 33, dtype=np.int64)
    xt = _process().sample(x0, 0.5, np.random.default_rng(0))
    assert xt.shape == x0.shape
    assert xt.dtype == np.int64


def test_t1_fully_masked() -> None:
    x0 = np.arange(1, 65, dtype=np.int64)
    xt = _process().sample(x0, 1.0, np.random.default_rng(0))
    assert np.all(xt == 0)


def test_t0_none_masked() -> None:
    x0 = np.arange(1, 65, dtype=np.int64)
    xt = _process().sample(x0, 0.0, np.random.default_rng(0))
    assert np.array_equal(xt, x0)


@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_empirical_mask_fraction_matches_rate(t: float) -> None:
    process = _process()
    x0 = np.ones(200_000, dtype=np.int64)
    xt = process.sample(x0, t, np.random.default_rng(123))
    frac = float((xt == 0).mean())
    assert abs(frac - process.mask_rate_at(t)) < 0.01


def test_determinism_same_seed() -> None:
    process = _process()
    x0 = np.arange(1, 129, dtype=np.int64)
    a = process.sample(x0, 0.5, np.random.default_rng(7))
    b = process.sample(x0, 0.5, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_t_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"t must be in"):
        _process().sample(np.array([1], dtype=np.int64), 1.5, np.random.default_rng(0))
