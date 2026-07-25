"""Invariants for noise schedules."""

from __future__ import annotations

import numpy as np
import pytest

from aether.diffusion.schedule import build_schedule

GRID = np.linspace(0.0, 1.0, 51, dtype=np.float64)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_endpoints(kind: str) -> None:
    schedule = build_schedule(kind)
    assert schedule.alpha(np.array([0.0]))[0] == pytest.approx(1.0)
    assert schedule.alpha(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_mask_rate_monotone_and_bounded(kind: str) -> None:
    mask_rate = build_schedule(kind).mask_rate(GRID)
    assert np.all(mask_rate >= -1e-9)
    assert np.all(mask_rate <= 1.0 + 1e-9)
    assert np.all(np.diff(mask_rate) >= -1e-9)  # non-decreasing


def test_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError, match="Unknown schedule"):
        build_schedule("nope")
