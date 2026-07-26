"""Warmup + cosine LR schedule values."""

from __future__ import annotations

import math

from aether.train.lr_schedule import cosine_warmup_lambda


def test_warmup_ramp_then_cosine() -> None:
    fn = cosine_warmup_lambda(warmup_steps=10, max_steps=110, min_ratio=0.1)
    assert fn(0) == 0.0
    assert math.isclose(fn(5), 0.5)
    assert math.isclose(fn(10), 1.0)  # peak at end of warmup
    # midpoint of cosine (step 60 = halfway through the 100 decay steps)
    assert math.isclose(fn(60), 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 0.5)), rel_tol=1e-6)
    assert math.isclose(fn(110), 0.1, rel_tol=1e-6)  # floor
    assert math.isclose(fn(500), 0.1, rel_tol=1e-6)  # clamped past max_steps
