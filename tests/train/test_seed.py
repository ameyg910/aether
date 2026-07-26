"""Deterministic seeding and RNG state round-trip."""

from __future__ import annotations

import random

import numpy as np
import torch

from aether.seed import get_rng_state, seed_everything, set_rng_state


def test_seed_is_reproducible() -> None:
    seed_everything(123)
    a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert a == b


def test_rng_state_round_trip() -> None:
    seed_everything(0)
    state = get_rng_state()
    first = torch.rand(3)
    set_rng_state(state)
    again = torch.rand(3)
    assert torch.equal(first, again)
