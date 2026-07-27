"""Deterministic seeding and RNG state capture across the whole stack.

Reproducibility on a preemptible cluster needs two things: a single seed that
initializes every RNG (python, numpy, torch, cuda), and the ability to snapshot
and restore those RNG states so a resumed run continues the *same* stochastic
stream rather than diverging.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU ByteTensors. ``torch.load(map_location="cuda")``
    # moves every tensor in a checkpoint to the GPU, these included, so coerce
    # back to keep checkpoints portable across devices.
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
