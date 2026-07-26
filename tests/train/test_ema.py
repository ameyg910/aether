"""EMA update math and store/restore semantics."""

from __future__ import annotations

import torch
from torch import nn

from aether.train.ema import EMA


def _linear() -> nn.Linear:
    m = nn.Linear(4, 4, bias=False)
    nn.init.ones_(m.weight)
    return m


def test_ema_tracks_moving_average() -> None:
    m = _linear()
    ema = EMA(m, decay=0.9)
    with torch.no_grad():
        m.weight.mul_(0.0)  # set params to 0
    ema.update(m)
    # shadow = 0.9*1 + 0.1*0 = 0.9
    assert torch.allclose(ema.shadow["weight"], torch.full((4, 4), 0.9))


def test_ema_store_and_restore() -> None:
    m = _linear()
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.mul_(2.0)  # live weights = 2
    ema.store(m)
    ema.copy_to(m)  # overwrite with shadow (=1)
    assert torch.allclose(m.weight, torch.ones(4, 4))
    ema.restore(m)  # bring back live weights (=2)
    assert torch.allclose(m.weight, torch.full((4, 4), 2.0))
