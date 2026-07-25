"""Noise schedules for absorbing-state discrete diffusion.

A schedule maps diffusion time ``t in [0, 1]`` to the *survival probability*
``alpha(t)`` = P(a token is NOT masked at time ``t``). It decreases from 1 at
``t=0`` (clean) to 0 at ``t=1`` (fully masked). The mask rate ``1 - alpha(t)``
is therefore monotonically increasing on ``[0, 1]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class NoiseSchedule(ABC):
    """Base class for monotone absorbing-diffusion noise schedules."""

    @abstractmethod
    def alpha(self, t: FloatArray) -> FloatArray:
        """Survival probability at time ``t`` (decreasing from 1 to 0)."""

    def mask_rate(self, t: FloatArray) -> FloatArray:
        """Masking probability at time ``t`` (increasing from 0 to 1)."""
        return np.asarray(1.0 - self.alpha(t), dtype=np.float64)


class LinearSchedule(NoiseSchedule):
    """``alpha(t) = 1 - t``. Mask rate rises linearly with time."""

    def alpha(self, t: FloatArray) -> FloatArray:
        return np.asarray(1.0 - t, dtype=np.float64)


class CosineSchedule(NoiseSchedule):
    """``alpha(t) = cos(pi/2 * t)^2``. Slower masking early, faster late."""

    def alpha(self, t: FloatArray) -> FloatArray:
        return np.asarray(np.cos(0.5 * np.pi * t) ** 2, dtype=np.float64)


_SCHEDULES: dict[str, type[NoiseSchedule]] = {
    "linear": LinearSchedule,
    "cosine": CosineSchedule,
}


def build_schedule(kind: str) -> NoiseSchedule:
    """Construct a schedule by name.

    Raises:
        ValueError: if ``kind`` is not a registered schedule.
    """
    try:
        return _SCHEDULES[kind]()
    except KeyError as exc:
        raise ValueError(f"Unknown schedule {kind!r}. Options: {sorted(_SCHEDULES)}") from exc
