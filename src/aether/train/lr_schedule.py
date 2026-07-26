"""Linear warmup followed by cosine decay to a floor.

Warmup stabilizes the noisy early steps; cosine decay to ``min_lr_ratio * lr``
gives a smooth, well-understood annealing. Implemented as a ``LambdaLR`` so its
state is captured by the scheduler's ``state_dict`` and thus survives a resume.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_lambda(
    warmup_steps: int, max_steps: int, min_ratio: float
) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def build_scheduler(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, min_ratio: float
) -> LambdaLR:
    return LambdaLR(optimizer, cosine_warmup_lambda(warmup_steps, max_steps, min_ratio))
