"""Minimal ancestral sampler for absorbing-state diffusion (stub).

Starts from an all-``[MASK]`` sequence and iteratively unmasks over ``steps``
discrete time points from ``t=1`` down to ``t=0``. At each step, currently masked
positions are unmasked with probability set by the schedule and filled by sampling
the model's predicted distribution (with ``[MASK]`` forbidden). Fast, parallel,
and cache-aware samplers arrive in Week 6; this exists to make the model runnable.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aether.diffusion.loss import _MASK_LOGIT_BIAS, alpha_of_t


@torch.no_grad()
def ancestral_sample(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    steps: int = 64,
    schedule: str = "linear",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate ``(batch, length)`` token ids by iterative unmasking."""
    device = device or torch.device("cpu")
    x = torch.full((batch, length), mask_token_id, device=device)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t_cur = times[i]
        t_next = times[i + 1]
        t_batch = t_cur.expand(batch)
        logits = model(x, t_batch)
        logits[..., mask_token_id] = _MASK_LOGIT_BIAS
        probs = torch.softmax(logits, dim=-1)

        a_cur = alpha_of_t(schedule, t_cur)
        a_next = alpha_of_t(schedule, t_next)
        unmask_p = ((a_next - a_cur) / (1.0 - a_cur)).clamp(0.0, 1.0)

        is_masked = x == mask_token_id
        draw = torch.rand(x.shape, generator=generator, device=device)
        do_unmask = is_masked & (draw < unmask_p)

        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=generator).reshape(batch, length)
        x = torch.where(do_unmask, sampled, x)

    # Fill any leftover masked positions with the model's argmax.
    if bool((x == mask_token_id).any()):
        logits = model(x, torch.zeros(batch, device=device))
        logits[..., mask_token_id] = _MASK_LOGIT_BIAS
        x = torch.where(x == mask_token_id, logits.argmax(dim=-1), x)
    return x
