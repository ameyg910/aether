"""SUBS masked-diffusion loss (MDLM).

The continuous-time NELBO for absorbing-state diffusion reduces to a *weighted sum
of masked-language-modeling cross-entropy losses*. Two SUBS properties make this
exact: carry-over unmasking means unmasked positions contribute zero to the loss,
so we only supervise masked positions; zero-masking means the model never predicts
the ``[MASK]`` token, enforced here by biasing its logit to a large negative value.

Per masked token the objective is a cross-entropy weighted by
``w(t) = -alpha'(t) / (1 - alpha(t))``, which diverges as ``t -> 0``; we sample
``t`` from ``[t_min, 1]`` to keep the weight finite (the numerically delicate part).
The torch schedule math here mirrors ``aether.diffusion.schedule`` (numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

_MASK_LOGIT_BIAS = -1e9


def alpha_of_t(schedule: str, t: Tensor) -> Tensor:
    """Survival probability ``alpha(t)`` (1 at t=0, 0 at t=1)."""
    if schedule == "linear":
        return 1.0 - t
    if schedule == "cosine":
        return torch.cos(0.5 * math.pi * t) ** 2
    raise ValueError(f"Unknown schedule {schedule!r}")


def d_alpha_dt(schedule: str, t: Tensor) -> Tensor:
    """Time derivative of the survival probability."""
    if schedule == "linear":
        return -torch.ones_like(t)
    if schedule == "cosine":
        return -(0.5 * math.pi) * torch.sin(math.pi * t)
    raise ValueError(f"Unknown schedule {schedule!r}")


def loss_weight(schedule: str, t: Tensor) -> Tensor:
    """MDLM per-token weight ``w(t) = -alpha'(t) / (1 - alpha(t))`` (>= 0)."""
    return -d_alpha_dt(schedule, t) / (1.0 - alpha_of_t(schedule, t))


def absorbing_corrupt(
    x0: Tensor,
    t: Tensor,
    mask_token_id: int,
    schedule: str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample ``x_t ~ q(x_t | x_0)``: mask each token with prob ``1 - alpha(t)``.

    Returns the corrupted ids and a boolean mask of which positions were masked.
    """
    rate = 1.0 - alpha_of_t(schedule, t)
    noise = torch.rand(x0.shape, generator=generator, device=x0.device)
    masked = noise < rate[:, None]
    x_t = torch.where(masked, torch.full_like(x0, mask_token_id), x0)
    return x_t, masked


def diffusion_loss_from_logits(
    logits: Tensor,
    x0: Tensor,
    masked: Tensor,
    t: Tensor,
    mask_token_id: int,
    schedule: str,
) -> Tensor:
    """Weighted masked-CE from precomputed logits (the differentiable core).

    Kept separate from corruption and the forward pass so gradients can be
    checked directly against this function.

    The per-sequence cross-entropy is **averaged over masked positions** rather
    than summed. Summing makes the loss (and its gradient) scale with the number
    of masked tokens, which varies with ``t`` and the schedule. That means the
    effective learning rate is different at every noise level, ``grad_norm``
    scales with sequence length, and a ``grad_clip`` tuned for one setting breaks
    at another. Averaging instead gives a loss in the familiar ~7-nats range,
    makes ``grad_clip=1.0`` meaningful, and makes results comparable to published
    MDLM numbers.
    """
    vocab = logits.shape[-1]
    bias = torch.zeros(vocab, dtype=logits.dtype, device=logits.device)
    bias[mask_token_id] = _MASK_LOGIT_BIAS  # SUBS zero-masking
    logp = torch.log_softmax(logits + bias, dim=-1)
    tok_logp = logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    # Sum CE over masked positions, then divide by the count. clamp(min=1)
    # guards the rare case where t is so small that no tokens were masked
    # (which cannot happen in practice given t_min=1e-3, but keeps autograd safe).
    n_masked = masked.sum(dim=1).clamp(min=1).to(logits.dtype)
    ce = -(tok_logp * masked.to(logits.dtype)).sum(dim=1) / n_masked
    weight = loss_weight(schedule, t)
    return (weight * ce).mean()


@dataclass
class LossOutput:
    """Loss plus the intermediates a trainer/logger may want."""

    loss: Tensor
    x_t: Tensor
    masked: Tensor
    t: Tensor


class MaskedDiffusionLoss(nn.Module):
    """Samples time, corrupts, runs the model, and returns the SUBS loss."""

    def __init__(self, mask_token_id: int, schedule: str = "linear", t_min: float = 1e-3) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id
        self.schedule = schedule
        self.t_min = t_min

    def sample_t(
        self, batch: int, device: torch.device, generator: torch.Generator | None
    ) -> Tensor:
        u = torch.rand(batch, generator=generator, device=device)
        return self.t_min + (1.0 - self.t_min) * u

    def forward(
        self,
        model: nn.Module,
        x0: Tensor,
        t: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LossOutput:
        if t is None:
            t = self.sample_t(x0.shape[0], x0.device, generator)
        x_t, masked = absorbing_corrupt(x0, t, self.mask_token_id, self.schedule, generator)
        logits = model(x_t, t)
        loss = diffusion_loss_from_logits(logits, x0, masked, t, self.mask_token_id, self.schedule)
        return LossOutput(loss=loss, x_t=x_t, masked=masked, t=t)
