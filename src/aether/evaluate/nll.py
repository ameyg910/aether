"""Likelihood evaluation: NELBO, nats/token, bits-per-dim, and perplexity.

Perplexity is subtler for a diffusion LM than for an autoregressive one. An AR
model factorizes ``log p(x) = sum_i log p(x_i | x_<i)`` exactly, so its perplexity
is an exact likelihood. A masked diffusion model has no such factorization; what
it optimizes is a *variational bound* on the negative log-likelihood:

    NELBO = E_{t ~ U(t_min, 1)} [ w(t) * sum_{i in masked(t)} -log p(x_i | x_t) ]

with ``w(t) = -alpha'(t) / (1 - alpha(t))``. Two consequences worth stating
plainly, because both are easy to get wrong:

1. **It is an upper bound on NLL, not the NLL.** Reported perplexity is therefore
   an *upper bound* on true perplexity, and is only comparable against other
   models evaluated under the same bound. Comparing a diffusion NELBO-perplexity
   directly against an AR model's exact perplexity flatters the AR model.

2. **It is a Monte Carlo estimate**, since the expectation over ``t`` has no
   closed form. Variance falls with more samples, so a small ``mc_samples``
   produces a noisy number that will not reproduce. This module uses *stratified*
   sampling of ``t`` -- one draw from each of ``n`` equal sub-intervals rather
   than ``n`` uniform draws -- which covers the time axis evenly and cuts
   variance substantially for the same compute.

Note the loss used for *training* averages cross-entropy over masked positions to
keep gradient magnitudes stable; the bound here deliberately **sums** instead,
because that is what the NELBO actually is. The two differ by a factor of the
masked-token count and must not be conflated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from aether.diffusion.loss import (
    _MASK_LOGIT_BIAS,
    absorbing_corrupt,
    loss_weight,
)

_LN2 = math.log(2.0)


@dataclass
class NLLResult:
    """Likelihood metrics for a dataset split."""

    nats_per_token: float
    bits_per_dim: float
    perplexity: float
    n_tokens: int
    n_sequences: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nats_per_token": self.nats_per_token,
            "bits_per_dim": self.bits_per_dim,
            "perplexity": self.perplexity,
            "n_tokens": self.n_tokens,
            "n_sequences": self.n_sequences,
        }


def stratified_times(
    n: int, batch: int, t_min: float, device: torch.device, generator: torch.Generator | None
) -> Iterator[Tensor]:
    """Yield ``n`` time tensors, one drawn from each equal sub-interval of [t_min, 1).

    Stratification is a free variance reduction: plain uniform sampling can leave
    whole regions of the time axis unvisited on a small budget, and the loss
    weight varies sharply near ``t=0``.
    """
    span = 1.0 - t_min
    for i in range(n):
        u = torch.rand(batch, generator=generator, device=device)
        yield t_min + span * (i + u) / n


@torch.no_grad()
def sequence_nelbo(
    model: nn.Module,
    x0: Tensor,
    mask_token_id: int,
    schedule: str = "linear",
    mc_samples: int = 8,
    t_min: float = 1e-3,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Monte Carlo NELBO in nats for each sequence in ``x0``; shape ``(B,)``."""
    batch = x0.shape[0]
    total = torch.zeros(batch, dtype=torch.float64, device=x0.device)

    for t in stratified_times(mc_samples, batch, t_min, x0.device, generator):
        x_t, masked = absorbing_corrupt(x0, t, mask_token_id, schedule, generator)
        logits = model(x_t, t)
        logits[..., mask_token_id] = _MASK_LOGIT_BIAS  # SUBS: never predict [MASK]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        # Sum over masked positions -- this is the bound, not the training mean.
        ce = -(tok_logp * masked.float()).sum(dim=1)
        total += (loss_weight(schedule, t) * ce).double()

    return total / mc_samples


@torch.no_grad()
def evaluate_nll(
    model: nn.Module,
    batches: Iterable[Tensor],
    mask_token_id: int,
    schedule: str = "linear",
    mc_samples: int = 8,
    t_min: float = 1e-3,
    device: torch.device | None = None,
    max_batches: int | None = None,
    generator: torch.Generator | None = None,
) -> NLLResult:
    """Estimate the NELBO over a dataset split.

    ``mc_samples`` trades estimate variance for compute: each sample costs one
    forward pass over the batch.
    """
    device = device or torch.device("cpu")
    was_training = model.training
    model.eval()

    nats = 0.0
    n_tokens = 0
    n_sequences = 0
    try:
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            x0 = batch.to(device)
            per_seq = sequence_nelbo(
                model, x0, mask_token_id, schedule, mc_samples, t_min, generator
            )
            nats += float(per_seq.sum())
            n_tokens += int(x0.numel())
            n_sequences += int(x0.shape[0])
    finally:
        if was_training:
            model.train()

    if n_tokens == 0:
        raise ValueError("no batches supplied to evaluate_nll")

    nats_per_token = nats / n_tokens
    return NLLResult(
        nats_per_token=nats_per_token,
        bits_per_dim=nats_per_token / _LN2,
        # Clamped before exp so a badly undertrained model reports inf rather than
        # overflowing to a nan that silently poisons downstream aggregation.
        perplexity=float(math.exp(min(nats_per_token, 700.0))),
        n_tokens=n_tokens,
        n_sequences=n_sequences,
    )
