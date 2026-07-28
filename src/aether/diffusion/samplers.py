"""Samplers for absorbing-state diffusion.

Generation runs the forward (masking) process backwards: start from an all-
``[MASK]`` sequence and unmask progressively until nothing is masked. The number
of model forward passes -- **NFE**, number of function evaluations -- is the knob
that trades compute for quality, and it is the reason diffusion LMs are
interesting: unlike an autoregressive model, which needs one forward pass per
token, a diffusion model can decide *how many* passes to spend on a sequence of
any length.

Two strategies are provided:

``ancestral``
    The faithful reverse process. At each step every masked position is unmasked
    independently with probability set by the noise schedule, and its value is
    drawn from the model's predicted distribution. Unbiased with respect to the
    learned distribution, but it commits to tokens in a random order -- including
    positions the model is deeply unsure about -- so it needs many steps to
    produce coherent text.

``confidence``
    Confidence-based parallel decoding (LLaDA / Fast-dLLM style). The schedule
    still decides *how many* tokens to unmask at each step, but instead of
    choosing them at random the sampler commits to the positions where the model
    is most confident, and leaves the uncertain ones masked for a later step when
    more context is available. Deciding easy tokens first and letting them inform
    the hard ones is what buys quality at low NFE.

Both share the SUBS constraint that the model may never emit ``[MASK]``, enforced
by biasing that logit before the softmax.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from aether.diffusion.loss import _MASK_LOGIT_BIAS, alpha_of_t

SAMPLERS = ("ancestral", "confidence")


@dataclass
class SamplerOutput:
    """Generated tokens plus the compute actually spent producing them."""

    tokens: Tensor
    nfe: int  # model forward passes


def _masked_probs(
    model: nn.Module, x: Tensor, t: Tensor, mask_token_id: int, temperature: float
) -> Tensor:
    """Model probabilities with ``[MASK]`` forbidden (SUBS zero-masking)."""
    logits = model(x, t)
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    logits[..., mask_token_id] = _MASK_LOGIT_BIAS
    return torch.softmax(logits, dim=-1)


def _sample_from(probs: Tensor, generator: torch.Generator | None) -> Tensor:
    """Draw one token per position from a ``(B, L, V)`` probability tensor."""
    batch, length, vocab = probs.shape
    flat = probs.reshape(-1, vocab)
    drawn = torch.multinomial(flat, 1, generator=generator)
    return drawn.reshape(batch, length)


class DenoiseState(NamedTuple):
    """One intermediate state of the reverse process."""

    step: int  # completed denoising steps
    total_steps: int
    tokens: Tensor  # (B, L); may still contain [MASK]
    n_masked: int  # positions still masked across the batch
    nfe: int  # forward passes spent so far


@torch.no_grad()
def _ancestral_iter(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    steps: int,
    schedule: str,
    device: torch.device,
    generator: torch.Generator | None,
    temperature: float,
) -> Iterator[DenoiseState]:
    """Reverse the absorbing process, unmasking at random per the schedule."""
    x = torch.full((batch, length), mask_token_id, device=device, dtype=torch.long)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    nfe = 0

    for i in range(steps):
        t_cur, t_next = times[i], times[i + 1]
        probs = _masked_probs(model, x, t_cur.expand(batch), mask_token_id, temperature)
        nfe += 1

        a_cur = alpha_of_t(schedule, t_cur)
        a_next = alpha_of_t(schedule, t_next)
        # Probability that a still-masked token gets revealed on this step.
        unmask_p = ((a_next - a_cur) / (1.0 - a_cur)).clamp(0.0, 1.0)

        is_masked = x == mask_token_id
        draw = torch.rand(x.shape, generator=generator, device=device)
        do_unmask = is_masked & (draw < unmask_p)
        x = torch.where(do_unmask, _sample_from(probs, generator), x)
        yield DenoiseState(i + 1, steps, x, int((x == mask_token_id).sum()), nfe)

    x, nfe = _finalize(model, x, mask_token_id, temperature, batch, device, nfe)
    yield DenoiseState(steps, steps, x, 0, nfe)


@torch.no_grad()
def _confidence_iter(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    steps: int,
    schedule: str,
    device: torch.device,
    generator: torch.Generator | None,
    temperature: float,
    greedy: bool = True,
) -> Iterator[DenoiseState]:
    """Unmask the most-confident positions first (parallel decoding)."""
    x = torch.full((batch, length), mask_token_id, device=device, dtype=torch.long)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    nfe = 0

    for i in range(steps):
        is_masked = x == mask_token_id
        n_masked = is_masked.sum(dim=1)
        if int(n_masked.max()) == 0:
            break

        probs = _masked_probs(model, x, times[i].expand(batch), mask_token_id, temperature)
        nfe += 1

        # How many should remain masked after this step, per the schedule.
        a_next = alpha_of_t(schedule, times[i + 1])
        target_masked = torch.round((1.0 - a_next) * length).long().expand(batch)
        # Always make progress: at least one token per step, never more than remain.
        n_reveal = torch.minimum((n_masked - target_masked).clamp(min=1), n_masked)

        confidence = probs.max(dim=-1).values.masked_fill(~is_masked, float("-inf"))
        # Rank masked positions by confidence; reveal the top ``n_reveal`` of each row.
        rank = confidence.argsort(dim=-1, descending=True).argsort(dim=-1)
        do_unmask = is_masked & (rank < n_reveal[:, None])

        values = probs.argmax(dim=-1) if greedy else _sample_from(probs, generator)
        x = torch.where(do_unmask, values, x)
        yield DenoiseState(i + 1, steps, x, int((x == mask_token_id).sum()), nfe)

    x, nfe = _finalize(model, x, mask_token_id, temperature, batch, device, nfe)
    yield DenoiseState(steps, steps, x, 0, nfe)


def _finalize(
    model: nn.Module,
    x: Tensor,
    mask_token_id: int,
    temperature: float,
    batch: int,
    device: torch.device,
    nfe: int,
) -> tuple[Tensor, int]:
    """Fill any position still masked at t=0 with the model's best guess."""
    if bool((x == mask_token_id).any()):
        probs = _masked_probs(
            model, x, torch.zeros(batch, device=device), mask_token_id, temperature
        )
        nfe += 1
        x = torch.where(x == mask_token_id, probs.argmax(dim=-1), x)
    return x, nfe


def iter_denoise(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    sampler: str = "ancestral",
    steps: int = 64,
    schedule: str = "linear",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> Iterator[DenoiseState]:
    """Yield every intermediate state of generation.

    Powers the streaming endpoint: a client can watch text emerge from noise
    instead of waiting for the final result. The non-streaming samplers consume
    this same generator, so there is exactly one implementation of each strategy.
    """
    device = device or torch.device("cpu")
    if sampler == "ancestral":
        return _ancestral_iter(
            model, batch, length, mask_token_id, steps, schedule, device, generator, temperature
        )
    if sampler == "confidence":
        return _confidence_iter(
            model, batch, length, mask_token_id, steps, schedule, device, generator, temperature
        )
    raise ValueError(f"Unknown sampler {sampler!r}; expected one of {SAMPLERS}")


def _consume(states: Iterator[DenoiseState]) -> SamplerOutput:
    final: DenoiseState | None = None
    for final in states:  # noqa: B007 - we want the last yielded state
        pass
    if final is None:  # pragma: no cover - steps >= 1 always yields
        raise RuntimeError("sampler produced no states")
    return SamplerOutput(tokens=final.tokens, nfe=final.nfe)


def ancestral_sample_full(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    steps: int = 64,
    schedule: str = "linear",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> SamplerOutput:
    """Reverse the absorbing process, unmasking at random per the schedule."""
    return _consume(
        _ancestral_iter(
            model,
            batch,
            length,
            mask_token_id,
            steps,
            schedule,
            device or torch.device("cpu"),
            generator,
            temperature,
        )
    )


def confidence_sample_full(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    steps: int = 64,
    schedule: str = "linear",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
    greedy: bool = True,
) -> SamplerOutput:
    """Unmask the most-confident positions first (parallel decoding)."""
    return _consume(
        _confidence_iter(
            model,
            batch,
            length,
            mask_token_id,
            steps,
            schedule,
            device or torch.device("cpu"),
            generator,
            temperature,
            greedy,
        )
    )


def sample(
    model: nn.Module,
    batch: int,
    length: int,
    mask_token_id: int,
    sampler: str = "ancestral",
    steps: int = 64,
    schedule: str = "linear",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> SamplerOutput:
    """Dispatch to a sampler by name."""
    if sampler == "ancestral":
        return ancestral_sample_full(
            model, batch, length, mask_token_id, steps, schedule, device, generator, temperature
        )
    if sampler == "confidence":
        return confidence_sample_full(
            model, batch, length, mask_token_id, steps, schedule, device, generator, temperature
        )
    raise ValueError(f"Unknown sampler {sampler!r}; expected one of {SAMPLERS}")


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
    """Back-compatible wrapper returning only the tokens."""
    return ancestral_sample_full(
        model, batch, length, mask_token_id, steps, schedule, device, generator
    ).tokens
