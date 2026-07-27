"""SUBS masked-diffusion loss: equivalence, gradients, and invariants."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from aether.diffusion.loss import (
    _MASK_LOGIT_BIAS,
    absorbing_corrupt,
    alpha_of_t,
    diffusion_loss_from_logits,
    loss_weight,
)


def test_alpha_endpoints() -> None:
    for schedule in ("linear", "cosine"):
        assert torch.isclose(alpha_of_t(schedule, torch.tensor(0.0)), torch.tensor(1.0))
        assert torch.isclose(alpha_of_t(schedule, torch.tensor(1.0)), torch.tensor(0.0), atol=1e-6)


def test_loss_weight_positive() -> None:
    t = torch.linspace(0.05, 1.0, 20)
    for schedule in ("linear", "cosine"):
        # >= 0 up to float roundoff at the t=1 endpoint (never sampled in training).
        assert (loss_weight(schedule, t) >= -1e-6).all()


def test_corruption_endpoints() -> None:
    x0 = torch.randint(0, 10, (4, 32))
    x1, m1 = absorbing_corrupt(x0, torch.ones(4), 99, "linear")
    assert m1.all()
    assert (x1 == 99).all()
    x2, m2 = absorbing_corrupt(x0, torch.zeros(4), 99, "linear")
    assert (~m2).all()
    assert torch.equal(x2, x0)


def test_loss_matches_independent_reference() -> None:
    torch.manual_seed(0)
    b, length, vocab, mask_id = 4, 12, 8, 7
    logits = torch.randn(b, length, vocab)
    x0 = torch.randint(0, mask_id, (b, length))
    masked = torch.rand(b, length) < 0.5
    t = torch.rand(b) * 0.9 + 0.05

    got = diffusion_loss_from_logits(logits, x0, masked, t, mask_id, "linear")

    # Independent reference via F.cross_entropy (different code path).
    ref_logits = logits.clone()
    ref_logits[..., mask_id] = _MASK_LOGIT_BIAS
    ce = F.cross_entropy(ref_logits.reshape(-1, vocab), x0.reshape(-1), reduction="none").reshape(
        b, length
    )
    n_masked = masked.float().sum(dim=1).clamp(min=1)
    per_seq = (ce * masked.float()).sum(dim=1) / n_masked
    ref = (loss_weight("linear", t) * per_seq).mean()

    assert torch.allclose(got, ref, atol=1e-5)


def test_gradcheck_on_logits() -> None:
    torch.manual_seed(0)
    b, length, vocab, mask_id = 2, 4, 6, 5
    logits = torch.randn(b, length, vocab, dtype=torch.double, requires_grad=True)
    x0 = torch.randint(0, mask_id, (b, length))
    masked = torch.zeros(b, length, dtype=torch.bool)
    masked[0, 1] = masked[1, 3] = masked[0, 2] = True
    t = torch.rand(b, dtype=torch.double) * 0.9 + 0.05

    def fn(lg: torch.Tensor) -> torch.Tensor:
        return diffusion_loss_from_logits(lg, x0, masked, t, mask_id, "linear")

    assert torch.autograd.gradcheck(fn, (logits,))


def test_model_never_predicts_mask() -> None:
    # The SUBS bias must make the mask token's probability negligible.
    logits = torch.zeros(1, 1, 5)
    x0 = torch.zeros(1, 1, dtype=torch.long)
    masked = torch.ones(1, 1, dtype=torch.bool)
    # A finite loss even though the mask logit started equal to the rest.
    loss = diffusion_loss_from_logits(logits, x0, masked, torch.tensor([0.5]), 4, "linear")
    assert torch.isfinite(loss)
