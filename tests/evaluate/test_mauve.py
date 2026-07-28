"""MAUVE divergence-frontier scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aether.evaluate.mauve import (
    divergence_frontier,
    hashed_ngram_features,
    mauve_score,
)


def test_identical_distributions_score_one() -> None:
    # The frontier collapses to a point here; the anchoring endpoints are what
    # make the area come out at 1.0 instead of 0.
    torch.manual_seed(0)
    seqs = torch.randint(0, 30, (40, 16))
    assert mauve_score(seqs, seqs) == pytest.approx(1.0, abs=1e-6)


def test_degenerate_generation_scores_near_zero() -> None:
    torch.manual_seed(0)
    reference = torch.randint(0, 30, (40, 16))
    collapsed = torch.zeros(40, 16, dtype=torch.long)
    assert mauve_score(reference, collapsed) < 0.05


def test_same_distribution_beats_different_distribution() -> None:
    torch.manual_seed(0)
    reference = torch.randint(0, 30, (60, 16))
    same = torch.randint(0, 30, (60, 16))
    narrow = torch.randint(0, 3, (60, 16))
    assert mauve_score(reference, same) > mauve_score(reference, narrow)


def test_score_is_bounded() -> None:
    torch.manual_seed(0)
    a = torch.randint(0, 30, (30, 16))
    b = torch.randint(0, 5, (30, 16))
    assert 0.0 <= mauve_score(a, b) <= 1.0


def test_frontier_includes_anchor_points() -> None:
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    xs, ys = divergence_frontier(p, q)
    assert xs[0] == pytest.approx(1.0)
    assert ys[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(0.0)
    assert ys[-1] == pytest.approx(1.0)


def test_features_are_deterministic_across_calls() -> None:
    # Python's built-in hash() is salted per process; the featurizer must not use it.
    seqs = torch.randint(0, 20, (5, 12))
    assert np.allclose(hashed_ngram_features(seqs), hashed_ngram_features(seqs))


def test_features_are_l2_normalized() -> None:
    seqs = torch.randint(0, 20, (6, 12))
    norms = np.linalg.norm(hashed_ngram_features(seqs), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
