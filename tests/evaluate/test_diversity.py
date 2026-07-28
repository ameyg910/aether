"""Diversity metrics."""

from __future__ import annotations

import math

import pytest
import torch

from aether.evaluate.diversity import (
    distinct_n,
    evaluate_diversity,
    repetition_rate,
    token_entropy,
)


def test_constant_sequence_is_maximally_degenerate() -> None:
    seqs = torch.zeros(3, 10, dtype=torch.long)
    assert distinct_n(seqs, 1) == pytest.approx(1 / 30)
    assert token_entropy(seqs) == pytest.approx(0.0)
    assert repetition_rate(seqs) == pytest.approx(1.0)


def test_all_unique_tokens_maximize_distinctness() -> None:
    seqs = torch.arange(12).reshape(1, 12)
    assert distinct_n(seqs, 1) == pytest.approx(1.0)
    assert repetition_rate(seqs) == pytest.approx(0.0)
    # Uniform over 12 symbols => entropy ln(12).
    assert token_entropy(seqs) == pytest.approx(math.log(12), rel=1e-9)


def test_distinct_n_counts_ngrams_not_tokens() -> None:
    # "0 1 0 1" has bigrams (0,1), (1,0), (0,1) -> 2 unique of 3.
    seqs = [[0, 1, 0, 1]]
    assert distinct_n(seqs, 2) == pytest.approx(2 / 3)


def test_evaluate_diversity_reports_every_metric() -> None:
    seqs = torch.randint(0, 20, (4, 16))
    result = evaluate_diversity(seqs).as_dict()
    assert set(result) == {
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "entropy",
        "repetition_rate",
    }
    assert all(v >= 0.0 for v in result.values())


def test_accepts_lists_and_tensors_alike() -> None:
    as_list = [[1, 2, 3], [4, 5, 6]]
    as_tensor = torch.tensor(as_list)
    assert distinct_n(as_list, 1) == distinct_n(as_tensor, 1)


def test_empty_input_is_zero_not_a_crash() -> None:
    assert distinct_n([], 1) == 0.0
    assert token_entropy([]) == 0.0
    assert repetition_rate([]) == 0.0
