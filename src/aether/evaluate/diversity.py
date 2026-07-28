"""Distributional metrics for generated text: distinct-n, entropy, repetition.

Likelihood alone does not tell you whether a model produces *interesting* text. A
sampler tuned for confidence can score well on perplexity while emitting the same
few high-probability tokens forever -- the classic degenerate mode. These metrics
catch that:

- **distinct-n** -- unique n-grams as a fraction of all n-grams. Falls toward zero
  when the model loops.
- **token entropy** -- Shannon entropy of the unigram distribution, in nats. A
  model collapsed onto a handful of tokens has low entropy regardless of how
  confident it is.
- **repetition rate** -- fraction of positions that simply copy the previous
  token, the most visible failure mode in short samples.

All operate on token ids rather than decoded strings, so they are tokenizer- and
language-agnostic.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from torch import Tensor


@dataclass
class DiversityResult:
    """Diversity metrics over a batch of generated sequences."""

    distinct_1: float
    distinct_2: float
    distinct_3: float
    entropy: float
    repetition_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "distinct_1": self.distinct_1,
            "distinct_2": self.distinct_2,
            "distinct_3": self.distinct_3,
            "entropy": self.entropy,
            "repetition_rate": self.repetition_rate,
        }


def _to_lists(sequences: Tensor | Sequence[Sequence[int]]) -> list[list[int]]:
    if isinstance(sequences, Tensor):
        return [[int(v) for v in row] for row in sequences.tolist()]
    return [list(map(int, row)) for row in sequences]


def distinct_n(sequences: Tensor | Sequence[Sequence[int]], n: int) -> float:
    """Unique n-grams / total n-grams, pooled across sequences (0.0-1.0)."""
    rows = _to_lists(sequences)
    seen: set[tuple[int, ...]] = set()
    total = 0
    for row in rows:
        for i in range(len(row) - n + 1):
            seen.add(tuple(row[i : i + n]))
            total += 1
    return len(seen) / total if total else 0.0


def token_entropy(sequences: Tensor | Sequence[Sequence[int]]) -> float:
    """Shannon entropy of the pooled unigram distribution, in nats."""
    counts: Counter[int] = Counter()
    for row in _to_lists(sequences):
        counts.update(row)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def repetition_rate(sequences: Tensor | Sequence[Sequence[int]]) -> float:
    """Fraction of positions equal to the token immediately before them."""
    repeats = 0
    total = 0
    for row in _to_lists(sequences):
        for a, b in pairwise(row):
            repeats += int(a == b)
            total += 1
    return repeats / total if total else 0.0


def evaluate_diversity(sequences: Tensor | Sequence[Sequence[int]]) -> DiversityResult:
    """Compute every diversity metric over one set of generated sequences."""
    return DiversityResult(
        distinct_1=distinct_n(sequences, 1),
        distinct_2=distinct_n(sequences, 2),
        distinct_3=distinct_n(sequences, 3),
        entropy=token_entropy(sequences),
        repetition_rate=repetition_rate(sequences),
    )
