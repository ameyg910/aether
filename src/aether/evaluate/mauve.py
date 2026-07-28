"""MAUVE: divergence-frontier comparison between generated and reference text.

MAUVE measures how far a model's output distribution ``Q`` sits from the human
reference distribution ``P``, capturing both failure directions at once:

- text the model produces that humans never would (low precision -- gibberish),
- text humans produce that the model never would (low recall -- mode collapse).

A single KL cannot express both. MAUVE instead sweeps a mixture
``R_lambda = lambda*P + (1-lambda)*Q`` across ``lambda in (0, 1)`` and traces the
curve of ``(exp(-c*KL(Q||R)), exp(-c*KL(P||R)))``. The area under that curve is
the score: 1.0 for identical distributions, near 0 for disjoint ones.

**On the featurizer.** Canonical MAUVE embeds text with a large pretrained model
(GPT-2 large) before clustering. That is a heavy, network-dependent dependency, so
the default here is a deterministic hashed n-gram featurizer: it captures local
lexical structure, needs no downloads, and makes the metric reproducible offline.
The divergence-frontier computation is the real algorithm; only the feature space
is an approximation. Scores are therefore comparable *between Aether runs* but not
against published MAUVE numbers. Pass your own ``featurizer`` to use embeddings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from torch import Tensor

Features = np.ndarray[Any, np.dtype[np.float64]]


def _to_lists(sequences: Tensor | Sequence[Sequence[int]]) -> list[list[int]]:
    if isinstance(sequences, Tensor):
        return [[int(v) for v in row] for row in sequences.tolist()]
    return [list(map(int, row)) for row in sequences]


def hashed_ngram_features(
    sequences: Tensor | Sequence[Sequence[int]], dim: int = 256, max_n: int = 2
) -> Features:
    """Deterministic bag-of-n-grams hashed into ``dim`` buckets, L2-normalized."""
    rows = _to_lists(sequences)
    out = np.zeros((len(rows), dim), dtype=np.float64)
    for r, row in enumerate(rows):
        for n in range(1, max_n + 1):
            for i in range(len(row) - n + 1):
                # Python's hash() is salted per process; use a fixed polynomial.
                h = 0
                for tok in row[i : i + n]:
                    h = (h * 1_000_003 + tok) % 2_147_483_647
                out[r, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norms, 1e-12)
    return out


def _kmeans(data: Features, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm with deterministic init; returns cluster assignments."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    k = max(1, min(k, n))
    centers = data[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # (n, k) squared distances via broadcasting.
        d = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = data[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


def _histogram(labels: np.ndarray, k: int, smoothing: float = 1e-9) -> Features:
    counts = np.bincount(labels, minlength=k).astype(np.float64) + smoothing
    counts /= counts.sum()
    return counts


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration across numpy versions.

    ``np.trapz`` was renamed to ``np.trapezoid`` in numpy 2.0 and the old name
    removed; support both so the metric works on either.
    """
    fn = getattr(np, "trapezoid", None)
    if fn is None:  # numpy < 2.0
        fn = np.trapz  # type: ignore[attr-defined]
    return float(fn(y, x))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def divergence_frontier(
    p: np.ndarray, q: np.ndarray, scaling_factor: float = 5.0, n_points: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the (recall-like, precision-like) frontier over mixture weights.

    The curve is anchored with the two limiting points ``(1, 0)`` and ``(0, 1)``,
    which correspond to the degenerate mixtures at each end of the sweep. Without
    them the frontier for two *identical* distributions collapses to the single
    point ``(1, 1)`` and the area under it computes as zero rather than one.
    """
    lambdas = np.linspace(1e-6, 1 - 1e-6, n_points)
    xs, ys = [1.0], [0.0]
    for lam in lambdas:
        r = lam * p + (1 - lam) * q
        xs.append(float(np.exp(-scaling_factor * _kl(q, r))))
        ys.append(float(np.exp(-scaling_factor * _kl(p, r))))
    xs.append(0.0)
    ys.append(1.0)
    return np.asarray(xs), np.asarray(ys)


def mauve_score(
    p_sequences: Tensor | Sequence[Sequence[int]],
    q_sequences: Tensor | Sequence[Sequence[int]],
    n_clusters: int | None = None,
    featurizer: Callable[[Tensor | Sequence[Sequence[int]]], Features] | None = None,
    scaling_factor: float = 5.0,
    seed: int = 0,
) -> float:
    """MAUVE between reference ``p`` and generated ``q`` sequences (0.0-1.0)."""
    featurize = featurizer or hashed_ngram_features
    p_feat, q_feat = featurize(p_sequences), featurize(q_sequences)
    joint = np.concatenate([p_feat, q_feat], axis=0)

    # Rule of thumb from the paper: clusters ~ n/10, bounded for tiny samples.
    k = n_clusters if n_clusters is not None else max(2, min(50, joint.shape[0] // 10))
    labels = _kmeans(joint, k, seed=seed)
    p_hist = _histogram(labels[: p_feat.shape[0]], k)
    q_hist = _histogram(labels[p_feat.shape[0] :], k)

    xs, ys = divergence_frontier(p_hist, q_hist, scaling_factor)
    order = np.argsort(xs)
    area = float(_trapezoid(ys[order], xs[order]))
    return max(0.0, min(1.0, area))
