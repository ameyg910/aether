"""Absorbing-state (masked) discrete diffusion forward process.

Toy/reference implementation over integer token-id arrays, used for tests and
visualization. Each position is independently replaced by ``mask_token_id`` with
probability ``mask_rate(t) = 1 - alpha(t)``. Week 3 adds a batched torch-tensor
path for training; the math here is the reference that path is checked against.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aether.diffusion.schedule import NoiseSchedule, build_schedule

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class AbsorbingForwardProcess:
    """Samples ``x_t ~ q(x_t | x_0)`` for absorbing-state diffusion."""

    schedule: NoiseSchedule
    mask_token_id: int = 0

    def mask_rate_at(self, t: float) -> float:
        """Scalar mask rate at time ``t``."""
        return float(self.schedule.mask_rate(np.asarray([t], dtype=np.float64))[0])

    def sample(self, x0: IntArray, t: float, rng: np.random.Generator) -> IntArray:
        """Sample the noised sequence at time ``t``.

        Args:
            x0: Clean token ids, shape ``(seq_len,)``.
            t: Diffusion time in ``[0, 1]``.
            rng: NumPy random generator (pass a seeded one for determinism).

        Returns:
            Token ids with masked positions set to ``mask_token_id``.
        """
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t must be in [0, 1], got {t}")
        rate = self.mask_rate_at(t)
        keep = rng.random(size=x0.shape) >= rate
        return np.asarray(np.where(keep, x0, np.int64(self.mask_token_id)), dtype=np.int64)


def _toy_encode(sentence: str, mask_token_id: int) -> tuple[IntArray, dict[int, str]]:
    """Whitespace toy tokenizer (a real BPE tokenizer arrives in Week 2)."""
    vocab: dict[str, int] = {}
    ids: list[int] = []
    next_id = mask_token_id + 1
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1
        ids.append(vocab[word])
    id_to_word = {i: w for w, i in vocab.items()}
    return np.asarray(ids, dtype=np.int64), id_to_word


def _render(x: IntArray, id_to_word: dict[int, str], mask_token_id: int) -> str:
    return " ".join(
        "\u2591\u2591\u2591" if int(tok) == mask_token_id else id_to_word[int(tok)]
        for tok in x.tolist()
    )


def main() -> None:
    """CLI: show a sentence being progressively masked as ``t`` grows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default="the cat sat on the mat")
    parser.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    mask_id = 0
    process = AbsorbingForwardProcess(build_schedule(args.schedule), mask_token_id=mask_id)
    x0, id_to_word = _toy_encode(args.sentence, mask_id)
    rng = np.random.default_rng(args.seed)

    print(f"schedule={args.schedule}  seed={args.seed}")
    print(f"t=0.00 | {_render(x0, id_to_word, mask_id)}  (mask 0%)")
    for t in args.steps:
        xt = process.sample(x0, float(t), rng)
        frac = float((xt == mask_id).mean())
        print(f"t={t:.2f} | {_render(xt, id_to_word, mask_id)}  (mask {frac * 100:.0f}%)")


if __name__ == "__main__":
    main()
