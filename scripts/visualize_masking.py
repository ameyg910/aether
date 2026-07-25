"""Plot the absorbing forward process mask rate vs diffusion time.

Overlays each schedule's theoretical mask rate ``1 - alpha(t)`` with the
empirical fraction of masked tokens from actually sampling the process, to show
they agree. Saves to ``docs/assets/mask_rate_vs_t.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from aether.diffusion.forward import AbsorbingForwardProcess
from aether.diffusion.schedule import build_schedule


def main() -> None:
    t = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    rng = np.random.default_rng(0)
    seq_len = 20_000

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for kind, color in (("linear", "#4C72B0"), ("cosine", "#C44E52")):
        schedule = build_schedule(kind)
        ax.plot(t, schedule.mask_rate(t), color=color, label=f"{kind} (theoretical)")
        process = AbsorbingForwardProcess(schedule)
        x0 = np.ones(seq_len, dtype=np.int64)
        sampled = t[::10]
        empirical = [float((process.sample(x0, float(tt), rng) == 0).mean()) for tt in sampled]
        ax.scatter(
            sampled, empirical, color=color, s=16, alpha=0.7, zorder=3, label=f"{kind} (empirical)"
        )

    ax.set_xlabel("diffusion time  t")
    ax.set_ylabel("mask rate  1 - alpha(t)")
    ax.set_title("Absorbing-state forward process: mask rate vs t")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "mask_rate_vs_t.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
