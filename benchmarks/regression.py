"""Fast regression benchmark with pinned thresholds.

Quality and performance regressions are silent by nature: nothing crashes, the
numbers just get worse. This module pins a tiny, fully deterministic benchmark and
a set of thresholds around it, so a change that breaks the samplers, the
likelihood scale, or the metric implementations fails a build instead of quietly
shipping.

It is designed to run in **seconds on CPU** so it can gate every pull request. It
uses an untrained model on purpose: the point is to detect changes in the
machinery, not in model quality, and an untrained model has *analytically known*
behaviour to anchor against (near-uniform likelihood, high-entropy ancestral
samples). A trained-checkpoint benchmark is a separate, slower job.

Run standalone::

    python benchmarks/regression.py            # exits non-zero on breach
    python benchmarks/regression.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from aether.config.schemas import ModelConfig
from aether.diffusion.samplers import sample
from aether.evaluate import evaluate_diversity, evaluate_nll, mauve_score
from aether.models.aether_model import AetherModel

# --- pinned configuration; changing these invalidates the thresholds ----------
SEED = 0
VOCAB_SIZE = 40
MASK_ID = 39
LENGTH = 32
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
BATCH = 8
STEPS = 16
MC_SAMPLES = 8
SCHEDULE = "linear"

# --- thresholds --------------------------------------------------------------
# Deliberately loose enough to survive platform float differences, tight enough
# that a real regression (wrong loss scale, collapsed sampler, broken metric)
# trips them. Each entry is (low, high), inclusive.
THRESHOLDS: dict[str, tuple[float, float]] = {
    # An untrained model is ~uniform over the non-mask vocabulary: ln(39) = 3.664.
    # A wrong normalisation in the NELBO moves this by orders of magnitude.
    "nll_nats_per_token": (2.5, 5.0),
    "nll_bits_per_dim": (3.6, 7.3),
    # Ancestral sampling from a random model should stay high-entropy; a collapse
    # here means the sampler stopped sampling and started arg-maxing.
    "ancestral_entropy": (2.5, 4.0),
    "ancestral_distinct_2": (0.5, 1.0),
    # Self-MAUVE must be exactly 1.0; anything else means the frontier broke.
    "self_mauve": (0.999, 1.0),
    # Generous ceiling: catches catastrophic slowdowns, not normal CI jitter.
    "sampler_latency_s": (0.0, 30.0),
}


def run_regression(device: torch.device | None = None) -> dict[str, Any]:
    """Execute the pinned benchmark and return its metrics."""
    device = device or torch.device("cpu")
    torch.manual_seed(SEED)
    model = (
        AetherModel(
            ModelConfig(
                vocab_size=VOCAB_SIZE,
                d_model=D_MODEL,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                max_seq_len=LENGTH,
            )
        )
        .to(device)
        .eval()
    )

    data = [
        torch.randint(
            0,
            MASK_ID,
            (BATCH, LENGTH),
            generator=torch.Generator().manual_seed(SEED + i),
        )
        for i in range(2)
    ]
    nll = evaluate_nll(
        model,
        data,
        MASK_ID,
        SCHEDULE,
        mc_samples=MC_SAMPLES,
        device=device,
        generator=torch.Generator(device=device).manual_seed(SEED),
    )

    start = time.perf_counter()
    ancestral = sample(
        model,
        BATCH,
        LENGTH,
        MASK_ID,
        sampler="ancestral",
        steps=STEPS,
        schedule=SCHEDULE,
        device=device,
        generator=torch.Generator(device=device).manual_seed(SEED),
    )
    latency = time.perf_counter() - start

    confidence = sample(
        model,
        BATCH,
        LENGTH,
        MASK_ID,
        sampler="confidence",
        steps=STEPS,
        schedule=SCHEDULE,
        device=device,
        generator=torch.Generator(device=device).manual_seed(SEED),
    )

    anc_div = evaluate_diversity(ancestral.tokens)
    conf_div = evaluate_diversity(confidence.tokens)

    return {
        "nll_nats_per_token": nll.nats_per_token,
        "nll_bits_per_dim": nll.bits_per_dim,
        "ancestral_entropy": anc_div.entropy,
        "ancestral_distinct_2": anc_div.distinct_2,
        "confidence_entropy": conf_div.entropy,
        "ancestral_nfe": float(ancestral.nfe),
        "confidence_nfe": float(confidence.nfe),
        "self_mauve": mauve_score(data[0], data[0]),
        "sampler_latency_s": latency,
        "_reference": {"uniform_nats": math.log(VOCAB_SIZE - 1)},
    }


def check_thresholds(metrics: dict[str, Any]) -> list[str]:
    """Return a list of human-readable threshold breaches (empty when clean)."""
    breaches = []
    for key, (low, high) in THRESHOLDS.items():
        if key not in metrics:
            breaches.append(f"{key}: MISSING from benchmark output")
            continue
        value = float(metrics[key])
        if not (low <= value <= high):
            breaches.append(f"{key}: {value:.4f} outside [{low}, {high}]")
    return breaches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None, help="write metrics to this path")
    args = ap.parse_args()

    metrics = run_regression()
    breaches = check_thresholds(metrics)

    for key in sorted(k for k in metrics if not k.startswith("_")):
        bounds = THRESHOLDS.get(key)
        window = f"  [{bounds[0]}, {bounds[1]}]" if bounds else ""
        print(f"  {key:26s} {float(metrics[key]):10.4f}{window}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.json}")

    if breaches:
        print("\nREGRESSION BENCHMARK FAILED:")
        for b in breaches:
            print(f"  - {b}")
        return 1
    print("\nregression benchmark OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
