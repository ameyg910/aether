"""Benchmark the NFE-quality tradeoff: how much does spending more sampling steps buy?

This is the measurement that justifies diffusion LMs. An autoregressive model needs
one forward pass per token, full stop. A diffusion model chooses: few steps and
fast, or many steps and better. This script sweeps the step count for both
samplers and records quality against latency.

Usage::

    # against a trained checkpoint
    python benchmarks/nfe_quality.py --checkpoint runs/my-run/checkpoints/latest.pt \\
        --data data/wikitext103 --steps 32 64 128 256 512

    # fast smoke run with an untrained model (CI / plumbing check)
    python benchmarks/nfe_quality.py --synthetic --steps 4 8 --samples 8

Latency is measured properly: warmup iterations first (to pay CUDA context and
allocator costs once), then several timed repeats reported as p50 and p95 rather
than a single mean, because sampler timings are right-skewed and a mean hides tail
behaviour.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aether.config.schemas import ModelConfig
from aether.data.datamodule import DiffusionDataModule
from aether.diffusion.samplers import SAMPLERS, sample
from aether.evaluate import evaluate_diversity, evaluate_nll, mauve_score
from aether.models.aether_model import AetherModel

DEFAULT_STEPS = (32, 64, 128, 256, 512)


@dataclass
class BenchRow:
    """One (sampler, steps) configuration."""

    sampler: str
    steps: int
    nfe: int
    latency_p50_s: float
    latency_p95_s: float
    tokens_per_s: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    entropy: float
    repetition_rate: float
    mauve: float | None = None


def _timed(fn: Any, warmup: int, repeats: int) -> tuple[list[float], Any]:
    """Run ``fn`` warmup times, then ``repeats`` timed times; return timings + last result."""
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    result = None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return timings, result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def benchmark_sampler(
    model: nn.Module,
    sampler: str,
    steps: int,
    *,
    batch: int,
    length: int,
    mask_token_id: int,
    schedule: str,
    device: torch.device,
    reference: torch.Tensor | None,
    warmup: int,
    repeats: int,
) -> BenchRow:
    """Measure one sampler at one step count."""
    generator = torch.Generator(device=device).manual_seed(0)

    def run() -> Any:
        return sample(
            model,
            batch=batch,
            length=length,
            mask_token_id=mask_token_id,
            sampler=sampler,
            steps=steps,
            schedule=schedule,
            device=device,
            generator=generator,
        )

    timings, out = _timed(run, warmup, repeats)
    tokens = out.tokens
    div = evaluate_diversity(tokens)
    p50 = statistics.median(timings)

    return BenchRow(
        sampler=sampler,
        steps=steps,
        nfe=out.nfe,
        latency_p50_s=p50,
        latency_p95_s=_percentile(timings, 95),
        tokens_per_s=(batch * length) / p50 if p50 > 0 else 0.0,
        mauve=mauve_score(reference, tokens) if reference is not None else None,
        **div.as_dict(),
    )


def render_table(rows: list[BenchRow]) -> str:
    """Markdown results table, grouped by sampler."""
    header = (
        "| sampler | steps | NFE | p50 (s) | p95 (s) | tok/s | "
        "distinct-1 | distinct-2 | entropy | rep. rate | MAUVE |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for r in sorted(rows, key=lambda r: (r.sampler, r.steps)):
        mauve = f"{r.mauve:.3f}" if r.mauve is not None else "n/a"
        lines.append(
            f"| {r.sampler} | {r.steps} | {r.nfe} | {r.latency_p50_s:.3f} | "
            f"{r.latency_p95_s:.3f} | {r.tokens_per_s:,.0f} | {r.distinct_1:.3f} | "
            f"{r.distinct_2:.3f} | {r.entropy:.3f} | {r.repetition_rate:.3f} | {mauve} |"
        )
    return "\n".join(lines)


def plot(rows: list[BenchRow], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"ancestral": "#c1121f", "confidence": "#457b9d"}

    for sampler in sorted({r.sampler for r in rows}):
        sub = sorted([r for r in rows if r.sampler == sampler], key=lambda r: r.nfe)
        ax1.plot(
            [r.nfe for r in sub],
            [r.entropy for r in sub],
            "o-",
            label=sampler,
            color=colors.get(sampler),
        )
        ax2.plot(
            [r.latency_p50_s for r in sub],
            [r.entropy for r in sub],
            "o-",
            label=sampler,
            color=colors.get(sampler),
        )

    ax1.set_xlabel("NFE (model forward passes)")
    ax1.set_ylabel("token entropy (nats)")
    ax1.set_title("Quality vs compute")
    ax1.set_xscale("log", base=2)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("latency p50 (s)")
    ax2.set_ylabel("token entropy (nats)")
    ax2.set_title("Quality vs latency")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def load_model(
    checkpoint: Path | None, synthetic: bool, vocab_size: int, length: int, device: torch.device
) -> tuple[nn.Module, int]:
    """Return a model and its mask-token id."""
    if synthetic or checkpoint is None:
        cfg = ModelConfig(
            vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, max_seq_len=length
        )
        return AetherModel(cfg).to(device).eval(), vocab_size - 1

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    # Recover geometry from the checkpoint so the benchmark needs no extra config.
    vocab, d_model = state["tok_emb.weight"].shape
    max_seq_len = state["pos_emb"].shape[1]
    n_layers = 1 + max(
        int(k.split(".")[1]) for k in state if k.startswith("blocks.") and k.count(".") > 1
    )
    cfg = ModelConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=max(1, d_model // 64),
        max_seq_len=max_seq_len,
    )
    model = AetherModel(cfg)
    model.load_state_dict(state)
    return model.to(device).eval(), vocab - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=None, help="prepared data dir for NLL + MAUVE")
    ap.add_argument("--synthetic", action="store_true", help="untrained model; plumbing only")
    ap.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    ap.add_argument("--samplers", nargs="+", default=list(SAMPLERS))
    ap.add_argument("--samples", type=int, default=32, help="sequences generated per config")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=64, help="synthetic mode only")
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/nfe_quality.json"))
    ap.add_argument("--plot", type=Path, default=Path("docs/assets/nfe_quality.png"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model, mask_id = load_model(
        args.checkpoint, args.synthetic, args.vocab_size, args.length, device
    )

    reference: torch.Tensor | None = None
    nll: dict[str, float | int] | None = None
    if args.data is not None:
        dm = DiffusionDataModule(str(args.data), split="val", batch_size=args.samples)
        val_batches = [torch.from_numpy(b.copy()) for b in dm.epoch_batches(0)]
        if val_batches:
            reference = val_batches[0][:, : args.length]
            nll = evaluate_nll(
                model, val_batches[:4], mask_id, args.schedule, mc_samples=8, device=device
            ).as_dict()
            print(f"NLL on val: {nll}")

    rows = [
        benchmark_sampler(
            model,
            sampler,
            steps,
            batch=args.samples,
            length=args.length,
            mask_token_id=mask_id,
            schedule=args.schedule,
            device=device,
            reference=reference,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for sampler in args.samplers
        for steps in args.steps
    ]
    for row in rows:
        print(
            f"  {row.sampler:11s} steps={row.steps:4d} "
            f"nfe={row.nfe:4d} p50={row.latency_p50_s:.3f}s"
        )

    payload: dict[str, Any] = {
        "config": {
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "synthetic": args.synthetic,
            "samples": args.samples,
            "length": args.length,
            "schedule": args.schedule,
            "device": str(device),
            "torch": torch.__version__,
        },
        "nll": nll,
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")

    print()
    print(render_table(rows))
    if not args.no_plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
