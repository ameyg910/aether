"""Summarize a scaling experiment: tokens/s and MFU versus GPU count.

Reads the ``manifest.json`` and ``metrics.jsonl`` that every run writes, and emits
a scaling plot plus a throughput table. Usage::

    python scripts/scaling_plot.py runs/ddp-1gpu runs/ddp-2gpu runs/ddp-3gpu
    python scripts/scaling_plot.py runs/ddp-*gpu --out docs/assets/scaling.png

Early steps are discarded before averaging: the first optimizer steps include
CUDA context creation, autotuning, and allocator warmup, and folding those into a
throughput number understates steady-state performance.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

WARMUP_LOGS = 2  # metric rows to drop before averaging


@dataclass
class RunSummary:
    name: str
    world_size: int
    tokens_per_sec: float
    mfu: float | None
    loss: float | None

    @property
    def mfu_pct(self) -> str:
        return f"{self.mfu * 100:.1f}%" if self.mfu is not None else "n/a"


def _median(rows: list[dict[str, float]], key: str) -> float | None:
    values = [r[key] for r in rows if key in r]
    return statistics.median(values) if values else None


def summarize_run(run_dir: Path) -> RunSummary | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    rows: list[dict[str, float]] = []
    for line in metrics_path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "tokens_per_sec" in obj:
            rows.append(obj)
    if not rows:
        return None
    rows = rows[WARMUP_LOGS:] or rows

    world_size = 1
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            world_size = int(json.loads(manifest.read_text()).get("world_size", 1))

    tps = _median(rows, "tokens_per_sec")
    if tps is None:
        return None
    return RunSummary(
        name=run_dir.name,
        world_size=world_size,
        tokens_per_sec=tps,
        mfu=_median(rows, "mfu"),
        loss=_median(rows, "loss"),
    )


def render_table(summaries: list[RunSummary], target_tokens: float) -> str:
    """Markdown throughput/cost report."""
    base = summaries[0].tokens_per_sec if summaries else 0.0
    lines = [
        "| run | GPUs | tokens/s | speedup | scaling eff. | MFU | est. time to target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        speedup = s.tokens_per_sec / base if base else 0.0
        ideal = s.world_size / summaries[0].world_size if summaries[0].world_size else 1
        eff = speedup / ideal if ideal else 0.0
        hours = target_tokens / s.tokens_per_sec / 3600 if s.tokens_per_sec else float("inf")
        lines.append(
            f"| {s.name} | {s.world_size} | {s.tokens_per_sec:,.0f} | "
            f"{speedup:.2f}x | {eff * 100:.0f}% | {s.mfu_pct} | {hours:.1f} h |"
        )
    return "\n".join(lines)


def plot(summaries: list[RunSummary], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpus = [s.world_size for s in summaries]
    tps = [s.tokens_per_sec for s in summaries]
    ideal = [tps[0] * (g / gpus[0]) for g in gpus]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(gpus, ideal, "--", color="grey", label="ideal linear")
    ax1.plot(gpus, tps, "o-", color="#c1121f", label="measured")
    ax1.set_xlabel("GPUs")
    ax1.set_ylabel("tokens / second")
    ax1.set_title("Throughput scaling")
    ax1.set_xticks(gpus)
    ax1.grid(alpha=0.3)
    ax1.legend()

    if all(s.mfu is not None for s in summaries):
        ax2.bar([str(g) for g in gpus], [s.mfu * 100 for s in summaries], color="#457b9d")  # type: ignore[misc]
        ax2.set_ylabel("MFU (%)")
        ax2.set_xlabel("GPUs")
        ax2.set_title("Model FLOPs Utilization")
        ax2.grid(alpha=0.3, axis="y")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "MFU unavailable\n(set train.device_peak_tflops)", ha="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories to compare")
    ap.add_argument("--out", type=Path, default=Path("docs/assets/scaling.png"))
    ap.add_argument(
        "--target-tokens",
        type=float,
        default=1e9,
        help="token budget used for the time-to-target column (default 1e9)",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    summaries = [s for s in (summarize_run(p) for p in args.runs) if s is not None]
    if not summaries:
        raise SystemExit("no runs with throughput metrics found")
    summaries.sort(key=lambda s: s.world_size)

    print(render_table(summaries, args.target_tokens))
    if not args.no_plot:
        plot(summaries, args.out)


if __name__ == "__main__":
    main()
