# Aether

[![CI](https://github.com/ameyg910/aether/actions/workflows/ci.yml/badge.svg)](https://github.com/ameyg910/aether/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000)](https://github.com/astral-sh/ruff)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-blue)](https://mypy-lang.org/)

**Aether is a production-grade platform for a modern masked (absorbing-state) diffusion
language model** — the MDLM/SUBS formulation that LLaDA and Dream scaled to challenge
autoregressive LLMs. This repository grows week by week from a typed research skeleton
into a served, containerized, observable, open-source framework.

> **Status:** Week 7 — the model is now a service. A FastAPI server exposes generation
> with dynamic batching, SSE streaming of the denoising process, versioned model loading
> with rollback, health/readiness probes, and Prometheus metrics. Behind it sits a typed
> evaluation harness, two samplers with an explicit NFE knob, and distributed training
> with MFU reporting. Containerization and deployment land in later weeks.

## What works today (Weeks 1–4)

- **Typed, config-driven core.** A Hydra structured-config system — fully typed and
  validated against dataclass schemas, with every field overridable from the CLI — drives
  each run. Noise schedules (linear, cosine) and the absorbing-state forward process are
  implemented and unit-tested. *(Week 1)*
- **Data pipeline.** GPT-2 and byte tokenizers, sequence packing, and sharded `uint16`
  memmap datasets with a content fingerprint (`dataset_hash`) for reproducibility. Builds
  WikiText-103 with a single command; a tiny offline corpus is available for tests.
  *(Week 2)*
- **Model + objective.** A bidirectional DiT denoiser with AdaLN-Zero time conditioning,
  and the MDLM/SUBS training loss — verified against `F.cross_entropy` for numerical
  equivalence and checked with `torch.autograd.gradcheck`. A single-batch overfit collapses
  the loss to ~0, confirming the model can fit. *(Week 3)*
- **Training system.** A config-driven `Trainer` with bf16 AMP, gradient accumulation,
  cosine + warmup LR, gradient clipping, and an EMA of the weights used for all sampling.
  Structured logging; pluggable experiment tracking (`jsonl` / `wandb` / `none`, so CI
  stays offline); a run manifest capturing git SHA, dataset hash, seed, and environment;
  and resumable checkpointing bundling `{model, ema, optimizer, scheduler, step, rng}` —
  with a **bit-for-bit resume test** in CI. *(Week 4)*

- **Distributed training.** DDP and FSDP behind a single `train.strategy` flag, with a
  size-based FSDP auto-wrap policy and optional activation checkpointing. Metrics reduce
  across ranks, side effects are rank-0 only, and gradient accumulation suppresses the
  all-reduce on all but the final micro-batch. Checkpoints are consolidated so they are
  **world-size independent** — one written by a 3-GPU run restores into a single process.
  Throughput and MFU are logged, and a scaling experiment turns several runs into a plot.
  *(Week 5)*

- **Evaluation and sampling.** A typed, config-driven harness computing the diffusion
  NELBO (nats/token, bits-per-dim, perplexity) with stratified time sampling, MAUVE via
  the divergence frontier, and diversity metrics (distinct-n, entropy, repetition). Two
  samplers — faithful `ancestral` and confidence-based parallel decoding — both reporting
  NFE, plus a benchmark sweeping the quality-vs-compute curve. *(Week 6)*

- **Inference service.** FastAPI with `/generate`, SSE streaming, `/health`, `/ready`,
  `/metrics`, and admin swap/rollback. An async dynamic batcher merges concurrent
  requests into single forward passes (**4.6× throughput** on eight concurrent requests),
  and a model registry loads checkpoints by version tag from local paths or the HF Hub so
  deploys are reproducible and rollback is instant. *(Week 7)*

Every commit is gated by `mypy --strict`, `ruff`, and `pytest` in CI — including a
multi-process DDP test that spawns real workers and asserts that ranks trained on
different data end with identical parameters, and a pinned regression benchmark that
fails the build when quality or performance silently drifts.

## Quickstart

```bash
git clone https://github.com/ameyg910/aether.git
cd aether
python -m venv .venv && source .venv/bin/activate
make install            # editable install + pre-commit hooks
make demo               # watch a sentence get progressively masked
make plot               # write docs/assets/mask_rate_vs_t.png
make config             # print the composed run configuration
make all                # lint + type-check + test
make data-debug         # build a tiny offline dataset (shards + manifest)
```

> For real WikiText-103: `pip install -e ".[data]"` then `make data`.

```bash
python -m aether.train.overfit   # single-batch overfit: loss collapses to ~0
aether-train train=debug data=local_debug   # tiny tracked training run

NPROC=3 scripts/launch/torchrun_local.sh    # 3-GPU DDP run
sbatch --account=... --partition=... scripts/launch/slurm_fsdp.sbatch   # FSDP on SLURM

aether-eval eval.checkpoint=runs/my-run/checkpoints/latest.pt   # full metrics report
make bench-regression                                          # pinned CI benchmark

pip install -e ".[serve]"
aether-serve serve.model_version=runs/my-run                    # inference API on :8000
```

Training is tracked, checkpointed, and resumable — see the [Training run](#training-run)
section below and [docs/training.md](docs/training.md).

### The forward process, in one command

```bash
python -m aether.diffusion.forward --sentence "the cat sat on the mat"
```

```
schedule=linear  seed=0
t=0.00 | the cat sat on the mat  (mask 0%)
t=0.25 | the cat ░░░ on the mat  (mask 17%)
t=0.50 | ░░░ cat ░░░ on ░░░ mat  (mask 50%)
t=0.75 | ░░░ ░░░ ░░░ on ░░░ ░░░  (mask 83%)
t=1.00 | ░░░ ░░░ ░░░ ░░░ ░░░ ░░░  (mask 100%)
```

![mask rate vs t](docs/assets/mask_rate_vs_t.png)

## Training run

A **55.5M-parameter** MDLM (`d_model=384`, `n_layers=6`, `n_heads=6`) trained on
WikiText-103 (GPT-2 tokenizer, 1024-token blocks) with bf16 AMP on a single H100 MIG
slice. The public dashboard shows a monotonically descending loss curve, decoded text
samples logged every 1000 steps, and a live resume-from-checkpoint that continues the
same curve without a discontinuity.

> **Public W&B dashboard:** https://wandb.ai/f20240973-bits-pilani/aether &nbsp;(run `aether-55m-v2`)

At this compute budget the model learns token frequency — common words surface in the
samples — but not yet long-range coherence; a converged language model is a much larger
compute problem. See the evaluation harness below for how that is now measured, plus
[docs/training.md](docs/training.md) for how to launch, resume, and read the dashboard,
and [docs/reviews/review-01.md](docs/reviews/review-01.md) for Engineering Review #1
(strengths, weaknesses, technical debt, and the refactors queued before Week 5).

## Scaling

The same entry point runs single-process, multi-GPU, or multi-node — rank and world
size come from the environment `torchrun`/SLURM sets:

```bash
NPROC=3 scripts/launch/torchrun_local.sh                 # DDP across 3 GPUs
python scripts/scaling_plot.py runs/ddp-*gpu             # throughput table + plot
```

`scripts/scaling_plot.py` reports tokens/s, speedup, scaling efficiency, MFU, and
estimated time-to-target per configuration, and writes `docs/assets/scaling.png`.
See [docs/cluster.md](docs/cluster.md) for the full guide and
[ADR-0003](docs/adr/0003-ddp-vs-fsdp.md) for the DDP-vs-FSDP tradeoff.

## Evaluation and the NFE tradeoff

**NFE** — number of function evaluations — is the count of model forward passes spent
generating a sequence, and it is what makes diffusion LMs interesting: an autoregressive
model needs one pass *per token* with no say in the matter, while a diffusion model
chooses how much compute to spend on a sequence of any length.

Two samplers make that tradeoff explicit:

| sampler | steps | NFE | p50 (s) | p95 (s) | tok/s | distinct-1 | distinct-2 | entropy | rep. rate | MAUVE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ancestral | 32 | 32 | 0.486 | 0.487 | 4,212 | 0.542 | 0.971 | 6.249 | 0.006 | 0.972 |
| ancestral | 64 | 64 | 0.974 | 0.974 | 2,103 | 0.544 | 0.971 | 6.242 | 0.007 | 0.979 |
| ancestral | 128 | 128 | 1.943 | 1.947 | 1,054 | 0.546 | 0.971 | 6.281 | 0.003 | 0.999 |
| ancestral | 256 | 256 | 3.888 | 3.891 | 527 | 0.539 | 0.954 | 6.209 | 0.011 | 0.989 |
| ancestral | 512 | 512 | 7.838 | 7.846 | 261 | 0.526 | 0.965 | 6.163 | 0.009 | 0.913 |
| confidence | 32 | 32 | 0.511 | 0.512 | 4,005 | 0.549 | 0.971 | 6.245 | 0.007 | 0.976 |
| confidence | 64 | 64 | 1.023 | 1.024 | 2,002 | 0.550 | 0.969 | 6.251 | 0.008 | 0.943 |
| confidence | 128 | 128 | 2.047 | 2.048 | 1,000 | 0.536 | 0.970 | 6.215 | 0.007 | 0.597 |
| confidence | 256 | 128 | 2.047 | 2.048 | 1,000 | 0.552 | 0.972 | 6.301 | 0.010 | 0.999 |
| confidence | 512 | 128 | 2.052 | 2.052 | 998 | 0.518 | 0.962 | 6.132 | 0.013 | 0.917 |

![quality vs compute](docs/assets/nfe_quality.png)

Held-out NELBO: **7.07 nats/token**, 10.20 bits-per-dim, perplexity bound 1178.

Latency scales linearly with NFE, as expected — one forward pass per step. MAUVE
peaks at 128 steps (0.999) and *declines* past it, so on this checkpoint the extra
compute beyond ~128 steps buys nothing.

**Confidence-based decoding collapsed on this checkpoint** — every position resolved
to the same token (distinct-2 = 0, repetition rate = 1.0), and NFE capped at 128 even
when 512 steps were requested, because the sampler revealed everything within the
first few passes. At `t=1` the input is entirely `[MASK]`, so the model predicts the
unconditional marginal identically at every position; those tokens become context and
reinforce themselves. Switching from argmax to sampling did not break the loop, which
suggests the distribution at 30k steps is too peaked for confidence ordering to help.
Ancestral is the sampler to ship for this model.

```bash
python benchmarks/nfe_quality.py \
  --checkpoint runs/my-run/checkpoints/latest.pt \
  --data data/wikitext103 --steps 32 64 128 256 512
```

<!-- Regenerate with the command above and paste the emitted table here. -->
| sampler | steps | NFE | p50 (s) | tok/s | distinct-2 | entropy | MAUVE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _pending — run the sweep against a trained checkpoint_ | | | | | | | |

> Evaluate `confidence` against a **trained** checkpoint. On an untrained model its
> argmax collapses immediately and it looks far worse than `ancestral` — a property of
> the random model, not of the sampler.

See [docs/evaluation.md](docs/evaluation.md) for the metric definitions and why
perplexity is subtler for a diffusion LM than for an AR model (short version: it is a
Monte Carlo estimate of a variational *bound*, not an exact likelihood).

## Serving

```bash
pip install -e ".[serve]"
aether-serve serve.model_version=runs/my-run       # OpenAPI docs at /docs

curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"n_samples":2,"length":64,"steps":64,"sampler":"confidence"}'

python examples/client_example.py --stream         # watch the text denoise live
```

The server holds arriving requests for a few milliseconds and runs whatever
accumulated as one forward pass. Every request pays up to `max_wait_ms` of extra
latency; in exchange they all share one pass, which is strongly net-positive under
concurrency:

| | wall clock | batch size per request |
| --- | ---: | ---: |
| 8 sequential requests | 0.61 s | 1 |
| 8 concurrent requests | 0.13 s | 8 |

Models load by **version tag** (`hf:owner/repo@revision`), so a deploy is reproducible
and `/admin/rollback` is instant. See [docs/serving.md](docs/serving.md) for the full
guide, including why liveness and readiness are separate probes.

## Why absorbing-state diffusion?

The forward process replaces tokens with an absorbing `[MASK]` state at a rate set by a
noise schedule; the learned reverse process unmasks them. The MDLM result is that the
training objective reduces to a weighted sum of masked-LM cross-entropy losses — stable
and cheap to train. See [ADR-0001](docs/adr/0001-absorbing-state-mdlm.md).

## Repository structure

```
src/aether/
  config/      # typed Hydra structured configs + loader
  diffusion/   # noise schedules, forward process, SUBS loss, samplers (NFE-aware)
  evaluate/    # NELBO/bpd, MAUVE, diversity metrics + `aether-eval` CLI
  serve/       # FastAPI app, dynamic batcher, model registry, metrics
  data/        # tokenizer, packing, sharding, datamodule
  models/      # bidirectional DiT denoiser (AdaLN time conditioning)
  train/       # config-driven Trainer: AMP, grad-accum, cosine schedule, EMA,
               #   resumable checkpointing, pluggable experiment tracking,
               #   distributed.py (DDP/FSDP), precision.py, mfu.py
  seed.py      # deterministic seeding + RNG state capture
configs/       # composable YAML run configuration (model / data / train / tracking)
benchmarks/    # NFE-quality sweep + pinned regression benchmark
loadtest/      # locust load-test driver
scripts/       # demo, visualization, scaling analysis
  launch/      # torchrun (local multi-GPU) + SLURM sbatch templates
tests/         # unit + invariant tests (incl. bit-for-bit resume)
docs/          # ADRs, data + training guides, engineering reviews
```

## Roadmap

| Week | Focus | Status |
| ---- | ----- | ------ |
| 1  | Typed skeleton, noise schedules, absorbing forward process | ✅ |
| 2  | Data pipeline: tokenizer, packing, sharded memmap datasets | ✅ |
| 3  | Bidirectional DiT denoiser + MDLM/SUBS loss | ✅ |
| 4  | Training system: AMP, EMA, cosine schedule, checkpointing, tracking | ✅ |
| 5  | Distributed training (DDP / FSDP), MFU + scaling | ✅ |
| 6  | Evaluation harness, samplers, NFE benchmark | ✅ |
| 7  | Serving: batching, registry, SSE, metrics | ✅ |
| 8  | Docker + CI/CD | ⏳ |
| 9  | Kubernetes + observability | ⏳ |
| 10 | Release + Hugging Face | ⏳ |

## Development

Fully typed (`mypy --strict`), linted and formatted with `ruff`, tested with `pytest`,
and configuration-driven via Hydra. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

Builds on ideas from MDLM (Sahoo et al., 2024), D3PM (Austin et al., 2021), and the
broader discrete-diffusion literature.
