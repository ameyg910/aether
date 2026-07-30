# Final Engineering Review (v1.0.0)

Written at the end of Week 10, after [Review #1](review-01.md) (Week 4) and
[Review #2](review-02.md) (Week 8).

## State of the platform

Aether is an end-to-end platform for masked diffusion language models: data
preparation through distributed training, a typed evaluation harness, an
inference service with dynamic batching and versioned model loading, multi-stage
container images, and a Helm chart that autoscales on queue depth with dashboards
and alerts committed as code. A 55.5M-parameter model trained on WikiText-103
reaches **7.14 nats/token** at **29.7% MFU** on one A6000, and serves at roughly
**15 req/s under 16 concurrent users** — about **12x** what serialized serving
would manage — with p50 latency of 660 ms. Every commit passes `mypy --strict`,
`ruff`, and 180+ tests, including a multi-process DDP test asserting that ranks
fed different data converge to identical parameters, a pinned regression
benchmark anchored to an analytically-known baseline, and a `kind` job that
installs the chart on a real cluster and probes the endpoints. The model itself
is small and undertrained on purpose: the platform is the artifact, and the model
is the fixture that keeps it honest.

## What is genuinely solid

- **The objective is verified, not assumed.** The SUBS loss is checked against
  `F.cross_entropy` for numerical equivalence and against `torch.autograd.gradcheck`
  for gradients, and a single-batch overfit drives the loss to ~0.
- **Reproducibility is mechanical.** Every run writes a manifest with git SHA,
  dataset hash, seed, and environment; checkpoints carry their own architecture
  config; `uv.lock` pins the full dependency graph; `make train-toy` reproduces a
  complete train-and-evaluate cycle from a clean clone with no GPU.
- **Failures surface where they are cheap.** The regression benchmark pins an
  untrained model's likelihood to `ln(vocab-1)` — an *external* anchor, not a
  self-consistency check. Deploy artifacts are validated against the metric names
  the service actually exports, so a renamed series fails a build instead of
  silently emptying a dashboard panel.
- **The serving design holds under load.** `/health` answered in 4 ms while the
  GPU was saturated, because the blocking model call runs in a worker thread.
  Liveness and readiness are separate probes for the reason they exist: a pod
  still loading weights is alive but not ready.

## The biggest technical debt

**Nothing was validated where it runs until very late.** This is the honest
headline. Four distinct bugs shipped because the sandbox they were written in
differed from the environment they ran in:

1. The batcher caught only the builtin `TimeoutError`; on Python 3.10
   `asyncio.TimeoutError` is a *different class*, so every request hung forever
   with no error. Introduced by a lint autofix that was correct for the
   configured target version and wrong for the deployment interpreter.
2. The config loader located `configs/` by walking up from `__file__`, which
   works in a source checkout and lands inside site-packages once installed.
3. `--help` was passed to a Hydra app, which tried to parse it as an override.
4. The load generator targeted a Service name the Helm chart does not create,
   requesting a sequence length the demo checkpoint could not serve, with
   `|| true` and `>/dev/null` hiding both.

Each was invisible where it was written and obvious where it ran. The mitigations
now in place — CI builds and smoke-tests the image, a `kind` job installs the
chart on a real cluster — are what caught them. The lesson generalizes: **the
thing you validate has to be the thing you ship**, and CI should test the *oldest*
supported Python, not only the newest.

Other outstanding debt:

- **FSDP has never run on hardware.** It cannot initialize on CPU, so it has no CI
  coverage, and three GPUs were never simultaneously free on the shared box. It is
  code-reviewed, not tested. Recorded in ADR-0003 and still true.
- **The `Trainer` was never split.** Review #1 committed to it before Week 5;
  Review #2 recorded it as knowingly deferred. It then absorbed distributed
  support, MFU accounting, and checkpoint bookkeeping. `fit()` now mixes the step
  loop with logging, sampling, and retention policy. Three reviews, still open.
- **No GPU in CI.** bf16 kernels, allocator behaviour under memory pressure, and
  NCCL collectives are exercised only by hand.
- **Config schema sprawl.** Eight top-level dataclasses in one file; `TrainConfig`
  alone has 31 fields spanning optimization, distribution, checkpointing, and
  sampling. Review #2 planned the split; it did not happen.
- **Numbers are hand-copied.** Benchmark tables are pasted into the README by
  hand. CPU-sandbox figures survived there for a week before real measurements
  replaced them. Nothing prevents a recurrence.

## What I would do with more time

**Prompt-conditioned infilling, first.** The model is unconditional, which
undersells the architecture badly. Masked diffusion is *natively* an infiller:
pin some positions to real tokens and the model fills the rest around them,
bidirectionally — something an AR model structurally cannot do. It is roughly a
15-line change and it is the difference between "generates WikiText-flavoured
noise" and a demo that shows why anyone should care about this model class.

**Then serving efficiency.** In order of expected value: KV-cache reuse across
denoising steps via block decoding, int8 weight-only quantization with
`torch.compile`, and continuous batching to replace the current fixed-window
batcher. The `log_softmax` over a 50k vocabulary is the single largest tensor in
both the training and inference step and the obvious target.

**With 10x the compute:** a larger model on more tokens is the boring answer, and
it is the right one — 55M parameters at ~3.9B tokens is far from converged for a
diffusion LM, which gets a training signal only on masked positions and so needs
substantially more compute than an AR model for comparable quality. Beyond that:
validate FSDP at 8+ GPUs and publish a real scaling curve; sweep noise schedules
(cosine is implemented and has never been compared against linear on a trained
model); and characterize the NFE-quality frontier properly with enough samples
that MAUVE stops being directional.

**What I would not do:** custom kernels. MFU is 29.7%, and the wins available
from batching, caching, and quantization are larger and cheaper. Optimizing the
inner loop before exhausting the structural wins would be the wrong order.

## Numbers at v1.0.0

| | |
| --- | --- |
| Source files / tests | 43 / 180 passing |
| Type coverage | `mypy --strict`, zero errors |
| Test coverage | 81% |
| CI jobs | 7 across 4 workflows |
| Model | 55.5M params, 30k steps, 7.14 nats/token |
| Training efficiency | 29.7% MFU, single A6000 |
| Serving | ~15 req/s @ 16 users, p50 660 ms, 0 failures |
| Batching gain | ~12x over serialized |
| Autoscaling | verified 1 -> 5 replicas under load on k3d |
