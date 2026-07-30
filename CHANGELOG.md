# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Release notes on GitHub are generated automatically from
[Conventional Commit](https://www.conventionalcommits.org/) messages; this file is
the curated, human-facing history. Both exist because generated notes are complete
but undifferentiated — they cannot tell you which of forty commits actually
changes how you use the project.

> **Note on versions below 0.8.0.** Aether was built in weekly milestones, and
> those milestones are recorded here because they are the project's real history.
> They were not published as tagged releases — `v0.8.0` is the first tag, and the
> first version with container images in the registry. Comparison links for
> earlier entries therefore point at the repository rather than at releases.

## [1.0.0] — Week 10

First stable release. The public API — HTTP contract, CLI entry points, Hydra
config keys, checkpoint format, and Prometheus metric names — is now covered by
semantic versioning.

### Added
- Model and generated model card published to the Hugging Face Hub, with an
  immutable revision tag so `hf:owner/repo@v1.0.0` is a reproducible deploy target.
- Public Gradio Space demo showing the denoising process and the NFE knob.
- MkDocs Material documentation site published to GitHub Pages, built with
  `--strict` so broken links fail the build.
- `make train-toy`: reproduces a full train-and-evaluate cycle from a clean clone
  with no GPU and no downloads.
- `CITATION.cff`, `ROADMAP.md`, and labelled good-first-issue template.
- Final Engineering Review (`docs/reviews/review-final.md`).

### Fixed
- The `hf:` version-tag path in the model registry had never been executed — every
  test used `local:` paths. Now covered, including revision parsing, explicit
  filenames, and rollback between pinned Hub revisions.

## [Unreleased]

### Added
- Multi-stage Docker images for serving and training, non-root, with the build
  toolchain excluded from the runtime layer.
- `docker compose up` stack: inference server, Prometheus, and a provisioned
  Grafana dashboard.
- CI matrix across Python 3.12/3.13 with coverage, plus image build and smoke test.
- Release workflow publishing versioned images to GHCR on `v*` tags, gated on the
  full test suite and the regression benchmark.
- `uv.lock` for reproducible dependency resolution.
- Engineering Review #2 (`docs/reviews/review-02.md`).

### Fixed
- Checkpoint pruning deleted checkpoints belonging to *other* runs, and — when a
  run started in a directory holding higher-numbered checkpoints — deleted the
  checkpoint it had just written, because retention was computed from a directory
  glob sorted by step number. Pruning is now scoped to the files the current run
  wrote.

## [0.7.0] — Week 7

### Added
- FastAPI inference service: `/generate`, SSE streaming of the denoising process,
  `/health`, `/ready`, `/metrics`, and admin swap/rollback.
- Async dynamic batcher merging concurrent requests into single forward passes;
  measured ~12x throughput under 16 concurrent users on one A6000.
- Model registry loading checkpoints by version tag from local paths or the HF
  Hub, with atomic swap and instant rollback.
- Prometheus instrumentation: latency histograms, queue depth, batch size,
  throughput, errors.
- Load-test driver (`loadtest/locustfile.py`) and client example.

### Fixed
- The batcher caught only the builtin `TimeoutError`. On Python 3.10
  `asyncio.TimeoutError` is a *distinct* class, so `wait_for` timeouts escaped,
  killed the batcher loop, and every subsequent request hung forever with no
  error. Both classes are now caught, and a failure while gathering a batch
  serves the requests already held instead of dropping their futures.
- Checkpoints now record the model architecture. Head count leaves no trace in
  any parameter shape, so a checkpoint without it could be loaded into a
  differently-shaped model that silently computed something else.

## [0.6.0] — Week 6

### Added
- Evaluation harness: diffusion NELBO (nats/token, bits-per-dim, perplexity bound)
  with stratified time sampling, MAUVE via the divergence frontier, and diversity
  metrics.
- Confidence-based parallel sampler alongside the ancestral sampler, both
  reporting NFE.
- NFE-quality benchmark sweeping the compute/quality curve.
- Pinned regression benchmark wired into CI.

### Fixed
- MAUVE returned 0.0 for identical distributions. The frontier collapses to a
  single point when P equals Q, and the area under a zero-width curve is zero;
  the curve is now anchored with its two limiting points.
- The SUBS loss summed cross-entropy over masked positions instead of averaging,
  inflating loss and gradients by roughly the sequence length. Gradient clipping
  therefore throttled nearly every update.

## [0.5.0] — Week 5

### Added
- DDP and FSDP behind a single `train.strategy` flag, with size-based FSDP
  auto-wrap and optional activation checkpointing.
- World-size-independent checkpoints: a checkpoint from a 3-GPU run restores into
  a single process.
- Throughput and MFU reporting; scaling-experiment tooling.
- `torchrun` and SLURM launch templates.

### Fixed
- RNG state is stored on CPU so checkpoints restore across devices.

## [0.4.0] — Week 4

### Added
- Config-driven `Trainer`: bf16 AMP, gradient accumulation, cosine+warmup LR,
  gradient clipping, EMA.
- Resumable checkpointing bundling model, EMA, optimizer, scheduler, step, and
  RNG state, with a bit-for-bit resume test.
- Pluggable experiment tracking (`jsonl` / `wandb` / `none`) and run manifests.

## [0.3.0] — Week 3

### Added
- Bidirectional DiT denoiser with AdaLN-Zero time conditioning.
- MDLM/SUBS training loss, verified against `F.cross_entropy` and `gradcheck`.

## [0.2.0] — Week 2

### Added
- Data pipeline: GPT-2 and byte tokenizers, sequence packing, sharded `uint16`
  memmap datasets with a content fingerprint.

## [0.1.0] — Week 1

### Added
- Typed Hydra configuration system, noise schedules, and the absorbing-state
  forward process.

[Unreleased]: https://github.com/ameyg910/aether/compare/main...HEAD
[0.7.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.6.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.5.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.4.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.3.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.2.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
[0.1.0]: https://github.com/ameyg910/aether  # weekly milestone, not tagged
