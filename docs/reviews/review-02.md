# Engineering Review #2 (end of Week 8)

Scope: the platform as of the first containerized release — trained model, served
API, reproducible build, automated CI/CD. This follows
[Review #1](review-01.md) at the end of Week 4.

## Strengths

- **Reproducible builds.** Multi-stage images keep the build toolchain out of the
  runtime layer; `uv.lock` pins the full transitive graph; the release workflow
  re-runs the entire gate before publishing, so no artifact reaches the registry
  without having passed it. `docker compose up` brings server, Prometheus, and a
  provisioned Grafana up on any machine.
- **Training is instrumented, not just executed.** Every run writes a manifest
  (git SHA, dataset hash, seed, environment) and a resolved config snapshot;
  checkpoints are world-size independent and resume bit-for-bit; MFU and
  throughput are reported alongside loss. The 55M run reached **29.7% MFU** on one
  A6000, which is a defensible number rather than a guess.
- **The model is measured.** A likelihood bound with stratified Monte Carlo
  sampling, MAUVE, diversity metrics, and an NFE-quality sweep. The reported
  7.14 nats/token is anchored against a theoretical baseline that CI checks on
  every commit.
- **Serving is production-shaped.** Dynamic batching (~12x throughput under 16
  concurrent users), separate liveness and readiness, Prometheus histograms rather
  than pre-computed quantiles, atomic model swap with in-memory rollback.
- **The test suite catches real bugs.** Multi-process DDP tests assert that ranks
  fed different data converge to identical parameters. The regression benchmark
  pins an untrained model's likelihood against `ln(vocab-1)` — an *external*
  anchor rather than a self-consistency check.

## Weaknesses

- **No orchestration.** `docker compose` is a single-host toy. There is no
  scheduler, no autoscaling, no rolling deploy, no service mesh, no ingress. A
  container dies and nothing brings it back beyond `restart: unless-stopped`.
  This is Week 9.
- **Dashboards are ad hoc.** One provisioned Grafana dashboard, no alerting rules,
  no SLOs, no recording rules. Nothing pages anyone when p95 doubles. The metrics
  exist; the operational layer on top of them does not.
- **FSDP remains unverified.** It cannot initialize on CPU, so unlike DDP it has
  no CI coverage and has never run on hardware. It is code-reviewed, not tested.
  Called out in ADR-0003 and still true.
- **Single-node everything.** The scaling experiment was never run: the shared
  A6000 box never had three free GPUs simultaneously. The DDP path is correct per
  CI but its scaling efficiency is unmeasured.
- **No GPU in CI.** Every automated test runs on CPU. bf16 kernels, CUDA
  allocator behaviour under memory pressure, and NCCL collectives are exercised
  only by hand.
- **Docker images are unvalidated in this environment.** They were written and
  linted but never built here; CI builds them on every push, which is where that
  gap actually closes.

## Technical debt

1. **Config schema sprawl.** `schemas.py` now holds six top-level dataclasses
   (`Model`, `Diffusion`, `Data`, `Train`, `Tracking`, `Eval`, `Serve`) in one
   file. `TrainConfig` alone has 27 fields spanning optimization, distribution,
   checkpointing, and sampling — concerns that change for unrelated reasons.
2. **Metrics are defined in two places.** `train/mfu.py` computes throughput for
   training; `serve/metrics.py` defines Prometheus series for inference. They
   share concepts (tokens/sec, latency) with no shared vocabulary, so the same
   quantity has two names.
3. **The serving/batcher boundary leaks.** `DynamicBatcher.submit` returns
   `(result, offset, count)` and the caller slices the tensor itself. The batcher
   knows about batch composition; the endpoint should not have to.
4. **The `Trainer` grew rather than shrank.** Review #1 committed to splitting it
   before Week 5. Only the precision extraction happened; distributed support,
   MFU, and checkpoint bookkeeping were then added on top. `fit()` is now ~70
   lines mixing the step loop with logging, sampling, and checkpoint policy.
5. **Version-specific behaviour is untested.** The batcher bug — builtin vs
   `asyncio.TimeoutError` — was introduced by a lint autofix that was correct for
   the configured target version and wrong for the deployment interpreter. CI
   tests 3.12 and 3.13; the failure was on 3.10.
6. **Two ways to load a model.** `benchmarks/nfe_quality.py` infers geometry from
   checkpoint shapes; `serve/registry.py` prefers the recorded config and falls
   back to inference. The inference path is a silent-wrong-answer risk for any
   checkpoint written before Week 7.
7. **Stale numbers in documentation.** The README carried CPU-sandbox batching
   figures until real A6000 measurements replaced them. Nothing prevents a
   recurrence; benchmark tables are hand-copied.

## Refactor plan (before Week 9)

1. **Split the config schema.** Break `schemas.py` into a package
   (`config/schemas/{model,data,train,serve}.py`), and decompose `TrainConfig`
   into `OptimizerConfig`, `DistributedConfig`, and `CheckpointConfig` composed
   into it. Keeps the flat Hydra override paths users already depend on while
   giving each concern a place to live.
2. **Consolidate metrics.** One `aether/metrics.py` owning metric *names*,
   units, and label conventions, with the training and serving layers importing
   from it. Kubernetes and alerting arrive next week and will hard-code these
   names; getting the vocabulary right now is much cheaper than renaming a series
   that dashboards already query.
3. **Tidy the serving/batcher boundary.** `submit()` should return the caller's
   own slice, not a tuple the caller has to index into. Removes tensor-slicing
   from the endpoint and makes the batcher independently substitutable — which
   matters if continuous batching replaces the current fixed-window design.

Deliberately deferred: splitting the `Trainer` (debt #4). It is the largest item
and the least urgent, since Weeks 9–10 touch serving and release rather than
training. Recording it as knowingly deferred rather than quietly dropping it,
which is what happened to it after Review #1.

## Metrics at review time

| | |
| --- | --- |
| Source files | 43 |
| Tests | 147 passing, 1 skipped |
| Type coverage | `mypy --strict`, zero errors |
| CI jobs | 3 (quality matrix, regression benchmark, image build) |
| Trained model | 55.5M params, 30k steps, 7.14 nats/token |
| Training efficiency | 29.7% MFU, single A6000 |
| Serving throughput | ~15 req/s at 16 concurrent users, p50 660 ms |
