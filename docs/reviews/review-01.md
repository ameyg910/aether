# Engineering Review #1 (end of Week 4)

Scope: the system as of the first tracked training run — typed model, SUBS loss,
data pipeline, and the Week-4 trainer.

## Strengths (production-grade)

- **Type safety end to end.** `mypy --strict` is clean across all source files,
  including the torch code; the pre-commit mypy hook runs in the project env so it
  matches CI exactly (no isolated-env version skew).
- **Reproducibility.** Every run writes a manifest (git SHA, dataset hash, seed,
  env) and a snapshot of the resolved config; seeding covers python/numpy/torch/cuda.
- **Resumability is real and tested.** Checkpoints bundle model + EMA + optimizer +
  scheduler + step + RNG state and restore bit-for-bit; a CI test proves a resumed
  run matches an uninterrupted one exactly.
- **Correctness proven, not assumed.** The loss has a numerical-equivalence test and
  a gradient check; the overfit collapses to ~0.
- **Decoupled, testable trainer.** The loop consumes a batch iterator and a tracker
  protocol, so it runs on CPU with synthetic data in CI and on real shards in prod.

## Weaknesses (fragile)

- **Single-GPU only.** No DDP/FSDP; no sharded optimizer state. Won't scale past one
  device — addressed in Week 5.
- **No evaluation harness.** We track training loss and eye-ball samples; there is no
  held-out perplexity/NLL metric or generation-quality eval — Week 6.
- **The sampler is a stub.** `ancestral_sample` is minimal and slow; sample quality
  during training is indicative, not trustworthy.
- **fp16 path is untested on hardware.** The GradScaler branch exists but only bf16
  and fp32 are exercised on CPU; fp16 needs a GPU smoke test.

## Technical debt

- Device/precision resolution lives inline in the `Trainer`; it should be a small
  reusable helper (also needed by eval and sampling).
- The `Trainer` is edging toward a God-object (owns optimizer, scheduler, EMA,
  scaler, tracking, sampling, checkpoint policy).
- Config resolution is split: `load_config()` for the library, `sys.argv` parsing in
  the CLI; these should share one path.
- `throughput` is reported in sequences/sec, not tokens/sec (the trainer doesn't
  know `block_size`); fine for now but worth threading through.
- No test asserts the manifest contents (git SHA / dataset hash capture).

## Refactoring plan (before Week 5)

1. **Extract device/precision handling** into `aether/train/precision.py` (resolve
   device, AMP dtype, autocast context, scaler) so the trainer, eval, and sampler
   share one implementation.
2. **Unify config resolution.** Have the CLI go through `load_config()` and pass a
   single typed `AetherConfig` everywhere, instead of re-parsing argv.
3. **Split the trainer God-object.** Pull checkpoint policy (rolling retention,
   naming) and the sample/log callbacks out of the loop into small collaborators,
   leaving `Trainer.fit` as just the step loop — this is the seam distributed
   training (Week 5) will need.

## Stretch (not blocking)

- Hyperparameter sweep over `lr`/`warmup_steps` (Hydra multirun or W&B Sweeps),
  logging the comparison.
