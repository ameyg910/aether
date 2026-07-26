# Training guide

The trainer turns the model + loss into a tracked, checkpointed, resumable system:
bf16 AMP, gradient accumulation, cosine+warmup LR, gradient clipping, EMA, a
periodic sample callback, and experiment tracking.

## Launch

```bash
# real run on wikitext-103 (build the data first: make data)
aether-train                                   # uses configs/config.yaml defaults

# quick local smoke run on the byte-tokenized debug corpus (build: make data-debug)
make train-debug
# == aether-train train=debug data=local_debug tracking.backend=jsonl
```

Every run creates `runs/<name>/` containing `manifest.json` (git SHA, dataset hash,
seed, env), `config.yaml` (the exact resolved config), `metrics.jsonl` (or a W&B
run), and `checkpoints/` (`latest.pt` plus a rolling window of `step_*.pt`).

## Key knobs (Hydra overrides)

- `train.precision=bf16|fp16|fp32` — bf16 is the default on Ampere/Hopper.
- `train.batch_size`, `train.grad_accum` — effective batch = `batch_size * grad_accum`.
- `train.lr`, `train.warmup_steps`, `train.min_lr_ratio` — the schedule.
- `train.ema_decay` — EMA is used for all sampling/eval.
- `train.max_steps`, `train.log_every`, `train.sample_every`, `train.ckpt_every`.

## Experiment tracking

`tracking.backend` selects the sink:

- `jsonl` (default) — appends metrics to `runs/<name>/metrics.jsonl`; no network.
- `wandb` — install the extra (`pip install -e ".[tracking]"`), then
  `aether-train tracking.backend=wandb tracking.project=aether`. Log in first with
  `wandb login`.
- `none` — discard everything (used in tests).

Logged per step group: `loss`, `lr`, `grad_norm`, `steps_per_sec`, `seqs_per_sec`,
and a decoded `sample` every `sample_every` steps (generated from the **EMA**
weights).

## Resume

Runs are resumable byte-for-byte after preemption — checkpoints bundle model, EMA,
optimizer, scheduler, step, and all RNG states.

```bash
aether-train train.resume=runs/<name>/checkpoints/latest.pt
# or point at the run dir; it resolves checkpoints/latest.pt
aether-train train.resume=runs/<name>
```

The resume path is verified in CI: `tests/train/test_checkpoint.py::test_resume_is_bit_for_bit`
trains, checkpoints, and shows that continuing from the checkpoint produces
parameters identical to an uninterrupted run.

## Public dashboard

<!-- Paste your public W&B run URL here after the first A6000 run. -->
Public run: _TODO — link your W&B run after the first training._
