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
- `train.grad_clip` — max gradient norm (see the note below on loss scale).
- `train.ema_decay` — EMA is used for all sampling/eval.
- `train.max_steps`, `train.log_every`, `train.sample_every`, `train.ckpt_every`.
- `train.out_dir` — where runs are written (set this to a persistent path on a cluster).

## Experiment tracking

`tracking.backend` selects the sink:

- `jsonl` (default) — appends metrics to `runs/<name>/metrics.jsonl`; no network.
- `wandb` — install the extra (`pip install -e ".[tracking]"`), then
  `aether-train tracking.backend=wandb tracking.project=aether`. Authenticate with
  `wandb login`, or set `WANDB_API_KEY` in the environment (better on shared machines).
- `none` — discard everything (used in tests).

Logged per step group: `loss`, `lr`, `grad_norm`, `steps_per_sec`, `seqs_per_sec`,
and a decoded `sample` every `sample_every` steps (generated from the **EMA** weights).

## Resume

Runs are resumable byte-for-byte after preemption — checkpoints bundle model, EMA,
optimizer, scheduler, step, and all RNG states (stored on CPU so a checkpoint written
on one device restores cleanly on another).

```bash
aether-train train.resume=runs/<name>/checkpoints/latest.pt
# or point at the run dir; it resolves checkpoints/latest.pt
aether-train train.resume=runs/<name>
```

The resume path is verified in CI: `tests/train/test_checkpoint.py::test_resume_is_bit_for_bit`
trains, checkpoints, and shows that continuing from the checkpoint produces parameters
identical to an uninterrupted run.

## Running on a shared cluster (SLURM)

Lessons from the reference run on an H100 MIG slice:

- **Write to persistent storage.** A compute node's local `runs/` may vanish when the job
  ends. Point `train.out_dir` at your home/scratch, e.g.
  `train.out_dir=$HOME/runs`.
- **Wrap the job in `tmux` on the *login* node**, then `srun` inside it, so an SSH
  disconnect never kills training. Detach with `Ctrl-B D`; reattach with `tmux attach`.
- **Authenticate W&B via env var**, not `wandb login` (which needs a writable `$HOME`):
  `export WANDB_API_KEY=...`.
- **Use a unique `WANDB_RUN_ID` per run**, and reuse the *same* id (with
  `WANDB_RESUME=allow`) only when genuinely continuing that run from a checkpoint — a
  reused id on a fresh run silently discards metrics until the step counter catches up.

Reference launch (55.5M model, single 12 GB MIG slice):

```bash
export WANDB_API_KEY=...          # from wandb.ai/authorize
export WANDB_RUN_ID=aether-55m-v2
export WANDB_RESUME=allow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

aether-train \
  model=medium data=wikitext103 \
  tracking.backend=wandb tracking.project=aether \
  train.run_name=aether-55m-v2 \
  train.max_steps=15000 \
  train.batch_size=4 train.grad_accum=8 \
  train.lr=1e-4 train.warmup_steps=200 train.min_lr_ratio=0.1 \
  train.precision=bf16 train.ema_decay=0.9999 \
  train.log_every=50 train.sample_every=1000 train.ckpt_every=1000 train.keep_last=3 \
  train.sample_length=128 train.sample_steps=128 \
  train.out_dir=$HOME/runs
```

## A note on loss scale

The SUBS loss currently **sums** cross-entropy over masked positions rather than averaging,
so both the loss and its gradients scale with the number of masked tokens (~sequence
length). Practical consequences: the loss reads in the thousands rather than the familiar
~7 nats/token, and `grad_norm` is correspondingly large — the default `grad_clip=1.0` is
far too aggressive at this scale, so raise it (e.g. `train.grad_clip=1000`) or it will
throttle nearly every update. Normalizing the loss per masked token is queued as a
refactor; it will restore the conventional scale, make `grad_clip=1.0` meaningful, and
make numbers comparable to published MDLM results.

## Public dashboard

Public run: https://wandb.ai/f20240973-bits-pilani/aether — the `aether-55m-v2` run shows
the loss curve descending, text samples logged every 1000 steps, and a live
resume-from-checkpoint.
