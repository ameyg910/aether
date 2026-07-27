# ADR-0003: DDP as the default, FSDP when memory demands it

- **Status:** accepted
- **Date:** Week 5
- **Supersedes:** none

## Context

Aether must scale beyond one GPU. Two PyTorch-native data-parallel strategies are
available, and they solve different problems:

**DDP** replicates the model on every rank. Each rank runs forward and backward on
its own slice of the batch, then a single all-reduce averages gradients so every
replica applies an identical update. Communication is one all-reduce of size
`|params|` per optimizer step, overlapped with the backward pass. Per-GPU memory is
unchanged by adding GPUs: every rank still holds the full parameters, gradients,
and optimizer state.

**FSDP** shards parameters, gradients, and optimizer state across ranks. Each layer's
parameters are all-gathered immediately before use and freed immediately after;
gradients are reduce-scattered so each rank keeps only its shard. Per-GPU memory
falls roughly linearly with world size, at the cost of substantially more
communication — an all-gather per FSDP unit per forward *and* per backward, rather
than one all-reduce per step.

The memory that matters is not just weights. With AdamW in mixed precision, one
parameter costs roughly 4 bytes (fp32 master) + 4 (gradient) + 8 (two optimizer
moments) ≈ 16 bytes before activations. Aether's current 55M-parameter model is
therefore under 1 GB of state — trivial on a 48 GB A6000. Activations at
`batch x 1024 x d_model` per layer dominate instead, and the `[MASK]`-biased
`log_softmax` over a 50k vocabulary is the single largest tensor in the step.

## Decision

**Default to DDP; offer FSDP behind a config flag; make the choice a one-word
change.**

`train.strategy` accepts `auto | none | ddp | fsdp`. `auto` — the default —
resolves to `ddp` when `world_size > 1` and `none` otherwise. Requesting any
distributed strategy with a single process resolves to `none`.

FSDP uses a **size-based auto-wrap policy** (`train.fsdp_min_params`, default 1M):
any submodule holding at least that many parameters becomes its own FSDP unit, so
all-gathers are scoped to one block at a time instead of the whole model. Gradients
reduce in fp32 even when parameters are bf16, because sharded reductions accumulate
error and the bandwidth saving is not worth the precision loss.

**Activation checkpointing** (`train.activation_checkpointing`) is orthogonal to
both strategies and available to either. It recomputes each block's activations
during the backward pass instead of storing them — roughly 30% more compute for a
large activation-memory saving. It is on by default in the `fsdp` config group,
because FSDP is chosen under memory pressure and recompute is the other half of
that trade.

## Consequences

**Good**

- The common case is fast by default. At 55M parameters DDP's single all-reduce
  is strictly cheaper than FSDP's per-layer gathers; defaulting to FSDP would pay
  for memory savings the model does not need.
- One command runs everywhere. `auto` means the same invocation works on a laptop,
  three A6000s, and a SLURM allocation.
- Checkpoints are strategy-independent. Model and optimizer state are consolidated
  through `torch.distributed.checkpoint.state_dict`, so a checkpoint written under
  FSDP restores under DDP or single-process and vice versa. Nothing about the
  allocation leaks into the artifact.
- The growth path is a flag, not a rewrite. Scaling the model past one device is
  `train.strategy=fsdp`.

**Bad / accepted costs**

- Two code paths to keep working. FSDP is exercised far less than DDP, so it will
  rot faster.
- **FSDP is not covered by CI.** FSDP requires a real accelerator — it refuses to
  initialize on CPU — so unlike DDP (verified in CI with `gloo` worker processes)
  it can only be validated on GPU hardware. This is a genuine gap, not an
  oversight: it means FSDP regressions surface on the cluster rather than in a pull
  request.
- **In-loop sampling is disabled under FSDP.** With sharded parameters no single
  rank holds the full model, and gathering them mid-training to run the sampler
  (with an EMA weight swap on top) is fiddly enough that the current code skips it
  and returns an empty sample. Sample from a saved full checkpoint instead.
- EMA under FSDP tracks each rank's local shard rather than a consolidated copy.
  This is consistent across ranks and correct on resume, but it is not the same
  object as the single-process EMA, and the difference is easy to forget.

## Alternatives considered

- **FSDP everywhere.** Simpler in that there is one path, but it makes the common
  case slower for no benefit, and it would have disabled in-loop sampling — a
  useful training signal — at 55M parameters where memory is not the constraint.
- **DeepSpeed ZeRO.** Comparable sharding with more mature stage-3 offload, but it
  adds a large dependency and its own config system alongside Hydra. PyTorch-native
  FSDP keeps the dependency surface small, which matters for a portfolio project
  meant to be read.
- **Tensor / pipeline parallelism.** The right answer when a *single layer* exceeds
  one device. Aether is nowhere near that, and both add far more complexity than
  sharding the optimizer state.
- **FSDP2 (`fully_shard`).** The newer per-parameter-sharding API, and where PyTorch
  is heading. Deferred because the size-based auto-wrap policy this week calls for
  is a first-class FSDP1 concept, and FSDP1 remains supported. Worth revisiting.

## Notes for later

The scaling experiment (`scripts/scaling_plot.py`) reports tokens/s and MFU per
world size. If DDP scaling efficiency drops off before three GPUs, the first
suspects are per-rank batch size (too little work to hide the all-reduce) and
input-pipeline throughput — not the strategy choice.
