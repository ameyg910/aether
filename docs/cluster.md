# Reproducing Aether training on a cluster

Aether runs the same entry point single-process, multi-GPU, and multi-node. Rank
and world size come from environment variables that `torchrun` (or SLURM) sets, so
nothing in the code needs to know how it was launched:

```bash
aether-train                                   # 1 process
torchrun --nproc_per_node=3 -m aether.train.cli train.strategy=ddp
sbatch --account=... --partition=... scripts/launch/slurm_fsdp.sbatch
```

## Choosing a strategy

`train.strategy` accepts `auto` (default), `none`, `ddp`, or `fsdp`.

| | DDP | FSDP |
| --- | --- | --- |
| What each rank holds | full model, grads, optimizer state | a *shard* of each |
| Memory per GPU | ~unchanged as GPUs are added | falls ~linearly with world size |
| Communication | one gradient all-reduce per step | all-gather params per layer, plus reduce-scatter |
| Use when | the model fits comfortably on one device | it does not |

`auto` resolves to `ddp` when there is more than one rank and `none` otherwise, so
one command works everywhere. Asking for `ddp`/`fsdp` with a single process
resolves to `none` — wrapping a lone process adds overhead and buys nothing.
See [ADR-0003](adr/0003-ddp-vs-fsdp.md) for the reasoning.

## Local multi-GPU (DDP)

```bash
NPROC=3 scripts/launch/torchrun_local.sh
NPROC=3 scripts/launch/torchrun_local.sh train.lr=1e-4 train.max_steps=2000
```

Every knob is an environment variable with a default (`NPROC`, `STRATEGY`,
`MODEL`, `DATA`, `RUN_NAME`, `OUT_DIR`, `TRACKING`, `MASTER_PORT`); trailing
arguments pass through to Hydra untouched.

On a *shared* box, pin yourself to the idle GPUs first — otherwise you will land
on a device someone else is already filling:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 scripts/launch/torchrun_local.sh
```

## SLURM (FSDP)

Partition, account, QoS, and GRES are deliberately **not** baked into the sbatch
file; pass them at submit time so the template is portable:

```bash
sbatch --account=<account> --partition=<partition> --qos=<qos> \
       --gres=gpu:<type>:<n> \
       scripts/launch/slurm_fsdp.sbatch
```

Environment variables tune the run without editing the file:

```bash
NPROC=4 MODEL=medium ACT_CKPT=true OUT_DIR=$HOME/runs \
VENV_PATH=$HOME/venv MODULES_TO_LOAD="cuda/12.4" \
sbatch --account=... --partition=... scripts/launch/slurm_fsdp.sbatch
```

The script derives a rendezvous endpoint from the first allocated node and a port
from the job id (so concurrent jobs never collide), prints the git SHA, Python and
torch versions, and the GPU inventory into the job log, and **auto-resumes**: if
`$OUT_DIR/$RUN_NAME/checkpoints/latest.pt` exists it restarts from there. That
makes a requeued (preempted) job safe to run unchanged.

### Cluster gotchas worth knowing

- **Write checkpoints to persistent storage.** A compute node's local disk may
  vanish when the job ends. Set `train.out_dir=$HOME/runs` (or your scratch path).
- **Wrap the job in `tmux` on the *login* node**, then `srun`/`sbatch` inside it,
  so an SSH disconnect never kills an interactive run.
- **Authenticate W&B via `WANDB_API_KEY`**, not `wandb login`, which needs a
  writable `$HOME` that shared clusters often do not give you.
- **Use one `WANDB_RUN_ID` per run.** Reusing an id for a *fresh* run makes W&B
  silently discard metrics until the step counter passes the old maximum.

## Checkpoints are world-size independent

Model and optimizer state go through `torch.distributed.checkpoint.state_dict`,
which consolidates FSDP shards and strips DDP's `module.` prefix. A checkpoint
written by a 3-GPU FSDP job therefore restores into a single-process run, and vice
versa — you are never locked into the allocation you started with.

Gathering that state is a *collective*: every rank calls `save_checkpoint`, and
only rank 0 writes the file. This is verified in CI by
`tests/train/test_distributed_ddp.py`, which spawns real worker processes over
`gloo`, trains them on **different** data, and asserts their parameters end
identical (proving gradients synchronized) and that the resulting checkpoint loads
into a lone process.

Resume as usual:

```bash
NPROC=3 scripts/launch/torchrun_local.sh train.resume=runs/ddp-3gpu
```

## Throughput and MFU

The trainer logs `tokens_per_sec` and `mfu` alongside loss. MFU (Model FLOPs
Utilization) is the fraction of the hardware's theoretical peak the run actually
uses, counting

```
flops_per_token = 6 * N + 12 * n_layers * d_model * seq_len
```

`6N` is the parameter matmuls (2 FLOPs/param forward, ~twice that backward); the
second term is attention, which grows with sequence length rather than parameter
count. Aether's denoiser is bidirectional, so the full score matrix is computed.

The MFU denominator is dense bf16 tensor-core peak, looked up from the CUDA device
name. For an unrecognised device — **including MIG slices**, which expose a
fraction of a GPU — set it explicitly, or MFU is omitted rather than reported
wrongly:

```bash
aether-train train.device_peak_tflops=77.4   # e.g. a half-size A6000 slice
```

## Scaling experiment

Run the same configuration at several world sizes, then summarize:

```bash
for n in 1 2 3; do
  NPROC=$n RUN_NAME=ddp-${n}gpu scripts/launch/torchrun_local.sh train.max_steps=200
done

python scripts/scaling_plot.py runs/ddp-1gpu runs/ddp-2gpu runs/ddp-3gpu \
  --out docs/assets/scaling.png
```

This prints a throughput table (tokens/s, speedup, scaling efficiency, MFU,
estimated time to a token budget) and writes the scaling plot. Warmup logs are
discarded before averaging, since the first steps include CUDA context creation
and allocator warmup.

### Reading the result

Scaling efficiency below 100% is normal and worth being able to explain. The usual
causes, roughly in order:

- **Gradient all-reduce** — communication grows with world size while per-GPU
  compute stays fixed. Larger per-rank batches amortize it.
- **Stragglers** — the step takes as long as the slowest rank, so a shared GPU or
  a noisy neighbour drags the whole job.
- **Small per-rank batches** — too little work per GPU to hide the collective.
- **Data loading** — if the input pipeline cannot feed N ranks, GPUs idle.

Gradient accumulation is already communication-aware: under DDP only the final
micro-batch of each optimizer step triggers the all-reduce (`no_sync()` suppresses
it for the rest), turning `grad_accum` all-reduces per step into one.
