#!/usr/bin/env bash
#SBATCH --job-name=aether-fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# FSDP training on a SLURM cluster. Account/partition/QoS/GRES are NOT baked in --
# pass them at submit time so this template works on any cluster:
#
#   sbatch --account=<acct> --partition=<part> --qos=<qos> \
#          --gres=gpu:<type>:<n> scripts/launch/slurm_fsdp.sbatch
#
# Override run settings with environment variables:
#   NPROC=4 MODEL=medium sbatch ... scripts/launch/slurm_fsdp.sbatch
#
# Under preemption SLURM re-queues the job; RESUME points at the run directory and
# training continues from the last checkpoint, so requeueing is safe by default.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR"

# --- environment -------------------------------------------------------------
# Load whatever your site needs; both are no-ops if unset.
if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
  # shellcheck disable=SC1091
  module load ${MODULES_TO_LOAD}
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

NPROC="${NPROC:-${SLURM_GPUS_ON_NODE:-1}}"
STRATEGY="${STRATEGY:-fsdp}"
MODEL="${MODEL:-medium}"
DATA="${DATA:-wikitext103}"
RUN_NAME="${RUN_NAME:-fsdp-${NPROC}gpu-${SLURM_JOB_ID:-local}}"
OUT_DIR="${OUT_DIR:-$PWD/runs}"
TRACKING="${TRACKING:-jsonl}"
ACT_CKPT="${ACT_CKPT:-true}"

# Rendezvous: pick the first allocated node as master, and derive a port from the
# job id so concurrent jobs on the same node never collide.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)"
MASTER_PORT="$(( 20000 + (${SLURM_JOB_ID:-0} % 20000) ))"
export MASTER_ADDR MASTER_PORT

# --- provenance: the run must be reconstructable from the log ----------------
echo ">> job=${SLURM_JOB_ID:-none} node=$(hostname) gpus=${NPROC}"
echo ">> git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo ">> python=$(python -V 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')"
echo ">> master=${MASTER_ADDR}:${MASTER_PORT} strategy=${STRATEGY}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Resume automatically if a checkpoint already exists (preemption-safe).
RESUME_ARG=()
if [[ -f "${OUT_DIR}/${RUN_NAME}/checkpoints/latest.pt" ]]; then
  echo ">> resuming from ${OUT_DIR}/${RUN_NAME}/checkpoints/latest.pt"
  RESUME_ARG=(train.resume="${OUT_DIR}/${RUN_NAME}")
fi

srun torchrun \
  --nnodes="${SLURM_NNODES:-1}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_id="${SLURM_JOB_ID:-0}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m aether.train.cli \
    model="${MODEL}" data="${DATA}" \
    train.strategy="${STRATEGY}" \
    train.activation_checkpointing="${ACT_CKPT}" \
    train.run_name="${RUN_NAME}" \
    train.out_dir="${OUT_DIR}" \
    tracking.backend="${TRACKING}" \
    "${RESUME_ARG[@]}" \
    "$@"
