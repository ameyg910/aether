#!/usr/bin/env bash
# Launch a single-node, multi-GPU DDP run with torchrun.
#
#   scripts/launch/torchrun_local.sh                       # all visible GPUs
#   NPROC=3 scripts/launch/torchrun_local.sh               # 3 GPUs
#   NPROC=1 scripts/launch/torchrun_local.sh train.lr=1e-4 # extra Hydra overrides
#
# Every setting is an environment variable with a default; nothing is hardcoded
# to one machine. Trailing arguments are passed through to Hydra untouched.
set -euo pipefail
cd "$(dirname "$0")/../.."

NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count() or 1)')}"
STRATEGY="${STRATEGY:-ddp}"
MODEL="${MODEL:-medium}"
DATA="${DATA:-wikitext103}"
RUN_NAME="${RUN_NAME:-ddp-${NPROC}gpu}"
OUT_DIR="${OUT_DIR:-runs}"
TRACKING="${TRACKING:-jsonl}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo ">> torchrun nproc_per_node=${NPROC} strategy=${STRATEGY} run=${RUN_NAME}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  -m aether.train.cli \
    model="${MODEL}" data="${DATA}" \
    train.strategy="${STRATEGY}" \
    train.run_name="${RUN_NAME}" \
    train.out_dir="${OUT_DIR}" \
    tracking.backend="${TRACKING}" \
    "$@"
