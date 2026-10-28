#!/bin/bash
# One base_train run of the nanochat-latent-think experiments on all local GPUs (see LATENT_THINK.md).
#   bash runs/latent_think.sh <run-name> [base_train args...]
#   bash runs/latent_think.sh d8-baseline --depth=8
#   bash runs/latent_think.sh d8-kda-hybrid --depth=8 --kda-hybrid --num-iterations=1920
# Environment (all optional):
#   NANOCHAT_BASE_DIR  data / tokenizer / checkpoints (default ~/.cache/nanochat, as nanochat; run runs/latent_think_setup.sh)
#   NPROC              GPUs to use (default: all visible)
#   DBS                rows of max-seq-len tokens per GPU per micro-step (default 16: fits d8 on 40 GB; FBT / thinking runs
#                      need 8; the total batch is unchanged, more micro-steps)
#   WANDB_PROJECT      wandb project (default nanochat-latent-think); WANDB=0 turns wandb off
#   PY                 python (default this repo's .venv)
# Log: runs/logs/<run-name>.log
set -e
name=$1; shift
cd "$(dirname "$0")/.."
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
nproc=${NPROC:-$(nvidia-smi -L | wc -l)}
run=$name; [ "${WANDB:-1}" = 0 ] && run=dummy
mkdir -p runs/logs
NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS} OMP_NUM_THREADS=1 \
  exec ${PY:-.venv/bin/python} -m torch.distributed.run --standalone --nproc_per_node=$nproc \
  -m scripts.base_train -- --device-batch-size=${DBS:-16} --run=$run --model-tag=$name \
  --wandb-project=${WANDB_PROJECT:-nanochat-latent-think} "$@" > runs/logs/$name.log 2>&1
