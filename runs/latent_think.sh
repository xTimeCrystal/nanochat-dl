#!/bin/bash
# nanochat-latent-think (wandb project) on the 8x A100 box: one base_train run on all 8 GPUs.
#   bash runs/latent_think.sh <run-name> [base_train args...]
#   bash runs/latent_think.sh d8-baseline --depth=8
# NANOCHAT_BASE_DIR is this project's own (checkpoints apart from the other sessions'), its data, tokenizer and eval
# bundle symlinked from the box's shared ~/.cache/nanochat (read only). The python env is the box's nanochat-dl venv
# (/workspace/nanochat_dl_val/.venv: the same pyproject.toml / uv.lock as this repo). 16 rows x 2048 per GPU = one
# micro-step of the default 262,144-token batch. Log: /root/scaling_laws/runs/nanochat_latent_think/<run>.log (pulled).
set -e
name=$1; shift
base=/root/nanochat_latent_think
mkdir -p $base /root/scaling_laws/runs/nanochat_latent_think
for d in base_data_climbmix tokenizer eval_bundle; do [ -e $base/$d ] || ln -s /root/.cache/nanochat/$d $base/$d; done
cd /root/scaling_laws/nanochat-dl
NANOCHAT_BASE_DIR=$base NCCL_P2P_LEVEL=SYS OMP_NUM_THREADS=1 \
  exec /workspace/nanochat_dl_val/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.base_train -- --device-batch-size=16 --run=$name --model-tag=$name --wandb-project=nanochat-latent-think "$@" \
  > /root/scaling_laws/runs/nanochat_latent_think/$name.log 2>&1
