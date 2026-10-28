#!/bin/bash
# Setup for the nanochat-latent-think runs on a fresh GPU machine (see LATENT_THINK.md): the environment (nanochat's uv
# env plus flash-linear-attention for the KDA layers), the ClimbMix shards and the tokenizer, as runs/speedrun.sh does.
#   bash runs/latent_think_setup.sh            # SHARDS=170 by default; a d8 run reads ~0.5B tokens, far fewer shards
set -e
cd "$(dirname "$0")/.."
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv
uv sync --extra gpu
uv pip install --python .venv/bin/python flash-linear-attention==0.5.1 pytest # (KDA kernels; the tests)
.venv/bin/python -m nanochat.dataset -n 8
.venv/bin/python -m nanochat.dataset -n ${SHARDS:-170} &
download=$!
.venv/bin/python -m scripts.tok_train
.venv/bin/python -m scripts.tok_eval
wait $download
.venv/bin/python -m pytest tests/test_thinking.py tests/test_kda.py -q
echo "setup done: data and tokenizer in $NANOCHAT_BASE_DIR"
