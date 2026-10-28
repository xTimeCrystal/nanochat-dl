# nanochat-latent-think

This branch (`thinking-tokens`) adds four things to nanochat's `gpt.py`. All are off by default, and each is a
`base_train` flag:

| Flag | What it does |
|---|---|
| `--fbt-passes=P` | Full-bandwidth transformer (FBT): P trunk passes. Pass k reads `norm(W_t e + W_h norm(h))`, where h is the previous pass's final state at the position before. h = 0 in pass 1 and at position 0. `W_t` starts at I and `W_h` at 0. Loss: the mean over the passes. |
| `--kda-hybrid` | Kimi K3's attention stack, with MHA in place of MLA. Three Kimi Delta Attention (KDA) layers, then one global attention layer, and the last layer always global. Attention layers use no positional encoding (NoPE), full context, and a sigmoid output gate. KDA layers use FLA's `chunk_kda` with K3's lower-bounded decay. `--kda-no-gate-params` drops K3's per-head scale A and bias b. |
| `--attn-gate` | K3's sigmoid output gate on every attention layer, with RoPE and windows unchanged. This is the control for the hybrid. |
| `--think-latents=L` (with `--fbt-passes>1`) | Thinking tokens: L latent positions after every real token, inside the FBT passes. `--think-schedule=stream\|grown`. `--think-random` draws each token's count from 0..L in training. `--think-eval-passes` sets the passes used at eval. |

Also new: `--seed` (weight init only; the data order is fixed).

## Setup and running

```bash
git clone -b thinking-tokens https://github.com/xTimeCrystal/nanochat-dl && cd nanochat-dl
bash runs/latent_think_setup.sh
```

The setup script does four things:
- builds the uv env and installs flash-linear-attention 0.5.1
- downloads the ClimbMix shards (SHARDS=170 by default)
- trains the tokenizer
- runs the CPU tests

To use wandb, log in first with `wandb login`, or set `WANDB=0` to turn it off.

`runs/latent_think.sh <name> [args]` runs one job on all local GPUs and logs to `runs/logs/<name>.log`. The d8
comparison runs all used the same 1,920 steps, i.e. 503M tokens at batch 262,144:

```bash
bash runs/latent_think.sh d8-baseline --depth=8
bash runs/latent_think.sh d8-attn-gate --depth=8 --attn-gate --num-iterations=1920
bash runs/latent_think.sh d8-fbt2 --depth=8 --fbt-passes=2
bash runs/latent_think.sh d8-kda-hybrid --depth=8 --kda-hybrid --num-iterations=1920
bash runs/latent_think.sh d8-kda-hybrid-noAb --depth=8 --kda-hybrid --kda-no-gate-params --num-iterations=1920
DBS=8 bash runs/latent_think.sh d8-kda-hybrid-fbt2 --depth=8 --kda-hybrid --fbt-passes=2 --num-iterations=1920
DBS=8 bash runs/latent_think.sh d8-kda-hybrid-think3r-p2 --depth=8 --kda-hybrid --fbt-passes=2 --think-latents=3 \
  --think-schedule=stream --think-random --num-iterations=1920
```

How to run them:
- **Matching steps:** `--num-iterations=1920` keeps the gated and hybrid runs, which have more parameters, on the
  baseline's tokens. Without it, nanochat's ratio-12 rule gives them more steps.
- **Memory:** `DBS=8` halves the rows per micro-step for the FBT runs. On A100-40GB they need it. The batch is
  unchanged.
- **Several runs in a row:** `bash runs/latent_think_queue.sh 0 "<name> <args>" "<name> <args>" ...`.

## Results so far

All runs: d8, 8×A100-40GB, val bpb on nanochat's val split.

| Run | Min val bpb, seeds 42 / 1 (mean) | Step time |
|---|---|---|
| baseline | 0.942947 / 0.942082 (0.94251) | 119 ms |
| attn-gate | 0.939331 / 0.937355 (0.93834) | 124 ms |
| fbt2 | 0.933988 / 0.934245 (0.93412) | 200 ms |
| kda-hybrid | 0.937923 / 0.937775 (0.93785) | 175 ms |
| kda-hybrid-noAb | 0.939532 / 0.941148 (0.94034) | 174 ms |
| kda-hybrid-fbt2 | 0.930770 / 0.929756 (0.93026) | 324 ms |
| kda-hybrid-think3r-p2 (3 thoughts at eval) | 0.942507 (one seed) | ~800 ms |

What these show:
- **The hybrid's gain is mostly the output gate.** Gated attention alone accounts for it, and the hybrid is level
  with gated attention, at 0.29× lower throughput.
- **CORE is noisy at d8.** Every CORE difference so far comes from BoolQ.
- **Thinking run, by thoughts per token** (eval only, 2 passes):

  | Thoughts per token | 0 | 1 | 2 | 3 | random 0–3 |
  |---|---|---|---|---|---|
  | Val bpb | 0.9890 | 0.9386 | **0.9355** | 0.9425 | 0.9367 |

  It is best at 2 thoughts, but still 0.005 behind the same stack without thoughts.
- **Long context:** the KDA hybrid's loss rises past its training length on 65k-token rows. Training at 4,096 tokens
  only moves the break to 4,096. Resetting the KDA state every training length (eval only) removes almost all of it.

## Tests

`python -m pytest tests/test_thinking.py tests/test_kda.py` (CPU). On a GPU, `test_kda.py` also checks `chunk_kda`
against an fp64 recurrence.
