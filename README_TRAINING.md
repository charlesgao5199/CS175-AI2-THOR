# Method 1 — PPO Training (CUDA / A100)

End-to-end recurrent PPO on iTHOR ObjectNav, defined in
[`src/method1/policy.py`](src/method1/policy.py) and trained by
[`scripts/train_method1.py`](scripts/train_method1.py).

Architecture (CS 175 proposal):

```
[RGB-D 224x224] ─► ResNet-18 (4-channel)  ┐
[target id]   ─►  Embedding(32)            ├─► Linear(560→512) ─► GRU(512)
[heading]     ─►  Linear(2→16)            ┘                               │
                                                                          ├─► Linear(512→6)   policy
                                                                          └─► Linear(512→1)   value
```

Reward: `+10` success, `-0.5` wrong STOP, `-0.01` step. Max 500 steps.
Scenes: FloorPlan1–20. Targets: Mug, Apple, Bowl, Laptop, Television.

---

## RunPod / Lambda — one-time setup

We've tested on **A100 80GB** with a CUDA 12.1 image. Pick an Ubuntu 22.04
template with PyTorch pre-installed if possible — saves ~10 min on first boot.

```bash
# 1. SSH in, then check the GPU
nvidia-smi

# 2. System libs AI2-THOR needs for off-screen rendering
sudo apt-get update
sudo apt-get install -y \
    xvfb ffmpeg \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libgl1 libgl1-mesa-dri libglib2.0-0 \
    libx11-6 libxext6 libxrender1 libxtst6 \
    libxi6 libxrandr2 libxcursor1 libxdamage1 \
    libxfixes3 libxcomposite1 libasound2 libnss3 libgbm1

# 3. Clone repo
git clone https://github.com/charlesgao5199/CS175-AI2-THOR.git ~/CS175-AI2-THOR
cd ~/CS175-AI2-THOR

# 4. Python env — use venv to avoid messing with the system interpreter
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 5. Install deps. Use the CUDA-specific index for torch wheels.
pip install -r requirements_gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 6. Smoke-test the simulator before training (uses xvfb-run to give it a
#    fake display)
xvfb-run -a python scripts/smoke_test_ai2thor.py --platform default
```

If smoke-test fails with a Vulkan/GL error, try `--platform cloud` (uses
AI2-THOR's CloudRendering backend, no X server needed):

```bash
python scripts/smoke_test_ai2thor.py --platform cloud
```

---

## Train

Full 2M-step run on a single A100 (≈ 8–12 hours):

```bash
# Foreground (you can detach with screen/tmux):
xvfb-run -a python scripts/train_method1.py \
    --total-steps 2000000 \
    --num-envs 4 \
    --rollout-steps 128 \
    --platform default

# Or with CloudRendering, no xvfb needed:
python scripts/train_method1.py \
    --total-steps 2000000 \
    --num-envs 4 \
    --platform cloud
```

Useful flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--total-steps`         | 2_000_000   | Env steps to train for. |
| `--num-envs`            | 4           | Parallel AI2-THOR controllers. Each one is ~500 MB RAM. |
| `--rollout-steps`       | 128         | Steps per env per PPO update. T·N transitions per update. |
| `--k-epochs`            | 4           | PPO update epochs over each rollout chunk. |
| `--num-minibatches`     | 2           | Minibatches per epoch (across envs, not time). |
| `--lr`                  | 3e-4        | Adam learning rate. |
| `--gamma`               | 0.99        | Discount. |
| `--gae-lambda`          | 0.95        | GAE λ. |
| `--clip-eps`            | 0.2         | PPO clip ε. |
| `--ent-coef`            | 0.01        | Entropy bonus coefficient. |
| `--checkpoint-interval` | 100_000     | Env steps between checkpoint saves. |
| `--log-interval`        | 10_000      | Env steps between CSV log rows. |
| `--no-pretrained`       | (off)       | Skip ImageNet ResNet18 init (slower convergence). |
| `--device`              | auto        | `auto` picks CUDA → MPS → CPU. |

### Outputs

```
checkpoints/method1/
  latest.pt                  # always overwritten — used by --resume
  step_000100000.pt          # 100K-step snapshot
  step_000200000.pt
  ...
logs/method1/
  training_log.csv           # 10K-step rows
  curves.png                 # produced by plot_training.py
```

The CSV columns are:

```
step,update,episodes,mean_reward,success_rate,episode_len,
value_loss,policy_loss,entropy,lr,wall_time_s
```

### Resume

Resume happens automatically — `latest.pt` is always loaded if present:

```bash
# Same command. If interrupted, picks up where it left off.
xvfb-run -a python scripts/train_method1.py --total-steps 2000000
```

To force a clean start, pass `--no-resume` (or delete `checkpoints/method1/latest.pt`).

---

## Monitor

In another shell:

```bash
# Live tail
tail -f logs/method1/training_log.csv

# Plot to PNG (after a few hundred K steps in)
python scripts/plot_training.py \
    --log logs/method1/training_log.csv \
    --out logs/method1/curves.png \
    --smooth-window 10
```

`curves.png` is a 2×3 grid: mean reward / success rate / episode length on
the top row; policy loss / value loss / entropy on the bottom.

---

## Plug back into the eval pipeline

The trained policy is loadable by
[`src/method1/navigator.py`](src/method1/navigator.py):

```python
from method1 import Method1Navigator

nav = Method1Navigator("checkpoints/method1/latest.pt")
# Drop into eval.runner.EpisodeRunner or scripts/run_small_eval.py just like
# Method2Navigator / Method3Navigator.
```

---

## Troubleshooting

**`InitialRandomSpawn` errors / `lastActionSuccess: False`** — some
FloorPlans don't support every randomization action. The trainer ignores
these silently; episodes still run.

**`RuntimeError: CUDA out of memory`** — drop `--num-envs` to 2, or
`--rollout-steps` to 64. Each rollout chunk is roughly
`T × N × (3 × 224 × 224 uint8 + 1 × 224 × 224 fp32)` ≈ `T × N × 350 KB`
in CPU RAM plus encoder activations on the GPU during the update.

**Episodes never end** — the env caps each episode at 500 steps even if
the policy never emits STOP, so training never deadlocks.

**Spawn worker crashes on first import** — Python `multiprocessing.spawn`
re-imports the script in each worker. Anything heavy (e.g., torch GPU
warmup) needs to be guarded by `if __name__ == "__main__":` — already the
case in `train_method1.py`.

**Headless rendering errors** — try `--platform cloud`, otherwise wrap the
invocation in `xvfb-run -a`. On RunPod, the cloud renderer is usually
faster and more stable.
