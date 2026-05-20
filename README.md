# CS175-AI2-THOR

Object-goal navigation experiments in AI2-THOR / ProcTHOR for CS 175.

The current project focus is **Method 1: end-to-end recurrent PPO training**.
Earlier random, heuristic, and coverage agents remain in the repo as reference
baselines and pipeline sanity checks.

## Methods

| Method | Main files | Status |
| --- | --- | --- |
| Method 1: recurrent PPO | `scripts/train_method1.py`, `src/method1/policy.py`, `src/method1/navigator.py` | Current training focus. |
| Method 2: semantic mapping + planning | `src/method2/`, `src/mapping/` | Implemented for small evaluation. |
| Method 3: semantic mapping + LLM guidance | `src/method3/`, `src/mapping/` | Implemented for small evaluation; requires an LLM API key. |
| Reference baselines | `scripts/run_*_agent.py`, `scripts/evaluate_*_agent.py` | Used to validate simulator, evaluation, visualization, and failure inspection. |

## Method 1 Training

Method 1 trains a neural policy directly from AI2-THOR interaction.

Inputs per step:

- RGB image
- depth image
- target object id, such as `Mug` or `Apple`
- heading/compass

Actions:

- `MoveAhead`
- `RotateLeft`
- `RotateRight`
- `LookUp`
- `LookDown`
- `STOP`

Reward:

- `+10` for successful `STOP` near the target
- `-0.5` for wrong `STOP`
- `-0.01` per step
- max episode length: 500 steps

Architecture:

```text
RGB-D 224x224 -> ResNet18
target id     -> embedding
heading       -> linear projection
combined      -> GRU memory
GRU output    -> policy head + value head
```

The trainer includes local stability fixes for long WSL/GPU runs:

- non-finite RGB-D/compass values are sanitized
- non-finite PPO loss/gradients are skipped
- AI2-THOR worker reset/step failures restart that worker's controller
- `--worker-start-delay` staggers simulator startup on local GPUs

## Environment Setup

### Windows WSL + RTX 2080

Use WSL2 with Ubuntu 22.04. From PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
wsl -d Ubuntu-22.04
```

Inside Ubuntu, confirm the GPU is visible:

```bash
nvidia-smi
```

Install system packages needed by AI2-THOR rendering:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  libvulkan1 vulkan-tools mesa-vulkan-drivers mesa-utils \
  libgl1 libgl1-mesa-dri libglib2.0-0 libgtk-3-0 \
  libx11-6 libxext6 libxrender1 libxtst6 libxi6 libxrandr2 \
  libxcursor1 libxdamage1 libxfixes3 libxcomposite1 \
  libasound2 libnss3 libgbm1
```

Create and activate the conda environment:

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
```

Install simulator and GPU training dependencies:

```bash
cd /mnt/c/Users/Charl/Desktop/AI2-THOR/CS175-AI2-THOR
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Confirm PyTorch can use the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
True
NVIDIA GeForce RTX 2080
```

Run the smoke test:

```bash
python scripts/smoke_test_ai2thor.py --platform default
```

Expected output includes:

```text
AI2-THOR smoke test passed with platform=default
frame_shape=(300, 300, 3)
depth_shape=(300, 300)
```

### RunPod / Lambda A100

For cloud training, an A100 80GB with a CUDA 12.1 Ubuntu 22.04 image is a good
target. A PyTorch-preinstalled image saves setup time.

```bash
nvidia-smi

sudo apt-get update
sudo apt-get install -y \
  xvfb ffmpeg \
  libvulkan1 vulkan-tools mesa-vulkan-drivers \
  libgl1 libgl1-mesa-dri libglib2.0-0 \
  libx11-6 libxext6 libxrender1 libxtst6 \
  libxi6 libxrandr2 libxcursor1 libxdamage1 \
  libxfixes3 libxcomposite1 libasound2 libnss3 libgbm1

git clone https://github.com/charlesgao5199/CS175-AI2-THOR.git ~/CS175-AI2-THOR
cd ~/CS175-AI2-THOR

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_gpu.txt --extra-index-url https://download.pytorch.org/whl/cu121

xvfb-run -a python scripts/smoke_test_ai2thor.py --platform default
```

If the default renderer fails on a headless cloud machine, try:

```bash
python scripts/smoke_test_ai2thor.py --platform cloud
```

### macOS

macOS can run simulator smoke tests and lightweight baselines with a native
Miniforge environment:

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
pip install ai2thor==5.0.0 procthor==0.0.1.dev2
python scripts/smoke_test_ai2thor.py --platform default
```

Apple Silicon Macs may need an x86_64/Rosetta environment or a Linux GPU
machine for simulator-heavy training.

## Training Commands

### Local RTX 2080 / WSL

The RTX 2080 can run 4 environments for short tests, but long runs are more
stable with 2 environments.

```bash
python scripts/train_method1.py \
  --device cuda \
  --platform default \
  --total-steps 2000000 \
  --num-envs 2 \
  --rollout-steps 128 \
  --lr 0.0001 \
  --max-grad-norm 0.25 \
  --checkpoint-interval 50000 \
  --log-interval 5000 \
  --checkpoints-dir checkpoints/method1_2080_2env \
  --logs-dir logs/method1_2080_2env \
  --worker-start-delay 15 \
  --no-resume
```

If interrupted, resume with the same command but remove `--no-resume`.

### A100 / RunPod

The A100 configuration can use more parallel simulators:

```bash
xvfb-run -a python scripts/train_method1.py \
  --total-steps 2000000 \
  --num-envs 4 \
  --rollout-steps 128 \
  --platform default
```

Or use CloudRendering:

```bash
python scripts/train_method1.py \
  --total-steps 2000000 \
  --num-envs 4 \
  --rollout-steps 128 \
  --platform cloud
```

## Monitoring

Tail the training log:

```bash
tail -f logs/method1_2080_2env/training_log.csv
```

Watch GPU usage:

```bash
nvidia-smi -l 1
```

Plot training curves:

```bash
python scripts/plot_training.py \
  --log logs/method1_2080_2env/training_log.csv \
  --out logs/method1_2080_2env/curves.png \
  --smooth-window 10
```

Training outputs:

```text
checkpoints/<run_name>/latest.pt
checkpoints/<run_name>/step_000050000.pt
logs/<run_name>/training_log.csv
logs/<run_name>/curves.png
```

The CSV columns are:

```text
step,update,episodes,mean_reward,success_rate,episode_len,
value_loss,policy_loss,entropy,lr,wall_time_s
```

## Evaluation

### Method 1 Checkpoint

After training, `src/method1/navigator.py` can load a checkpoint:

```python
from method1 import Method1Navigator

nav = Method1Navigator("checkpoints/method1_2080_2env/latest.pt")
```

This plugs into the shared `BaseNavigator` interface used by the evaluation
runner.

### Method 2 / Method 3 Small Evaluation

Run the small evaluation across Random, Method 2, and Method 3:

```bash
python scripts/run_small_eval.py
```

This writes:

```text
results/small_eval.json
results/llm_reasoning.json
results/maps/
results/videos/
```

Method 3 requires the Anthropic API dependency and credentials.

### Reference Baselines

These lightweight baselines are not the current research focus, but they are
useful for validating the simulator, visualization, and evaluation pipeline.

Run one random episode:

```bash
python scripts/run_random_agent.py --target Mug --max-steps 50 --save-dir outputs/random_mug --mp4
```

Evaluate random / heuristic / coverage on the same small grid:

```bash
python scripts/evaluate_random_agent.py \
  --scenes FloorPlan10 FloorPlan11 FloorPlan12 \
  --targets Mug Apple Bowl \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-steps 100 \
  --save-dir outputs/eval_random_3scenes_3targets_10seeds

python scripts/evaluate_heuristic_agent.py \
  --scenes FloorPlan10 FloorPlan11 FloorPlan12 \
  --targets Mug Apple Bowl \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-steps 100 \
  --save-dir outputs/eval_heuristic_3scenes_3targets_10seeds

python scripts/evaluate_coverage_agent.py \
  --scenes FloorPlan10 FloorPlan11 FloorPlan12 \
  --targets Mug Apple Bowl \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max-steps 100 \
  --save-dir outputs/eval_coverage_3scenes_3targets_10seeds
```

Generate reports:

```bash
python scripts/analyze_evaluation.py outputs/eval_random_3scenes_3targets_10seeds
python scripts/analyze_evaluation.py outputs/eval_heuristic_3scenes_3targets_10seeds
python scripts/analyze_evaluation.py outputs/eval_coverage_3scenes_3targets_10seeds
```

Inspect failed episodes:

```bash
python scripts/inspect_failures.py outputs/eval_coverage_3scenes_3targets_10seeds \
  --agent coverage \
  --limit 5 \
  --save-dir outputs/failure_inspection_coverage
```

## Outputs and Git

Generated experiment artifacts should stay under ignored directories:

```text
outputs/
logs/
checkpoints/
runs/
wandb/
```

Commit source code, configs, and README changes. Avoid committing frames, GIFs,
MP4s, CSVs, plots, logs, checkpoints, or generated JSON results unless the team
intentionally wants to share a small artifact.

## Troubleshooting

**RTX 2080 / WSL instability with 4 envs**

Use:

```bash
--num-envs 2 --rollout-steps 128 --worker-start-delay 15
```

Four environments can work for short tests, but long runs are more likely to hit
AI2-THOR reset timeouts or run close to the 8GB VRAM limit.

**CUDA out of memory**

Reduce `--num-envs` first. If needed, also reduce `--rollout-steps`:

```bash
--num-envs 2 --rollout-steps 64
```

**AI2-THOR timeout or stale simulator processes**

Check for stale simulator processes:

```bash
ps -eo pid,cmd | grep thor-Linux64 | grep -v grep
```

Stop stale processes before restarting training.

**Headless rendering errors**

On cloud machines, try `--platform cloud`. Otherwise wrap the command in
`xvfb-run -a`.

**Resume vs. clean start**

Resume is automatic if `latest.pt` exists in the checkpoint directory. To force
a clean run, pass `--no-resume` or choose a new checkpoint directory.
