# CS175-AI2-THOR

[中文 README](README.zh-CN.md)

Object-goal navigation experiments in AI2-THOR / ProcTHOR for CS 175.

The project compares:

- End-to-end RL with implicit memory.
- Semantic mapping plus classical planning.
- Semantic mapping plus LLM-guided exploration.

## Environment

Use Python 3.8 for now. AI2-THOR, ProcTHOR, AllenAct-style embodied AI code,
PyTorch, and detector libraries are sensitive to version mismatches.

The initial environment includes:

- Python 3.8
- AI2-THOR 5.0.0
- ProcTHOR 0.0.1.dev2
- Common scientific Python packages

Detic / Detectron2 should be added later after the team chooses a CUDA and
PyTorch target, because that stack is tightly version-coupled.

## Windows Setup

Use WSL2 with Ubuntu 22.04. This gives us a Linux environment while keeping the
repository in the Windows filesystem.

From PowerShell:

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

Install Miniforge:

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Restart the Ubuntu shell, then create the conda environment:

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
pip install ai2thor==5.0.0 procthor==0.0.1.dev2
```

Run the smoke test from the repository:

```bash
cd /mnt/c/Users/Charl/Desktop/AI2-THOR/CS175-AI2-THOR
python scripts/smoke_test_ai2thor.py --platform default
```

Expected output:

```text
AI2-THOR smoke test passed with platform=default
scene=FloorPlan10
frame_shape=(300, 300, 3)
depth_shape=(300, 300)
```

For graphics diagnostics:

```bash
echo $DISPLAY
glxinfo -B
```

A working WSL setup should report an accelerated OpenGL renderer, for example:

```text
OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 2080)
```

## macOS Setup

Mac users should start with a native Miniforge environment and run the same
smoke test:

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
pip install ai2thor==5.0.0 procthor==0.0.1.dev2
python scripts/smoke_test_ai2thor.py --platform default
```

Apple Silicon Macs may need an x86_64/Rosetta environment or a Linux GPU
machine for simulator-heavy experiments.

## Smoke Test

The smoke test is:

```bash
python scripts/smoke_test_ai2thor.py --platform default
```

It creates an AI2-THOR controller, executes one `RotateRight` action, and prints
RGB/depth frame shapes plus the agent position.

## Random Baseline

Run one random ObjectNav episode:

```bash
python scripts/run_random_agent.py --config configs/random_agent.yaml
```

Override the scene, target, seed, or step budget from the command line:

```bash
python scripts/run_random_agent.py --scene FloorPlan10 --target Mug --seed 1 --max-steps 100
```

The script prints a JSON episode summary with success, step count, final agent
position, action counts, and target visibility information.

To save first-person frames, per-step metadata, and a top-down trajectory plot:

```bash
python scripts/run_random_agent.py --target Mug --max-steps 50 --save-dir outputs/random_mug
```

Add `--mp4` to export an MP4 video:

```bash
python scripts/run_random_agent.py --target Mug --max-steps 50 --save-dir outputs/random_mug --mp4
```

The random baseline records RGB frames by default. Use `--render-depth` only if
an experiment specifically needs depth frames; use the smoke test to verify
RGB-D simulator support.

This writes:

```text
outputs/random_mug/episode.json
outputs/random_mug/episode.gif
outputs/random_mug/episode.mp4
outputs/random_mug/trajectory.png
outputs/random_mug/frames/step_0000.png
```

To create a GIF from an existing run without rerunning the simulator:

```bash
python scripts/render_episode_gif.py outputs/random_mug
```

To create an MP4 from an existing run:

```bash
python scripts/render_episode_mp4.py outputs/random_mug
```
