# CS175-AI2-THOR

[English README](README.md)

这是一个用于 CS 175 的 AI2-THOR / ProcTHOR ObjectNav 项目。我们的目标是让智能体在室内环境中根据目标物体类别进行导航，例如寻找 `Mug`、`Apple`、`Chair` 等物体。

项目计划比较几类方法：

- 带隐式记忆的端到端强化学习方法
- 语义地图加传统规划方法
- 语义地图加 LLM 引导探索方法

## 环境

目前建议使用 Python 3.8。AI2-THOR、ProcTHOR、AllenAct 风格的 embodied AI 代码、PyTorch 和 detector 库都比较容易受到版本不匹配的影响，所以先固定一个稳定的基础环境。

当前基础环境包含：

- Python 3.8
- AI2-THOR 5.0.0
- ProcTHOR 0.0.1.dev2
- 常用科学计算 Python 包

Detic / Detectron2 之后再加入。它们和 CUDA、PyTorch 版本绑定很紧，等团队确认 CUDA 和 PyTorch 目标版本后再配置会更稳。

## Windows 设置

Windows 用户推荐使用 WSL2 + Ubuntu 22.04。这样可以在 Windows 电脑上运行 Linux 环境，同时保留项目文件在 Windows 文件系统中。

在 PowerShell 中运行：

```powershell
wsl --install -d Ubuntu-22.04
wsl -d Ubuntu-22.04
```

进入 Ubuntu 后，先确认 GPU 可见：

```bash
nvidia-smi
```

安装 AI2-THOR 渲染所需的系统依赖：

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

安装 Miniforge：

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

重启 Ubuntu shell，然后创建 conda 环境：

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
pip install ai2thor==5.0.0 procthor==0.0.1.dev2
```

进入项目目录并运行 smoke test：

```bash
cd /mnt/c/Users/Charl/Desktop/AI2-THOR/CS175-AI2-THOR
python scripts/smoke_test_ai2thor.py --platform default
```

期望看到类似输出：

```text
AI2-THOR smoke test passed with platform=default
scene=FloorPlan10
frame_shape=(300, 300, 3)
depth_shape=(300, 300)
```

如果需要检查图形渲染状态，可以运行：

```bash
echo $DISPLAY
glxinfo -B
```

正常情况下，WSL 应该能看到一个加速的 OpenGL renderer，例如：

```text
OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 2080)
```

## macOS 设置

macOS 用户可以先使用原生 Miniforge 环境，并运行同一个 smoke test：

```bash
mamba create -n ai2thor-objectnav python=3.8 -y
conda activate ai2thor-objectnav
pip install ai2thor==5.0.0 procthor==0.0.1.dev2
python scripts/smoke_test_ai2thor.py --platform default
```

Apple Silicon Mac 可能需要 x86_64 / Rosetta 环境，或者使用 Linux GPU 机器来跑更重的 simulator 实验。

## Smoke Test

Smoke test 用来确认环境是否搭建成功：

```bash
python scripts/smoke_test_ai2thor.py --platform default
```

它会创建一个 AI2-THOR controller，执行一次 `RotateRight` 动作，并打印 RGB/depth frame 的 shape 和 agent 位置。这个测试不是算法实验，而是确认 simulator、渲染和 Python 环境都能正常工作。

## Random Baseline

运行一个随机 ObjectNav episode：

```bash
python scripts/run_random_agent.py --config configs/random_agent.yaml
```

也可以从命令行覆盖 scene、target、seed 或最大步数：

```bash
python scripts/run_random_agent.py --scene FloorPlan10 --target Mug --seed 1 --max-steps 100
```

脚本会输出一个 JSON summary，包括是否成功、走了多少步、最终位置、动作统计和目标物体可见性信息。

如果想保存第一人称画面、每一步 metadata 和俯视轨迹图：

```bash
python scripts/run_random_agent.py --target Mug --max-steps 50 --save-dir outputs/random_mug
```

如果还想导出 MP4：

```bash
python scripts/run_random_agent.py --target Mug --max-steps 50 --save-dir outputs/random_mug --mp4
```

random baseline 默认只记录 RGB frames。只有在实验明确需要 depth frame 时才使用 `--render-depth`；RGB-D 支持可以用 smoke test 单独验证。

运行后会生成：

```text
outputs/random_mug/episode.json
outputs/random_mug/episode.gif
outputs/random_mug/episode.mp4
outputs/random_mug/trajectory.png
outputs/random_mug/frames/step_0000.png
```

注意：`outputs/` 已经被 `.gitignore` 忽略，不会被提交到 GitHub。

## 从已有 Frames 生成可视化

如果已经跑过 episode，只想从现有 frames 生成 GIF，不需要重跑 simulator：

```bash
python scripts/render_episode_gif.py outputs/random_mug
```

从现有 frames 生成 MP4：

```bash
python scripts/render_episode_mp4.py outputs/random_mug
```

MP4 导出依赖 `ffmpeg`。如果缺少 `ffmpeg`，在 Ubuntu 中运行：

```bash
sudo apt-get install -y ffmpeg
```

## Batch Evaluation

运行多个 random baseline episodes，并输出整体指标：

```bash
python scripts/evaluate_random_agent.py \
  --scenes FloorPlan10 FloorPlan11 FloorPlan12 \
  --targets Mug Apple Bowl \
  --seeds 0 1 2 \
  --max-steps 100 \
  --save-dir outputs/eval_random
```

运行后会生成：

```text
outputs/eval_random/results.csv
outputs/eval_random/summary.json
```

`results.csv` 每一行对应一个 episode。`summary.json` 会汇总总 episode 数、成功率、平均步数、错误数量，并按 scene 和 target 分组统计。

从 evaluation 目录生成更容易阅读的报告和图：

```bash
python scripts/analyze_evaluation.py outputs/eval_random
```

运行后会生成：

```text
outputs/eval_random/analysis.md
outputs/eval_random/success_by_scene.png
outputs/eval_random/success_by_target.png
outputs/eval_random/steps_by_target.png
outputs/eval_random/success_by_scene_target.png
```

重跑失败的 evaluation episodes，并保存可视化诊断：

```bash
python scripts/inspect_failures.py outputs/eval_random --limit 5 --save-dir outputs/failure_inspection
```

每个被检查的失败案例会生成一个文件夹，里面包含：

```text
episode.json
episode.gif
trajectory.png
frames/
inspection.json
```

可以用 `--scene` 或 `--target` 只检查某一部分，例如：

```bash
python scripts/inspect_failures.py outputs/eval_random --target Bowl --limit 3
```
