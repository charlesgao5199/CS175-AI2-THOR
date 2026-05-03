# CS175-AI2-THOR

Object-goal navigation experiments in AI2-THOR / ProcTHOR for CS 175.

The project compares:

- End-to-end RL with implicit memory.
- Semantic mapping plus classical planning.
- Semantic mapping plus LLM-guided exploration.

## Development Environment

Use the VS Code Dev Containers extension so the editor attaches directly to the
Docker environment. This keeps Python, AI2-THOR, ProcTHOR, and PyTorch versions
consistent across team members.

The first environment is intentionally small:

- Python 3.8
- AI2-THOR 5.0.0
- ProcTHOR 0.0.1.dev2
- CPU PyTorch 1.10.2
- Common scientific Python packages

Detic / Detectron2 should be added in a second pass after the team chooses the
CUDA target, because Detic depends tightly on the PyTorch, torchvision,
Detectron2, and CUDA combination.

## VS Code

1. Install Docker Desktop.
2. Install the VS Code extension `Dev Containers`.
3. Open this repository in VS Code.
4. Run `Dev Containers: Reopen in Container`.
5. VS Code should use this interpreter inside the container:

   ```text
   /opt/conda/envs/ai2thor-objectnav/bin/python
   ```

## Smoke Test

After the container opens, run:

```bash
python scripts/smoke_test_ai2thor.py
```

If default rendering fails on a headless machine, try:

```bash
xvfb-run -a python scripts/smoke_test_ai2thor.py --platform default
```

GPU training and Detic inference will likely need a CUDA-enabled Docker setup.
When that becomes necessary, add GPU runtime args such as `--gpus=all` to the
Dev Container configuration and switch the PyTorch package set from CPU to a
CUDA build.
