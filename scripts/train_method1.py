"""Method 1 PPO training — self-contained, single-file, ready for a CUDA GPU.

Trains the ResNet-18 + GRU policy defined in ``src/method1/policy.py`` on
iTHOR ObjectNav (FloorPlan1-20, 5 target categories) using a clean recurrent
PPO loop. Auto-resumes from ``checkpoints/method1/latest.pt`` if interrupted.

Run on a CUDA box (RunPod / Lambda):

    xvfb-run -a python scripts/train_method1.py \\
        --total-steps 2_000_000 --num-envs 4 --rollout-steps 128

See ``README.md`` for setup, training, monitoring, and troubleshooting.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Make `import method1`, `import shared`, etc. work without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from method1.policy import (  # noqa: E402
    DEFAULT_TARGETS,
    HIDDEN_DIM,
    MAX_DEPTH_M,
    NUM_ACTIONS,
    Method1Policy,
    initial_hidden,
)
from shared.interfaces import Action  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_SCENES: Tuple[str, ...] = tuple(f"FloorPlan{i}" for i in range(1, 21))

ACTION_TO_THOR = {
    Action.MOVE_AHEAD: "MoveAhead",
    Action.ROTATE_LEFT: "RotateLeft",
    Action.ROTATE_RIGHT: "RotateRight",
    Action.LOOK_UP: "LookUp",
    Action.LOOK_DOWN: "LookDown",
}

# Reward shaping (from CS 175 proposal).
R_SUCCESS = +10.0
R_WRONG_STOP = -0.5
R_STEP = -0.01
MAX_EPISODE_STEPS = 500
SUCCESS_DISTANCE_M = 1.0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def _linux_process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            try:
                state = next(
                    (line for line in status.read_text(errors="replace").splitlines()
                     if line.startswith("State:")),
                    "",
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                return False
            return "\tZ" not in state
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_child_pid_map() -> Dict[int, List[int]]:
    if os.name != "posix":
        return {}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return {}

    children: Dict[int, List[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            ppid_line = next(
                (line for line in (entry / "status").read_text(errors="replace").splitlines()
                 if line.startswith("PPid:")),
                "",
            )
            ppid = int(ppid_line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def _linux_descendant_pids(root_pid: int) -> List[int]:
    children = _linux_child_pid_map()
    descendants: List[int] = []
    seen: Set[int] = set()
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        stack.extend(children.get(pid, ()))
    return descendants


def _terminate_linux_process_tree(
    root_pid: Optional[int],
    *,
    include_root: bool,
    grace_s: float = 1.0,
) -> None:
    """Terminate a Linux/WSL process tree without touching the whole process group."""
    if root_pid is None or os.name != "posix" or not Path("/proc").exists():
        return

    pids = _linux_descendant_pids(int(root_pid))
    if include_root:
        pids.append(int(root_pid))
    unique_pids = list(dict.fromkeys(pid for pid in pids if pid > 0))
    if not unique_pids:
        return

    for pid in unique_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"[worker] permission denied terminating pid {pid}: {exc}", flush=True)

    deadline = time.time() + max(0.0, grace_s)
    while time.time() < deadline:
        if not any(_linux_process_is_alive(pid) for pid in unique_pids):
            return
        time.sleep(0.05)

    for pid in unique_pids:
        if not _linux_process_is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"[worker] permission denied killing pid {pid}: {exc}", flush=True)


class AI2ThorObjectNavEnv:
    """Minimal gym-style ObjectNav env on iTHOR.

    Not a real ``gymnasium.Env`` to avoid pulling in the dependency; it just
    exposes ``reset()`` / ``step()`` with the contract that ``SubprocVecEnv``
    expects.
    """

    def __init__(
        self,
        scenes: Tuple[str, ...] = DEFAULT_SCENES,
        targets: Tuple[str, ...] = DEFAULT_TARGETS,
        width: int = 224,
        height: int = 224,
        platform: str = "default",
        seed: int = 0,
        server_timeout: float = 100.0,
    ) -> None:
        self.scenes = tuple(scenes)
        self.targets = tuple(targets)
        self.target_to_id = {t: i for i, t in enumerate(self.targets)}
        self.width = width
        self.height = height
        self.platform = platform
        self.server_timeout = float(server_timeout)
        self.rng = np.random.default_rng(seed)
        self._controller = None
        self._current_scene = ""
        self._current_target = ""
        self._steps = 0
        self._start_pos: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._target_positions: List[Dict[str, float]] = []

    # ----- lifecycle ------------------------------------------------------- #

    def _ensure_controller(self):
        if self._controller is not None:
            return
        # Imported here so the parent process never instantiates AI2-THOR.
        from ai2thor.controller import Controller  # type: ignore

        kwargs: Dict[str, Any] = {
            "scene": self.scenes[0],
            "width": self.width,
            "height": self.height,
            "renderDepthImage": True,
            "server_timeout": self.server_timeout,
        }
        if self.platform == "cloud":
            from ai2thor.platform import CloudRendering  # type: ignore

            kwargs["platform"] = CloudRendering
        self._controller = Controller(**kwargs)

    def close(self) -> None:
        controller = self._controller
        if controller is not None:
            unity_pids = self._controller_unity_pids()
            for pid in unity_pids:
                _terminate_linux_process_tree(pid, include_root=True, grace_s=0.5)
            try:
                controller.stop()
            except Exception:
                pass
            for pid in unity_pids:
                _terminate_linux_process_tree(pid, include_root=True, grace_s=0.5)
            self._controller = None

    def _controller_unity_pids(self) -> List[int]:
        if self._controller is None:
            return []
        pids: List[int] = []
        unity_pid = getattr(self._controller, "unity_pid", None)
        if isinstance(unity_pid, int) and unity_pid > 0:
            pids.append(unity_pid)
        server = getattr(self._controller, "server", None)
        unity_proc = getattr(server, "unity_proc", None)
        proc_pid = getattr(unity_proc, "pid", None)
        if isinstance(proc_pid, int) and proc_pid > 0:
            pids.append(proc_pid)
        return list(dict.fromkeys(pids))

    # ----- observation ----------------------------------------------------- #

    def _build_obs(self) -> Dict[str, np.ndarray]:
        event = self._controller.last_event
        rgb = np.asarray(event.frame, dtype=np.uint8)              # (H, W, 3)
        depth = event.depth_frame
        depth = (np.asarray(depth, dtype=np.float32)
                 if depth is not None else np.zeros((self.height, self.width), dtype=np.float32))
        if depth.shape != (self.height, self.width):
            depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0)
        depth = np.clip(depth, 0.0, MAX_DEPTH_M).astype(np.float32, copy=False)
        agent = event.metadata["agent"]
        heading_rad = float(np.deg2rad(agent["rotation"]["y"]))
        compass = np.array([np.sin(heading_rad), np.cos(heading_rad)], dtype=np.float32)
        target_id = self.target_to_id[self._current_target]
        return {
            "rgb": rgb,                                              # (H, W, 3) uint8
            "depth": depth,                                          # (H, W) float32
            "target_id": np.int64(target_id),                        # scalar
            "compass": compass,                                      # (2,) float32
        }

    # ----- gym API --------------------------------------------------------- #

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self._ensure_controller()
        self._current_scene = str(self.rng.choice(self.scenes))
        self._current_target = str(self.rng.choice(self.targets))
        self._controller.reset(scene=self._current_scene)
        self._steps = 0
        event = self._controller.last_event
        agent = event.metadata["agent"]
        self._start_pos = dict(agent["position"])
        self._target_positions = [
            {"x": o["position"]["x"], "y": o["position"]["y"], "z": o["position"]["z"]}
            for o in event.metadata.get("objects", [])
            if o.get("objectType") == self._current_target
        ]
        info = {
            "scene": self._current_scene,
            "target": self._current_target,
            "n_target_instances": len(self._target_positions),
        }
        return self._build_obs(), info

    def _min_distance(self, p: Dict[str, float]) -> float:
        if not self._target_positions:
            return 0.0
        return float(min(np.hypot(p["x"] - t["x"], p["z"] - t["z"])
                         for t in self._target_positions))

    def step(self, action_id: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self._controller is None:
            raise RuntimeError("reset() must be called before step()")
        self._steps += 1
        action = Action(int(action_id))
        info: Dict[str, Any] = {}
        terminated = False

        if action == Action.STOP:
            cur = self._controller.last_event.metadata["agent"]["position"]
            d = self._min_distance(cur)
            success = bool(self._target_positions) and d <= SUCCESS_DISTANCE_M
            reward = R_SUCCESS if success else R_WRONG_STOP
            terminated = True
            info.update({
                "success": success,
                "stop_issued": True,
                "d_final": d,
                "episode_steps": self._steps,
            })
        else:
            self._controller.step(action=ACTION_TO_THOR[action])
            reward = R_STEP
            if self._steps >= MAX_EPISODE_STEPS:
                cur = self._controller.last_event.metadata["agent"]["position"]
                d = self._min_distance(cur)
                terminated = True
                info.update({
                    "success": False,
                    "stop_issued": False,
                    "d_final": d,
                    "episode_steps": self._steps,
                })

        return self._build_obs(), float(reward), terminated, info


# --------------------------------------------------------------------------- #
# SubprocVecEnv (multiprocessing rollouts)
# --------------------------------------------------------------------------- #

class WorkerError(RuntimeError):
    """Raised when an AI2-THOR worker cannot recover cleanly."""


def _reset_with_retries(
    env: AI2ThorObjectNavEnv,
    context: str,
    max_attempts: int,
    retry_delay_s: float,
):
    last_error: Optional[BaseException] = None
    max_attempts = max(1, int(max_attempts))
    for attempt in range(1, max_attempts + 1):
        try:
            return env.reset()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[worker] {context} reset failed on attempt {attempt}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            env.close()
            if attempt < max_attempts:
                time.sleep(min(30.0, max(0.0, retry_delay_s) * float(attempt)))
    raise WorkerError(f"{context} reset failed after {max_attempts} attempts") from last_error


def _step_with_recovery(
    env: AI2ThorObjectNavEnv,
    action_id: int,
    max_reset_retries: int,
    reset_retry_delay_s: float,
):
    try:
        return env.step(action_id)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[worker] step failed; resetting environment: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        env.close()
        obs, reset_info = _reset_with_retries(
            env,
            "step recovery",
            max_reset_retries,
            reset_retry_delay_s,
        )
        info = {
            "success": False,
            "stop_issued": False,
            "episode_steps": env._steps,
            "env_error": f"{type(exc).__name__}: {exc}",
            "reset_info": reset_info,
        }
        return obs, R_WRONG_STOP, True, info


def _worker(conn, env_kwargs):
    worker_id = int(env_kwargs.pop("worker_id", 0))
    startup_delay_s = float(env_kwargs.pop("startup_delay_s", 0.0))
    max_reset_retries = int(env_kwargs.pop("max_reset_retries", 3))
    reset_retry_delay_s = float(env_kwargs.pop("reset_retry_delay_s", 3.0))
    if startup_delay_s > 0:
        print(
            f"[worker {worker_id}] waiting {startup_delay_s:.1f}s before AI2-THOR startup",
            flush=True,
        )
        time.sleep(startup_delay_s)
    env = AI2ThorObjectNavEnv(**env_kwargs)
    try:
        while True:
            cmd, data = conn.recv()
            try:
                if cmd == "reset":
                    obs, info = _reset_with_retries(
                        env,
                        "initial",
                        max_reset_retries,
                        reset_retry_delay_s,
                    )
                    conn.send(("ok", (obs, info)))
                elif cmd == "step":
                    obs, reward, done, info = _step_with_recovery(
                        env,
                        data,
                        max_reset_retries,
                        reset_retry_delay_s,
                    )
                    if done and "reset_info" not in info:
                        final_info = info
                        obs, reset_info = _reset_with_retries(
                            env,
                            "episode boundary",
                            max_reset_retries,
                            reset_retry_delay_s,
                        )
                        info = {**final_info, "reset_info": reset_info}
                    conn.send(("ok", (obs, reward, done, info)))
                elif cmd == "close":
                    env.close()
                    conn.close()
                    return
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[worker {worker_id}] unrecoverable error: {msg}", flush=True)
                try:
                    conn.send(("error", msg))
                except Exception:
                    pass
                return
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        env.close()


class SubprocVecEnv:
    """Parallel env collector using one Python subprocess per env.

    Each worker owns its own AI2-THOR Controller. Auto-resets on episode end
    so the rollout loop never sees a "padding" timestep.
    """

    def __init__(
        self,
        env_kwargs_list: List[Dict[str, Any]],
        response_timeout_s: float = 900.0,
    ) -> None:
        self.n = len(env_kwargs_list)
        self.response_timeout_s = max(1.0, float(response_timeout_s))
        self._parents: List = []
        self._procs: List[Process] = []
        for env_kwargs in env_kwargs_list:
            parent, child = Pipe()
            p = Process(target=_worker, args=(child, env_kwargs), daemon=True)
            p.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(p)

    def _stop_process(self, proc: Process) -> None:
        _terminate_linux_process_tree(proc.pid, include_root=False, grace_s=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        _terminate_linux_process_tree(proc.pid, include_root=False, grace_s=0.5)
        if proc.is_alive():
            try:
                proc.kill()
            except AttributeError:
                proc.terminate()
            proc.join(timeout=2)
        _terminate_linux_process_tree(proc.pid, include_root=False, grace_s=0.5)

    def _recv_result(self, idx: int, context: str):
        parent = self._parents[idx]
        proc = self._procs[idx]
        if not parent.poll(self.response_timeout_s):
            self._stop_process(proc)
            raise WorkerError(
                f"{context}: worker {idx} timed out after "
                f"{self.response_timeout_s:.0f}s"
            )
        try:
            status, payload = parent.recv()
        except EOFError as exc:
            raise WorkerError(f"{context}: worker {idx} exited before replying") from exc
        if status == "error":
            self._stop_process(proc)
            raise WorkerError(f"{context}: worker {idx} failed: {payload}")
        if status != "ok":
            raise WorkerError(f"{context}: worker {idx} sent unknown status {status!r}")
        return payload

    def reset(self) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
        for p in self._parents:
            p.send(("reset", None))
        results = [self._recv_result(i, "reset") for i in range(self.n)]
        obs = [r[0] for r in results]
        info = [r[1] for r in results]
        return obs, info

    def step(self, actions: np.ndarray):
        for p, a in zip(self._parents, actions):
            p.send(("step", int(a)))
        results = [self._recv_result(i, "step") for i in range(self.n)]
        obs = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.bool_)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self) -> None:
        for p in self._parents:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            self._stop_process(proc)
        for p in self._parents:
            try:
                p.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Rollout buffer
# --------------------------------------------------------------------------- #

@dataclass
class RolloutBuffer:
    T: int
    N: int
    device: torch.device

    rgb: torch.Tensor = field(init=False)      # (T, N, 3, 224, 224) uint8
    depth: torch.Tensor = field(init=False)    # (T, N, 1, 224, 224) float32
    target: torch.Tensor = field(init=False)   # (T, N) long
    compass: torch.Tensor = field(init=False)  # (T, N, 2) float32
    actions: torch.Tensor = field(init=False)  # (T, N) long
    log_probs: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    hidden0: torch.Tensor = field(init=False)  # (1, N, H) hidden state BEFORE step 0

    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        T, N = self.T, self.N
        d = self.device
        # Keep RGB on CPU (uint8) to save GPU memory; ship per-minibatch.
        self.rgb = torch.zeros(T, N, 3, 224, 224, dtype=torch.uint8)
        self.depth = torch.zeros(T, N, 1, 224, 224, dtype=torch.float32)
        self.target = torch.zeros(T, N, dtype=torch.long)
        self.compass = torch.zeros(T, N, 2, dtype=torch.float32)
        self.actions = torch.zeros(T, N, dtype=torch.long)
        self.log_probs = torch.zeros(T, N, dtype=torch.float32)
        self.values = torch.zeros(T, N, dtype=torch.float32)
        self.rewards = torch.zeros(T, N, dtype=torch.float32)
        self.dones = torch.zeros(T, N, dtype=torch.float32)
        self.hidden0 = torch.zeros(1, N, HIDDEN_DIM, device=d)
        self.advantages = torch.zeros(T, N, dtype=torch.float32)
        self.returns = torch.zeros(T, N, dtype=torch.float32)

    def insert(
        self, t: int, obs_batch: List[Dict[str, np.ndarray]], action: torch.Tensor,
        log_prob: torch.Tensor, value: torch.Tensor, reward: np.ndarray, done: np.ndarray,
    ) -> None:
        for n, obs in enumerate(obs_batch):
            # Resize to 224x224 if env didn't already.
            rgb = torch.from_numpy(obs["rgb"]).permute(2, 0, 1).unsqueeze(0).float()
            if rgb.shape[-2:] != (224, 224):
                rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
            self.rgb[t, n] = rgb.squeeze(0).to(torch.uint8)
            depth_np = np.nan_to_num(obs["depth"], nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0)
            depth_np = np.clip(depth_np, 0.0, MAX_DEPTH_M).astype(np.float32, copy=False)
            depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
            if depth.shape[-2:] != (224, 224):
                depth = F.interpolate(depth, size=(224, 224), mode="bilinear", align_corners=False)
            self.depth[t, n] = depth.squeeze(0)
            self.target[t, n] = int(obs["target_id"])
            self.compass[t, n] = torch.from_numpy(obs["compass"])
        self.actions[t] = action.cpu()
        self.log_probs[t] = log_prob.cpu()
        self.values[t] = value.cpu()
        self.rewards[t] = torch.from_numpy(reward)
        self.dones[t] = torch.from_numpy(done.astype(np.float32))

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        last_value = last_value.cpu()
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.N)
        next_value = last_value
        for t in reversed(range(self.T)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_value = self.values[t]
        self.advantages = adv
        self.returns = adv + self.values


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(
            "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
        )
        print(f"[trainer] device={self.device}")

        self.targets = tuple(args.targets)
        self.scenes = tuple(args.scenes)

        env_kwargs_list = [
            {
                "worker_id": i,
                "startup_delay_s": i * args.worker_start_delay,
                "scenes": self.scenes,
                "targets": self.targets,
                "width": 224,
                "height": 224,
                "platform": args.platform,
                "seed": args.seed + i,
                "server_timeout": args.server_timeout,
                "max_reset_retries": args.max_reset_retries,
                "reset_retry_delay_s": args.reset_retry_delay,
            }
            for i in range(args.num_envs)
        ]
        self.envs = SubprocVecEnv(env_kwargs_list, response_timeout_s=args.worker_timeout)

        self.policy = Method1Policy(
            num_targets=len(self.targets),
            pretrained_encoder=not args.no_pretrained,
            target_categories=self.targets,
        ).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=args.lr, eps=1e-5)

        self.rollout = RolloutBuffer(T=args.rollout_steps, N=args.num_envs, device=self.device)

        self.total_steps = 0
        self.update_idx = 0
        self.checkpoints_dir = Path(args.checkpoints_dir)
        self.logs_dir = Path(args.logs_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "training_log.csv"
        if not self.log_path.exists():
            self.log_path.write_text(
                "step,update,episodes,mean_reward,success_rate,episode_len,"
                "value_loss,policy_loss,entropy,lr,wall_time_s\n"
            )

        # Sliding window of recent episode stats.
        self.ep_returns: deque[float] = deque(maxlen=100)
        self.ep_successes: deque[float] = deque(maxlen=100)
        self.ep_lengths: deque[int] = deque(maxlen=100)
        self.ep_running = np.zeros(args.num_envs, dtype=np.float32)
        self.ep_running_len = np.zeros(args.num_envs, dtype=np.int32)
        self.n_episodes = 0
        self._last_log_step = 0

        # Resume?
        latest = self.checkpoints_dir / "latest.pt"
        if args.resume and latest.exists():
            self._load_checkpoint(latest)
        elif args.resume:
            print(f"[trainer] --resume set but {latest} doesn't exist; starting fresh")

        # Initial reset.
        try:
            obs, _info = self.envs.reset()
        except BaseException:
            self.envs.close()
            raise
        self._last_obs = obs
        self._hidden = initial_hidden(args.num_envs, self.device)

        self._stop_requested = False
        # Let Ctrl+C raise KeyboardInterrupt so it can break blocking worker waits.
        signal.signal(signal.SIGTERM, self._sig)

    def _sig(self, *_args) -> None:
        print("\n[trainer] termination requested; will save checkpoint and exit")
        self._stop_requested = True

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self) -> None:
        ckpt = {
            "step": self.total_steps,
            "update": self.update_idx,
            "n_episodes": self.n_episodes,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "targets": list(self.targets),
            "scenes": list(self.scenes),
            "config": vars(self.args),
        }
        named = self.checkpoints_dir / f"step_{self.total_steps:09d}.pt"
        latest = self.checkpoints_dir / "latest.pt"
        torch.save(ckpt, named)
        torch.save(ckpt, latest)
        print(f"[trainer] checkpoint saved: {named.name}")

    def _load_checkpoint(self, path: Path) -> None:
        print(f"[trainer] resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:  # noqa: BLE001
            print(f"[trainer] could not restore optimizer state: {e}; starting optimizer fresh")
        self.total_steps = int(ckpt.get("step", 0))
        self.update_idx = int(ckpt.get("update", 0))
        self.n_episodes = int(ckpt.get("n_episodes", 0))
        print(f"[trainer]   step={self.total_steps:,} update={self.update_idx:,} "
              f"episodes={self.n_episodes:,}")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log_row(self, value_loss: float, policy_loss: float, entropy: float,
                 wall_time_s: float) -> None:
        mr = float(np.mean(self.ep_returns)) if self.ep_returns else 0.0
        sr = float(np.mean(self.ep_successes)) if self.ep_successes else 0.0
        el = float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0
        lr = self.optimizer.param_groups[0]["lr"]
        with self.log_path.open("a") as fh:
            fh.write(f"{self.total_steps},{self.update_idx},{self.n_episodes},"
                     f"{mr:.4f},{sr:.4f},{el:.1f},"
                     f"{value_loss:.4f},{policy_loss:.4f},{entropy:.4f},"
                     f"{lr:.6f},{wall_time_s:.1f}\n")
        print(f"[step {self.total_steps:>9,}] eps={self.n_episodes:>5} "
              f"mean_R={mr:+.3f} success={sr:.2%} len={el:.0f} "
              f"v_loss={value_loss:.3f} p_loss={policy_loss:.3f} ent={entropy:.3f} "
              f"wall={wall_time_s:.0f}s")

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #

    def _gather_obs_tensors(self, obs_list: List[Dict[str, np.ndarray]]):
        rgb = torch.stack([
            F.interpolate(
                torch.from_numpy(o["rgb"]).permute(2, 0, 1).unsqueeze(0).float(),
                size=(224, 224), mode="bilinear", align_corners=False,
            ).squeeze(0).to(torch.uint8)
            for o in obs_list
        ]).to(self.device)
        depth = torch.stack([
            F.interpolate(
                torch.from_numpy(
                    np.clip(
                        np.nan_to_num(o["depth"], nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0),
                        0.0,
                        MAX_DEPTH_M,
                    ).astype(np.float32, copy=False)
                ).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode="bilinear", align_corners=False,
            ).squeeze(0)
            for o in obs_list
        ]).to(self.device)
        target = torch.tensor([int(o["target_id"]) for o in obs_list],
                              dtype=torch.long, device=self.device)
        compass = torch.tensor(
            np.stack([o["compass"] for o in obs_list]),
            dtype=torch.float32, device=self.device,
        )
        return rgb, depth, target, compass

    def _collect_rollout(self) -> torch.Tensor:
        """Run T rollout steps with the current policy. Returns last_value for GAE."""
        rb = self.rollout
        # Save hidden state at the start of the chunk for the training pass.
        rb.hidden0 = self._hidden.detach()

        obs_list = self._last_obs
        for t in range(rb.T):
            rgb, depth, target, compass = self._gather_obs_tensors(obs_list)
            action, log_prob, value, new_hidden = self.policy.select_action(
                rgb, depth, target, compass, self._hidden, deterministic=False,
            )
            actions_np = action.cpu().numpy()
            next_obs, reward, done, infos = self.envs.step(actions_np)

            rb.insert(t, obs_list, action, log_prob, value, reward, done)

            # Track episode stats.
            self.ep_running += reward
            self.ep_running_len += 1
            for n, d in enumerate(done):
                if d:
                    self.n_episodes += 1
                    self.ep_returns.append(float(self.ep_running[n]))
                    self.ep_lengths.append(int(self.ep_running_len[n]))
                    self.ep_successes.append(float(infos[n].get("success", False)))
                    self.ep_running[n] = 0.0
                    self.ep_running_len[n] = 0

            # Reset hidden state for envs whose episodes just ended.
            done_mask = torch.from_numpy(1.0 - done.astype(np.float32)).to(self.device)
            self._hidden = new_hidden * done_mask.view(1, -1, 1)
            obs_list = next_obs

        self._last_obs = obs_list
        self.total_steps += rb.T * rb.N

        # last_value for bootstrap.
        rgb, depth, target, compass = self._gather_obs_tensors(obs_list)
        with torch.no_grad():
            _logits, last_value, _h = self.policy.step(
                rgb, depth, target, compass, self._hidden
            )
        return last_value

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #

    def _update(self) -> Tuple[float, float, float]:
        """Run K epochs of PPO over the rollout. Expects ``rb.advantages`` and
        ``rb.returns`` to already be filled in (caller invokes ``compute_gae``).
        """
        a = self.args
        rb = self.rollout
        N = rb.N
        env_indices = np.arange(N)
        adv_norm = rb.advantages
        adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)
        adv_norm = torch.nan_to_num(adv_norm, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        total_p, total_v, total_e = 0.0, 0.0, 0.0
        n_updates = 0
        for _epoch in range(a.k_epochs):
            np.random.shuffle(env_indices)
            mb_size = max(1, N // a.num_minibatches)
            for start in range(0, N, mb_size):
                mb = env_indices[start:start + mb_size]
                if mb.size == 0:
                    continue
                mb_t = torch.tensor(mb, dtype=torch.long)

                rgb = rb.rgb[:, mb_t].to(self.device, non_blocking=True)
                depth = rb.depth[:, mb_t].to(self.device, non_blocking=True)
                target = rb.target[:, mb_t].to(self.device, non_blocking=True)
                compass = rb.compass[:, mb_t].to(self.device, non_blocking=True)
                actions = rb.actions[:, mb_t].to(self.device, non_blocking=True)
                old_log_probs = rb.log_probs[:, mb_t].to(self.device, non_blocking=True)
                advantages = adv_norm[:, mb_t].to(self.device, non_blocking=True)
                returns = torch.nan_to_num(
                    rb.returns[:, mb_t].to(self.device, non_blocking=True),
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
                dones = rb.dones[:, mb_t].to(self.device, non_blocking=True)
                h0 = rb.hidden0[:, mb_t]

                new_log_probs, new_values, entropy = self.policy.evaluate_sequence(
                    rgb, depth, target, compass, actions, h0, dones
                )

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1.0 - a.clip_eps, 1.0 + a.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + a.vf_coef * value_loss + a.ent_coef * entropy_loss
                if not torch.isfinite(loss):
                    print("[trainer] warning: skipped PPO minibatch with non-finite loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), a.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    print("[trainer] warning: skipped PPO minibatch with non-finite gradients")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()

                total_p += policy_loss.item()
                total_v += value_loss.item()
                total_e += entropy.mean().item()
                n_updates += 1

        if n_updates == 0:
            return 0.0, 0.0, 0.0
        return total_v / n_updates, total_p / n_updates, total_e / n_updates

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        a = self.args
        t0 = time.time()
        next_log_at = ((self.total_steps // a.log_interval) + 1) * a.log_interval
        next_ckpt_at = ((self.total_steps // a.checkpoint_interval) + 1) * a.checkpoint_interval
        steps_per_update = a.rollout_steps * a.num_envs

        while self.total_steps < a.total_steps and not self._stop_requested:
            last_value = self._collect_rollout()
            self.rollout.compute_gae(last_value, a.gamma, a.gae_lambda)
            value_loss, policy_loss, entropy = self._update()
            self.update_idx += 1

            wall = time.time() - t0
            if self.total_steps >= next_log_at:
                self._log_row(value_loss, policy_loss, entropy, wall)
                next_log_at += a.log_interval
            if self.total_steps >= next_ckpt_at:
                self._save_checkpoint()
                next_ckpt_at += a.checkpoint_interval

        # Final save before exit (even on interrupt).
        self._save_checkpoint()
        wall = time.time() - t0
        self._log_row(value_loss=0.0, policy_loss=0.0, entropy=0.0, wall_time_s=wall)
        self.envs.close()
        print(f"[trainer] done. total_steps={self.total_steps:,} "
              f"updates={self.update_idx:,} episodes={self.n_episodes:,} "
              f"wall={wall:.0f}s")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Method 1 PPO training")

    # Environment
    p.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES),
                   help="iTHOR FloorPlanXX scenes to sample from.")
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                   help="Target object categories.")
    p.add_argument("--platform", choices=("default", "cloud"), default="default",
                   help="AI2-THOR rendering platform. Use 'cloud' on headless servers.")

    # Optimization
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--k-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # IO
    p.add_argument("--checkpoints-dir", default="checkpoints/method1")
    p.add_argument("--logs-dir", default="logs/method1")
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--log-interval", type=int, default=10_000)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--worker-start-delay", type=float, default=5.0,
                   help="Seconds to stagger AI2-THOR worker startup. Useful on WSL/local GPUs.")
    p.add_argument("--max-reset-retries", type=int, default=3,
                   help="Maximum AI2-THOR reset attempts before saving and exiting.")
    p.add_argument("--reset-retry-delay", type=float, default=3.0,
                   help="Base seconds between AI2-THOR reset retries.")
    p.add_argument("--worker-timeout", type=float, default=900.0,
                   help="Seconds the trainer waits for a worker response before killing it.")
    p.add_argument("--server-timeout", type=float, default=100.0,
                   help="Seconds AI2-THOR waits for a backend response before reset recovery.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip ImageNet-pretrained ResNet18 init.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Auto-resume from <checkpoints-dir>/latest.pt if present.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # spawn is safer than fork for AI2-THOR + CUDA.
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Quick summary so the run is reproducible from the log.
    print("[trainer] config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    trainer: Optional[Trainer] = None
    try:
        trainer = Trainer(args)
        trainer.train()
    except KeyboardInterrupt:
        print("\n[trainer] caught KeyboardInterrupt; saving checkpoint before exit")
        if trainer is not None:
            trainer._save_checkpoint()
        return 130
    except WorkerError as exc:
        print(f"\n[trainer] worker failed; saving checkpoint before exit: {exc}")
        if trainer is not None:
            trainer._save_checkpoint()
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n[trainer] fatal error; saving checkpoint before exit: {type(exc).__name__}: {exc}")
        if trainer is not None:
            trainer._save_checkpoint()
        return 1
    finally:
        if trainer is not None:
            trainer.envs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
