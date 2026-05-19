"""Small-scale evaluation: Random vs. Method2 vs. Method3 on 10 episodes.

Run:
    conda activate objectnav
    python scripts/run_small_eval.py

Outputs:
    results/small_eval.json          metrics for every episode and aggregate
    results/llm_reasoning.json       full Method3 reasoning + usage trace
    results/maps/<method>_<scene>_<target>.png   per-episode semantic map
    results/videos/<method>_ep<NN>_<scene>_<target>.mp4   first-person video
                                                          (first 3 episodes only)
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from ai2thor.controller import Controller  # noqa: E402

from eval import EpisodeRunner  # noqa: E402
from mapping import SemanticMapBuilder, visualize_map  # noqa: E402
from method2 import Method2Navigator  # noqa: E402
from method3 import Method3Navigator  # noqa: E402
from shared.interfaces import (  # noqa: E402
    Action,
    BaseNavigator,
    Observation,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

EPISODES = [
    {"scene": "FloorPlan1",  "target": "Mug"},
    {"scene": "FloorPlan2",  "target": "Mug"},
    {"scene": "FloorPlan3",  "target": "Apple"},
    {"scene": "FloorPlan4",  "target": "Apple"},
    {"scene": "FloorPlan5",  "target": "Television"},
    {"scene": "FloorPlan6",  "target": "Television"},
    {"scene": "FloorPlan7",  "target": "Laptop"},
    {"scene": "FloorPlan8",  "target": "Laptop"},
    {"scene": "FloorPlan9",  "target": "Bowl"},
    {"scene": "FloorPlan10", "target": "Bowl"},
]
SEED_BASE = 42
MAX_STEPS = 200
VIDEO_EPISODES_PER_METHOD = 3
SUCCESS_DISTANCE = 1.0      # m
FRAME_W = 300
FRAME_H = 300
VIDEO_FPS = 10

# Claude Haiku 4.5 pricing (USD / million tokens) — Jan 2026.
HAIKU_INPUT_PRICE_PER_MTOK = 1.0
HAIKU_OUTPUT_PRICE_PER_MTOK = 5.0

RESULTS_DIR = PROJECT_ROOT / "results"
MAPS_DIR = RESULTS_DIR / "maps"
VIDEOS_DIR = RESULTS_DIR / "videos"
for d in (MAPS_DIR, VIDEOS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Random baseline
# --------------------------------------------------------------------------- #

class RandomNavigator(BaseNavigator):
    """Uniform-random action baseline.

    Picks each non-STOP action with equal probability, and emits STOP with
    a low fixed probability so episodes can terminate before MAX_STEPS.
    """

    _NON_STOP = (
        Action.MOVE_AHEAD,
        Action.ROTATE_LEFT,
        Action.ROTATE_RIGHT,
        Action.LOOK_UP,
        Action.LOOK_DOWN,
    )

    def __init__(self, seed: int = 0, stop_prob: float = 0.01) -> None:
        self.seed = seed
        self.stop_prob = stop_prob
        self._rng = random.Random(seed)
        self.last_reasoning: str = ""

    def reset(self, target_object: str) -> None:
        # Reseed deterministically per episode so the trace is reproducible.
        self._rng = random.Random(self.seed)
        self.last_reasoning = ""

    def act(self, obs: Observation) -> Action:
        if self._rng.random() < self.stop_prob:
            self.last_reasoning = "random STOP"
            return Action.STOP
        a = self._rng.choice(self._NON_STOP)
        self.last_reasoning = f"random:{a.name}"
        return a


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #

ACTION_TO_THOR = {
    Action.MOVE_AHEAD: "MoveAhead",
    Action.ROTATE_LEFT: "RotateLeft",
    Action.ROTATE_RIGHT: "RotateRight",
    Action.LOOK_UP: "LookUp",
    Action.LOOK_DOWN: "LookDown",
}


def _event_to_obs(event, target: str, start_pos: dict) -> Observation:
    agent = event.metadata["agent"]
    pos = agent["position"]
    heading = float(np.deg2rad(agent["rotation"]["y"]))
    dx = pos["x"] - start_pos["x"]
    dz = pos["z"] - start_pos["z"]
    rgb = np.asarray(event.frame, dtype=np.uint8)
    depth = event.depth_frame
    depth = (np.asarray(depth, dtype=np.float32)
             if depth is not None else np.zeros(rgb.shape[:2], dtype=np.float32))

    # See eval.runner.EpisodeRunner._event_to_obs for matching logic.
    DET_RANGE_M = 3.0
    FOV_HALF_RAD = np.deg2rad(45.0)
    forward = (float(np.sin(heading)), float(np.cos(heading)))
    visible = []
    for obj in event.metadata.get("objects", []):
        op = obj["position"]
        d_dx = op["x"] - pos["x"]
        d_dz = op["z"] - pos["z"]
        dist = float(np.hypot(d_dx, d_dz))
        if obj.get("visible"):
            in_view = True
        elif dist <= DET_RANGE_M and dist > 0:
            cos_angle = (forward[0] * d_dx + forward[1] * d_dz) / dist
            in_view = cos_angle >= float(np.cos(FOV_HALF_RAD))
        else:
            in_view = False
        if in_view:
            visible.append({
                "name": obj["objectType"],
                "dx": d_dx,
                "dz": d_dz,
            })

    return Observation(
        rgb=rgb,
        depth=depth,
        compass=np.array([heading, float(np.hypot(dx, dz))], dtype=np.float32),
        target_object=target,
        visible_objects=visible,
    )


def _target_positions(event, target: str) -> list:
    out = []
    for o in event.metadata.get("objects", []):
        if o.get("objectType") == target:
            p = o["position"]
            out.append({"x": p["x"], "y": p["y"], "z": p["z"]})
    return out


def _min_distance_xz(pos: dict, targets: list) -> float:
    if not targets:
        return 0.0
    return float(min(np.hypot(pos["x"] - t["x"], pos["z"] - t["z"]) for t in targets))


def run_episode(
    controller: Controller,
    navigator: BaseNavigator,
    scene: str,
    target: str,
    seed: int,
    max_steps: int,
    capture_video: bool,
    method_name: str,
) -> dict:
    """Run one episode end-to-end. Returns a dict with metrics + traces."""
    controller.reset(scene=scene)
    # Randomize object placement deterministically for the seed.
    try:
        controller.step(action="InitialRandomSpawn", randomSeed=int(seed),
                        forceVisible=False, numPlacementAttempts=5)
    except Exception:
        # Some FloorPlans / ai2thor versions don't support this; safe to ignore.
        pass

    event = controller.last_event
    start_pos = dict(event.metadata["agent"]["position"])
    targets = _target_positions(event, target)
    d_init = _min_distance_xz(start_pos, targets) if targets else 0.0

    # Separate map JUST for visualization (so the Random baseline can still
    # render a map). Method2/3 maintain their own internally; we visualize
    # this shared builder instead, which gives a consistent rendering across
    # methods.
    viz_builder = SemanticMapBuilder()

    navigator.reset(target)

    frames: list = []
    reasoning_log: list = []
    trajectory: list = [(start_pos["x"], start_pos["z"])]
    d_taken = 0.0
    success = False
    stopped = False
    t_start = time.time()

    for step in range(max_steps):
        obs = _event_to_obs(event, target, start_pos)
        viz_builder.update(obs)
        if capture_video:
            frames.append(np.asarray(event.frame, dtype=np.uint8).copy())

        t0 = time.time()
        try:
            action = navigator.act(obs)
        except Exception as e:  # navigator crashed mid-episode
            print(f"    [error] navigator.act raised {type(e).__name__}: {e}")
            traceback.print_exc()
            break
        act_dt = time.time() - t0

        entry = {
            "step": step,
            "action": action.name,
            "reasoning": getattr(navigator, "last_reasoning", ""),
            "agent_pos": [event.metadata["agent"]["position"]["x"],
                          event.metadata["agent"]["position"]["z"]],
            "act_duration_s": round(act_dt, 4),
        }
        usage = getattr(navigator, "last_usage", None)
        if usage:
            entry["usage"] = dict(usage)
        reasoning_log.append(entry)

        if action == Action.STOP:
            stopped = True
            cur = event.metadata["agent"]["position"]
            success = bool(targets) and _min_distance_xz(cur, targets) <= SUCCESS_DISTANCE
            break

        event = controller.step(action=ACTION_TO_THOR[action])
        new = event.metadata["agent"]["position"]
        d_taken += float(np.hypot(new["x"] - trajectory[-1][0], new["z"] - trajectory[-1][1]))
        trajectory.append((new["x"], new["z"]))

    duration_s = time.time() - t_start
    final = event.metadata["agent"]["position"]
    d_final = _min_distance_xz(final, targets) if targets else 0.0

    d_shortest = d_init  # Euclidean approximation
    spl = 0.0
    if success and d_taken > 0:
        spl = d_shortest / max(d_taken, d_shortest)
    soft_spl = 0.0
    if d_init > 0:
        progress = max(0.0, (d_init - d_final) / d_init)
        soft_spl = progress * (d_shortest / max(d_taken, d_shortest, 1e-6))

    # Visualize the semantic map for this episode.
    map_path = MAPS_DIR / f"{method_name}_{scene}_{target}.png"
    try:
        visualize_map(viz_builder.map, out_path=str(map_path),
                      title=f"{method_name} | {scene} | target={target}")
    except Exception as e:
        print(f"    [warn] map render failed: {e}")

    # Save video if requested.
    video_path: Optional[Path] = None
    if capture_video and frames:
        try:
            import imageio.v2 as imageio
            video_path = VIDEOS_DIR / f"{method_name}_{scene}_{target}.mp4"
            imageio.mimsave(str(video_path), frames, fps=VIDEO_FPS, macro_block_size=1)
        except Exception as e:
            print(f"    [warn] video save failed: {e}")

    return {
        "method": method_name,
        "scene": scene,
        "target": target,
        "seed": seed,
        "success": bool(success),
        "spl": float(spl),
        "soft_spl": float(soft_spl),
        "num_steps": len(reasoning_log),
        "duration_s": float(duration_s),
        "d_init": float(d_init),
        "d_final": float(d_final),
        "d_taken": float(d_taken),
        "n_target_instances": len(targets),
        "stop_issued": stopped,
        "trajectory": trajectory,
        "map_path": str(map_path.relative_to(PROJECT_ROOT)) if map_path.exists() else None,
        "video_path": (str(video_path.relative_to(PROJECT_ROOT))
                       if video_path is not None else None),
        "reasoning_log": reasoning_log,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    success = np.mean([r["success"] for r in results])
    spl = np.mean([r["spl"] for r in results])
    soft_spl = np.mean([r["soft_spl"] for r in results])
    avg_steps = np.mean([r["num_steps"] for r in results])
    avg_time = np.mean([r["duration_s"] for r in results])
    return {
        "n": n,
        "success": float(success),
        "spl": float(spl),
        "soft_spl": float(soft_spl),
        "avg_steps": float(avg_steps),
        "avg_time_s": float(avg_time),
    }


def print_table(rows: list) -> None:
    """rows: list of (method_name, aggregate_dict)."""
    print()
    print("Method      | Success | SPL   | SoftSPL | Avg Steps | Avg Time")
    print("------------|---------|-------|---------|-----------|---------")
    for name, agg in rows:
        print(f"{name:<11} | {agg['success']:<7.2f} | {agg['spl']:<5.3f} | "
              f"{agg['soft_spl']:<7.3f} | {agg['avg_steps']:<9.1f} | "
              f"{agg['avg_time_s']:.1f}s")
    print()


def method3_cost(results: list) -> dict:
    """Sum Claude Haiku 4.5 usage across all Method3 episodes."""
    in_tok = 0
    out_tok = 0
    calls = 0
    for r in results:
        for entry in r["reasoning_log"]:
            u = entry.get("usage")
            if not u:
                continue
            in_tok += int(u.get("input_tokens", 0))
            out_tok += int(u.get("output_tokens", 0))
            calls += 1
    cost = (in_tok * HAIKU_INPUT_PRICE_PER_MTOK
            + out_tok * HAIKU_OUTPUT_PRICE_PER_MTOK) / 1e6
    return {
        "calls": calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_cost_usd": float(cost),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def make_navigator(method_name: str, seed: int) -> BaseNavigator:
    if method_name == "Random":
        return RandomNavigator(seed=seed)
    if method_name == "Method2":
        return Method2Navigator()
    if method_name == "Method3":
        return Method3Navigator()
    raise ValueError(f"unknown method: {method_name}")


def main() -> int:
    print(f"Launching AI2-THOR controller (one shared instance for all episodes)...")
    controller = Controller(scene="FloorPlan1", width=FRAME_W, height=FRAME_H,
                            renderDepthImage=True)

    all_results: dict[str, list] = {}
    aggregates: list = []

    for method_name in ("Random", "Method2", "Method3"):
        print(f"\n{'#' * 60}\n# {method_name}\n{'#' * 60}")
        method_results: list = []
        for ep_idx, ep in enumerate(EPISODES):
            capture_video = ep_idx < VIDEO_EPISODES_PER_METHOD
            seed = SEED_BASE + ep_idx
            scene, target = ep["scene"], ep["target"]
            print(f"  [{method_name}][ep {ep_idx:02d}] scene={scene} target={target} "
                  f"seed={seed} video={capture_video}")
            navigator = make_navigator(method_name, seed)
            try:
                result = run_episode(
                    controller=controller,
                    navigator=navigator,
                    scene=scene,
                    target=target,
                    seed=seed,
                    max_steps=MAX_STEPS,
                    capture_video=capture_video,
                    method_name=method_name,
                )
            except Exception as e:
                print(f"    [error] {type(e).__name__}: {e}")
                traceback.print_exc()
                result = {
                    "method": method_name, "scene": scene, "target": target,
                    "seed": seed, "success": False, "spl": 0.0, "soft_spl": 0.0,
                    "num_steps": 0, "duration_s": 0.0, "d_init": 0.0, "d_final": 0.0,
                    "d_taken": 0.0, "n_target_instances": 0, "stop_issued": False,
                    "trajectory": [], "map_path": None, "video_path": None,
                    "reasoning_log": [], "error": f"{type(e).__name__}: {e}",
                }
            print(f"     -> success={result['success']} spl={result['spl']:.3f} "
                  f"soft_spl={result['soft_spl']:.3f} steps={result['num_steps']} "
                  f"time={result['duration_s']:.1f}s")
            method_results.append(result)

        all_results[method_name] = method_results
        agg = aggregate(method_results)
        aggregates.append((method_name, agg))
        print(f"\nResults so far:")
        print_table(aggregates)

    controller.stop()

    # Save JSON. Strip the trajectory + reasoning_log out of the metrics file
    # (it's bulky); keep them in llm_reasoning.json for Method3.
    metrics_payload: dict = {"config": {
        "episodes": EPISODES,
        "max_steps": MAX_STEPS,
        "success_distance_m": SUCCESS_DISTANCE,
        "seed_base": SEED_BASE,
        "video_episodes_per_method": VIDEO_EPISODES_PER_METHOD,
    }, "aggregates": {name: agg for name, agg in aggregates}, "episodes_per_method": {}}
    for name, results in all_results.items():
        compact = []
        for r in results:
            r2 = {k: v for k, v in r.items() if k not in ("reasoning_log", "trajectory")}
            compact.append(r2)
        metrics_payload["episodes_per_method"][name] = compact

    cost = method3_cost(all_results.get("Method3", []))
    metrics_payload["method3_cost"] = cost

    out = RESULTS_DIR / "small_eval.json"
    out.write_text(json.dumps(metrics_payload, indent=2))
    print(f"\nSaved metrics to {out.relative_to(PROJECT_ROOT)}")

    # Method3 reasoning log (separate, possibly large).
    reasoning_payload = {
        "model": "claude-haiku-4-5-20251001",
        "cost": cost,
        "episodes": [
            {
                "scene": r["scene"], "target": r["target"], "seed": r["seed"],
                "success": r["success"], "spl": r["spl"], "soft_spl": r["soft_spl"],
                "reasoning_log": r["reasoning_log"],
            }
            for r in all_results.get("Method3", [])
        ],
    }
    reasoning_out = RESULTS_DIR / "llm_reasoning.json"
    reasoning_out.write_text(json.dumps(reasoning_payload, indent=2))
    print(f"Saved Method3 reasoning to {reasoning_out.relative_to(PROJECT_ROOT)}")

    # Final summary
    print("\n=== FINAL RESULTS ===")
    print_table(aggregates)
    print(f"Method3 API: {cost['calls']} calls, "
          f"{cost['input_tokens']} input + {cost['output_tokens']} output tokens, "
          f"total cost = ${cost['total_cost_usd']:.4f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
