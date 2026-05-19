"""Integration test for the ObjectNav project.

Verifies that each component (mapping, Method2, Method3, eval) is wired up
correctly with AI2-THOR. Each subtest is isolated — if one fails, the test
prints the full traceback and continues with the next component.

Run:
    conda activate objectnav
    python src/integration_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

RESULTS: list[tuple[str, bool, str]] = []  # (component, passed, message)


def run_subtest(name: str, fn: Callable[[], Optional[str]]) -> None:
    """Run a subtest, printing PASS/FAIL and capturing the result.

    The subtest function may return an optional short status string to
    include in the PASS line. Any exception is caught and reported.
    """
    print(f"\n{'=' * 70}")
    print(f"[TEST] {name}")
    print("=" * 70)
    try:
        msg = fn() or ""
        print(f"[PASS] {name}  {msg}")
        RESULTS.append((name, True, msg))
    except Exception as e:  # noqa: BLE001  — integration test wants all errors
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Make `import shared`, `import mapping`, ... work when running this file
# directly (e.g. `python src/integration_test.py`).
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shared.interfaces import (  # noqa: E402
    Action,
    BaseNavigator,
    EpisodeResult,
    Observation,
    SemanticMap,
)


def event_to_observation(event, target_object: str, start_pos: dict) -> Observation:
    """Convert an AI2-THOR event into our Observation dataclass.

    Compass = (heading_rad, distance_from_start_meters).
    """
    agent = event.metadata["agent"]
    pos = agent["position"]
    rot_deg = agent["rotation"]["y"]
    heading_rad = float(np.deg2rad(rot_deg))
    dx = pos["x"] - start_pos["x"]
    dz = pos["z"] - start_pos["z"]
    distance = float(np.hypot(dx, dz))

    rgb = np.asarray(event.frame, dtype=np.uint8)
    depth = event.depth_frame
    if depth is None:
        # depth rendering not enabled — use zeros so downstream still runs
        depth = np.zeros(rgb.shape[:2], dtype=np.float32)
    else:
        depth = np.asarray(depth, dtype=np.float32)

    return Observation(
        rgb=rgb,
        depth=depth,
        compass=np.array([heading_rad, distance], dtype=np.float32),
        target_object=target_object,
    )


ACTION_TO_THOR = {
    Action.MOVE_AHEAD: "MoveAhead",
    Action.ROTATE_LEFT: "RotateLeft",
    Action.ROTATE_RIGHT: "RotateRight",
    Action.LOOK_UP: "LookUp",
    Action.LOOK_DOWN: "LookDown",
    Action.STOP: "Done",
}


# --------------------------------------------------------------------------- #
# AI2-THOR controller (shared between subtests)
# --------------------------------------------------------------------------- #

CONTROLLER = None
SCENE_LABEL = "<none>"
TARGET_OBJECT = "Mug"
START_POS: dict = {"x": 0.0, "y": 0.0, "z": 0.0}


def launch_controller() -> str:
    """Try ProcTHOR via `prior`; fall back to FloorPlan1 if unavailable."""
    global CONTROLLER, SCENE_LABEL, START_POS

    from ai2thor.controller import Controller

    scene_label = "FloorPlan1"
    init_scene = "FloorPlan1"

    # Optional: try ProcTHOR (only if user has `prior` installed)
    try:
        import prior  # type: ignore

        dataset = prior.load_dataset("procthor-10k")
        house = dataset["train"][0]
        init_scene = house  # ai2thor accepts a house JSON as `scene`
        scene_label = "procthor-10k:train[0]"
        print(f"  Loaded ProcTHOR scene: {scene_label}")
    except Exception as e:
        print(f"  ProcTHOR unavailable ({type(e).__name__}: {e}); using FloorPlan1.")

    CONTROLLER = Controller(
        scene=init_scene,
        width=300,
        height=300,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )
    SCENE_LABEL = scene_label
    agent = CONTROLLER.last_event.metadata["agent"]
    START_POS = dict(agent["position"])
    print(f"  Scene OK — agent at {START_POS}, heading={agent['rotation']['y']}°")
    return f"scene={scene_label}"


# --------------------------------------------------------------------------- #
# Subtest 1 — SemanticMapBuilder
# --------------------------------------------------------------------------- #

def test_semantic_map_builder() -> str:
    from mapping import SemanticMapBuilder  # type: ignore  # noqa: F401

    builder = SemanticMapBuilder()
    event = CONTROLLER.last_event
    obs = event_to_observation(event, TARGET_OBJECT, START_POS)
    smap = builder.update(obs)

    assert isinstance(smap, SemanticMap), (
        f"SemanticMapBuilder.update must return a SemanticMap, got {type(smap)}"
    )
    explored = int(smap.explored.sum())
    return f"explored={explored} cells, grid_size={smap.grid_size}"


# --------------------------------------------------------------------------- #
# Subtest 2 — Method2Navigator (30 steps)
# --------------------------------------------------------------------------- #

def _run_navigator_loop(nav: BaseNavigator, label: str, num_steps: int = 30) -> str:
    """Drive a navigator for N steps, printing each step.

    Returns a one-line summary suitable for the PASS message.
    """
    counts = {a: 0 for a in Action}
    stopped_at = None

    for step in range(num_steps):
        obs = event_to_observation(CONTROLLER.last_event, TARGET_OBJECT, START_POS)
        action = nav.act(obs)
        counts[action] += 1
        # Reasoning trace: some LLM navigators expose `last_reasoning`
        reasoning = getattr(nav, "last_reasoning", None)
        rtail = f" :: {reasoning}" if reasoning else ""
        print(f"  [{label} step {step:02d}] action={action.name}{rtail}")

        if action == Action.STOP:
            stopped_at = step
            break
        thor_action = ACTION_TO_THOR[action]
        CONTROLLER.step(action=thor_action)

    summary = ", ".join(f"{a.name}={n}" for a, n in counts.items() if n)
    stop_note = f", stopped@{stopped_at}" if stopped_at is not None else ""
    return summary + stop_note


def test_method2_navigator() -> str:
    from method2 import Method2Navigator  # type: ignore

    nav = Method2Navigator()
    nav.reset(TARGET_OBJECT)
    return _run_navigator_loop(nav, "M2", num_steps=30)


# --------------------------------------------------------------------------- #
# Subtest 3 — Method3Navigator (30 steps, LLM)
# --------------------------------------------------------------------------- #

def test_method3_navigator() -> str:
    from method3 import Method3Navigator  # type: ignore

    nav = Method3Navigator()
    nav.reset(TARGET_OBJECT)
    return _run_navigator_loop(nav, "M3", num_steps=30)


# --------------------------------------------------------------------------- #
# Subtest 4 — EpisodeRunner + Method2Navigator
# --------------------------------------------------------------------------- #

def test_episode_runner() -> str:
    from eval import EpisodeRunner  # type: ignore
    from method2 import Method2Navigator  # type: ignore

    runner = EpisodeRunner(CONTROLLER)
    nav = Method2Navigator()
    result = runner.run(nav, target_object=TARGET_OBJECT, scene_id=SCENE_LABEL)

    assert isinstance(result, EpisodeResult), (
        f"EpisodeRunner.run must return EpisodeResult, got {type(result)}"
    )
    print(f"  EpisodeResult: success={result.success} spl={result.spl:.3f} "
          f"soft_spl={result.soft_spl:.3f} steps={result.num_steps} "
          f"target={result.target_object} scene={result.scene_id}")
    return f"success={result.success} spl={result.spl:.3f} steps={result.num_steps}"


# --------------------------------------------------------------------------- #
# Subtest 5 — Map visualization to results/test_map.png
# --------------------------------------------------------------------------- #

def test_save_map_viz() -> str:
    out = RESULTS_DIR / "test_map.png"

    # Try the project visualizer first; if it isn't implemented yet, fall back
    # to a minimal matplotlib rendering so this subtest is still useful.
    smap: Optional[SemanticMap] = None
    try:
        from mapping import SemanticMapBuilder  # type: ignore

        builder = SemanticMapBuilder()
        obs = event_to_observation(CONTROLLER.last_event, TARGET_OBJECT, START_POS)
        smap = builder.update(obs)
    except Exception as e:
        print(f"  SemanticMapBuilder unavailable ({type(e).__name__}: {e}) — "
              "using empty placeholder map.")
        smap = SemanticMap(
            grid_size=64,
            explored=np.zeros((64, 64), dtype=bool),
            traversable=np.zeros((64, 64), dtype=bool),
        )

    visualizer_used = "fallback-matplotlib"
    try:
        from mapping import visualize_map  # type: ignore

        visualize_map(smap, out_path=str(out))
        visualizer_used = "mapping.visualize_map"
    except Exception as e:
        print(f"  mapping.visualize_map unavailable ({type(e).__name__}: {e}) "
              "— rendering with matplotlib.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        # Compose: gray = unexplored, white = explored+traversable, black =
        # explored+obstacle.
        canvas = np.full(smap.explored.shape + (3,), 0.5, dtype=np.float32)
        canvas[smap.explored & smap.traversable] = 1.0
        canvas[smap.explored & ~smap.traversable] = 0.0
        ax.imshow(canvas, origin="lower")
        ax.set_title(f"Semantic map — {SCENE_LABEL}\nexplored={int(smap.explored.sum())} cells")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)

    assert out.exists() and out.stat().st_size > 0, f"map png was not written to {out}"
    return f"wrote {out.relative_to(PROJECT_ROOT)} via {visualizer_used}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("Integration test starting.")
    run_subtest("launch AI2-THOR controller", launch_controller)

    if CONTROLLER is None:
        print("\nNo controller available — skipping component tests.")
        return _summarize()

    run_subtest("SemanticMapBuilder.update(one obs)", test_semantic_map_builder)
    run_subtest("Method2Navigator 30 steps", test_method2_navigator)
    run_subtest("Method3Navigator 30 steps", test_method3_navigator)
    run_subtest("EpisodeRunner + Method2Navigator", test_episode_runner)
    run_subtest("save map visualization", test_save_map_viz)

    try:
        CONTROLLER.stop()
    except Exception:  # noqa: BLE001
        pass

    return _summarize()


def _summarize() -> int:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    for name, ok, msg in RESULTS:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}  {msg}")
    print("-" * 70)
    print(f"  {n_pass} passed, {n_fail} failed, {len(RESULTS)} total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
