"""Evaluate the random ObjectNav baseline over multiple episodes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objectnav.config import load_simple_yaml
from objectnav.env import ObjectNavEnv
from objectnav.random_agent import run_random_episode


RESULT_FIELDS = [
    "episode_id",
    "scene",
    "target_object_type",
    "seed",
    "max_steps",
    "success",
    "stop_reason",
    "steps_taken",
    "last_action",
    "last_action_success",
    "final_x",
    "final_y",
    "final_z",
    "target_total_instances",
    "target_visible_instances",
    "closest_distance",
    "closest_visible_distance",
    "error",
]


def _episode_row(
    episode_id: int,
    scene: str,
    target_object_type: str,
    seed: int,
    max_steps: int,
    summary: Dict[str, Any] | None = None,
    error: str = "",
) -> Dict[str, Any]:
    if summary is None:
        return {
            "episode_id": episode_id,
            "scene": scene,
            "target_object_type": target_object_type,
            "seed": seed,
            "max_steps": max_steps,
            "success": False,
            "stop_reason": "error",
            "steps_taken": "",
            "last_action": "",
            "last_action_success": "",
            "final_x": "",
            "final_y": "",
            "final_z": "",
            "target_total_instances": "",
            "target_visible_instances": "",
            "closest_distance": "",
            "closest_visible_distance": "",
            "error": error,
        }

    final_position = summary["final_position"]
    target_observation = summary["target_observation"]
    return {
        "episode_id": episode_id,
        "scene": summary["scene"],
        "target_object_type": summary["target_object_type"],
        "seed": summary["seed"],
        "max_steps": summary["max_steps"],
        "success": summary["success"],
        "stop_reason": summary["stop_reason"],
        "steps_taken": summary["steps_taken"],
        "last_action": summary["last_action"],
        "last_action_success": summary["last_action_success"],
        "final_x": final_position["x"],
        "final_y": final_position["y"],
        "final_z": final_position["z"],
        "target_total_instances": target_observation["total_instances"],
        "target_visible_instances": target_observation["visible_instances"],
        "closest_distance": target_observation["closest_distance"],
        "closest_visible_distance": target_observation["closest_visible_distance"],
        "error": "",
    }


def _success_rate(rows: Iterable[Dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["success"] is True) / len(rows)


def _average_steps(rows: Iterable[Dict[str, Any]], successes_only: bool) -> float | None:
    selected_steps = [
        int(row["steps_taken"])
        for row in rows
        if row["steps_taken"] != "" and (not successes_only or row["success"] is True)
    ]
    if not selected_steps:
        return None
    return mean(selected_steps)


def _group_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    return {
        group_key: {
            "episodes": len(group_rows),
            "successes": sum(1 for row in group_rows if row["success"] is True),
            "success_rate": _success_rate(group_rows),
            "average_steps_all": _average_steps(group_rows, successes_only=False),
            "average_steps_successes": _average_steps(group_rows, successes_only=True),
        }
        for group_key, group_rows in sorted(grouped.items())
    }


def _summary_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "successes": sum(1 for row in rows if row["success"] is True),
        "errors": sum(1 for row in rows if row["error"]),
        "success_rate": _success_rate(rows),
        "average_steps_all": _average_steps(rows, successes_only=False),
        "average_steps_successes": _average_steps(rows, successes_only=True),
        "by_scene": _group_summary(rows, "scene"),
        "by_target": _group_summary(rows, "target_object_type"),
    }


def _write_results(save_dir: Path, rows: List[Dict[str, Any]]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "results.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (save_dir / "summary.json").write_text(
        json.dumps(_summary_payload(rows), indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/random_agent.yaml")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--platform", choices=("default", "cloud"))
    parser.add_argument("--save-dir", default="outputs/eval_random")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    scenes = args.scenes or [str(config["scene"])]
    targets = args.targets or [str(config["target_object_type"])]
    max_steps = args.max_steps or int(config.get("max_steps", 100))
    platform = args.platform or str(config.get("platform", "default"))

    rows: List[Dict[str, Any]] = []
    episode_id = 0
    for scene in scenes:
        for target_object_type in targets:
            for seed in args.seeds:
                episode_id += 1
                env = ObjectNavEnv(
                    scene=scene,
                    target_object_type=target_object_type,
                    width=int(config.get("width", 300)),
                    height=int(config.get("height", 300)),
                    visibility_distance=float(config.get("visibility_distance", 1.5)),
                    platform=platform,
                    render_depth=bool(config.get("render_depth", False)),
                )
                try:
                    summary = run_random_episode(
                        env=env,
                        max_steps=max_steps,
                        seed=seed,
                    ).to_dict()
                    row = _episode_row(
                        episode_id=episode_id,
                        scene=scene,
                        target_object_type=target_object_type,
                        seed=seed,
                        max_steps=max_steps,
                        summary=summary,
                    )
                except Exception as exc:
                    row = _episode_row(
                        episode_id=episode_id,
                        scene=scene,
                        target_object_type=target_object_type,
                        seed=seed,
                        max_steps=max_steps,
                        error=str(exc),
                    )
                finally:
                    env.stop()

                rows.append(row)
                print(
                    json.dumps(
                        {
                            "episode_id": row["episode_id"],
                            "scene": row["scene"],
                            "target": row["target_object_type"],
                            "seed": row["seed"],
                            "success": row["success"],
                            "steps_taken": row["steps_taken"],
                            "stop_reason": row["stop_reason"],
                        },
                        sort_keys=True,
                    )
                )

    save_dir = Path(args.save_dir)
    _write_results(save_dir=save_dir, rows=rows)
    print(json.dumps(_summary_payload(rows), indent=2, sort_keys=True))
    print(f"wrote {save_dir / 'results.csv'}")
    print(f"wrote {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
