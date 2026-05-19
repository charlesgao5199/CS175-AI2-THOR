"""Shared evaluation result helpers for ObjectNav baselines."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


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


def episode_row(
    episode_id: int,
    scene: str,
    target_object_type: str,
    seed: int,
    max_steps: int,
    summary: Optional[Dict[str, Any]] = None,
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


def success_rate(rows: Iterable[Dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["success"] is True) / len(rows)


def average_steps(rows: Iterable[Dict[str, Any]], successes_only: bool) -> Optional[float]:
    selected_steps = [
        int(row["steps_taken"])
        for row in rows
        if row["steps_taken"] != "" and (not successes_only or row["success"] is True)
    ]
    if not selected_steps:
        return None
    return mean(selected_steps)


def group_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    return {
        group_key: {
            "episodes": len(group_rows),
            "successes": sum(1 for row in group_rows if row["success"] is True),
            "success_rate": success_rate(group_rows),
            "average_steps_all": average_steps(group_rows, successes_only=False),
            "average_steps_successes": average_steps(
                group_rows, successes_only=True
            ),
        }
        for group_key, group_rows in sorted(grouped.items())
    }


def summary_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "successes": sum(1 for row in rows if row["success"] is True),
        "errors": sum(1 for row in rows if row["error"]),
        "success_rate": success_rate(rows),
        "average_steps_all": average_steps(rows, successes_only=False),
        "average_steps_successes": average_steps(rows, successes_only=True),
        "by_scene": group_summary(rows, "scene"),
        "by_target": group_summary(rows, "target_object_type"),
    }


def write_results(save_dir: Path, rows: List[Dict[str, Any]]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "results.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (save_dir / "summary.json").write_text(
        json.dumps(summary_payload(rows), indent=2, sort_keys=True)
    )
