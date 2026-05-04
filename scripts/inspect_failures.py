"""Replay failed evaluation episodes and save visual diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objectnav.config import load_simple_yaml
from objectnav.env import ObjectNavEnv
from objectnav.heuristic_agent import run_sweep_move_episode
from objectnav.random_agent import run_random_episode
from objectnav.recording import EpisodeRecorder


def _load_rows(results_path: Path) -> List[Dict[str, str]]:
    with results_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _is_failure(row: Dict[str, str]) -> bool:
    return row.get("success", "").lower() != "true"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe.strip("_") or "unknown"


def _episode_dir_name(row: Dict[str, str], agent: str) -> str:
    episode_id = int(row["episode_id"])
    scene = _safe_name(row["scene"])
    target = _safe_name(row["target_object_type"])
    seed = _safe_name(row["seed"])
    return f"{agent}_episode_{episode_id:04d}_{scene}_{target}_seed{seed}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _inspect_row(
    row: Dict[str, str],
    config: Dict[str, Any],
    save_dir: Path,
    agent: str,
    gif_fps: int,
    export_mp4: bool,
    mp4_fps: Optional[int],
    scan_rotations: int,
    recovery_turns: Optional[int],
) -> Dict[str, Any]:
    episode_dir = save_dir / _episode_dir_name(row, agent)
    max_steps = int(row["max_steps"])
    seed = int(row["seed"])
    scene = row["scene"]
    target_object_type = row["target_object_type"]

    recorder = EpisodeRecorder(
        save_dir=episode_dir,
        target_object_type=target_object_type,
    )
    env = ObjectNavEnv(
        scene=scene,
        target_object_type=target_object_type,
        width=int(config.get("width", 300)),
        height=int(config.get("height", 300)),
        visibility_distance=float(config.get("visibility_distance", 1.5)),
        platform=str(config.get("platform", "default")),
        render_depth=bool(config.get("render_depth", False)),
    )

    replay_summary: Optional[Dict[str, Any]] = None
    replay_error = ""
    try:
        if agent == "random":
            replay_summary = run_random_episode(
                env=env,
                max_steps=max_steps,
                seed=seed,
                recorder=recorder,
            ).to_dict()
        elif agent == "heuristic":
            replay_summary = run_sweep_move_episode(
                env=env,
                max_steps=max_steps,
                seed=seed,
                scan_rotations=scan_rotations,
                recovery_turns=recovery_turns,
                recorder=recorder,
            ).to_dict()
        else:
            raise ValueError(f"Unsupported agent: {agent}")
        recorder.write_metadata(replay_summary)
    except Exception as exc:
        replay_error = str(exc)
    finally:
        env.stop()

    if recorder.records:
        recorder.write_trajectory()
        recorder.write_gif(fps=gif_fps)
        if export_mp4:
            recorder.write_mp4(fps=mp4_fps or gif_fps)

    inspection = {
        "agent": agent,
        "source_result": row,
        "replay_summary": replay_summary,
        "replay_error": replay_error,
        "replay_success": bool(replay_summary and replay_summary["success"]),
    }
    _write_json(episode_dir / "inspection.json", inspection)
    if replay_error:
        _write_json(episode_dir / "error.json", {"error": replay_error})

    return {
        "agent": agent,
        "source_episode_id": row["episode_id"],
        "scene": scene,
        "target_object_type": target_object_type,
        "seed": seed,
        "source_stop_reason": row["stop_reason"],
        "source_steps_taken": row["steps_taken"],
        "replay_success": inspection["replay_success"],
        "replay_stop_reason": (
            replay_summary["stop_reason"] if replay_summary is not None else "error"
        ),
        "replay_steps_taken": (
            replay_summary["steps_taken"] if replay_summary is not None else ""
        ),
        "path": str(episode_dir),
    }


def _write_index(save_dir: Path, records: List[Dict[str, Any]]) -> None:
    table_rows = [
        [
            record["agent"],
            str(record["source_episode_id"]),
            record["scene"],
            record["target_object_type"],
            str(record["seed"]),
            record["source_stop_reason"],
            str(record["source_steps_taken"]),
            str(record["replay_success"]),
            record["replay_stop_reason"],
            str(record["replay_steps_taken"]),
            record["path"],
        ]
        for record in records
    ]
    index = "\n\n".join(
        [
            "# Failure Inspection",
            _markdown_table(
                [
                    "agent",
                    "source episode",
                    "scene",
                    "target",
                    "seed",
                    "source stop",
                    "source steps",
                    "replay success",
                    "replay stop",
                    "replay steps",
                    "path",
                ],
                table_rows,
            ),
            "",
        ]
    )
    (save_dir / "index.md").write_text(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", help="Directory containing results.csv.")
    parser.add_argument(
        "--agent",
        choices=("random", "heuristic"),
        default="random",
        help="Agent policy used to replay the failed episodes.",
    )
    parser.add_argument(
        "--config",
        help="Agent config path. Defaults to the matching random or heuristic config.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--scene")
    parser.add_argument("--target")
    parser.add_argument("--scan-rotations", type=int)
    parser.add_argument("--recovery-turns", type=int)
    parser.add_argument("--save-dir", default="outputs/failure_inspection")
    parser.add_argument("--gif-fps", type=int, default=4)
    parser.add_argument("--mp4", action="store_true", help="Also export MP4 videos.")
    parser.add_argument("--mp4-fps", type=int, help="MP4 frame rate. Defaults to GIF FPS.")
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    rows = [row for row in _load_rows(evaluation_dir / "results.csv") if _is_failure(row)]
    if args.scene:
        rows = [row for row in rows if row["scene"] == args.scene]
    if args.target:
        rows = [row for row in rows if row["target_object_type"] == args.target]
    if args.limit > 0:
        rows = rows[: args.limit]

    config_path = args.config or (
        "configs/heuristic_agent.yaml"
        if args.agent == "heuristic"
        else "configs/random_agent.yaml"
    )
    config = load_simple_yaml(config_path)
    scan_rotations = args.scan_rotations or int(config.get("scan_rotations", 4))
    recovery_turns = (
        args.recovery_turns
        if args.recovery_turns is not None
        else config.get("recovery_turns")
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    inspected_records = []
    for row in rows:
        record = _inspect_row(
            row=row,
            config=config,
            save_dir=save_dir,
            agent=args.agent,
            gif_fps=args.gif_fps,
            export_mp4=args.mp4,
            mp4_fps=args.mp4_fps,
            scan_rotations=scan_rotations,
            recovery_turns=recovery_turns,
        )
        inspected_records.append(record)
        print(json.dumps(record, sort_keys=True))

    _write_index(save_dir, inspected_records)
    print(f"wrote {save_dir / 'index.md'}")


if __name__ == "__main__":
    main()
