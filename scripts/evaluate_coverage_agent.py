"""Evaluate the coverage-oriented ObjectNav baseline over multiple episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objectnav.config import load_simple_yaml
from objectnav.coverage_agent import run_coverage_episode
from objectnav.env import ObjectNavEnv
from objectnav.evaluation import episode_row, summary_payload, write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coverage_agent.yaml")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--platform", choices=("default", "cloud"))
    parser.add_argument("--scan-rotations", type=int)
    parser.add_argument("--scan-interval", type=int)
    parser.add_argument("--loop-window", type=int)
    parser.add_argument("--revisit-threshold", type=int)
    parser.add_argument("--cell-size", type=float)
    parser.add_argument("--save-dir", default="outputs/eval_coverage")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    scenes = args.scenes or [str(config["scene"])]
    targets = args.targets or [str(config["target_object_type"])]
    max_steps = (
        args.max_steps if args.max_steps is not None else int(config.get("max_steps", 100))
    )
    platform = args.platform or str(config.get("platform", "default"))
    scan_rotations = (
        args.scan_rotations
        if args.scan_rotations is not None
        else int(config.get("scan_rotations", 4))
    )
    scan_interval = (
        args.scan_interval
        if args.scan_interval is not None
        else int(config.get("scan_interval", 1))
    )
    loop_window = (
        args.loop_window
        if args.loop_window is not None
        else int(config.get("loop_window", 8))
    )
    revisit_threshold = (
        args.revisit_threshold
        if args.revisit_threshold is not None
        else int(config.get("revisit_threshold", 3))
    )
    cell_size = (
        args.cell_size
        if args.cell_size is not None
        else float(config.get("cell_size", 0.25))
    )

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
                    summary = run_coverage_episode(
                        env=env,
                        max_steps=max_steps,
                        seed=seed,
                        scan_rotations=scan_rotations,
                        scan_interval=scan_interval,
                        loop_window=loop_window,
                        revisit_threshold=revisit_threshold,
                        cell_size=cell_size,
                    ).to_dict()
                    row = episode_row(
                        episode_id=episode_id,
                        scene=scene,
                        target_object_type=target_object_type,
                        seed=seed,
                        max_steps=max_steps,
                        summary=summary,
                    )
                except Exception as exc:
                    row = episode_row(
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
    write_results(save_dir=save_dir, rows=rows)
    print(json.dumps(summary_payload(rows), indent=2, sort_keys=True))
    print(f"wrote {save_dir / 'results.csv'}")
    print(f"wrote {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
