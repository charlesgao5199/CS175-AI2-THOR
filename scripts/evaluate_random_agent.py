"""Evaluate the random ObjectNav baseline over multiple episodes."""

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
from objectnav.env import ObjectNavEnv
from objectnav.evaluation import episode_row, summary_payload, write_results
from objectnav.random_agent import run_random_episode


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
