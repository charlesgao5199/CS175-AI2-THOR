"""Run a heuristic ObjectNav baseline episode in AI2-THOR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objectnav.config import load_simple_yaml
from objectnav.env import ObjectNavEnv
from objectnav.heuristic_agent import run_sweep_move_episode
from objectnav.recording import EpisodeRecorder


def _merged_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_simple_yaml(args.config)
    overrides = {
        "scene": args.scene,
        "target_object_type": args.target,
        "platform": args.platform,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "scan_rotations": args.scan_rotations,
        "recovery_turns": args.recovery_turns,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.render_depth:
        config["render_depth"] = True
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/heuristic_agent.yaml")
    parser.add_argument("--scene")
    parser.add_argument("--target")
    parser.add_argument("--platform", choices=("default", "cloud"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scan-rotations", type=int)
    parser.add_argument("--recovery-turns", type=int)
    parser.add_argument(
        "--render-depth",
        action="store_true",
        help="Request AI2-THOR depth frames. RGB-only is more stable for visualization.",
    )
    parser.add_argument("--save-dir", help="Optional directory for frames and plots.")
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=4,
        help="GIF frame rate when --save-dir is set. Use 0 to skip GIF export.",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also export episode.mp4 when --save-dir is set. Requires ffmpeg.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        help="MP4 frame rate. Defaults to --gif-fps.",
    )
    args = parser.parse_args()

    config = _merged_config(args)
    if args.mp4 and not args.save_dir:
        raise ValueError("--mp4 requires --save-dir")

    recorder = (
        EpisodeRecorder(
            save_dir=Path(args.save_dir),
            target_object_type=str(config["target_object_type"]),
        )
        if args.save_dir
        else None
    )
    env = ObjectNavEnv(
        scene=str(config["scene"]),
        target_object_type=str(config["target_object_type"]),
        width=int(config.get("width", 300)),
        height=int(config.get("height", 300)),
        visibility_distance=float(config.get("visibility_distance", 1.5)),
        platform=str(config.get("platform", "default")),
        render_depth=bool(config.get("render_depth", False)),
    )

    try:
        summary = run_sweep_move_episode(
            env=env,
            max_steps=int(config.get("max_steps", 100)),
            seed=int(config.get("seed", 0)),
            scan_rotations=int(config.get("scan_rotations", 4)),
            recovery_turns=config.get("recovery_turns"),
            recorder=recorder,
        )
    finally:
        env.stop()

    summary_dict = summary.to_dict()
    if recorder is not None:
        recorder.write_metadata(summary_dict)
        recorder.write_trajectory()
        recorder.write_gif(fps=args.gif_fps)
        if args.mp4:
            recorder.write_mp4(fps=args.mp4_fps or args.gif_fps)

    print(json.dumps(summary_dict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
