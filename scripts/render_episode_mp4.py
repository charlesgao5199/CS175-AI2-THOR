"""Create an MP4 from frames saved by run_random_agent.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objectnav.recording import write_mp4_from_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", help="Directory containing frames/step_*.png.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--output", help="Output MP4 path. Defaults to episode.mp4.")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    output_path = Path(args.output) if args.output else episode_dir / "episode.mp4"
    mp4_path = write_mp4_from_frames(
        frames_dir=episode_dir / "frames",
        output_path=output_path,
        fps=args.fps,
    )
    if mp4_path is None:
        raise RuntimeError(f"No frames found in {episode_dir / 'frames'}")

    print(mp4_path)


if __name__ == "__main__":
    main()
