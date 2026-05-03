"""Minimal AI2-THOR smoke test for the Docker dev environment."""

from __future__ import annotations

import argparse
from contextlib import suppress

from ai2thor.controller import Controller


def build_controller(scene: str, width: int, height: int, platform: str) -> Controller:
    kwargs = {
        "scene": scene,
        "width": width,
        "height": height,
        "renderDepthImage": True,
    }

    if platform == "cloud":
        from ai2thor.platform import CloudRendering

        kwargs["platform"] = CloudRendering

    return Controller(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan10")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument(
        "--platform",
        choices=("auto", "cloud", "default"),
        default="auto",
        help="Use cloud rendering first by default, then fall back to default rendering.",
    )
    args = parser.parse_args()

    platforms = ["cloud", "default"] if args.platform == "auto" else [args.platform]
    last_error: Exception | None = None

    for platform in platforms:
        controller = None
        try:
            controller = build_controller(args.scene, args.width, args.height, platform)
            event = controller.step(action="RotateRight")
            print(f"AI2-THOR smoke test passed with platform={platform}")
            print(f"scene={args.scene}")
            print(f"frame_shape={event.frame.shape}")
            print(f"depth_shape={event.depth_frame.shape}")
            print(f"agent_position={event.metadata['agent']['position']}")
            return
        except Exception as exc:
            last_error = exc
            print(f"platform={platform} failed: {exc}")
        finally:
            if controller is not None:
                with suppress(Exception):
                    controller.stop()

    raise RuntimeError("AI2-THOR smoke test failed for all platforms") from last_error


if __name__ == "__main__":
    main()
