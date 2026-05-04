"""Episode recording and visualization utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from PIL import Image


def write_gif_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 4,
) -> Optional[Path]:
    if fps <= 0:
        return None

    frame_paths = sorted(frames_dir.glob("step_*.png"))
    if not frame_paths:
        return None

    duration_ms = max(1, round(1000 / fps))
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def write_mp4_from_frames(
    frames_dir: Path,
    output_path: Path,
    fps: int = 4,
) -> Optional[Path]:
    if fps <= 0:
        return None

    frame_paths = sorted(frames_dir.glob("step_*.png"))
    if not frame_paths:
        return None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to export MP4. Install it with: "
            "sudo apt-get install -y ffmpeg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_pattern = str(frames_dir / "step_%04d.png")
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg failed to export MP4:\n{error.stderr}") from error

    return output_path


@dataclass
class StepRecord:
    step: int
    action: Optional[str]
    last_action_success: Optional[bool]
    position: Dict[str, float]
    rotation: Dict[str, float]
    visible_objects: List[str]
    target_visible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "last_action_success": self.last_action_success,
            "position": self.position,
            "rotation": self.rotation,
            "visible_objects": self.visible_objects,
            "target_visible": self.target_visible,
        }


@dataclass
class EpisodeRecorder:
    save_dir: Path
    target_object_type: str
    records: List[StepRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._clear_previous_outputs()

    @property
    def frames_dir(self) -> Path:
        return self.save_dir / "frames"

    def _clear_previous_outputs(self) -> None:
        for artifact_name in (
            "episode.json",
            "episode.gif",
            "episode.mp4",
            "trajectory.png",
        ):
            artifact_path = self.save_dir / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()

        for frame_path in self.frames_dir.glob("step_*.png"):
            frame_path.unlink()

    def record(self, event: Any, step: int, action: Optional[str]) -> None:
        metadata = event.metadata
        visible_objects = sorted(
            {
                obj.get("objectType", "Unknown")
                for obj in metadata.get("objects", [])
                if obj.get("visible", False)
            }
        )
        target_visible = self.target_object_type in visible_objects

        self.records.append(
            StepRecord(
                step=step,
                action=action,
                last_action_success=metadata.get("lastActionSuccess"),
                position=metadata["agent"]["position"],
                rotation=metadata["agent"]["rotation"],
                visible_objects=visible_objects,
                target_visible=target_visible,
            )
        )

        frame_path = self.frames_dir / f"step_{step:04d}.png"
        Image.fromarray(event.frame).save(frame_path)

    def write_metadata(self, summary: Dict[str, Any]) -> None:
        payload = {
            "summary": summary,
            "steps": [record.to_dict() for record in self.records],
        }
        (self.save_dir / "episode.json").write_text(json.dumps(payload, indent=2))

    def write_trajectory(self) -> None:
        if not self.records:
            return

        xs = [record.position["x"] for record in self.records]
        zs = [record.position["z"] for record in self.records]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, zs, marker="o", linewidth=1.5, markersize=3)
        ax.scatter(xs[0], zs[0], c="green", label="start", s=60, zorder=3)
        ax.scatter(xs[-1], zs[-1], c="red", label="end", s=60, zorder=3)

        for record in self.records:
            if record.target_visible:
                ax.scatter(
                    record.position["x"],
                    record.position["z"],
                    c="gold",
                    edgecolor="black",
                    label="target visible",
                    s=80,
                    zorder=4,
                )
                break

        ax.set_title("Agent Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        ax.legend(deduped.values(), deduped.keys())

        fig.tight_layout()
        fig.savefig(self.save_dir / "trajectory.png", dpi=160)
        plt.close(fig)

    def write_gif(self, fps: int = 4) -> Optional[Path]:
        return write_gif_from_frames(
            frames_dir=self.frames_dir,
            output_path=self.save_dir / "episode.gif",
            fps=fps,
        )

    def write_mp4(self, fps: int = 4) -> Optional[Path]:
        return write_mp4_from_frames(
            frames_dir=self.frames_dir,
            output_path=self.save_dir / "episode.mp4",
            fps=fps,
        )
