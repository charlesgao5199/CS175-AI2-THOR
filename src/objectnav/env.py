"""AI2-THOR environment wrapper for object-goal navigation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ai2thor.controller import Controller


DEFAULT_ACTIONS: Sequence[str] = (
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
)


@dataclass(frozen=True)
class TargetObservation:
    """Target-object visibility information for one AI2-THOR event."""

    target_object_type: str
    total_instances: int
    visible_instances: int
    success: bool
    closest_distance: Optional[float]
    closest_visible_distance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_object_type": self.target_object_type,
            "total_instances": self.total_instances,
            "visible_instances": self.visible_instances,
            "success": self.success,
            "closest_distance": self.closest_distance,
            "closest_visible_distance": self.closest_visible_distance,
        }


def _platform_value(platform: str):
    if platform == "default":
        return None
    if platform == "cloud":
        from ai2thor.platform import CloudRendering

        return CloudRendering
    raise ValueError(f"Unsupported platform: {platform}")


def _object_distance(obj: Dict[str, Any]) -> Optional[float]:
    distance = obj.get("distance")
    if isinstance(distance, (int, float)):
        return float(distance)
    return None


def target_observation(
    event: Any, target_object_type: str, visibility_distance: float
) -> TargetObservation:
    target_objects = [
        obj
        for obj in event.metadata.get("objects", [])
        if obj.get("objectType") == target_object_type
    ]
    visible_objects = [obj for obj in target_objects if obj.get("visible", False)]

    all_distances = [
        distance
        for distance in (_object_distance(obj) for obj in target_objects)
        if distance is not None
    ]
    visible_distances = [
        distance
        for distance in (_object_distance(obj) for obj in visible_objects)
        if distance is not None
    ]

    success = any(
        _object_distance(obj) is None or _object_distance(obj) <= visibility_distance
        for obj in visible_objects
    )

    return TargetObservation(
        target_object_type=target_object_type,
        total_instances=len(target_objects),
        visible_instances=len(visible_objects),
        success=success,
        closest_distance=min(all_distances) if all_distances else None,
        closest_visible_distance=min(visible_distances) if visible_distances else None,
    )


class ObjectNavEnv:
    """Thin AI2-THOR Controller wrapper with ObjectNav-oriented helpers."""

    def __init__(
        self,
        scene: str,
        target_object_type: str,
        width: int = 300,
        height: int = 300,
        visibility_distance: float = 1.5,
        platform: str = "default",
        render_depth: bool = True,
    ) -> None:
        self.scene = scene
        self.target_object_type = target_object_type
        self.width = width
        self.height = height
        self.visibility_distance = visibility_distance
        self.platform = platform
        self.render_depth = render_depth
        self.controller: Optional[Controller] = None

    def start(self) -> Any:
        kwargs: Dict[str, Any] = {
            "scene": self.scene,
            "width": self.width,
            "height": self.height,
            "renderDepthImage": self.render_depth,
        }
        platform_value = _platform_value(self.platform)
        if platform_value is not None:
            kwargs["platform"] = platform_value

        self.controller = Controller(**kwargs)
        return self.controller.last_event

    def stop(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def step(self, action: str) -> Any:
        if self.controller is None:
            raise RuntimeError("ObjectNavEnv.start() must be called before step().")
        return self.controller.step(action=action)

    def observe_target(self, event: Any) -> TargetObservation:
        return target_observation(
            event=event,
            target_object_type=self.target_object_type,
            visibility_distance=self.visibility_distance,
        )

    def __enter__(self) -> "ObjectNavEnv":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

