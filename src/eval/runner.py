"""Episode runner — drives a BaseNavigator inside AI2-THOR and computes metrics.

Metrics implemented:
  - Success: agent emits STOP within `success_distance` of any instance of
    the target object category in the scene.
  - SPL: Success-weighted Path Length
        SPL = success * d_shortest / max(d_taken, d_shortest)
  - SoftSPL: progress-weighted variant
        SoftSPL = max(0, (d_init - d_final) / d_init) *
                  (d_shortest / max(d_taken, d_shortest))

`d_shortest` is approximated with Euclidean distance for now (a TODO is to
use the controller's geodesic GetShortestPath service).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from shared.interfaces import Action, BaseNavigator, EpisodeResult, Observation


ACTION_TO_THOR = {
    Action.MOVE_AHEAD: "MoveAhead",
    Action.ROTATE_LEFT: "RotateLeft",
    Action.ROTATE_RIGHT: "RotateRight",
    Action.LOOK_UP: "LookUp",
    Action.LOOK_DOWN: "LookDown",
}


class EpisodeRunner:
    """Drives a navigator through one AI2-THOR episode and returns the result."""

    def __init__(
        self,
        controller,
        max_steps: int = 100,
        success_distance: float = 1.0,
    ) -> None:
        """Create the runner.

        Args:
            controller: an already-initialized ``ai2thor.controller.Controller``.
            max_steps: episode time limit. The navigator's STOP terminates earlier.
            success_distance: meters of Euclidean tolerance for success.
        """
        self.controller = controller
        self.max_steps = max_steps
        self.success_distance = success_distance

    def run(
        self,
        navigator: BaseNavigator,
        target_object: str,
        scene_id: str = "",
    ) -> EpisodeResult:
        """Run one episode and return an EpisodeResult."""
        navigator.reset(target_object)

        event = self.controller.last_event
        start_pos = dict(event.metadata["agent"]["position"])
        target_positions = self._target_positions(event, target_object)

        d_init = (
            self._min_distance(start_pos, target_positions) if target_positions else 0.0
        )
        d_shortest = d_init  # Euclidean approximation; replace with GetShortestPath later

        trajectory: list = [(start_pos["x"], start_pos["z"])]
        d_taken = 0.0
        num_steps = 0
        success = False
        stop_issued = False

        for _ in range(self.max_steps):
            obs = self._event_to_obs(event, target_object, start_pos)
            action = navigator.act(obs)
            num_steps += 1

            if action == Action.STOP:
                stop_issued = True
                cur = event.metadata["agent"]["position"]
                if target_positions:
                    success = self._min_distance(cur, target_positions) <= self.success_distance
                break

            event = self.controller.step(action=ACTION_TO_THOR[action])
            new = event.metadata["agent"]["position"]
            d_taken += float(np.hypot(new["x"] - trajectory[-1][0], new["z"] - trajectory[-1][1]))
            trajectory.append((new["x"], new["z"]))

        # Final metrics
        final = event.metadata["agent"]["position"]
        d_final = self._min_distance(final, target_positions) if target_positions else 0.0

        spl = 0.0
        if success and d_taken > 0:
            spl = d_shortest / max(d_taken, d_shortest)

        soft_spl = 0.0
        if d_init > 0:
            progress = max(0.0, (d_init - d_final) / d_init)
            denom = max(d_taken, d_shortest, 1e-6)
            soft_spl = progress * (d_shortest / denom)

        return EpisodeResult(
            success=bool(success),
            spl=float(spl),
            soft_spl=float(soft_spl),
            num_steps=num_steps,
            trajectory=trajectory,
            target_object=target_object,
            scene_id=scene_id,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _event_to_obs(event, target_object: str, start_pos: dict) -> Observation:
        """Build an Observation from a controller event."""
        agent = event.metadata["agent"]
        pos = agent["position"]
        heading_rad = float(np.deg2rad(agent["rotation"]["y"]))
        dx = pos["x"] - start_pos["x"]
        dz = pos["z"] - start_pos["z"]
        distance = float(np.hypot(dx, dz))

        rgb = np.asarray(event.frame, dtype=np.uint8)
        depth = event.depth_frame
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.float32)
        else:
            depth = np.asarray(depth, dtype=np.float32)

        # Ground-truth detection stand-in. AI2-THOR's `visible` flag is very
        # strict (requires interactability), so we also accept anything in
        # front of the agent (within a 90° cone) at ≤ 3m as "detected".
        DET_RANGE_M = 3.0
        FOV_HALF_RAD = np.deg2rad(45.0)
        forward = (float(np.sin(heading_rad)), float(np.cos(heading_rad)))
        visible_objects = []
        for obj in event.metadata.get("objects", []):
            op = obj["position"]
            d_dx = op["x"] - pos["x"]
            d_dz = op["z"] - pos["z"]
            dist = float(np.hypot(d_dx, d_dz))
            if obj.get("visible"):
                in_view = True
            elif dist <= DET_RANGE_M and dist > 0:
                # Cosine of angle between forward and (dx, dz).
                cos_angle = (forward[0] * d_dx + forward[1] * d_dz) / dist
                in_view = cos_angle >= float(np.cos(FOV_HALF_RAD))
            else:
                in_view = False
            if in_view:
                visible_objects.append({
                    "name": obj["objectType"],
                    "dx": d_dx,
                    "dz": d_dz,
                })

        return Observation(
            rgb=rgb,
            depth=depth,
            compass=np.array([heading_rad, distance], dtype=np.float32),
            target_object=target_object,
            visible_objects=visible_objects,
        )

    @staticmethod
    def _target_positions(event, target_object: str) -> list:
        """All instance positions in the scene matching the target category."""
        out = []
        for obj in event.metadata.get("objects", []):
            if obj.get("objectType") == target_object:
                p = obj["position"]
                out.append({"x": p["x"], "y": p["y"], "z": p["z"]})
        return out

    @staticmethod
    def _min_distance(pos: dict, targets: list) -> float:
        """Min Euclidean (x,z) distance from `pos` to any target position."""
        if not targets:
            return 0.0
        return float(min(np.hypot(pos["x"] - t["x"], pos["z"] - t["z"]) for t in targets))
