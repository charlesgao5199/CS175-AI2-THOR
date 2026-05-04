"""Simple non-random ObjectNav baseline agents."""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional

from objectnav.env import ObjectNavEnv
from objectnav.random_agent import EpisodeSummary
from objectnav.recording import EpisodeRecorder


class SweepMoveAgent:
    """Scan at each location, move forward, and rotate when blocked."""

    def __init__(
        self,
        seed: int = 0,
        scan_rotations: int = 4,
        recovery_turns: Optional[int] = None,
    ) -> None:
        self.rotate_action = "RotateRight" if seed % 2 == 0 else "RotateLeft"
        self.scan_rotations = scan_rotations
        self.recovery_turns = recovery_turns if recovery_turns is not None else 1 + seed % 2
        self.last_action: Optional[str] = None
        self.queue: Deque[str] = deque(self._scan_actions() + ["MoveAhead"])

    def _scan_actions(self) -> list[str]:
        return ["LookDown"] + [self.rotate_action] * self.scan_rotations + ["LookUp"]

    def act(self, last_action_success: Optional[bool]) -> str:
        if self.last_action == "MoveAhead":
            if last_action_success is False:
                self.queue = deque([self.rotate_action] * self.recovery_turns + ["MoveAhead"])
            else:
                self.queue = deque(self._scan_actions() + ["MoveAhead"])

        if not self.queue:
            self.queue = deque(self._scan_actions() + ["MoveAhead"])

        action = self.queue.popleft()
        self.last_action = action
        return action


def run_sweep_move_episode(
    env: ObjectNavEnv,
    max_steps: int = 100,
    seed: int = 0,
    scan_rotations: int = 4,
    recovery_turns: Optional[int] = None,
    recorder: Optional[EpisodeRecorder] = None,
) -> EpisodeSummary:
    agent = SweepMoveAgent(
        seed=seed,
        scan_rotations=scan_rotations,
        recovery_turns=recovery_turns,
    )
    action_counts: Counter[str] = Counter()
    last_action: Optional[str] = None
    last_action_success: Optional[bool] = None

    event = env.start()
    if recorder is not None:
        recorder.record(event=event, step=0, action=None)

    observation = env.observe_target(event)
    steps_taken = 0

    if observation.success:
        stop_reason = "target_visible_at_start"
    else:
        stop_reason = "max_steps_reached"
        for _ in range(max_steps):
            last_action = agent.act(last_action_success=last_action_success)
            action_counts[last_action] += 1
            event = env.step(last_action)
            steps_taken += 1
            if recorder is not None:
                recorder.record(event=event, step=steps_taken, action=last_action)
            last_action_success = event.metadata.get("lastActionSuccess")
            observation = env.observe_target(event)
            if observation.success:
                stop_reason = "target_found"
                break

    final_position = event.metadata["agent"]["position"]

    return EpisodeSummary(
        scene=env.scene,
        target_object_type=env.target_object_type,
        seed=seed,
        max_steps=max_steps,
        steps_taken=steps_taken,
        success=observation.success,
        stop_reason=stop_reason,
        final_position=final_position,
        last_action=last_action,
        last_action_success=last_action_success,
        target_observation=observation,
        action_counts=dict(action_counts),
    )
