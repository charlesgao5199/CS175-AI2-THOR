"""Coverage-oriented ObjectNav baseline with lightweight position memory."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, Optional, Tuple

from objectnav.env import ObjectNavEnv
from objectnav.random_agent import EpisodeSummary
from objectnav.recording import EpisodeRecorder


GridCell = Tuple[int, int]


class CoverageAgent:
    """Explore by scanning, moving forward, and escaping repeated cells."""

    def __init__(
        self,
        seed: int = 0,
        scan_rotations: int = 4,
        scan_interval: int = 1,
        loop_window: int = 8,
        revisit_threshold: int = 3,
        cell_size: float = 0.25,
    ) -> None:
        self.primary_rotate = "RotateRight" if seed % 2 == 0 else "RotateLeft"
        self.secondary_rotate = "RotateLeft" if self.primary_rotate == "RotateRight" else "RotateRight"
        self.scan_rotations = scan_rotations
        self.scan_interval = max(1, scan_interval)
        self.loop_window = max(2, loop_window)
        self.revisit_threshold = max(2, revisit_threshold)
        self.cell_size = cell_size

        self.queue: Deque[str] = deque(self._scan_actions() + ["MoveAhead"])
        self.last_action: Optional[str] = None
        self.last_cell: Optional[GridCell] = None
        self.visit_counts: Counter[GridCell] = Counter()
        self.recent_cells: Deque[GridCell] = deque(maxlen=self.loop_window)
        self.moves_since_scan = 0
        self.move_failures = 0
        self.loop_escapes = 0

    def _scan_actions(self) -> list[str]:
        return ["LookDown"] + [self.primary_rotate] * self.scan_rotations + ["LookUp"]

    def _cell_from_event(self, event: Any) -> GridCell:
        position: Dict[str, float] = event.metadata["agent"]["position"]
        return (
            round(position["x"] / self.cell_size),
            round(position["z"] / self.cell_size),
        )

    def _observe(self, event: Any) -> None:
        cell = self._cell_from_event(event)
        if cell == self.last_cell:
            return

        self.last_cell = cell
        self.visit_counts[cell] += 1
        self.recent_cells.append(cell)
        if self.last_action == "MoveAhead":
            self.moves_since_scan += 1

    def _is_looping(self) -> bool:
        if self.last_cell is None:
            return False
        if self.visit_counts[self.last_cell] >= self.revisit_threshold:
            return True
        if len(self.recent_cells) < self.loop_window:
            return False
        return len(set(self.recent_cells)) <= max(2, self.loop_window // 3)

    def _recovery_rotate(self) -> str:
        if (self.move_failures + self.loop_escapes) % 2 == 0:
            return self.primary_rotate
        return self.secondary_rotate

    def _queue_scan_then_move(self, force_scan: bool = False) -> None:
        should_scan = force_scan or self.moves_since_scan >= self.scan_interval
        scan_actions = self._scan_actions() if should_scan else []
        if should_scan:
            self.moves_since_scan = 0
        self.queue = deque(scan_actions + ["MoveAhead"])

    def _queue_recovery(self) -> None:
        self.move_failures += 1
        turns = 1 + (self.move_failures % 3)
        self.queue = deque([self._recovery_rotate()] * turns + ["MoveAhead"])

    def _queue_loop_escape(self) -> None:
        self.loop_escapes += 1
        turns = 1 + (self.loop_escapes % 3)
        rotate = self.secondary_rotate if self.loop_escapes % 2 else self.primary_rotate
        self.queue = deque([rotate] * turns + self._scan_actions() + ["MoveAhead"])
        self.moves_since_scan = 0

    def act(self, event: Any, last_action_success: Optional[bool]) -> str:
        self._observe(event)

        if self.last_action == "MoveAhead":
            if last_action_success is False:
                self._queue_recovery()
            else:
                self.move_failures = 0
                if self._is_looping():
                    self._queue_loop_escape()
                else:
                    self._queue_scan_then_move()

        if not self.queue:
            self._queue_scan_then_move(force_scan=True)

        action = self.queue.popleft()
        self.last_action = action
        return action


def run_coverage_episode(
    env: ObjectNavEnv,
    max_steps: int = 100,
    seed: int = 0,
    scan_rotations: int = 4,
    scan_interval: int = 1,
    loop_window: int = 8,
    revisit_threshold: int = 3,
    cell_size: float = 0.25,
    recorder: Optional[EpisodeRecorder] = None,
) -> EpisodeSummary:
    agent = CoverageAgent(
        seed=seed,
        scan_rotations=scan_rotations,
        scan_interval=scan_interval,
        loop_window=loop_window,
        revisit_threshold=revisit_threshold,
        cell_size=cell_size,
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
            last_action = agent.act(
                event=event,
                last_action_success=last_action_success,
            )
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
