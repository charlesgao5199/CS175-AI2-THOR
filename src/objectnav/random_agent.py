"""Random baseline for AI2-THOR ObjectNav episodes."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from objectnav.env import DEFAULT_ACTIONS, ObjectNavEnv, TargetObservation


@dataclass(frozen=True)
class EpisodeSummary:
    scene: str
    target_object_type: str
    seed: int
    max_steps: int
    steps_taken: int
    success: bool
    stop_reason: str
    final_position: Dict[str, float]
    last_action: Optional[str]
    last_action_success: Optional[bool]
    target_observation: TargetObservation
    action_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "target_object_type": self.target_object_type,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "steps_taken": self.steps_taken,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "final_position": self.final_position,
            "last_action": self.last_action,
            "last_action_success": self.last_action_success,
            "target_observation": self.target_observation.to_dict(),
            "action_counts": dict(self.action_counts),
        }


class RandomAgent:
    def __init__(self, actions: Sequence[str] = DEFAULT_ACTIONS, seed: int = 0) -> None:
        self.actions = tuple(actions)
        self.rng = random.Random(seed)

    def act(self) -> str:
        return self.rng.choice(self.actions)


def run_random_episode(
    env: ObjectNavEnv,
    max_steps: int = 100,
    seed: int = 0,
    actions: Sequence[str] = DEFAULT_ACTIONS,
) -> EpisodeSummary:
    agent = RandomAgent(actions=actions, seed=seed)
    action_counts: Counter[str] = Counter()
    last_action: Optional[str] = None
    last_action_success: Optional[bool] = None

    event = env.start()
    observation = env.observe_target(event)
    steps_taken = 0

    if observation.success:
        stop_reason = "target_visible_at_start"
    else:
        stop_reason = "max_steps_reached"
        for _ in range(max_steps):
            last_action = agent.act()
            action_counts[last_action] += 1
            event = env.step(last_action)
            steps_taken += 1
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

