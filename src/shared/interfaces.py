"""Shared interfaces for all navigation methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    """Discrete action space matching AI2-THOR."""
    MOVE_AHEAD = 0
    ROTATE_LEFT = 1
    ROTATE_RIGHT = 2
    LOOK_UP = 3
    LOOK_DOWN = 4
    STOP = 5


@dataclass
class Observation:
    """Single timestep observation from AI2-THOR."""
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: np.ndarray          # (H, W) float32, meters
    compass: np.ndarray        # (2,) orientation and displacement from start
    target_object: str         # e.g. "Microwave"


@dataclass
class SemanticMap:
    """2D top-down semantic grid map."""
    grid_size: int                              # M (map is M x M)
    explored: np.ndarray                        # (M, M) bool
    traversable: np.ndarray                     # (M, M) bool
    object_map: dict = field(default_factory=dict)  # (x,y) -> set of object class strings
    agent_pos: tuple = (0, 0)                   # (x, y) grid coords
    agent_rot: float = 0.0                      # heading in degrees

    def mark_explored(self, x: int, y: int, traversable: bool, objects: Optional[set] = None) -> None:
        """Mark a cell as explored with optional object detections."""
        self.explored[x, y] = True
        self.traversable[x, y] = traversable
        if objects:
            if (x, y) not in self.object_map:
                self.object_map[(x, y)] = set()
            self.object_map[(x, y)].update(objects)

    def has_target(self, target: str) -> Optional[tuple]:
        """Return grid coords of target if seen, else None."""
        for pos, objs in self.object_map.items():
            if target in objs:
                return pos
        return None


@dataclass
class EpisodeResult:
    """Result of a single navigation episode."""
    success: bool
    spl: float
    soft_spl: float
    num_steps: int
    trajectory: list = field(default_factory=list)
    target_object: str = ""
    scene_id: str = ""


class BaseNavigator(ABC):
    """Abstract base class all methods must implement."""

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """Given an observation, return an action."""
        ...

    @abstractmethod
    def reset(self, target_object: str) -> None:
        """Reset for a new episode with a new target."""
        ...
