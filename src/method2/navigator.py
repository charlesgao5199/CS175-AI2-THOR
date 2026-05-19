"""Method 2 — Semantic Map + Classical Planner.

Maintains a SemanticMap, finds frontiers (free cells adjacent to unexplored),
plans a path via BFS (shortest-path on a unit-cost grid → equivalent to A*
with constant heuristic), and converts the first step of the path into an
AI2-THOR action.

Object detection is not yet wired in, so the navigator effectively does pure
frontier exploration. When the target object is added to ``SemanticMap.object_map``
(e.g., via ``SemanticMapBuilder.add_detection``), the navigator will switch
to navigating to that location.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from mapping import MapConfig, SemanticMapBuilder
from shared.interfaces import Action, BaseNavigator, Observation, SemanticMap


class Method2Navigator(BaseNavigator):
    """Frontier explorer over a semantic map, with BFS path planning."""

    def __init__(
        self,
        map_config: Optional[MapConfig] = None,
        rotate_step_deg: float = 90.0,
        # With AI2-THOR's 90° rotation step the agent can be at most 45° off
        # from any direction, so the move-ahead alignment tolerance must be
        # at least 45° to avoid endless oscillation.
        forward_tolerance_deg: float = 45.0,
        max_steps: int = 500,
        stop_distance_m: float = 0.9,
    ) -> None:
        """Create the navigator.

        Args:
            map_config: configuration for the internal SemanticMapBuilder.
            rotate_step_deg: AI2-THOR's default rotation magnitude per action.
            forward_tolerance_deg: how aligned with the next cell we must be
                before issuing MOVE_AHEAD; otherwise we rotate to align.
            max_steps: episode cap after which the navigator emits STOP.
        """
        self.builder = SemanticMapBuilder(map_config)
        self.rotate_step_deg = rotate_step_deg
        self.forward_tolerance_rad = np.deg2rad(forward_tolerance_deg)
        self.max_steps = max_steps
        self.stop_distance_m = stop_distance_m
        self.target_object = ""
        self._step = 0
        self.last_reasoning: str = ""

    def reset(self, target_object: str) -> None:
        """Reset internal state for a new episode."""
        self.builder.reset()
        self.target_object = target_object
        self._step = 0
        self.last_reasoning = ""

    def act(self, obs: Observation) -> Action:
        """Update the map and return the next action."""
        self._step += 1
        smap = self.builder.update(obs)

        if self._step > self.max_steps:
            self.last_reasoning = "max_steps reached → STOP"
            return Action.STOP

        # 1. If the target was detected, commit to it: STOP if close enough,
        # otherwise head greedily in its direction. (BFS over the agent-local
        # map often fails to find a route to a freshly-detected object cell
        # because the cells between haven't been classified as traversable
        # yet — falling back to exploration would lose progress, so we use
        # a direct heading-based action instead.)
        target_cell = smap.has_target(self.target_object)
        if target_cell is not None:
            cell_size = self.builder.config.cell_size
            ai, aj = smap.agent_pos
            ti, tj = target_cell
            dist_m = float(np.hypot(ti - ai, tj - aj)) * cell_size
            if dist_m <= self.stop_distance_m:
                self.last_reasoning = (
                    f"target {self.target_object} dist={dist_m:.2f}m "
                    f"≤ {self.stop_distance_m}m → STOP"
                )
                return Action.STOP
            self.last_reasoning = (
                f"target {self.target_object} at {target_cell} dist={dist_m:.2f}m → head toward"
            )
            return self._path_to_action([smap.agent_pos, target_cell], smap)

        # 2. Frontier-driven exploration.
        frontier = self._compute_frontier(smap)
        n_frontier = int(frontier.sum())
        if n_frontier > 0:
            path = self._bfs_path(smap, smap.agent_pos, goal_mask=frontier)
            if path is not None and len(path) >= 2:
                self.last_reasoning = (
                    f"frontier_cells={n_frontier}, path len={len(path)} -> next={path[1]}"
                )
                return self._path_to_action(path, smap)
            self.last_reasoning = f"frontier_cells={n_frontier} but unreachable; rotate"
        else:
            self.last_reasoning = "no frontier visible; rotate to widen view"

        # 3. No reachable frontier — rotate to broaden the view.
        return Action.ROTATE_RIGHT

    # ------------------------------------------------------------------ #
    # Planning helpers
    # ------------------------------------------------------------------ #

    def _compute_frontier(self, smap: SemanticMap) -> np.ndarray:
        """Cells that are explored+traversable and adjacent to unexplored."""
        unexplored = ~smap.explored
        adj_unexplored = np.zeros_like(unexplored)
        adj_unexplored[:-1, :] |= unexplored[1:, :]
        adj_unexplored[1:, :] |= unexplored[:-1, :]
        adj_unexplored[:, :-1] |= unexplored[:, 1:]
        adj_unexplored[:, 1:] |= unexplored[:, :-1]
        return smap.explored & smap.traversable & adj_unexplored

    def _bfs_path(
        self,
        smap: SemanticMap,
        start: tuple,
        goal_cells: Optional[set] = None,
        goal_mask: Optional[np.ndarray] = None,
    ) -> Optional[list]:
        """BFS shortest path from ``start`` to the first cell matching the goal.

        Walks on traversable cells; either ``goal_cells`` (a set of (i, j))
        or ``goal_mask`` (boolean array) defines acceptable goals.
        """
        M = smap.grid_size
        # Allow the agent's own cell to start the search even if it wasn't
        # marked traversable (the agent stands somewhere).
        traversable = smap.traversable.copy()
        traversable[start] = True

        visited = np.zeros((M, M), dtype=bool)
        visited[start] = True
        parent: dict = {}
        q: deque = deque([start])

        def is_goal(cell: tuple) -> bool:
            if goal_cells is not None and cell in goal_cells:
                return True
            if goal_mask is not None and goal_mask[cell]:
                return True
            return False

        while q:
            cell = q.popleft()
            if cell != start and is_goal(cell):
                path = [cell]
                while cell in parent:
                    cell = parent[cell]
                    path.append(cell)
                path.reverse()
                return path
            ci, cj = cell
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < M and 0 <= nj < M and not visited[ni, nj] and traversable[ni, nj]:
                    visited[ni, nj] = True
                    parent[(ni, nj)] = cell
                    q.append((ni, nj))
        return None

    def _path_to_action(self, path: list, smap: SemanticMap) -> Action:
        """Turn the first step of ``path`` into an AI2-THOR action.

        The map is world-aligned, the agent is at the grid center, and
        ``smap.agent_rot`` is the agent's world heading in degrees. ``gi``
        corresponds to world x (right when heading=0), ``gj`` corresponds to
        world z (forward when heading=0).
        """
        start_i, start_j = path[0]
        next_i, next_j = path[1]
        d_i = next_i - start_i
        d_j = next_j - start_j

        # World-frame direction from agent toward next cell.
        target_angle = float(np.arctan2(d_i, d_j))  # 0 = +z (north), π/2 = +x (east)
        agent_angle = float(np.deg2rad(smap.agent_rot))
        rel = target_angle - agent_angle
        # Normalize to (-π, π]
        rel = (rel + np.pi) % (2 * np.pi) - np.pi

        if abs(rel) <= self.forward_tolerance_rad:
            return Action.MOVE_AHEAD
        return Action.ROTATE_RIGHT if rel > 0 else Action.ROTATE_LEFT
