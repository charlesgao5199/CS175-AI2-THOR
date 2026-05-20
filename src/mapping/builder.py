"""Semantic map builder: depth → agent-centered top-down occupancy grid.

The builder projects per-pixel depth into a top-down grid centered on the
agent. Cells are classified as ground (traversable), obstacle (not
traversable), or unknown. The grid is agent-local: x_grid increases to the
agent's right, y_grid increases in front of the agent. The map accumulates
across calls — repeated `update()` calls OR with new observations widen the
explored area within the agent-centered window.

This is intentionally minimal — it does not yet integrate object detection
(Detic). `object_map` stays empty unless callers populate it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.interfaces import Observation, SemanticMap


@dataclass
class MapConfig:
    """Configuration for the semantic map.

    Defaults give a 10m x 10m grid at 10cm resolution.
    """
    grid_size: int = 100
    cell_size: float = 0.1            # meters per grid cell
    camera_height: float = 1.5        # AI2-THOR default agent camera height (m)
    hfov_deg: float = 90.0            # AI2-THOR default horizontal FOV
    max_range: float = 5.0            # ignore depth beyond this (m)
    ground_threshold: float = 0.2     # height ≤ this → ground/traversable (m)
    ceiling_threshold: float = 2.0    # height ≥ this → ignore (m)


class SemanticMapBuilder:
    """Builds and accumulates a 2D top-down SemanticMap from RGB-D observations."""

    def __init__(self, config: MapConfig | None = None) -> None:
        """Create a builder; the internal map is lazily allocated on first update."""
        self.config = config or MapConfig()
        self._map: SemanticMap | None = None

    @property
    def map(self) -> SemanticMap:
        """Return the current SemanticMap; allocates if not yet built."""
        if self._map is None:
            self._map = self._empty_map()
        return self._map

    def reset(self) -> None:
        """Clear all accumulated state — call between episodes."""
        self._map = None

    def update(self, obs: Observation) -> SemanticMap:
        """Project obs.depth into the top-down grid and update the map.

        Returns the accumulated SemanticMap. The agent is always at the grid
        center; rotation around the agent is applied via obs.compass[0].
        """
        if self._map is None:
            self._map = self._empty_map()

        # Object detections are stored at agent-relative grid coordinates. The
        # moment the agent moves, world-fixed objects shift to a different
        # grid cell, so any stale entries from past frames become invalid.
        # Clearing here keeps object_map a "current frame" view; the explored
        # and traversable masks still accumulate (good enough for navigation).
        self._map.object_map.clear()

        depth = obs.depth
        if depth.size == 0:
            return self._map

        cfg = self.config
        H, W = depth.shape
        fov = np.deg2rad(cfg.hfov_deg)
        fx = (W / 2.0) / np.tan(fov / 2.0)
        fy = fx
        cx_pix = W / 2.0
        cy_pix = H / 2.0

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        z = depth.astype(np.float32)
        valid = np.isfinite(z) & (z > 0.1) & (z < cfg.max_range)
        if not valid.any():
            return self._map

        # Camera frame: x right, y down, z forward
        x_cam = (us - cx_pix) * z / fx
        y_cam = (vs - cy_pix) * z / fy
        height_above_ground = cfg.camera_height - y_cam

        in_height = (height_above_ground > -0.5) & (height_above_ground < cfg.ceiling_threshold)
        is_ground = (height_above_ground <= cfg.ground_threshold) & valid & in_height
        is_obstacle = (height_above_ground > cfg.ground_threshold) & valid & in_height

        # Rotate (x_cam, z) into world (rotation around vertical by heading)
        heading_rad = float(obs.compass[0]) if obs.compass.size > 0 else 0.0
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        x_world = x_cam * cos_h + z * sin_h
        z_world = -x_cam * sin_h + z * cos_h

        # Agent at grid center
        cx_grid = cfg.grid_size // 2
        cy_grid = cfg.grid_size // 2
        gi = np.round(x_world / cfg.cell_size + cx_grid).astype(np.int32)
        gj = np.round(z_world / cfg.cell_size + cy_grid).astype(np.int32)
        in_grid = (gi >= 0) & (gi < cfg.grid_size) & (gj >= 0) & (gj < cfg.grid_size)

        m = self._map

        # 1. Agent footprint: the cells under and immediately around the agent
        # are always free (the agent stands there).
        foot = 2  # cells of half-width — ~0.2m at default resolution
        i_lo = max(0, cx_grid - foot)
        i_hi = min(cfg.grid_size, cx_grid + foot + 1)
        j_lo = max(0, cy_grid - foot)
        j_hi = min(cfg.grid_size, cy_grid + foot + 1)
        m.explored[i_lo:i_hi, j_lo:j_hi] = True
        m.traversable[i_lo:i_hi, j_lo:j_hi] = True

        # 2. Ray-cast: any visible depth hit means the cells along the ray
        # from the agent to the hit are free space. We approximate by
        # sampling a few interpolated cells along each ray.
        hit_mask = in_grid & (is_ground | is_obstacle)
        hi_i = gi[hit_mask]
        hi_j = gj[hit_mask]
        for k in (1, 2, 3, 4):  # 4 samples between agent and hit (exclusive endpoint)
            t = k / 5.0
            ri = (cx_grid + (hi_i - cx_grid) * t).astype(np.int32)
            rj = (cy_grid + (hi_j - cy_grid) * t).astype(np.int32)
            in_g = (ri >= 0) & (ri < cfg.grid_size) & (rj >= 0) & (rj < cfg.grid_size)
            m.explored[ri[in_g], rj[in_g]] = True
            m.traversable[ri[in_g], rj[in_g]] = True

        # 3. Ground hits → explored + traversable (overrides ray-cast).
        gmask = is_ground & in_grid
        m.explored[gi[gmask], gj[gmask]] = True
        m.traversable[gi[gmask], gj[gmask]] = True

        # 4. Obstacle hits → explored + NOT traversable (overrides everything).
        omask = is_obstacle & in_grid
        m.explored[gi[omask], gj[omask]] = True
        m.traversable[gi[omask], gj[omask]] = False

        # 5. Ingest object detections. Each entry carries the world-frame
        # offset from the agent in meters; the grid is agent-centered and
        # world-aligned so this maps directly to (gi, gj).
        for det in (obs.visible_objects or []):
            name = det.get("name")
            if not name:
                continue
            dx = float(det.get("dx", 0.0))
            dz = float(det.get("dz", 0.0))
            oi = int(round(dx / cfg.cell_size + cx_grid))
            oj = int(round(dz / cfg.cell_size + cy_grid))
            if 0 <= oi < cfg.grid_size and 0 <= oj < cfg.grid_size:
                m.mark_explored(oi, oj, traversable=True, objects={name})

        m.agent_pos = (cx_grid, cy_grid)
        m.agent_rot = float(np.rad2deg(heading_rad))
        return m

    def add_detection(self, x_world_m: float, z_world_m: float, label: str) -> None:
        """Optionally annotate a world-frame point with an object label.

        Hook for future Detic / instance-seg integration. No-op for now if
        the point is outside the grid.
        """
        if self._map is None:
            self._map = self._empty_map()
        cfg = self.config
        gi = int(round(x_world_m / cfg.cell_size + cfg.grid_size // 2))
        gj = int(round(z_world_m / cfg.cell_size + cfg.grid_size // 2))
        if 0 <= gi < cfg.grid_size and 0 <= gj < cfg.grid_size:
            self._map.mark_explored(gi, gj, traversable=True, objects={label})

    def _empty_map(self) -> SemanticMap:
        cfg = self.config
        return SemanticMap(
            grid_size=cfg.grid_size,
            explored=np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool),
            traversable=np.zeros((cfg.grid_size, cfg.grid_size), dtype=bool),
            agent_pos=(cfg.grid_size // 2, cfg.grid_size // 2),
            agent_rot=0.0,
        )
