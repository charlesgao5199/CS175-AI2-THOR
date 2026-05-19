"""Top-down semantic map visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from shared.interfaces import SemanticMap


def visualize_map(smap: SemanticMap, out_path: Optional[str] = None, title: str = "Semantic Map"):
    """Render a SemanticMap as a top-down image.

    Color scheme:
      gray   = unexplored
      white  = explored + traversable
      black  = explored + obstacle
      red dot = agent position, oriented by agent_rot

    If `out_path` is provided, the figure is saved and the Figure is returned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas = np.full(smap.explored.shape + (3,), 0.55, dtype=np.float32)  # unexplored
    free = smap.explored & smap.traversable
    obstacle = smap.explored & ~smap.traversable
    canvas[free] = 1.0
    canvas[obstacle] = 0.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(canvas, origin="lower")
    ax.set_title(
        f"{title}\nexplored={int(smap.explored.sum())} "
        f"free={int(free.sum())} obstacle={int(obstacle.sum())}"
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Agent marker
    ax_x, ax_y = smap.agent_pos
    ax.plot(ax_y, ax_x, marker="o", color="red", markersize=8)
    # Heading indicator
    heading_rad = np.deg2rad(smap.agent_rot)
    dx = np.sin(heading_rad) * 5
    dy = np.cos(heading_rad) * 5
    ax.arrow(ax_y, ax_x, dy, dx, color="red", head_width=2, length_includes_head=True)

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    return fig
