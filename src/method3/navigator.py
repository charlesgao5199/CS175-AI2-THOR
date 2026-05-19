"""Method 3 — Semantic Map + LLM-Guided Exploration (Claude API).

At each step the navigator:
  1. Updates its SemanticMap from the observation.
  2. Renders a compact text description + a small ASCII window of the map
     centered on the agent.
  3. Calls Claude (3.5 Haiku, for cost reasons) with the target object and
     the map description, asking for ONE action and a brief reasoning.
  4. Parses ``ACTION: <NAME>`` out of the model response.

Requires the ``ANTHROPIC_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np

from mapping import MapConfig, SemanticMapBuilder
from shared.interfaces import Action, BaseNavigator, Observation, SemanticMap


# Per the project constraints (CLAUDE.md): use the cheapest Haiku available
# to keep costs low. The user's account doesn't have access to Claude 3.5 Haiku,
# so we use Claude Haiku 4.5 (the only Haiku model their key can call).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 256
REQUEST_TIMEOUT_S = 30.0  # per-request cap so the script never hangs indefinitely

SYSTEM_PROMPT = """You are an embodied agent in a household environment, navigating to find a target object.

You can take exactly ONE of these actions per turn:
  MOVE_AHEAD     - move forward 0.25m
  ROTATE_LEFT    - rotate 90° counter-clockwise
  ROTATE_RIGHT   - rotate 90° clockwise
  LOOK_UP        - tilt camera up
  LOOK_DOWN      - tilt camera down
  STOP           - declare you have reached the target

DECISION PRIORITY — apply in order, do not skip steps:
1. If the summary contains "TARGET seen" AND distance ≤ 10 cells: choose STOP. (10 cells = 1m,
   the success threshold. Do not keep exploring.)
2. If the summary contains "TARGET seen" AND distance > 10 cells: head toward the 'T' on
   the local map. Pick the rotation that points your heading at 'T'; if you are already roughly
   facing 'T', choose MOVE_AHEAD. Do not get sidetracked by exploring unrelated cells.
3. Otherwise (target not seen): explore. Prefer MOVE_AHEAD when forward is free; rotate
   toward the largest patch of unexplored ('.') cells.

Reply in this exact format:
  REASONING: <one short sentence>
  ACTION: <ACTION_NAME>
"""


class Method3Navigator(BaseNavigator):
    """LLM-guided navigator (Claude API + ReAct-style single-turn reasoning)."""

    def __init__(
        self,
        map_config: Optional[MapConfig] = None,
        model: str = DEFAULT_MODEL,
        window_radius: int = 7,
        max_steps: int = 500,
    ) -> None:
        """Create the navigator.

        Args:
            map_config: configuration for the internal SemanticMapBuilder.
            model: Anthropic model ID. Defaults to Claude 3.5 Haiku.
            window_radius: half-width (in cells) of the ASCII map window
                included in each LLM prompt.
            max_steps: episode cap.
        """
        self.builder = SemanticMapBuilder(map_config)
        self.model = model
        self.window_radius = window_radius
        self.max_steps = max_steps
        self.target_object = ""
        self._step = 0
        self._client = None  # lazy init
        self.last_reasoning: str = ""
        self.last_raw_response: str = ""
        self.last_usage: dict = {}  # {"input_tokens": int, "output_tokens": int}

    def reset(self, target_object: str) -> None:
        """Reset internal state for a new episode."""
        self.builder.reset()
        self.target_object = target_object
        self._step = 0
        self.last_reasoning = ""
        self.last_raw_response = ""

    def act(self, obs: Observation) -> Action:
        """Update the map, query Claude, and return the chosen action."""
        self._step += 1
        smap = self.builder.update(obs)

        if self._step > self.max_steps:
            self.last_reasoning = "max_steps reached → STOP"
            return Action.STOP

        client = self._get_client()
        user_prompt = self._build_user_prompt(smap)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            # Network hiccup / timeout / rate limit. Don't take down the eval —
            # log a fallback action and let the next step try again.
            self.last_reasoning = f"API error: {type(e).__name__}: {e}"
            self.last_raw_response = ""
            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "error": str(e)}
            return Action.ROTATE_RIGHT
        text = "".join(
            getattr(block, "text", "") for block in response.content
        ).strip()
        self.last_raw_response = text
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "input_tokens": int(getattr(usage, "input_tokens", 0)),
                "output_tokens": int(getattr(usage, "output_tokens", 0)),
            }
        reasoning, action = self._parse_response(text)
        self.last_reasoning = reasoning or "(no reasoning in response)"
        return action

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_client(self):
        """Lazily import and construct the Anthropic client.

        Raises a clear error if ANTHROPIC_API_KEY isn't set.
        """
        if self._client is not None:
            return self._client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Method3 requires it to call the "
                "Claude API. Export it in your shell before running."
            )
        import anthropic  # type: ignore

        # Cap each request at REQUEST_TIMEOUT_S so a hung connection can't
        # stall the whole eval. The SDK also retries transient errors.
        self._client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S, max_retries=2)
        return self._client

    def _build_user_prompt(self, smap: SemanticMap) -> str:
        """Render the user prompt with target, heading, summary, and ASCII map."""
        summary = self._describe_surroundings(smap)
        ascii_map = self._render_ascii_window(smap)
        return (
            f"Target object: {self.target_object}\n"
            f"Current heading: {smap.agent_rot:.0f}°  "
            f"(0°=north/forward, 90°=east, 180°=south, 270°=west)\n"
            f"Step: {self._step}\n"
            f"\n"
            f"Surroundings summary:\n{summary}\n"
            f"\n"
            f"Local map ({2 * self.window_radius + 1}×{2 * self.window_radius + 1} cells, "
            f"each 0.1m, agent at center, oriented to world frame):\n"
            f"  A=you  F=free  X=obstacle  .=unexplored  T=target\n"
            f"{ascii_map}\n"
            f"\n"
            f"What action should you take next?"
        )

    def _describe_surroundings(self, smap: SemanticMap) -> str:
        """Brief textual breakdown of what's around the agent."""
        ax, ay = smap.agent_pos
        # 5m windows in the four agent-relative directions, in world-aligned grid.
        heading_rad = np.deg2rad(smap.agent_rot)
        # Probe each direction by stepping along the agent's forward / right.
        directions = {
            "forward": (np.sin(heading_rad), np.cos(heading_rad)),
            "right": (np.cos(heading_rad), -np.sin(heading_rad)),
            "back": (-np.sin(heading_rad), -np.cos(heading_rad)),
            "left": (-np.cos(heading_rad), np.sin(heading_rad)),
        }
        lines = []
        for name, (di, dj) in directions.items():
            lines.append(f"  {name}: {self._probe(smap, ax, ay, di, dj)}")
        target_pos = smap.has_target(self.target_object)
        if target_pos is not None:
            lines.append(f"  TARGET seen at grid {target_pos} (distance "
                         f"{np.hypot(target_pos[0] - ax, target_pos[1] - ay):.1f} cells)")
        return "\n".join(lines)

    @staticmethod
    def _probe(smap: SemanticMap, ax: int, ay: int, di: float, dj: float, max_cells: int = 30) -> str:
        """Walk along a direction and report the first thing we encounter."""
        for k in range(1, max_cells + 1):
            i = int(round(ax + di * k))
            j = int(round(ay + dj * k))
            if not (0 <= i < smap.grid_size and 0 <= j < smap.grid_size):
                return f"out of map ({k * 0.1:.1f}m)"
            if not smap.explored[i, j]:
                return f"unexplored at {k * 0.1:.1f}m"
            if not smap.traversable[i, j]:
                return f"obstacle at {k * 0.1:.1f}m"
        return f"free for {max_cells * 0.1:.1f}m+"

    def _render_ascii_window(self, smap: SemanticMap) -> str:
        """Small ASCII window of the map, centered on the agent."""
        r = self.window_radius
        ax, ay = smap.agent_pos
        rows = []
        target_pos = smap.has_target(self.target_object)
        # Render with +i down so it looks like a normal grid; world-aligned.
        for i in range(ax - r, ax + r + 1):
            chars = []
            for j in range(ay - r, ay + r + 1):
                if i == ax and j == ay:
                    chars.append("A")
                elif target_pos is not None and (i, j) == target_pos:
                    chars.append("T")
                elif not (0 <= i < smap.grid_size and 0 <= j < smap.grid_size):
                    chars.append("?")
                elif not smap.explored[i, j]:
                    chars.append(".")
                elif smap.traversable[i, j]:
                    chars.append("F")
                else:
                    chars.append("X")
            rows.append("  " + "".join(chars))
        return "\n".join(rows)

    _ACTION_RE = re.compile(r"ACTION\s*[:=]\s*([A-Z_]+)", re.IGNORECASE)
    _REASONING_RE = re.compile(r"REASONING\s*[:=]\s*(.+?)(?:\n|$)", re.IGNORECASE)

    @classmethod
    def _parse_response(cls, text: str) -> tuple[str, Action]:
        """Extract (reasoning, action) from Claude's reply."""
        reasoning_match = cls._REASONING_RE.search(text)
        action_match = cls._ACTION_RE.search(text)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        if action_match:
            name = action_match.group(1).upper().strip()
            try:
                return reasoning, Action[name]
            except KeyError:
                pass
        # Couldn't parse — default to a safe explore action.
        reasoning = (reasoning + " [parse fallback]").strip()
        return reasoning, Action.ROTATE_RIGHT
