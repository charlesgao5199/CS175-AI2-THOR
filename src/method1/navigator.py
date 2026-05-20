"""Method 1 inference-time navigator. Loads a PPO checkpoint and plugs into the
eval pipeline as a ``BaseNavigator``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from shared.interfaces import Action, BaseNavigator, Observation

from method1.policy import DEFAULT_TARGETS, Method1Policy, initial_hidden


def _select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Method1Navigator(BaseNavigator):
    """End-to-end RL policy from Method 1.

    Loads the policy + target vocabulary from a saved checkpoint. ``act()``
    runs one forward pass through the encoder + GRU + policy head. The
    recurrent state is carried across the episode and reset by ``reset()``.

    By default actions are sampled greedily (argmax). Pass ``deterministic=False``
    to sample from the categorical distribution (useful for diversity during
    qualitative evaluation).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        deterministic: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Method1 checkpoint not found: {self.checkpoint_path}. "
                "Train one with scripts/train_method1.py first."
            )
        self.device = _select_device(device)
        self.deterministic = deterministic

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.target_categories = tuple(ckpt.get("targets", DEFAULT_TARGETS))
        self.policy = Method1Policy(
            num_targets=len(self.target_categories),
            pretrained_encoder=False,  # weights come from the checkpoint
            target_categories=self.target_categories,
        ).to(self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()

        self._target_to_id = {t: i for i, t in enumerate(self.target_categories)}
        self._hidden: Optional[torch.Tensor] = None
        self._target_id: int = 0
        self._step: int = 0
        self.last_reasoning: str = ""

    # ------------------------------------------------------------------ #
    # BaseNavigator API
    # ------------------------------------------------------------------ #

    def reset(self, target_object: str) -> None:
        if target_object not in self._target_to_id:
            # Unknown category — fall back to id 0 so the episode still runs.
            # Calling code can detect this via last_reasoning.
            self.last_reasoning = (
                f"WARNING: target '{target_object}' not in trained vocab "
                f"{self.target_categories}; defaulting to id 0"
            )
            self._target_id = 0
        else:
            self.last_reasoning = ""
            self._target_id = self._target_to_id[target_object]
        self._hidden = initial_hidden(batch_size=1, device=self.device)
        self._step = 0

    def act(self, obs: Observation) -> Action:
        rgb, depth = self._preprocess(obs)
        target = torch.tensor([self._target_id], dtype=torch.long, device=self.device)
        heading = float(obs.compass[0]) if obs.compass.size > 0 else 0.0
        compass = torch.tensor(
            [[float(np.sin(heading)), float(np.cos(heading))]],
            dtype=torch.float32,
            device=self.device,
        )

        action_t, _log_prob, value, self._hidden = self.policy.select_action(
            rgb, depth, target, compass, self._hidden, deterministic=self.deterministic
        )
        action_id = int(action_t.item())
        self._step += 1
        self.last_reasoning = (
            f"step={self._step} target_id={self._target_id} "
            f"value={float(value.item()):+.2f} action={Action(action_id).name}"
        )
        return Action(action_id)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, obs: Observation):
        """Resize the RGB-D observation to the policy's 224×224 input."""
        rgb = torch.as_tensor(obs.rgb, dtype=torch.uint8, device=self.device)
        if rgb.ndim == 3:
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        rgb = F.interpolate(rgb.float(), size=(224, 224), mode="bilinear",
                            align_corners=False).to(torch.uint8)

        depth = torch.as_tensor(obs.depth, dtype=torch.float32, device=self.device)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        depth = F.interpolate(depth, size=(224, 224), mode="bilinear",
                              align_corners=False)
        return rgb, depth
