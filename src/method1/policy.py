"""Method 1 policy network: ResNet18 (RGB-D) + target embed + compass + GRU + PPO heads.

Shared by ``scripts/train_method1.py`` (training) and
``src.method1.navigator.Method1Navigator`` (inference). Keeping the architecture
in one module guarantees train/eval consistency.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import models


# ImageNet normalization for the RGB channels. The depth channel is normalized
# separately (0..MAX_DEPTH_M → 0..1) in the env preprocessor.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Architecture sizes — change here, not in caller code.
TARGET_EMB_DIM = 32
COMPASS_DIM = 16
COMBINED_DIM = 512
HIDDEN_DIM = 512
NUM_ACTIONS = 6     # matches shared.interfaces.Action

# Defaults for the project. Override at train time via CLI; the trained model
# remembers its own values inside the checkpoint.
DEFAULT_TARGETS: Tuple[str, ...] = ("Mug", "Apple", "Bowl", "Laptop", "Television")
MAX_DEPTH_M = 5.0


def make_resnet18_4channel(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """ResNet-18 with first conv accepting RGB + depth (4 channels).

    When pretrained weights are available we initialize the new depth channel
    of conv1 with the per-spatial mean of the RGB weights — a standard trick
    that keeps activations at roughly the pretrained scale.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = models.resnet18(weights=weights)
    old_conv1 = resnet.conv1
    new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3] = old_conv1.weight
        new_conv1.weight[:, 3:4] = old_conv1.weight.mean(dim=1, keepdim=True)
    resnet.conv1 = new_conv1
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
    return backbone, 512


class Method1Policy(nn.Module):
    """Recurrent actor-critic for ObjectNav.

    Inputs per timestep:
      rgb     : (B, 3, 224, 224) uint8 [0..255]
      depth   : (B, 1, 224, 224) float32 meters (will be clipped + scaled)
      target  : (B,) long  — target category id
      compass : (B, 2) float32 — (sin(heading), cos(heading))

    Plus the recurrent state ``hidden : (1, B, HIDDEN_DIM)``.
    """

    def __init__(
        self,
        num_targets: int,
        pretrained_encoder: bool = True,
        target_categories: Sequence[str] = DEFAULT_TARGETS,
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.target_categories = tuple(target_categories)

        # Visual encoder
        self.encoder, vis_dim = make_resnet18_4channel(pretrained=pretrained_encoder)
        # Target + compass
        self.target_embed = nn.Embedding(num_targets, TARGET_EMB_DIM)
        self.compass_proj = nn.Linear(2, COMPASS_DIM)
        # Combine
        self.combine = nn.Sequential(
            nn.Linear(vis_dim + TARGET_EMB_DIM + COMPASS_DIM, COMBINED_DIM),
            nn.ReLU(inplace=True),
        )
        # Recurrent core
        self.gru = nn.GRU(input_size=COMBINED_DIM, hidden_size=HIDDEN_DIM, batch_first=False)
        # Heads
        self.policy_head = nn.Linear(HIDDEN_DIM, NUM_ACTIONS)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        # Pre-allocated normalization tensors (registered as buffers so they
        # follow the module across .to(device)).
        self.register_buffer(
            "_rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # ------------------------------------------------------------------ #
    # Pre-processing
    # ------------------------------------------------------------------ #

    def _prep_visual(self, rgb_u8: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
        """Build a 4-channel float input from raw observation tensors."""
        rgb = rgb_u8.to(self._rgb_mean.dtype) / 255.0
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        rgb = (rgb - self._rgb_mean) / self._rgb_std
        depth = depth_m.to(self._rgb_mean.dtype)
        depth = torch.nan_to_num(depth, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0)
        depth = depth.clamp(min=0.0, max=MAX_DEPTH_M) / MAX_DEPTH_M
        return torch.cat([rgb, depth], dim=1)

    def _embed_step(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        target: torch.Tensor,
        compass: torch.Tensor,
    ) -> torch.Tensor:
        x = self._prep_visual(rgb, depth)
        v = self.encoder(x).flatten(1)
        t = self.target_embed(target)
        compass = torch.nan_to_num(compass, nan=0.0, posinf=0.0, neginf=0.0)
        c = F.relu(self.compass_proj(compass), inplace=True)
        return self.combine(torch.cat([v, t, c], dim=-1))

    # ------------------------------------------------------------------ #
    # Forward APIs
    # ------------------------------------------------------------------ #

    def step(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        target: torch.Tensor,
        compass: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single rollout step. Returns (logits, value, new_hidden)."""
        feat = self._embed_step(rgb, depth, target, compass)            # (B, COMBINED_DIM)
        out, new_hidden = self.gru(feat.unsqueeze(0), hidden)            # (1, B, HIDDEN_DIM)
        out = out.squeeze(0)
        return self.policy_head(out), self.value_head(out).squeeze(-1), new_hidden

    def evaluate_sequence(
        self,
        rgb: torch.Tensor,      # (T, B, 3, 224, 224) uint8
        depth: torch.Tensor,    # (T, B, 1, 224, 224) float32
        target: torch.Tensor,   # (T, B) long
        compass: torch.Tensor,  # (T, B, 2) float32
        actions: torch.Tensor,  # (T, B) long
        h0: torch.Tensor,       # (1, B, HIDDEN_DIM)
        dones: torch.Tensor,    # (T, B) float — 1.0 if env reset BEFORE this step
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a T-step rollout chunk for PPO. Backprops through the GRU.

        ``dones[t]`` marks transitions where the env reset *before* timestep t,
        so we zero the hidden state at that boundary to keep the GRU coherent.
        """
        T, B = actions.shape
        flat_rgb = rgb.view(T * B, *rgb.shape[2:])
        flat_depth = depth.view(T * B, *depth.shape[2:])
        flat_target = target.view(T * B)
        flat_compass = compass.view(T * B, 2)
        feats = self._embed_step(flat_rgb, flat_depth, flat_target, flat_compass)
        feats = feats.view(T, B, COMBINED_DIM)

        outputs = []
        h = h0
        for t in range(T):
            # Reset hidden state on episode boundaries.
            if t > 0:
                mask = (1.0 - dones[t]).view(1, B, 1)
                h = h * mask
            o, h = self.gru(feats[t : t + 1], h)
            outputs.append(o)
        out = torch.cat(outputs, dim=0)  # (T, B, HIDDEN_DIM)

        logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    @torch.no_grad()
    def select_action(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        target: torch.Tensor,
        compass: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) one action. Returns (action, log_prob, value, hidden)."""
        logits, value, new_hidden = self.step(rgb, depth, target, compass, hidden)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden


def initial_hidden(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(1, batch_size, HIDDEN_DIM, device=device)
