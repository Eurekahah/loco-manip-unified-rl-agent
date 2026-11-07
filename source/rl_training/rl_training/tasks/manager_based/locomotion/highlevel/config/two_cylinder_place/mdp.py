# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific rewards for the two-cylinder pick-and-place scene."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import compute_base_height_rel_to_feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_height_for_place_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    low_height: float = 0.40,
    high_height: float = 0.52,
    lift_transition_margin: float = 0.10,
    tolerance: float = 0.07,
) -> torch.Tensor:
    """Reward lowering at the source and raising the body for the high target.

    The desired body height switches once the object has been lifted near the
    high platform. This makes the vertical body motion useful without forcing a
    fixed posture throughout the whole episode.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]

    feet_cfg = SceneEntityCfg("robot", body_names=".*wheel")
    feet_cfg.resolve(env.scene)
    height = compute_base_height_rel_to_feet(env, asset_cfg, feet_cfg)

    target_height = target.data.root_pos_w[:, 2]
    lifted_for_high_place = obj.data.root_pos_w[:, 2] > (target_height - lift_transition_margin)
    desired_height = torch.where(
        lifted_for_high_place,
        torch.full_like(height, high_height),
        torch.full_like(height, low_height),
    )
    return torch.exp(-torch.square((height - desired_height) / max(tolerance, 1e-6)))
