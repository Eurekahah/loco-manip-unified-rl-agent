# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific MDP helpers for P3 PeekOver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place import (
    mdp as low_shelf_mdp,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FAR_TABLE_CENTER_X = 2.15
TARGET_SIDE_OFFSET_Y = 0.34
TRAY_SIDE_OFFSET_Y = 0.30
TARGET_INITIAL_Z = 0.54
TRAY_CENTER_Z = 0.46
OBJECT_BASE_ROT = (0.7071068, 0.7071068, 0.0, 0.0)


def reset_peek_over_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    x_jitter: float = 0.04,
    y_jitter: float = 0.02,
    yaw_jitter: float = 0.15,
) -> None:
    """Randomize the target side and place the tray on the opposite side."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    object_asset: RigidObject = env.scene[object_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    origins = env.scene.env_origins[env_ids]
    count = env_ids.numel()

    side = torch.where(
        torch.rand(count, device=env.device) > 0.5,
        torch.ones(count, device=env.device),
        -torch.ones(count, device=env.device),
    )
    x = FAR_TABLE_CENTER_X + (torch.rand(count, device=env.device) * 2.0 - 1.0) * x_jitter
    object_y = side * (
        TARGET_SIDE_OFFSET_Y
        + (torch.rand(count, device=env.device) * 2.0 - 1.0) * y_jitter
    )
    tray_y = -side * TRAY_SIDE_OFFSET_Y

    object_pos = torch.stack(
        (x, object_y, torch.full_like(x, TARGET_INITIAL_Z)),
        dim=-1,
    ) + origins
    target_pos = torch.stack(
        (x, tray_y, torch.full_like(x, TRAY_CENTER_Z)),
        dim=-1,
    ) + origins

    yaw = (torch.rand(count, device=env.device) * 2.0 - 1.0) * yaw_jitter
    zeros = torch.zeros_like(yaw)
    yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    base_quat = torch.tensor(OBJECT_BASE_ROT, device=env.device).unsqueeze(0).expand(count, -1)
    object_quat = math_utils.quat_mul(yaw_quat, base_quat)
    target_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).unsqueeze(0).expand(count, -1)

    object_state = torch.zeros(count, 13, device=env.device)
    object_state[:, :3] = object_pos
    object_state[:, 3:7] = object_quat
    target_state = torch.zeros(count, 13, device=env.device)
    target_state[:, :3] = target_pos
    target_state[:, 3:7] = target_quat

    object_asset.write_root_state_to_sim(object_state, env_ids=env_ids)
    target_asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)


def peek_over_place_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return low_shelf_mdp.object_placed_on_target(
        env,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
        robot_cfg=robot_cfg,
        xy_threshold=0.10,
        height_threshold=0.045,
        target_height_offset=0.09,
        lift_height=0.04,
        hold_duration=1.0,
    )
