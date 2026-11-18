# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific MDP helpers for P1 RearPick."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place import (
    mdp as low_shelf_mdp,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


REAR_TABLE_CENTER_X = -1.35
TARGET_SIDE_OFFSET_Y = 0.24
TRAY_SIDE_OFFSET_Y = 0.22
TARGET_INITIAL_Z = 0.645
TRAY_CENTER_Z = 0.56
OBJECT_BASE_ROT = (0.7071068, 0.7071068, 0.0, 0.0)


def reset_rear_pick_layout(
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
    x = REAR_TABLE_CENTER_X + (torch.rand(count, device=env.device) * 2.0 - 1.0) * x_jitter
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
    # The tray is kinematic. Writing root velocity into a kinematic PhysX body
    # raises PxRigidDynamic::setLinearVelocity errors, so only move its pose.
    target_asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)


def rear_pick_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a coarse 0-1 progress signal for logging and evaluation."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    robot_to_object = torch.linalg.vector_norm(
        obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    object_to_target = torch.linalg.vector_norm(
        obj.data.root_pos_w[:, :3] - target.data.root_pos_w[:, :3], dim=-1
    )
    approach_progress = torch.exp(-robot_to_object / 1.2)
    transport_progress = torch.exp(-object_to_target / 0.8)
    return 0.35 * approach_progress + 0.65 * transport_progress


def object_in_front_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 1.2,
    cosine_threshold: float = 0.6,
) -> torch.Tensor:
    """Reward local reacquisition once the target enters the forward region."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    rel = obj.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    distance = torch.linalg.vector_norm(rel, dim=-1)
    forward = torch.zeros_like(rel)
    forward[:, 0] = 1.0
    forward = math_utils.quat_apply(robot.data.root_quat_w, forward)
    cosine = torch.nn.functional.cosine_similarity(rel, forward, dim=-1)
    return ((distance < distance_threshold) & (cosine > cosine_threshold)).float()


def rear_pick_place_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Placement success for the P1 tray geometry."""
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
