# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import compute_base_height_rel_to_feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*wheel"),
) -> torch.Tensor:
    """Return ``[height_above_wheels, pitch, roll]`` for the robot body."""
    robot: Articulation = env.scene[asset_cfg.name]
    height = compute_base_height_rel_to_feet(env, asset_cfg, feet_cfg)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    return torch.stack((height, pitch, roll), dim=-1)


def gripper_joint_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: tuple[str, str] = ("gripper_joint1", "gripper_joint2"),
) -> torch.Tensor:
    """Return the two prismatic gripper positions in the robot joint order."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_names)
    return robot.data.joint_pos[:, joint_ids]


def object_to_target_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    return torch.linalg.vector_norm(obj.data.root_pos_w[:, :3] - target.data.root_pos_w[:, :3], dim=-1)


def object_to_target_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    sensitivity: float = 6.0,
) -> torch.Tensor:
    """Dense reward for moving the object toward the upper target pad."""
    return torch.exp(-sensitivity * object_to_target_distance(env, object_cfg, target_cfg))


def body_pitch_reach_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    desired_pitch: float = -0.15,
    tolerance: float = 0.12,
    object_height_gate: float = 0.45,
) -> torch.Tensor:
    """Reward forward body pitch while the object is still in the low shelf region."""
    robot: Articulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    gate = (obj.data.root_pos_w[:, 2] < object_height_gate).float()
    return gate * torch.exp(-torch.square((pitch - desired_pitch) / max(tolerance, 1e-6)))


def object_placed_on_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.035,
    target_height_offset: float = 0.025,
    lift_height: float = 0.04,
    hold_duration: float = 0.5,
    gripper_open_threshold: float = 0.03,
) -> torch.Tensor:
    """Success requires a prior lift, target alignment, low speed, and an open gripper."""
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    current_pos = obj.data.root_pos_w[:, :3]
    target_pos = target.data.root_pos_w[:, :3]
    xy_dist = torch.linalg.vector_norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
    height_err = torch.abs(current_pos[:, 2] - target_pos[:, 2] - target_height_offset)
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)

    gripper_ids, _ = robot.find_joints(("gripper_joint1", "gripper_joint2"))
    gripper_open = torch.all(torch.abs(robot.data.joint_pos[:, gripper_ids]) > gripper_open_threshold, dim=-1)

    if not hasattr(env, "_low_shelf_place_init_height"):
        env._low_shelf_place_init_height = current_pos[:, 2].clone()
        env._low_shelf_place_lifted_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._low_shelf_place_timer = torch.zeros(env.num_envs, device=env.device)

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._low_shelf_place_init_height[just_reset] = current_pos[just_reset, 2].clone()
        env._low_shelf_place_lifted_once[just_reset] = False
        env._low_shelf_place_timer[just_reset] = 0.0

    lifted = current_pos[:, 2] - env._low_shelf_place_init_height > lift_height
    env._low_shelf_place_lifted_once |= lifted

    placed = (
        env._low_shelf_place_lifted_once
        & (xy_dist < xy_threshold)
        & (height_err < height_threshold)
        & (speed < 0.08)
        & gripper_open
    )
    env._low_shelf_place_timer = torch.where(
        placed,
        env._low_shelf_place_timer + env.step_dt,
        torch.zeros_like(env._low_shelf_place_timer),
    )
    return env._low_shelf_place_timer >= hold_duration

