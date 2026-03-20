# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_rotate_inverse,
    yaw_quat,
    wrap_to_pi,
)

from .utils import robot_root_pos_w, robot_root_quat_w, object_root_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

# =============================================================================
# Rewards
# =============================================================================

def distance_to_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Potential-based dense reward: encourages the robot to get closer to the target.

    Uses a Gaussian kernel:  r = exp(-dist² / (2·std²))
    Value is 1.0 at the target, smoothly decays to 0 at far distances.
    This avoids the 1/dist singularity and keeps gradients well-behaved.

    Args:
        threshold: Distance (m) below which the robot is considered "at target".
                   Used only for clipping — does not affect gradient.
        std:       Width of the Gaussian kernel. Larger = more long-range reward signal.

    Returns shape (N,).
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    # Horizontal distance only (ignore z for flat navigation)
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]  # (N, 2)
    dist = torch.norm(diff, dim=-1)                    # (N,)

    reward = torch.exp(-dist**2 / (2.0 * std**2))     # (N,)  ∈ (0, 1]
    return reward


def distance_to_target_potential(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Potential-based shaping reward: Φ(s_t-1) - Φ(s_t),
    where Φ(s) = -dist(robot, target).

    Positive when the robot moves closer, negative when it moves away.
    This is theoretically grounded (Ng et al. 1999) and does not change
    the optimal policy.

    Requires env to cache previous distance.  Uses env.extras for storage.

    Returns shape (N,).
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    curr_dist = torch.norm(diff, dim=-1)  # (N,)

    # Retrieve previous distance (stored at end of last step)
    key = f"_prev_dist_{target_cfg.name}"
    if key not in env.extras:
        # First call: no previous distance → zero shaping
        env.extras[key] = curr_dist.clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev_dist = env.extras[key]           # (N,)
    shaping   = prev_dist - curr_dist     # positive = getting closer
    env.extras[key] = curr_dist.clone()   # update for next step

    return shaping


def heading_to_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    std: float = 0.8,
) -> torch.Tensor:
    """
    Dense reward for facing the target.

    r = exp(-angle² / (2·std²))

    Value is 1.0 when perfectly aligned, decays for larger angular error.

    Args:
        std: Angular width in radians. Default 0.8 rad ≈ 46°.

    Returns shape (N,).
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    robot_quat_w = robot_root_quat_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    rel_pos_w      = target_pos_w[:, :2] - robot_pos_w[:, :2]  # (N, 2)
    target_angle_w = torch.atan2(rel_pos_w[:, 1], rel_pos_w[:, 0])  # (N,)

    heading_quat = yaw_quat(robot_quat_w)
    robot_yaw    = 2.0 * torch.atan2(heading_quat[:, 3], heading_quat[:, 0])  # (N,)

    angle_error = wrap_to_pi(target_angle_w - robot_yaw)  # (N,)  ∈ [-π, π]
    reward = torch.exp(-angle_error**2 / (2.0 * std**2))  # (N,)

    # Mask out when robot is already very close (heading irrelevant at target)
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)
    reward = torch.where(dist < 0.3, torch.zeros_like(reward), reward)

    return reward


def reach_target_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.4,
) -> torch.Tensor:
    """
    Sparse bonus reward: +1.0 for each step the robot is within `threshold` metres
    of the target (horizontal distance).

    Combine with a termination condition so this fires at most once per episode,
    or leave it as a per-step bonus if you want the robot to stay near the target.

    Returns shape (N,).
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)  # (N,)

    return (dist < threshold).float()  # (N,)  ∈ {0.0, 1.0}

def slow_down_near_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    distance_threshold: float = 0.6,
    vel_max: float = 0.5,
) -> torch.Tensor:
    """
    当机器人进入目标附近 distance_threshold 范围内时，
    速度越小奖励越高，最大奖励为 1.0，速度超过 vel_max 时奖励为 0。
    
    Returns: (N,) float tensor
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    # 水平距离
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)  # (N,)

    # 线速度大小
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel = robot.data.root_lin_vel_w[:, :2]       # (N, 2) 水平速度
    speed = torch.norm(lin_vel, dim=-1)              # (N,)

    # 仅在距离足够近时激活
    in_range = dist < distance_threshold             # (N,) bool

    # 速度奖励：线性从1（speed=0）降到0（speed>=vel_max），截断到[0,1]
    vel_reward = torch.clamp(1.0 - speed / vel_max, min=0.0, max=1.0)

    return (vel_reward * in_range.float())           # (N,)


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """
    Penalty proportional to the maximum contact force above `threshold`.

    r = -max(0, max_force - threshold)

    Uses IsaacLab's ContactSensor data.

    Args:
        sensor_cfg: SceneEntityCfg pointing to a ContactSensor on the robot.
        threshold:  Force (N) below which no penalty is applied.

    Returns shape (N,).
    """
    from isaaclab.sensors import ContactSensor

    sensor_cfg.resolve(env.scene)
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    # net_forces_w: (N, num_bodies, 3)
    forces     = sensor.data.net_forces_w
    force_norm = torch.norm(forces, dim=-1)          # (N, num_bodies)
    max_force  = force_norm.max(dim=-1).values       # (N,)

    penalty = torch.clamp(max_force - threshold, min=0.0)
    return -penalty  # negative reward




