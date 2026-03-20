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
# =============================================================================
# Terminations
# =============================================================================
 
def reached_target(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.4,
    vel_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Terminate (done=True) when the robot is within `threshold` metres of the
    target (horizontal distance only) AND the robot's horizontal speed is
    below `vel_threshold` m/s.

    Returns bool tensor of shape (N,).
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)  # (N,)

    robot: Articulation = env.scene[robot_cfg.name]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)  # (N,)

    return (dist < threshold) & (speed < vel_threshold)  # (N,) bool
 