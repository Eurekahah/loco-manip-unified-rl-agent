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
    threshold: float = 0.6,
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
    # print(f"Distance to target: {dist}")  # 调试用，观察距离分布

    return dist < threshold  # (N,) bool

def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    height_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Terminate (done=True) when the object is dropped, defined as the object's
    height above the table being less than `height_threshold`.

    Returns bool tensor of shape (N,).
    """
    object_pos_w = object_root_pos_w(env, object_cfg)
    object_height = object_pos_w[:, 2]  # (N,)

    # print(f"Current object height: {object_height}")  # 调试用，观察高度分布
    # print(f"Height threshold: {height_threshold}")

    return object_height < height_threshold  # (N,) bool
 