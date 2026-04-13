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
 

def object_held_for_duration(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    hold_duration: float = 5.0,
) -> torch.Tensor:
    """
    物体被持续举起超过 hold_duration 秒后触发终止。
    
    内部维护一个每环境的计时器（存在 env 的自定义属性中），
    每步当物体高于 initial_height + minimal_height 时累加 dt，
    否则清零；累计时间超过 hold_duration 则返回 True（触发终止）。

    Args:
        env: RL 环境实例
        minimal_height: 判定"举起"的最小高度阈值（相对于初始高度），单位：米
        object_cfg: 物体的 SceneEntityCfg
        hold_duration: 需要持续举起的时间，单位：秒，默认 5.0s
    Returns:
        shape (num_envs,) 的 bool 张量，True 表示该环境触发终止
    """
    object_asset = env.scene[object_cfg.name]
    device = env.device

    # ---------- 计时器初始化（懒加载，只初始化一次）----------
    _TIMER_KEY = "_hold_object_timer"
    if not hasattr(env, _TIMER_KEY):
        setattr(env, _TIMER_KEY, torch.zeros(env.num_envs, device=device))
    hold_timer: torch.Tensor = getattr(env, _TIMER_KEY)

    # ---------- 判断当前是否处于"举起"状态 ----------
    current_height = object_asset.data.root_pos_w[:, 2]          # (N,)
    initial_height = object_asset.data.default_root_state[:, 2]  # (N,)
    lifted_height  = current_height - initial_height              # (N,)
    is_lifted = lifted_height > minimal_height                    # (N,) bool

    # ---------- 更新计时器 ----------
    dt = env.step_dt  # 单步时间，单位秒
    hold_timer += dt * is_lifted.float()   # 举起则累加
    hold_timer *= is_lifted.float()        # 未举起则清零（等价于 where）

    # ---------- reset 时清零对应环境的计时器 ----------
    # env.termination_manager 会在 episode 结束后自动 reset 环境，
    # 但计时器是自定义属性，需要手动响应 reset_buf
    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf.bool()
        hold_timer[reset_mask] = 0.0

    # ---------- 写回（原地操作已修改，这步是防御性保证） ----------
    setattr(env, _TIMER_KEY, hold_timer)

    # ---------- 终止条件 ----------
    return hold_timer >= hold_duration