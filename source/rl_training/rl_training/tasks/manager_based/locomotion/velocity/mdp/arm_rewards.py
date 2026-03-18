# =============================================================================
# arm_rewards.py
# 机械臂奖励函数 —— 适配四足+机械臂平台，EE差分控制（IK），目标由
# HeightInvariantEECommandCfg采样
# =============================================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_arm_weight(env: ManagerBasedRLEnv, command_name: str | None) -> torch.Tensor:
    """
    通用helper：从 CommandManager 取臂部权重。
    若 command_name 为 None，返回全1（不缩放）。
    """
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    return env.command_manager.get_command(command_name)[:, 0]  # (N,)



# =============================================================================
# 1. 位置跟踪奖励
# =============================================================================

def ee_position_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_name: str = "arm_link6",
    std: float = 0.15,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    EE位置跟踪，使用高斯核：exp(-||pos_error||² / (2*std²))
    越接近目标奖励越高，最大为1。
    
    Args:
        std: 高斯核标准差，控制奖励的"宽度"，建议0.1~0.2
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 7): pos(3) + quat(4)

    # 获取EE在世界坐标系下的位姿
    body_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, body_idx, :]      # (N, 3)
    ee_quat_w = robot.data.body_quat_w[:, body_idx, :]    # (N, 4)

    # 目标位姿（世界系）
    target_pos_w = command[:, :3]
    
    # 计算位置误差
    pos_error = torch.norm(target_pos_w - ee_pos_w, dim=-1)  # (N,)
    reward = torch.exp(-pos_error**2 / (2 * std**2))  # (N,)
    weight = _get_arm_weight(env, arm_weight_command_name)
    # print(f"Position tracking reward: mean={reward.mean().item():.4f}, "f"pos_error: mean={pos_error.mean().item():.4f}, "f"arm_weight: mean={weight.mean().item():.4f}")
    return reward * weight
    
    return torch.exp(-pos_error**2 / (2 * std**2)) * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 2. 姿态跟踪奖励
# =============================================================================

def ee_orientation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_name: str = "arm_link6",
    std: float = 0.5,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    EE姿态跟踪，使用高斯核：exp(-||quat_error||² / (2*std²))
    quat_error_magnitude返回旋转角误差（弧度）。
    
    Args:
        std: 高斯核标准差，单位弧度，建议0.3~0.8
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    body_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_quat_w = robot.data.body_quat_w[:, body_idx, :]    # (N, 4) wxyz

    target_quat_w = command[:, 3:7]                        # (N, 4) wxyz

    # 计算最小旋转角误差（弧度）
    angle_error = quat_error_magnitude(ee_quat_w, target_quat_w)  # (N,)

    return torch.exp(-angle_error**2 / (2 * std**2)) * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 3. 到达目标的稀疏奖励
# =============================================================================

def ee_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_name: str = "arm_link6",
    pos_threshold: float = 0.05,      # 5cm
    angle_threshold: float = 0.2,     # ~11.5°
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    同时满足位置和姿态阈值时给稀疏奖励（0或1）。
    可用于触发抓取或作为额外激励。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    body_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_pos_w   = robot.data.body_pos_w[:, body_idx, :]
    ee_quat_w  = robot.data.body_quat_w[:, body_idx, :]

    target_pos_w  = command[:, :3]
    target_quat_w = command[:, 3:7]

    pos_error   = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    angle_error = quat_error_magnitude(ee_quat_w, target_quat_w)

    reached = (pos_error < pos_threshold) & (angle_error < angle_threshold)
    return reached.float() * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 4. 抓取成功奖励
# =============================================================================

def grasp_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_name: str = "arm_link6",
    grasp_distance_threshold: float = 0.08,   # EE到物体的距离阈值
    lift_height_threshold: float = 0.05,       # 物体抬升高度阈值
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    抓取成功奖励：
    - 条件1: EE与物体距离 < grasp_distance_threshold（接触判断）
    - 条件2: 物体高于初始位置 lift_height_threshold（验证有效抓取）
    
    如果没有物体检测，可只用条件1作为"接近物体"的密集奖励。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject   = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_pos_w  = robot.data.body_pos_w[:, body_idx, :]    # (N, 3)
    obj_pos_w = obj.data.root_pos_w                       # (N, 3)

    # EE到物体距离
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)      # (N,)

    # 物体是否被抬起（相对于场景初始z坐标）
    # 如果有记录物体初始高度，可替换为 obj_init_pos_w[:, 2]
    obj_lifted = obj_pos_w[:, 2] > (obj.data.default_root_state[:, 2] + lift_height_threshold)

    grasped = (dist < grasp_distance_threshold) & obj_lifted
    return grasped.float() * _get_arm_weight(env, arm_weight_command_name)


def ee_approach_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_name: str = "arm_link6",
    std: float = 0.1,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    密集的接近物体奖励（抓取前引导），用高斯核。
    可与 grasp_success 配合使用，在接近阶段提供稠密引导。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject   = env.scene[object_cfg.name]

    body_idx = robot.find_bodies(ee_frame_name)[0][0]
    ee_pos_w  = robot.data.body_pos_w[:, body_idx, :]
    obj_pos_w = obj.data.root_pos_w

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    return torch.exp(-dist**2 / (2 * std**2)) * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 5. 关节力矩惩罚
# =============================================================================

def arm_joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_names: list[str] | None = None,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    机械臂关节力矩L2惩罚，防止过大出力。
    返回值为负（惩罚），在cfg中权重设为负数。
    
    Args:
        arm_joint_names: 机械臂关节名列表，None则使用asset_cfg.joint_ids
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if arm_joint_names is not None:
        joint_ids = robot.find_joints(arm_joint_names)[0]
        torques = robot.data.applied_torque[:, joint_ids]
    elif asset_cfg.joint_ids is not None:
        torques = robot.data.applied_torque[:, asset_cfg.joint_ids]
    else:
        torques = robot.data.applied_torque

    return torch.sum(torques**2, dim=-1) * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 6. 关节速度惩罚
# =============================================================================

def arm_joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_names: list[str] | None = None,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    机械臂关节速度L2惩罚，鼓励平滑运动。
    返回值为负（惩罚），在cfg中权重设为负数。
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if arm_joint_names is not None:
        joint_ids = robot.find_joints(arm_joint_names)[0]
        vel = robot.data.joint_vel[:, joint_ids]
    elif asset_cfg.joint_ids is not None:
        vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        vel = robot.data.joint_vel

    return torch.sum(vel**2, dim=-1) * _get_arm_weight(env, arm_weight_command_name)


# =============================================================================
# 7. 关节加速度惩罚（可选，进一步平滑）
# =============================================================================

def arm_joint_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_names: list[str] | None = None,
    arm_weight_command_name: str | None = None,
) -> torch.Tensor:
    """
    机械臂关节加速度L2惩罚，抑制抖动。
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if arm_joint_names is not None:
        joint_ids = robot.find_joints(arm_joint_names)[0]
        acc = robot.data.joint_acc[:, joint_ids]
    elif asset_cfg.joint_ids is not None:
        acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
    else:
        acc = robot.data.joint_acc

    return torch.sum(acc**2, dim=-1) * _get_arm_weight(env, arm_weight_command_name)