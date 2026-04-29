# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_rotate_inverse,
    quat_apply,
    yaw_quat,
    wrap_to_pi,
    quat_error_magnitude,
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

def distance_to_target_reward_shift(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    lam: float = 2.0,   # λ 越大，收敛越快（建议 3.0–5.0）
) -> torch.Tensor:
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1).clamp(min=1e-3)   # 防除零
    reward = 1.0 - torch.exp(-lam / dist)              # ∈ (0, 1)
    return reward

# def distance_to_target_reward(env, robot_cfg, target_cfg, std=1.0):
#     robot_pos_w  = robot_root_pos_w(env, robot_cfg)
#     target_pos_w = object_root_pos_w(env, target_cfg)
#     diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
#     dist = torch.norm(diff, dim=-1)
    
#     # 用 1/(1+dist) 代替 Gaussian
#     # 0.9m → 0.526,  0.7m → 0.588,  0.0m → 1.0
#     # 越近梯度越大，越有动力冲进去
#     reward = 1.0 / (1.0 + dist)
#     return reward


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
    penalty_scale: float = 1.0,
) -> torch.Tensor:
    """
    当机器人进入目标附近 distance_threshold 范围内时：
      - 速度 <= vel_max：线性奖励 [0, 1]，速度越小奖励越高
      - 速度 >  vel_max：线性惩罚（负值），超速越多惩罚越重
    
    奖励/惩罚曲线（speed 轴）：
    
      1.0 |*
          | *
      0.0 |----*------*--------> speed
          |  vel_max  *
     -1.0 |            *
          |             * (惩罚随超速线性增大)

    Returns: (N,) float tensor
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    # 水平距离
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)  # (N,)

    # 线速度大小
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel = robot.data.root_lin_vel_w[:, :2]  # (N, 2) 水平速度
    speed   = torch.norm(lin_vel, dim=-1)        # (N,)

    # 仅在距离足够近时激活
    in_range = (dist < distance_threshold).float()  # (N,)

    # ---- 奖励/惩罚逻辑 ----
    # 合规区间 [0, vel_max]：reward ∈ [0, 1]，speed=0 时最高
    reward  = torch.clamp(1.0 - speed / vel_max, min=0.0, max=1.0)

    # 超速区间 (vel_max, +∞)：penalty < 0，超速量越大惩罚越重
    # 超速量归一化：excess = (speed - vel_max) / vel_max
    excess  = torch.clamp(speed - vel_max, min=0.0) / vel_max
    penalty = -excess * penalty_scale               # 负值

    # 叠加：合规时 penalty=0，超速时 reward=0
    combined = reward + penalty                     # 两段函数自然拼接
    
    return combined * in_range

def reach_target_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float = 0.7,
    vel_good: float = 0.1,    # 低于此速度：满分
    vel_bad: float = 0.5,     # 高于此速度：负分上限
) -> torch.Tensor:
    """
    仅在到达目标的终止帧触发，根据速度给连续奖励：
    
    speed:   0         vel_good      vel_bad       +∞
             |            |             |
    reward:  +1.0  ----  +1.0  \  0.0  \ -1.0 ----  -1.0
                               线性下降   线性下降（钳位）
    
    非终止帧返回 0，不干扰日常训练。
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    dist = torch.norm(
        target_pos_w[:, :2] - robot_pos_w[:, :2], dim=-1
    )  # (N,)

    robot: Articulation = env.scene[robot_cfg.name]
    speed = torch.norm(
        robot.data.root_lin_vel_w[:, :2], dim=-1
    )  # (N,)

    # ── 连续速度评分 ──────────────────────────────────────────
    # 段1: speed ∈ [0, vel_good]         → reward = 1.0
    # 段2: speed ∈ [vel_good, vel_bad]   → reward 从 1.0 线性降到 -1.0
    # 段3: speed ∈ [vel_bad, +∞)         → reward = -1.0（钳位）
    t = (speed - vel_good) / (vel_bad - vel_good + 1e-6)  # 0→1
    t = torch.clamp(t, 0.0, 1.0)
    vel_score = 1.0 - 2.0 * t   # 1.0 → -1.0

    # 仅在到达目标帧激活（用 reset_buf 或直接检查距离）
    in_target = (dist < threshold).float()  # (N,)
    # print(speed)

    return vel_score * in_target


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # print("force_matrix_w:", contact_sensor.data.force_matrix_w)
    # print("force_matrix_w shape:", contact_sensor.data.force_matrix_w.shape if contact_sensor.data.force_matrix_w is not None else "None")
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
   
    # if sensor_cfg.name == "arm_contact_forces" and reward.sum() > 0:
    #     print(f"Net contact forces: {net_contact_forces[:, :, sensor_cfg.body_ids]}")
    #     print(f"Undesired contacts: {reward}")
    return reward

def gripper_object_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # force_matrix_w shape: [N, H, num_sensor_bodies, 3]
    force_matrix = contact_sensor.data.force_matrix_w

    # ✅ 用 sensor 内部局部索引，而不是全局 body_ids
    # contact_sensor.body_names 是 sensor 追踪的 body 列表
    # 找到目标 body 在 sensor 内部的索引
    target_body_name = sensor_cfg.body_names  # 例如 "arm_link7"
    if isinstance(target_body_name, str):
        target_body_name = [target_body_name]
    
    # 在 sensor.body_names 中查找局部索引
    local_ids = [
        contact_sensor.body_names.index(name)
        for name in target_body_name
        if name in contact_sensor.body_names
    ]

    if not local_ids:
        # 找不到目标 body，返回零奖励
        print(f"[WARN] body {target_body_name} not found in sensor bodies: {contact_sensor.body_names}")
        return torch.zeros(force_matrix.shape[0], device=force_matrix.device)

    # print(f"Gripper-object contact sensor '{sensor_cfg.name}'")
    # print(f"Tracking bodies: {contact_sensor.body_names}")
    # print(f"Local IDs: {local_ids}")
    # print(f"force_matrix_w shape: {force_matrix.shape}")          # [N, H, 2, 3]
    # print(f"body_names count: {len(contact_sensor.body_names)}")
    # shape: [N, H, len(local_ids), 3]
    finger_forces = force_matrix[:, local_ids, :,  :]

    # 计算力的大小并取历史最大值
    force_norm = torch.norm(finger_forces, dim=-1)   # [N, H, len(local_ids)]
    N = force_norm.shape[0]
    max_force = force_norm.view(N, -1).max(dim=-1)[0]  # [N]

    is_contact = max_force > threshold
    reward = is_contact.float()

    # if reward.sum() > 0:
    #     print(f"[{target_body_name}] object contact max force: {max_force[is_contact]}")

    return reward

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object_asset = env.scene[object_cfg.name]
    robot_asset = env.scene[ee_frame_cfg.name]

    # 物体位置：有 body_ids 用 body_pos_w，否则用 root_pos_w
    
    object_pos_w = object_asset.data.root_pos_w[:, :3]

    # EE 位置：必须指定 body_names，body_ids 由框架解析
    if ee_frame_cfg.body_ids is not None:
        ee_pos_w = robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]
    else:
        # fallback：取最后一个 body（不推荐，应确保 body_names 已配置）
        ee_pos_w = robot_asset.data.body_pos_w[:, -1, :]

    distance = torch.norm(object_pos_w - ee_pos_w, dim=-1)
    reward = torch.exp(-distance ** 2 / (2 * std ** 2))

    return reward


def delta_action_l2_near_target(
    env,
    action_name: str,
    object_cfg,
    distance_threshold: float = 0.2,
):
    """
    Penalize delta action magnitude when EE is close to object.

    Returns:
        reward: (num_envs,)
    """
    object_asset = env.scene[object_cfg.name]
    # ---------------------------
    # 1. 获取 delta action
    # ---------------------------
    action_term = env.action_manager.get_term(action_name)
    delta_action = action_term._delta_action  # (N, action_dim)

    # L2 norm
    action_l2 = torch.sum(delta_action ** 2, dim=-1)  # (N,)

    # ---------------------------
    # 2. 获取 EE 和 object 距离
    # ---------------------------
    object_pos_w = object_asset.data.root_pos_w[:, :3]  # (N, 3)

    # 默认用 EE（arm_link6）
    ee_pos = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("arm_link6"), :]

    dist = torch.norm(ee_pos - object_pos_w, dim=-1)  # (N,)

    # ---------------------------
    # 3. gating（只在接近时生效）
    # ---------------------------
    near_mask = (dist < distance_threshold).float()

    # ---------------------------
    # 4. reward
    # ---------------------------
    reward = action_l2 * near_mask

    return reward

def ee_velocity_l2(
    env,
    ee_frame_cfg,
):
    """
    Penalize end-effector velocity magnitude.

    Returns:
        reward: (num_envs,)
    """

    robot = env.scene[ee_frame_cfg.name]

    # ---------------------------
    # 1. 找到 EE index
    # ---------------------------
    body_name = ee_frame_cfg.body_names[0]
    body_id = robot.data.body_names.index(body_name)

    # ---------------------------
    # 2. 获取 EE 速度
    # ---------------------------
    # linear velocity (world frame)
    ee_vel = robot.data.body_lin_vel_w[:, body_id, :]  # (N, 3)

    # L2 norm
    vel_l2 = torch.sum(ee_vel ** 2, dim=-1)  # (N,)

    # ---------------------------
    # 3. reward
    # ---------------------------
    reward = vel_l2

    return reward

def object_ee_symmetric_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    min_finger_dist: float,          # 新增：夹爪最小张开距离（米）
    object_cfg: SceneEntityCfg,
    ee_frame_cfg_finger1: SceneEntityCfg,
    ee_frame_cfg_finger2: SceneEntityCfg,
) -> torch.Tensor:
    """
    对称夹爪对准奖励：
    1. 两指到物体距离之和最小（整体靠近）
    2. 两指到物体距离之差最小（对称）
    3. 两指间距必须大于 min_finger_dist（防止夹爪闭合作弊）
    """
    object_asset = env.scene[object_cfg.name]
    robot_asset = env.scene[ee_frame_cfg_finger1.name]

    object_pos_w = object_asset.data.root_pos_w[:, :3]

    def get_finger_pos(ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
        if ee_frame_cfg.body_ids is not None:
            return robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]
        else:
            return robot_asset.data.body_pos_w[:, -1, :]

    pos1 = get_finger_pos(ee_frame_cfg_finger1)  # [N, 3]
    pos2 = get_finger_pos(ee_frame_cfg_finger2)  # [N, 3]

    # print(f"Finger1 pos: {pos1}")
    # print(f"Finger2 pos: {pos2}")

    dist1 = torch.norm(object_pos_w - pos1, dim=-1)   # [N]
    dist2 = torch.norm(object_pos_w - pos2, dim=-1)   # [N]

    # 1. 平均距离奖励
    mean_dist = (dist1 + dist2) / 2.0
    reward_proximity = torch.exp(-mean_dist ** 2 / (2 * std ** 2))

    # 2. 对称性奖励
    asymmetry = (dist1 - dist2).abs()
    reward_symmetry = torch.exp(-asymmetry ** 2 / (2 * std ** 2))

    # 3. 夹爪张开门控：两指间距必须 > min_finger_dist 才给奖励
    finger_span = torch.norm(pos1 - pos2, dim=-1)     # [N]
    # print(f"Finger span: {finger_span}")
    gate_open = (finger_span > min_finger_dist).float()

    # print(f"gate_open: {gate_open}")
    return reward_proximity * reward_symmetry * gate_open

def forward_velocity_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "pre_trained_pick_action",
    target_cfg: SceneEntityCfg = None,
    cmd_proximity_gate: float = 0.8,    # cmd_pos 距离目标小于此值时才惩罚速度
    stop_penalty_dist: float = 0.5,     # 底盘已到位时也不再惩罚（可选，与接近奖励对齐）
) -> torch.Tensor:
    """
    前向速度惩罚，带双重门控：
    1. cmd_pos 门控：HL 下发的目标点离 object 足够近时，说明机器人正在做精细接近，才惩罚速度
    2. 到位门控（可选）：底盘已到达抓取距离后，速度惩罚归零（机器人应切换抓取，不再导航）
    """
    action_term = env.action_manager.get_term(action_name)
    vx = action_term.ll_command[:, 0]              # v_x 前向速度指令

    # ---- 门控 1：cmd_pos 距离门控 ----
    gate_cmd = torch.ones_like(vx)                 # 默认全惩罚（无 target_cfg 时退化）
    if target_cfg is not None:
        object_pos_w = object_root_pos_w(env, target_cfg)
        cmd_pos_w    = action_term.ll_command[:, 3:6]          # HL 下发的目标位置
        cmd_dist     = torch.norm(cmd_pos_w - object_pos_w, dim=-1)
        gate_cmd     = (cmd_dist < cmd_proximity_gate).float() # 接近目标点时 = 1

    return gate_cmd * vx.pow(2)

def lateral_velocity_penalty(env: ManagerBasedRLEnv, action_name: str = "pre_trained_nav_action") -> torch.Tensor:
    """惩罚侧向速度命令 v_y（ll_command[:, 1]）"""
    action_term = env.action_manager.get_term(action_name)
    vy = action_term.ll_command[:, 1]  # v_y
    return vy.pow(2)


def angular_velocity_penalty(env: ManagerBasedRLEnv, action_name: str = "pre_trained_nav_action") -> torch.Tensor:
    """惩罚角速度命令 omega_z（ll_command[:, 2]）"""
    action_term = env.action_manager.get_term(action_name)
    wz = action_term.ll_command[:, 2]  # omega_z
    return wz.pow(2)

def _measure_ee_offset(robot):
    """临时调用来测量 body_offset，确认后删除"""
    
    # ---- 获取 body indices ----
    # IsaacLab 2.3.0 用 find_bodies 返回 (indices, names)
    link6_indices, _  = robot.find_bodies("arm_link6")
    link7_indices, _  = robot.find_bodies("arm_link7")
    link8_indices, _  = robot.find_bodies("arm_link8")

    link6_idx = link6_indices[0]
    link7_idx = link7_indices[0]
    link8_idx = link8_indices[0]

    # ---- 世界坐标 (num_envs, 3) ----
    link6_pos_w = robot.data.body_pos_w[:, link6_idx, :]   # wrist
    link7_pos_w = robot.data.body_pos_w[:, link7_idx, :]   # finger1
    link8_pos_w = robot.data.body_pos_w[:, link8_idx, :]   # finger2

    link6_quat_w = robot.data.body_quat_w[:, link6_idx, :] # (w,x,y,z)

    # ---- 夹爪中心 = link7 和 link8 的中点 ----
    gripper_center_w = (link7_pos_w + link8_pos_w) / 2.0

    # ---- 计算 link6 -> gripper_center 在 link6 局部坐标系下的偏移 ----
    from isaaclab.utils.math import quat_inv, quat_rotate
    
    # link6 坐标系的逆旋转
    link6_quat_inv = quat_inv(link6_quat_w)  # (num_envs, 4)
    
    # 世界坐标系下的位移向量
    diff_w = gripper_center_w - link6_pos_w  # (num_envs, 3)
    
    # 转换到 link6 局部坐标系
    diff_local = quat_rotate(link6_quat_inv, diff_w)  # (num_envs, 3)

    # ---- 只打印第一个 env 的结果 ----
    print("=" * 50)
    print(f"[link6]  world pos  : {link6_pos_w[0].cpu().numpy()}")
    print(f"[link6]  world quat : {link6_quat_w[0].cpu().numpy()}  (w,x,y,z)")
    print(f"[link7]  world pos  : {link7_pos_w[0].cpu().numpy()}")
    print(f"[link8]  world pos  : {link8_pos_w[0].cpu().numpy()}")
    print(f"[gripper center] world pos : {gripper_center_w[0].cpu().numpy()}")
    print(f"[body_offset] pos (in link6 frame) : {diff_local[0].cpu().numpy()}")
    print("=" * 50)

def get_max_force(env: ManagerBasedRLEnv,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper function to extract the maximum contact force for specified bodies from a ContactSensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w  # [N, H, num_bodies, 3]

    target_body_name = sensor_cfg.body_names
    if isinstance(target_body_name, str):
        target_body_name = [target_body_name]

    local_ids = [
        contact_sensor.body_names.index(name)
        for name in target_body_name
        if name in contact_sensor.body_names
    ]
    if not local_ids:
        return torch.zeros(force_matrix.shape[0], device=force_matrix.device)

    # ✅ 修正：第2维才是 body
    finger_forces = force_matrix[:, :, local_ids, :]  # [N, H, len(local_ids), 3]
    # print(f"Sensor '{sensor_cfg.name}' tracking bodies: {contact_sensor.body_names}, local_ids: {local_ids}")
    # print(f"force_matrix_w shape: {force_matrix.shape}, finger_forces shape: {finger_forces.shape}")
    force_norm = torch.norm(finger_forces, dim=-1)     # [N, H, len(local_ids)]
    N = force_norm.shape[0]
    max_force = force_norm.view(N, -1).max(dim=-1)[0]  # [N]
    return max_force


def gripper_both_fingers_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg_finger1: SceneEntityCfg,
    sensor_cfg_finger2: SceneEntityCfg,
) -> torch.Tensor:
    """双指同时接触才给奖励（乘法耦合，消除单指靠墙局部最优）"""


    max_force1 = get_max_force(env, sensor_cfg_finger1)  # [N]
    max_force2 = get_max_force(env, sensor_cfg_finger2)  # [N]

    # print(f"Finger1 max force: {max_force1}")
    # print(f"Finger2 max force: {max_force2}")
    contact1 = (max_force1 > threshold).float()
    contact2 = (max_force2 > threshold).float()

    # 乘法耦合：两指都接触才得奖励
    return contact1 * contact2

def gripper_contact_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg_finger1: SceneEntityCfg,
    sensor_cfg_finger2: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚两指接触力不对称（返回负值，配合负权重使用或直接用正权重+负号）"""

    max_force1 = get_max_force(env, sensor_cfg_finger1)  # [N]
    max_force2 = get_max_force(env, sensor_cfg_finger2)  # [N]

    asymmetry = (max_force1 - max_force2).abs()          # [N]
    # 归一化防止数值过大
    return asymmetry / (max_force1 + max_force2 + 1e-6)

def gripper_contact_symmetric_grasp(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg_finger1: SceneEntityCfg,
    sensor_cfg_finger2: SceneEntityCfg,
    ee_frame_cfg_finger1: SceneEntityCfg,  # 新增：用于获取夹爪位置
    ee_frame_cfg_finger2: SceneEntityCfg,  # 新增：用于获取夹爪位置
    object_cfg: SceneEntityCfg,
    min_finger_dist: float = 0.02,          # 新增：夹爪最小张开距离（米）
    gripper_action_name: str = "gripper_action",
    cmd_proximity_gate: float = 0.1,          # cmd_pos 到物体距离阈值
    action_term_name: str = "pre_trained_pick_action",  # 用于取 cmd_pos
) -> torch.Tensor:
    
    # ---- 获取夹爪闭合状态 ----
    gripper_term = env.action_manager.get_term(gripper_action_name)
    is_close_cmd = torch.all(
        gripper_term._processed_actions == gripper_term._close_command, dim=-1
    ).float()  # [N,]

    # ---- 接触力门控 ----
    max_force1 = get_max_force(env, sensor_cfg_finger1)
    max_force2 = get_max_force(env, sensor_cfg_finger2)

    contact1 = (max_force1 > threshold).float()
    contact2 = (max_force2 > threshold).float()
    gate_contact = contact1 * contact2

    # ---- 对称性奖励 ----
    asymmetry = (max_force1 - max_force2).abs()
    reward_symmetry = 1.0 - asymmetry / (max_force1 + max_force2 + 1e-6)

    # ---- 夹爪张开距离门控（防止完全闭合时接触作弊）----
    robot_asset = env.scene[ee_frame_cfg_finger1.name]

    def get_finger_pos(ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
        if ee_frame_cfg.body_ids is not None:
            return robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]
        else:
            return robot_asset.data.body_pos_w[:, -1, :]

    pos1 = get_finger_pos(ee_frame_cfg_finger1)  # [N, 3]
    pos2 = get_finger_pos(ee_frame_cfg_finger2)  # [N, 3]

    finger_span = torch.norm(pos1 - pos2, dim=-1)        # [N]
    gate_open = (finger_span > min_finger_dist).float()  # [N,]

    # ----cmd_pos 到物体距离门控 ----
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.ll_command[:, 3:6]                        # [N, 3]
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w             # [N, 3]  ← 需在参数里加 object_cfg
    cmd_dist  = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)              # [N,]
    gate_close_cmd  = (cmd_dist < cmd_proximity_gate).float()                # [N,]

    # ---- 综合门控 ----
    return gate_contact * reward_symmetry * is_close_cmd * gate_open * gate_close_cmd

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    cmd_proximity_gate: float = 0.2,          # cmd_pos 到物体距离阈值
    action_term_name: str = "pre_trained_pick_action",  # 用于取 cmd_pos
) -> torch.Tensor:
    """
    物体被抬起的奖励。

    物体高于每次 reset 后记录的初始高度 + minimal_height 时给予奖励。
    奖励为连续值：超出越多奖励越高（clamp 在 [0, 1] 内）。

    Args:
        env: RL 环境实例
        minimal_height: 物体需要被抬起的最小高度（相对于 reset 后实际高度），单位：米
        object_cfg: 物体的 SceneEntityCfg
    Returns:
        shape (num_envs,) 的奖励张量，值域 [0, 1]
    """
    object_asset = env.scene[object_cfg.name]
    current_height = object_asset.data.root_pos_w[:, 2]  # (N,)

    # ---------- 记录每次 reset 后的实际初始高度 ----------
    # 用 env 上的自定义属性存储，key 加上 object 名称避免多物体冲突
    cache_key = f"_lifted_init_height_{object_cfg.name}"

    if not hasattr(env, cache_key):
        # 首次调用：用当前高度初始化（shape: (num_envs,)，存在 GPU 上）
        setattr(env, cache_key, current_height.clone())

    init_height: torch.Tensor = getattr(env, cache_key)

    # env.episode_length_buf == 1 表示该 env 刚刚被 reset（第 1 步）
    just_reset = env.episode_length_buf == 1  # (N,) bool
    if just_reset.any():
        init_height[just_reset] = current_height[just_reset].clone()
    # ------------------------------------------------------

    lifted_height = current_height - init_height  # (N,)
    # print(f"Current height: {current_height}, Initial height: {init_height}, Lifted height: {lifted_height}")

    # ----cmd_pos 到物体距离门控 ----
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.ll_command[:, 3:6]                        # [N, 3]
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w             # [N, 3]  ← 需在参数里加 object_cfg
    cmd_dist  = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)              # [N,]
    gate_close_cmd  = (cmd_dist < cmd_proximity_gate).float()                # [N,]

    reward = torch.clamp(
        (lifted_height - minimal_height) / minimal_height,
        min=0.0,
        max=1.0,
    ) * gate_close_cmd
    return reward

def cmd_pos_to_object_reward(
    env: ManagerBasedRLEnv,
    action_term_name: str = "pre_trained_pick_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_sigma: float = 0.1,
    use_shaped: bool = True,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.ll_command[:, 3:6]
    
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    
    pos_dist = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)  # (N,)

    # print(f"Command position(world): {cmd_pos_w}")
    # print(f"Object position(world): {obj_pos_w}")
    # print(f"Position distance: {pos_dist}")
    if use_shaped:
        # ✅ 方案A：线性 + Gaussian 混合
        # 远处线性引导（始终有梯度），近处Gaussian精确奖励
        linear_reward   = 1.0 / (1.0 + pos_dist)                               # 始终有信号
        gaussian_reward = torch.exp(-pos_dist.pow(2) / (2 * pos_sigma ** 2))    # 近处精确
        reward = 0.3 * linear_reward + 0.7 * gaussian_reward
        # print(f"Linear reward: {linear_reward}, Gaussian reward: {gaussian_reward}, Combined reward: {reward}")
    else:
        # 原始Gaussian（梯度消失，不推荐）
        reward = torch.exp(-pos_dist.pow(2) / (2 * pos_sigma ** 2))
    
    return reward

# ─────────────────────────────────────────────────────────────────────────────
# Potential-based shaping helpers
# 所有函数共用同一套"历史最近距离"存储逻辑，key 按 (reward_name, env_id) 区分
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_init_min(env, key: str, init_val: torch.Tensor) -> torch.Tensor:
    """
    取出存储在 env.extras[key] 的历史最近值张量。
    首次调用或 reset 后用 init_val 初始化。
    episode_length_buf == 1 → 该 env 刚被 reset，用当前值覆盖。
    """
    if key not in env.extras:
        env.extras[key] = init_val.clone()

    stored: torch.Tensor = env.extras[key]

    just_reset = env.episode_length_buf == 1          # (N,) bool
    if just_reset.any():
        stored[just_reset] = init_val[just_reset].clone()

    return stored


def distance_to_target_reward_shift_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    sensitivity: float = 20.0,
    penalty_scale: float = 0.0,
    stop_reward_dist: float = 0.73,    # 最优抓取距离（甜甜圈半径）
    overshoot_penalty_scale: float = 1.0,  # 太近时的惩罚强度
) -> torch.Tensor:
    """
    势能塑形版底盘接近奖励，带过近惩罚：
    - dist > stop_reward_dist：进步奖励（tanh 塑形）
    - dist ≈ stop_reward_dist：零（最优位置）
    - dist < stop_reward_dist：惩罚（太近不利于抓取）
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1).clamp(min=1e-3)

    # -------- 区域一：dist > stop_reward_dist，进步奖励 --------
    key = f"_min_chassis_dist_{target_cfg.name}"
    min_dist = _get_or_init_min(env, key, dist)

    progress   = (min_dist - dist).clamp(min=0.0)
    regression = (dist - min_dist).clamp(min=0.0)

    # 只在还未到达最优距离时更新历史最近（防止过冲后继续"记录"更近的错误距离）
    not_overshot = (dist >= stop_reward_dist)
    env.extras[key] = torch.where(
        not_overshot,
        torch.minimum(min_dist, dist),
        min_dist   # 已过近，冻结历史最近，不再更新
    )

    far_gate = (dist > stop_reward_dist).float()
    approach_reward = far_gate * (
        torch.tanh(sensitivity * progress)
        - penalty_scale * torch.tanh(sensitivity * regression)
    )

    # -------- 区域二：dist < stop_reward_dist，过近惩罚 --------
    overshoot = (stop_reward_dist - dist).clamp(min=0.0)   # 超出量，>0 才生效
    overshoot_penalty = overshoot_penalty_scale * torch.tanh(sensitivity * overshoot)

    return approach_reward - overshoot_penalty





def heading_to_target_reward_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    std: float = 0.3,
    sensitivity: float = 30.0,
    penalty_scale: float = 0.0,
    near_dist_gate: float = 0.3,   # 距离小于此值时朝向奖励关闭（已到位）
) -> torch.Tensor:
    """
    势能塑形版朝向奖励。
    只有角度误差比历史最小值更小时给正奖励。
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    robot_quat_w = robot_root_quat_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, target_cfg)

    rel_pos_w      = target_pos_w[:, :2] - robot_pos_w[:, :2]
    target_angle_w = torch.atan2(rel_pos_w[:, 1], rel_pos_w[:, 0])

    heading_quat = yaw_quat(robot_quat_w)
    robot_yaw    = 2.0 * torch.atan2(heading_quat[:, 3], heading_quat[:, 0])

    angle_err = wrap_to_pi(target_angle_w - robot_yaw).abs()   # (N,) ∈ [0, π]

    key = f"_min_heading_err_{target_cfg.name}"
    min_err = _get_or_init_min(env, key, angle_err)

    progress   = (min_err - angle_err).clamp(min=0.0)
    regression = (angle_err - min_err).clamp(min=0.0)

    env.extras[key] = torch.minimum(min_err, angle_err)

    # 距离太近时朝向无意义，关闭
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)
    gate = (dist >= near_dist_gate).float()

    reward = (torch.tanh(sensitivity * progress) - penalty_scale * torch.tanh(sensitivity * regression)) * gate
    return reward


def cmd_pos_to_object_reward_progress(
    env: ManagerBasedRLEnv,
    action_term_name: str = "pre_trained_pick_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensitivity: float = 50.0,
    penalty_scale: float = 0.0,
) -> torch.Tensor:
    """
    势能塑形版 EE 命令位置接近奖励。
    只有 cmd_pos 比历史最近更接近物体时给正奖励。
    """
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w   = action_term.ll_command[:, 3:6]                  # (N, 3)

    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w                                # (N, 3)

    dist = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)               # (N,)

    key = f"_min_cmd_dist_{object_cfg.name}"
    min_dist = _get_or_init_min(env, key, dist)

    progress   = (min_dist - dist).clamp(min=0.0)
    regression = (dist - min_dist).clamp(min=0.0)

    env.extras[key] = torch.minimum(min_dist, dist)

    reward = (torch.tanh(sensitivity * progress) - penalty_scale * torch.tanh(sensitivity * regression))

    return reward


def object_ee_symmetric_alignment_progress(
    env: ManagerBasedRLEnv,
    min_finger_dist: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg_finger1: SceneEntityCfg,
    ee_frame_cfg_finger2: SceneEntityCfg,
    action_term_name: str = "pre_trained_pick_action",
    cmd_proximity_gate: float = 0.3,   # cmd_pos 距物体多近才激活夹爪对准奖励
    sensitivity: float = 50.0,
    penalty_scale: float = 0.0,
) -> torch.Tensor:
    object_asset = env.scene[object_cfg.name]
    robot_asset  = env.scene[ee_frame_cfg_finger1.name]
    object_pos_w = object_asset.data.root_pos_w[:, :3]

    def get_finger_pos(cfg):
        if cfg.body_ids is not None:
            return robot_asset.data.body_pos_w[:, cfg.body_ids[0], :]
        return robot_asset.data.body_pos_w[:, -1, :]

    pos1 = get_finger_pos(ee_frame_cfg_finger1)
    pos2 = get_finger_pos(ee_frame_cfg_finger2)

    dist1 = torch.norm(object_pos_w - pos1, dim=-1)
    dist2 = torch.norm(object_pos_w - pos2, dim=-1)
    mean_dist = (dist1 + dist2) / 2.0

    # 夹爪张开门控
    finger_span = torch.norm(pos1 - pos2, dim=-1)
    gate_open   = (finger_span > min_finger_dist).float()

    # ---- 核心新增：cmd_pos 距离门控 ----
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w   = action_term.ll_command[:, 3:6]
    cmd_dist    = torch.norm(cmd_pos_w - object_pos_w, dim=-1)
    gate_cmd    = (cmd_dist < cmd_proximity_gate).float()   # (N,)

    # 未进入门控区域时直接返回 0，不更新历史最近（避免污染 min_dist）
    key = f"_min_finger_mean_dist_{object_cfg.name}"
    if key not in env.extras:
        env.extras[key] = torch.full((env.num_envs,), float('inf'), device=env.device)
    min_dist = env.extras[key]

    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        min_dist[just_reset] = float('inf')   # reset 后重置为无穷大

    progress   = (min_dist - mean_dist).clamp(min=0.0)
    regression = (mean_dist - min_dist).clamp(min=0.0)

    # 只在 gate 激活时更新历史最近，gate 未激活时 min_dist 保持不变
    active = gate_cmd * gate_open                           # (N,)
    new_min = torch.where(active.bool(), 
                          torch.minimum(min_dist, mean_dist),
                          min_dist)
    env.extras[key] = new_min

    reward = (torch.tanh(sensitivity * progress)
            - penalty_scale * torch.tanh(sensitivity * regression))

    return reward * gate_open * gate_cmd


def gripper_contact_symmetric_grasp_progress(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg_finger1: SceneEntityCfg,
    sensor_cfg_finger2: SceneEntityCfg,
    ee_frame_cfg_finger1: SceneEntityCfg,
    ee_frame_cfg_finger2: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    min_finger_dist: float = 0.02,
    gripper_action_name: str = "gripper_action",
    cmd_proximity_gate: float = 0.1,
    action_term_name: str = "pre_trained_pick_action",
    sensitivity: float = 10.0,   # tanh 灵敏度
) -> torch.Tensor:

    # ---- 原有门控逻辑（保持不变）----
    gripper_term = env.action_manager.get_term(gripper_action_name)
    is_close_cmd = torch.all(
        gripper_term._processed_actions == gripper_term._close_command, dim=-1
    ).float()

    max_force1 = get_max_force(env, sensor_cfg_finger1)
    max_force2 = get_max_force(env, sensor_cfg_finger2)
    contact1 = (max_force1 > threshold).float()
    contact2 = (max_force2 > threshold).float()
    gate_contact = contact1 * contact2                          # 双指同时接触

    asymmetry = (max_force1 - max_force2).abs()
    reward_symmetry = 1.0 - asymmetry / (max_force1 + max_force2 + 1e-6)

    robot_asset = env.scene[ee_frame_cfg_finger1.name]
    def get_finger_pos(cfg):
        if cfg.body_ids is not None:
            return robot_asset.data.body_pos_w[:, cfg.body_ids[0], :]
        return robot_asset.data.body_pos_w[:, -1, :]

    pos1 = get_finger_pos(ee_frame_cfg_finger1)
    pos2 = get_finger_pos(ee_frame_cfg_finger2)
    finger_span = torch.norm(pos1 - pos2, dim=-1)
    gate_open = (finger_span > min_finger_dist).float()

    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.ll_command[:, 3:6]
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w
    cmd_dist  = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)
    gate_close_cmd = (cmd_dist < cmd_proximity_gate).float()

    # ---- 综合接触质量分（0~1）----
    contact_quality = gate_contact * reward_symmetry * is_close_cmd \
                    * gate_open * gate_close_cmd                 # (N,)

    # ---- 首次接触检测：只在从"未接触"→"接触"的跳变时给奖励 ----
    key = "_grasp_contact_prev"
    if key not in env.extras:
        env.extras[key] = torch.zeros(env.num_envs, device=env.device)

    prev_contact = env.extras[key]                              # (N,)  0 or 1
    curr_contact = (contact_quality > 0.0).float()             # (N,)

    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        prev_contact[just_reset] = 0.0

    # 上一步没接触、这一步接触 → 首次接触
    first_contact = ((prev_contact < 0.5) & (curr_contact > 0.5)).float()  # (N,)

    env.extras[key] = curr_contact.clone()

    # ---- 奖励：首次接触一次性奖励 ----
    reward = torch.tanh(sensitivity * first_contact)
    return reward

def cmd_pos_tracking_penalty(
    env: ManagerBasedRLEnv,
    action_term_name: str = "pre_trained_pick_action",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="arm_link6"),
    sensitivity: float = 20.0,
    gate_dist: float = 1.0,         # cmd 距物体超过此值时不惩罚（底盘阶段 cmd 乱飘是正常的）
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    """
    惩罚 cmd_pos 与当前 EE 实际位置的偏差。
    偏差越大说明低层执行器跟踪越困难，policy 生成了不可执行的命令。
    
    只在 cmd 已经靠近物体时激活（底盘导航阶段 cmd 和 EE 偏差大是正常的）。
    """
    # ---- EE 实际位置 ----
    robot_asset = env.scene[ee_frame_cfg.name]
    if ee_frame_cfg.body_ids is not None:
        ee_pos_w = robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]
    else:
        ee_pos_w = robot_asset.data.body_pos_w[:, -1, :]               # (N, 3)

    # ---- cmd_pos ----
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w   = action_term.ll_command[:, 3:6]                      # (N, 3)

    # ---- 跟踪误差 ----
    tracking_err = torch.norm(cmd_pos_w - ee_pos_w, dim=-1)            # (N,)

    # ---- gate：只在 cmd 已经靠近物体时激活 ----
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    cmd_dist  = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)
    gate      = (cmd_dist < gate_dist).float()                         # (N,)

    # tanh 压缩，tracking_err 越大惩罚越接近 -1
    penalty = torch.tanh(sensitivity * tracking_err)

    return penalty * gate

def base_vel_cmd_action_l1_near_object(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.8,
) -> torch.Tensor:
    """
    机体距物体在 distance_threshold 范围内时，对 policy 输出的 ll_command
    施加 L1 范数惩罚，抑制大幅指令抖动。
    范围外返回 0。
    """
    # 机体与物体的水平距离
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)
    target_pos_w = object_root_pos_w(env, object_cfg)
    diff = target_pos_w[:, :2] - robot_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)                          # (N,)

    gate = (dist < distance_threshold).float()               # (N,)

    # policy 输出的原始指令 L1 范数
    action_term = env.action_manager.get_term(action_term_name)
    l1  = action_term.ll_command[:,:2].abs().sum(dim=-1)                              # (N,)

    return l1 * gate

def ee_delta_pose_l1_near_object(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.2,
) -> torch.Tensor:
    """
    当 EE 与物体距离小于 distance_threshold 时，对 policy 输出的位姿增量指令
    ll_command[:, 3:6] 的 L1 范数进行惩罚。
    返回值 = L1_norm * gate，建议在总奖励中减去此项（正值表示惩罚强度）。
    """
    # ----- 获取 EE 位置 -----
    robot_asset = env.scene[ee_frame_cfg.name]
    if ee_frame_cfg.body_ids is not None:
        ee_pos_w = robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]   # (N,3)
    else:
        # 默认取最后一个 body（通常是 end-effector）
        ee_pos_w = robot_asset.data.body_pos_w[:, -1, :]                         # (N,3)

    # ----- 获取物体位置 -----
    object_asset = env.scene[object_cfg.name]
    obj_pos_w = object_asset.data.root_pos_w[:, :3]                              # (N,3)

    # ----- 三维欧氏距离门控 -----
    diff = obj_pos_w - ee_pos_w                                                  # (N,3)
    dist = torch.norm(diff, dim=-1)                                              # (N,)
    gate = (dist < distance_threshold).float()                                   # (N,)

    # ----- 获取 policy 输出的原始动作 -----
    action_term = env.action_manager.get_term(action_term_name)
    raw = action_term.raw_actions                                                # (N, action_dim)

    # ----- 计算位置增量指令 (索引 3~5) 的 L1 范数 -----
    # 假设 raw_actions 的维度至少为 6，前三维可能是基座速度或其它，3:6 为位姿增量
    delta_pose = raw[:, 3:6]                                                     # (N,3)
    l1_norm = delta_pose.abs().sum(dim=-1)                                       # (N,)

    return l1_norm * gate

def object_is_lifted_progress(
    env, object_cfg,
    cmd_proximity_gate=0.2,
    action_term_name="pre_trained_pick_action",
    sensitivity: float = 50.0,
) -> torch.Tensor:
    object_asset = env.scene[object_cfg.name]
    current_height = object_asset.data.root_pos_w[:, 2]

    cache_key = f"_lifted_init_height_{object_cfg.name}"
    if not hasattr(env, cache_key):
        setattr(env, cache_key, current_height.clone())
    init_height = getattr(env, cache_key)
    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        init_height[just_reset] = current_height[just_reset].clone()

    lifted_height = (current_height - init_height).clamp(min=0.0)

    # cmd 门控
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.ll_command[:, 3:6]
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w
    cmd_dist  = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)
    gate_close_cmd = (cmd_dist < cmd_proximity_gate).float()

    # 改为进步版：只奖励高度刷新历史最高
    height_key = f"_max_lifted_height_{object_cfg.name}"
    if height_key not in env.extras:
        env.extras[height_key] = torch.zeros(env.num_envs, device=env.device)
    max_height = env.extras[height_key]
    if just_reset.any():
        max_height[just_reset] = 0.0

    progress = (lifted_height - max_height).clamp(min=0.0)
    env.extras[height_key] = torch.maximum(max_height, lifted_height)

    return torch.tanh(sensitivity * progress) * gate_close_cmd


def ee_orientation_to_object_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,          # 通常是 arm_link6
    sensitivity: float = 50.0,             # 进步敏感度
    dist_gate: float = 0.01,               # 距离门控，小于此值不计算奖励
    # 可选 cmd 门控（与 object_is_lifted_progress 风格一致，默认关闭）
    use_cmd_gate: bool = False,
    action_term_name: Optional[str] = None,
    cmd_proximity_gate: float = 0.2,
) -> torch.Tensor:
    """
    步进版 EE z轴方向对准物体奖励。
    只奖励余弦相似度相比历史最高值的进步，并使用 tanh 压缩。
    """
    robot_asset = env.scene[ee_frame_cfg.name]
    object_asset = env.scene[object_cfg.name]

    # ---------- 位置 / 姿态 ----------
    if ee_frame_cfg.body_ids is not None:
        ee_pos_w = robot_asset.data.body_pos_w[:, ee_frame_cfg.body_ids[0], :]   # (N,3)
        ee_quat_w = robot_asset.data.body_quat_w[:, ee_frame_cfg.body_ids[0], :] # (N,4)
    else:
        ee_pos_w = robot_asset.data.body_pos_w[:, -1, :]
        ee_quat_w = robot_asset.data.body_quat_w[:, -1, :]

    obj_pos_w = object_asset.data.root_pos_w[:, :3]                              # (N,3)

    # ---------- EE z轴世界方向 ----------
    z_axis_local = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis_local[:, 2] = 1.0                                                    # 局部 [0,0,1]
    ee_z_dir_world = quat_apply(ee_quat_w, z_axis_local)                        # (N,3)

    # ---------- 指向物体的方向 ----------
    obj_dir = obj_pos_w - ee_pos_w                                               # (N,3)
    obj_dist = torch.norm(obj_dir, dim=-1)                                       # (N,)

    # 当前余弦相似度（仅远距离有效）
    cos_sim = torch.zeros(env.num_envs, device=env.device)
    far_mask = obj_dist >= dist_gate
    if far_mask.any():
        obj_dir_unit = obj_dir[far_mask] / obj_dist[far_mask].unsqueeze(-1)
        cos_sim[far_mask] = F.cosine_similarity(
            ee_z_dir_world[far_mask], obj_dir_unit, dim=-1
        )  # ∈ [-1, 1]

    # ---------- 步进奖励缓存 ----------
    cache_key = f"_best_cosine_{ee_frame_cfg.name}_{object_cfg.name}"
    if not hasattr(env, cache_key):
        setattr(env, cache_key, torch.full((env.num_envs,), -1.0, device=env.device))
    best_cos = getattr(env, cache_key)

    # 重置逻辑：新 episode 时将 best_cos 重置为当前余弦值（首次进步为0）
    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        # 注意：重置时只有远距离的环境才用当前余弦，近距离的用 -1（避免负进步）
        reset_val = torch.where(far_mask[just_reset], cos_sim[just_reset], 
                                torch.full_like(cos_sim[just_reset], -1.0))
        best_cos[just_reset] = reset_val

    # ---------- 可选 cmd 门控 ----------
    gate = torch.ones(env.num_envs, device=env.device)
    if use_cmd_gate:
        if action_term_name is None:
            raise ValueError("Must provide action_term_name when use_cmd_gate=True")
        action_term = env.action_manager.get_term(action_term_name)
        cmd_pos_w = action_term.ll_command[:, 3:6]          # 假设 cmd 在动作的后三维
        cmd_dist = torch.norm(cmd_pos_w - obj_pos_w, dim=-1)
        gate = (cmd_dist < cmd_proximity_gate).float()

    # ---------- 计算进步奖励 ----------
    progress = (cos_sim - best_cos).clamp(min=0.0)          # 只奖励正向进步
    reward = torch.tanh(sensitivity * progress) * gate

    # 更新历史最佳余弦（仅更新远距离的环境，近距离保持不变）
    if far_mask.any():
        best_cos[far_mask] = torch.maximum(best_cos[far_mask], cos_sim[far_mask])

    return reward