# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_rotate_inverse,
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

def reward_delta_scale(env: ManagerBasedRLEnv,
                       action_name: str = "pre_trained_nav_action",
                       object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                       d_max: float = 0.5,
                       ) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    delta_scale = action_term.raw_actions[:,10]  # (N, action_dim)
    target_pos_w = action_term.raw_actions[:, 3:6]  # (N, 3)
    object_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]

    dist = torch.norm(object_pos - target_pos_w, dim=-1)

    target_scale = torch.clamp(dist / d_max, 0.0, 1.0)

    reward = -torch.nn.functional.smooth_l1_loss(
        delta_scale, target_scale, reduction="none"
    )

    reward -= 0.01 * delta_scale

    return reward

def gripper_state_stage_reward(
    env: ManagerBasedRLEnv,
    gripper_action_term_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="arm_link6"),
    approach_dist: float = 0.15,
    grasp_dist: float = 0.15,
) -> torch.Tensor:
    robot_asset = env.scene[ee_frame_cfg.name]

    # ── EE 位置：处理 body_ids 为 slice / list / None 三种情况 ──
    body_ids = ee_frame_cfg.body_ids
    if body_ids is None:
        ee_pos_w = robot_asset.data.body_pos_w[:, -1, :]
    elif isinstance(body_ids, slice):
        # slice → 取解析后的第一个真实索引
        all_ids = list(range(robot_asset.data.body_pos_w.shape[1]))
        resolved = all_ids[body_ids]
        ee_pos_w = robot_asset.data.body_pos_w[:, resolved[0], :]
    else:
        # list / tuple
        ee_pos_w = robot_asset.data.body_pos_w[:, body_ids[0], :]

    # ── 物体位置 ──────────────────────────────────────────────────
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    # ── 夹爪指令 ──────────────────────────────────────────────────
    action_term = env.action_manager.get_term(gripper_action_term_name)
    gripper_cmd = action_term.raw_actions[:, 0]
    gripper_closed = (gripper_cmd > 0.5).float()
    gripper_open   = 1.0 - gripper_closed

    print(f"gripper_cmd: {gripper_cmd}")
    # ── 距离 & 阶段奖励 ───────────────────────────────────────────
    dist = torch.norm(ee_pos_w - obj_pos_w, dim=-1)

    should_open  = (dist > approach_dist).float()
    should_close = (dist < grasp_dist).float()

    print(f"should_close: {should_close}")
    print(f"should_open: {should_open}")

    reward  =  should_open  * gripper_open   * 0.2
    reward += should_close  * gripper_closed * 0.5
    penalty =  should_open  * gripper_closed * (-0.2)

    return reward + penalty

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

def forward_velocity_penalty(env: ManagerBasedRLEnv, action_name: str = "pre_trained_nav_action") -> torch.Tensor:
    """惩罚前向速度命令 v_x（raw_actions[:, 0]）"""
    action_term = env.action_manager.get_term(action_name)
    vx = action_term.raw_actions[:, 0]  # v_x
    return vx.pow(2)

def lateral_velocity_penalty(env: ManagerBasedRLEnv, action_name: str = "pre_trained_nav_action") -> torch.Tensor:
    """惩罚侧向速度命令 v_y（raw_actions[:, 1]）"""
    action_term = env.action_manager.get_term(action_name)
    vy = action_term.raw_actions[:, 1]  # v_y
    return vy.pow(2)


def angular_velocity_penalty(env: ManagerBasedRLEnv, action_name: str = "pre_trained_nav_action") -> torch.Tensor:
    """惩罚角速度命令 omega_z（raw_actions[:, 2]）"""
    action_term = env.action_manager.get_term(action_name)
    wz = action_term.raw_actions[:, 2]  # omega_z
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
    min_finger_dist: float = 0.02,          # 新增：夹爪最小张开距离（米）
    gripper_action_name: str = "gripper_action",
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

    # ---- 综合门控 ----
    return gate_contact * reward_symmetry * is_close_cmd * gate_open

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
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

    reward = torch.clamp(
        (lifted_height - minimal_height) / minimal_height,
        min=0.0,
        max=1.0,
    )
    return reward

def cmd_pos_to_object_reward(
    env: ManagerBasedRLEnv,
    action_term_name: str = "pre_trained_pick_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_sigma: float = 0.1,
    use_shaped: bool = True,
) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_term_name)
    cmd_pos_w = action_term.raw_actions[:, 3:6]
    
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

