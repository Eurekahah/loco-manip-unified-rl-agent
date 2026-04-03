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

def lateral_velocity_penalty(env, robot_cfg):
    # 惩罚 v_y 分量
    
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_y = robot.data.root_lin_vel_b[:, 1]
    return torch.abs(lin_vel_y)  # v_y 绝对值

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

    if reward.sum() > 0:
        print(f"[{target_body_name}] object contact max force: {max_force[is_contact]}")

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

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    物体被抬起的奖励。
    
    物体高于初始放置高度（spawn height）+ minimal_height 时给予奖励。
    奖励为连续值：超出越多奖励越高（clamp 在 [0, 1] 内），
    避免纯稀疏奖励带来的训练困难。

    Args:
        env: RL 环境实例
        minimal_height: 物体需要被抬起的最小高度（相对于初始高度），单位：米
        object_cfg: 物体的 SceneEntityCfg

    Returns:
        shape (num_envs,) 的奖励张量，值域 [0, 1]
    """
    object_asset = env.scene[object_cfg.name]

    # 当前物体 z 轴高度
    current_height = object_asset.data.root_pos_w[:, 2]  # (N,)

    # 初始高度（reset 时记录，存储在 extras 或直接用 default_root_state）
    # IsaacLab 中 default_root_state 的第 2 列是初始 z
    initial_height = object_asset.data.default_root_state[:, 2]  # (N,)

    # 相对抬起高度
    lifted_height = current_height - initial_height  # (N,)

    # 连续奖励：超过 minimal_height 才开始给分，线性增长后 clamp
    # 你也可以换成纯 bool：(lifted_height > minimal_height).float()
    # if lifted_height.sum() > 1e-4:
    #     print(f"Current height: {current_height}, Initial height: {initial_height}, Lifted height: {lifted_height}")
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
    
    if use_shaped:
        # ✅ 方案A：线性 + Gaussian 混合
        # 远处线性引导（始终有梯度），近处Gaussian精确奖励
        linear_reward   = 1.0 / (1.0 + pos_dist)                               # 始终有信号
        gaussian_reward = torch.exp(-pos_dist.pow(2) / (2 * pos_sigma ** 2))    # 近处精确
        reward = 0.3 * linear_reward + 0.7 * gaussian_reward
    else:
        # 原始Gaussian（梯度消失，不推荐）
        reward = torch.exp(-pos_dist.pow(2) / (2 * pos_sigma ** 2))
    
    return reward