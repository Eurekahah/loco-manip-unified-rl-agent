# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from .utils import compute_base_height_rel_to_feet


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        # 小线速度指令设置为0
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.0).unsqueeze(1)
    
    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["end_error_lin_vel"] = torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        self.metrics["end_error_ang_vel"] = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])

@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

def cart2sphere(xyz: torch.Tensor) -> torch.Tensor:
    """
    笛卡尔坐标 -> 球坐标

    Args:
        xyz: shape (num_envs, 3)，每行为 [x, y, z]

    Returns:
        lpy: shape (num_envs, 3)，每行为 [l, pitch, yaw]
            l     - 径向距离 r，范围 [0, +∞)
            pitch - 仰角（elevation），从 XY 平面向上为正，范围 [-π/2, π/2]
            yaw   - 方位角（azimuth），从 X 轴沿 Y 轴方向，范围 (-π, π]
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    l = torch.norm(xyz, dim=-1)                  # 径向距离
    pitch = torch.asin(torch.clamp(z / (l + 1e-8), -1.0, 1.0))  # 仰角
    yaw = torch.atan2(y, x)                      # 方位角

    return torch.stack([l, pitch, yaw], dim=-1)


def sphere2cart(lpy: torch.Tensor) -> torch.Tensor:
    """
    球坐标 -> 笛卡尔坐标

    Args:
        lpy: shape (num_envs, 3)，每行为 [l, pitch, yaw]
            l     - 径向距离
            pitch - 仰角，范围 [-π/2, π/2]
            yaw   - 方位角，范围 (-π, π]

    Returns:
        xyz: shape (num_envs, 3)，每行为 [x, y, z]
    """
    l, pitch, yaw = lpy[:, 0], lpy[:, 1], lpy[:, 2]

    cos_pitch = torch.cos(pitch)
    x = l * cos_pitch * torch.cos(yaw)
    y = l * cos_pitch * torch.sin(yaw)
    z = l * torch.sin(pitch)

    return torch.stack([x, y, z], dim=-1)

from dataclasses import dataclass, field

class HeightInvariantEECommand(mdp.UniformPoseCommand):
    
    cfg: HeightInvariantEECommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        num_envs = env.num_envs
        device = env.device

        self.pose_end_cart = torch.zeros(num_envs, 7, device=device)
        self.pose_command_b = torch.zeros(num_envs, 7, device=device)

        self.ee_start_pos_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_pos_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_pos_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)

        # 碰撞盒 limits 转为 tensor，移到对应设备
        self.collision_lower_limits = torch.tensor(
            cfg.collision_lower_limits, dtype=torch.float32, device=self.device
        )  # (3,) 或 (K, 3)
        self.collision_upper_limits = torch.tensor(
            cfg.collision_upper_limits, dtype=torch.float32, device=self.device
        )

        self.underground_limit = cfg.underground_limit

        # 路径插值采样点 t ∈ [0, 1]，shape: (T,)
        self.collision_check_t = torch.linspace(
            0.0, 1.0, cfg.num_collision_check_samples, device=self.device
        )

        self.num_collision_check_samples = cfg.num_collision_check_samples
        self.max_resample_attempts = cfg.max_resample_attempts
        self.arm_base_link_idx = env.scene["robot"].data.body_names.index(self.cfg.arm_base_link_name)


    def _resample_command(self, env_ids):
        # 1. 获取当前 height-invariant 坐标系
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, env_ids)

        # 2. 球坐标采样新目标
        self._resample_ee_goal(env_ids)

        # 3. 转换到当前 root 坐标系
        pos_local = self.pose_end_cart[env_ids, :3]
        quat_local = self.pose_end_cart[env_ids, 3:]

        pos_world = math_utils.quat_apply(quat_yaw, pos_local) + origin_pos
        quat_world = math_utils.quat_mul(quat_yaw, quat_local)

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, pos_world, quat_world,
        )
        self.pose_command_b[env_ids] = torch.cat([target_pos_b, target_quat_b], dim=-1)
        
    
    def _update_command(self):
        pass

    def collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        检查 ee_start_pos_sphere → ee_end_pos_sphere 路径是否与 AABB 碰撞盒或地面相交。
        返回: collision_mask (len(env_ids),)  True = 碰撞，需要重采样
        """
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, env_ids)

        # 球坐标插值（在球坐标空间做 lerp）
        # ee_start_pos_sphere / ee_end_pos_sphere shape: (N, 3)，假设为 (r, θ, φ)
        t = self.collision_check_t  # (T,)
        
        # 球坐标空间线性插值: (1, N, 3) + (T, 1, 1) * delta → (T, N, 3)
        sphere_start = self.ee_start_pos_sphere[env_ids]  # (N, 3)
        sphere_end   = self.ee_end_pos_sphere[env_ids]    # (N, 3)
        
        path_sphere = sphere_start.unsqueeze(0) + t[:, None, None] * (
            sphere_end - sphere_start
        ).unsqueeze(0)  # (T, N, 3)

        # 转换到笛卡尔坐标: 先 reshape 成 (T*N, 3)，转换后再 reshape 回 (T, N, 3)
        T, N, _ = path_sphere.shape
        path_cart_local = sphere2cart(path_sphere.reshape(T * N, 3)).reshape(T, N, 3)  # (T, N, 3)

        # local_to_world 期望输入 (N, 3)，这里批量处理 (T*N, 3)
        origin_pos_rep = origin_pos.unsqueeze(0).expand(T, N, 3).reshape(T * N, 3)
        quat_yaw_rep   = quat_yaw.unsqueeze(0).expand(T, N, 4).reshape(T * N, 4)
        
        path_pts = (
            math_utils.quat_apply(quat_yaw_rep, path_cart_local.reshape(T * N, 3))
            + origin_pos_rep
        ).reshape(T, N, 3)  # (T, N, 3)

        # ── AABB 碰撞检测 ──────────────────────────────────────────
        upper = self.collision_upper_limits
        lower = self.collision_lower_limits

        if upper.dim() == 1:
            in_box = torch.logical_and(
                torch.all(path_pts < upper, dim=-1),
                torch.all(path_pts > lower, dim=-1),
            )  # (T, N)
            collision_mask = torch.any(in_box, dim=0)  # (N,)
        else:
            pts = path_pts.unsqueeze(2)  # (T, N, 1, 3)
            in_box = torch.logical_and(
                torch.all(pts < upper, dim=-1),
                torch.all(pts > lower, dim=-1),
            )  # (T, N, K)
            collision_mask = torch.any(
                in_box.reshape(T, N, -1), dim=(0, 2)
            )  # (N,)

        # ── 地下检测 ──────────────────────────────────────────────
        underground_mask = torch.any(
            path_pts[..., 2] < self.underground_limit, dim=0
        )  # (N,)

        return collision_mask | underground_mask

    
    def get_height_invariant_base_frame(self, env: ManagerBasedEnv, env_ids):
        """
        构造 height-roll-pitch-invariant 坐标系：
        - 原点：base 在世界坐标系中的 XY 位置，Z 固定为某个参考平面（如地面 + 固定偏移）
        - 朝向：只保留 yaw，去除 roll 和 pitch
        """
        # 获取机械臂arm_base的位置 
        arm_base_link_pos_w = env.scene["robot"].data.body_pos_w[env_ids, self.arm_base_link_idx]
        # 获取基座的姿态
        base_quat_w = env.scene["robot"].data.root_quat_w[env_ids]  # (N, 4) wxyz

        # 提取 yaw 角（去除 roll/pitch）
        _, _, yaw = math_utils.euler_xyz_from_quat(base_quat_w)  # (N,)
        
        # 重新构造只含 yaw 的四元数
        zeros = torch.zeros_like(yaw)
        quat_yaw_only = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)  # (N, 4)

        # 原点：base 的 XY，Z 固定（arm base 的固定高度偏移）
        # 论文中是 arm base 沿 Z 轴的固定平面
        origin_pos = arm_base_link_pos_w.clone()
        origin_pos[..., 2] = self.cfg.sampled_height  # FIXED_ARM_BASE_HEIGHT  # 固定参考高度
        return origin_pos, quat_yaw_only
    
    def _resample_ee_goal_orn(self, env_ids):
        """
        在锥形范围内采样 EE 目标姿态：
        - 锥轴对齐到目标位置向量方向（sphere2cart(ee_end_pos_sphere)）；
        - 锥张角（pitch/yaw 范围）随径向距离 p_l 增大而收窄；
        - roll（绕锥轴自转）不受距离影响，仍在 o_roll 范围内均匀采样。
        """
        n = len(env_ids)

        # ---- 1. 根据径向距离 p_l 计算锥角收缩系数 ----
        p_l = self.ee_end_pos_sphere[env_ids, 0]  # (N,) 径向距离，此时位置应已采样完毕
        l_min, l_max = self.cfg.ranges.p_l
        frac = ((p_l - l_min) / max(l_max - l_min, 1e-6)).clamp(0.0, 1.0)  # 距离越远 frac 越大

        # 近处 scale=1（用满 o_pitch/o_yaw 范围），远处收缩到 min_scale
        min_scale = getattr(self.cfg.ranges, "orn_cone_min_scale", 0.1)
        scale = 1.0 - frac * (1.0 - min_scale)  # (N,)

        def _sample_scaled(rng, scale):
            lo, hi = rng  # 假设范围以 0 为中心（如 (-a, a)），直接按 scale 缩放上下界
            lo_s, hi_s = lo * scale, hi * scale
            u = torch.rand(n, device=self.device)
            return lo_s + u * (hi_s - lo_s)

        r_roll  = _sample_scaled(self.cfg.ranges.o_roll, scale)
        r_pitch = _sample_scaled(self.cfg.ranges.o_pitch, scale)
        # yaw（绕最终 z 轴自转）不影响 z 轴指向，不缩放
        r_yaw   = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.o_yaw)


        quat_local = math_utils.quat_from_euler_xyz(r_roll, r_pitch, r_yaw)  # (N,4) 锥内局部偏转

        # ---- 2. 计算锥轴对齐旋转：把默认参考方向对齐到位置向量方向 ----
        pos_dir = math_utils.normalize(sphere2cart(self.ee_end_pos_sphere[env_ids]))  # (N,3)
        ref_axis = torch.zeros_like(pos_dir)
        ref_axis[:, 2] = 1.0  # 与原实现里 roll=pitch=yaw=0 时的默认朝向保持一致（局部 z 轴）

        q_align = self._quat_from_two_vectors(ref_axis, pos_dir)  # (N,4) wxyz

        # ---- 3. 组合：先做锥内局部偏转，再整体对齐到位置方向 ----
        self.ee_end_orn_quat[env_ids] = math_utils.quat_mul(q_align, quat_local)

    def _quat_from_two_vectors(self, v_from: torch.Tensor, v_to: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """计算把 v_from 旋转到 v_to 的最短路径四元数 (wxyz)，v_from/v_to 形状为 (N,3)。"""
        v_from = math_utils.normalize(v_from)
        v_to = math_utils.normalize(v_to)
        dot = (v_from * v_to).sum(-1, keepdim=True)          # (N,1)
        cross = torch.cross(v_from, v_to, dim=-1)             # (N,3)

        quat = torch.cat([1.0 + dot, cross], dim=-1)          # (N,4) 未归一化

        # 处理 v_from 与 v_to 几乎完全反向的退化情况（叉积接近零）
        parallel_neg = dot.squeeze(-1) < (-1.0 + eps)
        if parallel_neg.any():
            idx = parallel_neg.nonzero(as_tuple=True)[0]
            v = v_from[idx]
            aux = torch.zeros_like(v)
            aux[:, 0] = 1.0
            collinear = v[:, 0].abs() > 0.9
            aux[collinear] = torch.tensor([0.0, 1.0, 0.0], device=v.device)
            ortho = math_utils.normalize(torch.cross(v, aux, dim=-1))
            quat[idx] = torch.cat([torch.zeros(len(idx), 1, device=v.device), ortho], dim=-1)

        return math_utils.normalize(quat)

    def _resample_ee_goal_sphere(self, env_ids):
        """
        在球坐标系中均匀采样末端执行器目标
        """
        # ---------- 位置（球坐标系） ----------
        n = len(env_ids)
        self.ee_end_pos_sphere[env_ids] = torch.stack([
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_l),
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_pitch),
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_yaw),
        ], dim=-1)  # (N, 3)
    
    def _resample_ee_goal(self, env_ids):
        init_env_ids = env_ids.clone()
        # 起点优先使用上一轮的终点球坐标，保证插值命令连续且起点远离当前 EE。
        # 仅当上一轮终点为零向量时（首次 reset）才回退到真实 EE 位置。
        prev_end_valid = self.ee_end_pos_sphere[env_ids].norm(dim=-1) > 0  # (N,)
        if prev_end_valid.all():
            self.ee_start_pos_sphere[env_ids] = self.ee_end_pos_sphere[env_ids].clone()
        else:
            # 部分环境是首次 reset，分别处理
            ids_valid   = env_ids[prev_end_valid]
            ids_invalid = env_ids[~prev_end_valid]

            if len(ids_valid) > 0:
                self.ee_start_pos_sphere[ids_valid] = self.ee_end_pos_sphere[ids_valid].clone()

            if len(ids_invalid) > 0:
                ee_pos_w = self.robot.data.body_pos_w[ids_invalid, self.body_idx]
                origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, ids_invalid)
                quat_yaw_inv = math_utils.quat_conjugate(quat_yaw)
                ee_pos_local = math_utils.quat_apply(quat_yaw_inv, ee_pos_w - origin_pos)
                self.ee_start_pos_sphere[ids_invalid] = cart2sphere(ee_pos_local)

        for i in range(self.cfg.max_resample_attempts):
            self._resample_ee_goal_sphere(env_ids)
            collision_mask = self.collision_check(env_ids)
            env_ids = env_ids[collision_mask]
            if len(env_ids) == 0:
                break
        self._resample_ee_goal_orn(init_env_ids)
        self.ee_end_pos_cart[init_env_ids,:] = sphere2cart(self.ee_end_pos_sphere[init_env_ids,:])
        self.pose_end_cart[init_env_ids] = torch.cat(
            [self.ee_end_pos_cart[init_env_ids], self.ee_end_orn_quat[init_env_ids]], dim=-1
        )  # (N, 7)
    
    @property
    def command(self) -> torch.Tensor:
        """返回当前命令（实现抽象属性方法）"""
        return self.pose_command_b
    
    @property
    def command_local(self) -> torch.Tensor:
        """height-invariant local 坐标，给 obs 用"""
        return self.pose_command_b
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "sample_frame_visualizer"):
                # -- goal pose
                self.sample_frame_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                
            # set their visibility to true
            self.sample_frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "sample_frame_visualizer"):
                self.sample_frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, torch.arange(self._env.num_envs))
        self.sample_frame_visualizer.visualize(origin_pos,quat_yaw)
        
    
    

@configclass
class HeightInvariantEECommandCfg(mdp.UniformPoseCommandCfg):

    class_type: type = HeightInvariantEECommand

    sampled_height: float = MISSING

    arm_base_link_name: str = MISSING

    @configclass
    class Ranges:
        # 球坐标采样位置范围
        p_l: tuple[float, float] = MISSING       # position_半径 l

        p_pitch: tuple[float, float] = MISSING   # position_pitch p

        p_yaw: tuple[float, float] = MISSING     # position_yaw y

        # 姿态rpy采样范围
        o_roll: tuple[float,float] = MISSING     # orientation_roll

        o_pitch: tuple[float,float] = MISSING    # orientation_pitch

        o_yaw: tuple[float,float] = MISSING      # orientation_yaw

        orn_cone_min_scale: float = MISSING      # 锥桶最小收缩范围

    ranges: Ranges = MISSING

    collision_lower_limits: list = field(default_factory=lambda: [-0.3, -0.3, 0.0])
    collision_upper_limits: list = field(default_factory=lambda: [ 0.3,  0.3, 0.5])
    underground_limit: float = 0.05          # EE z 低于此值视为穿地
    num_collision_check_samples: int = 10    # 路径插值采样点数
    max_resample_attempts: int = 10          # 最大重采样次数
   
    

class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class ArmWeightCommand(CommandTerm):
    cfg: ArmWeightCommandCfg

    def __init__(self, cfg: ArmWeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._weight = torch.ones(env.num_envs, 1, device=env.device)
        self._max_weight: float = cfg.init_max_weight
        self._min_weight: float = 0.0

    # ------------------------------------------------------------------
    # 课程接口
    # ------------------------------------------------------------------
    def set_max_weight(self, value: float):
        self._max_weight = float(torch.tensor(value).clamp(0.0, 1.0))
        # min 不能超过 max
        self._min_weight = min(self._min_weight, self._max_weight)

    def set_min_weight(self, value: float):
        self._min_weight = float(torch.tensor(value).clamp(0.0, self._max_weight))

    def get_max_weight(self) -> float:
        return self._max_weight

    def get_min_weight(self) -> float:
        return self._min_weight

    # ------------------------------------------------------------------
    # CommandTerm 接口
    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self._weight

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: torch.Tensor):
        low  = self._min_weight
        high = max(self._max_weight, low + 1e-6)
        # 先生成新值，再写回
        new_vals = torch.empty(len(env_ids), device=self._weight.device).uniform_(low, high)
        self._weight[env_ids, 0] = new_vals


@configclass
class ArmWeightCommandCfg(CommandTermCfg):
    class_type: type = ArmWeightCommand

    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    init_max_weight: float = 0.0
    """训练开始时 max_weight 的初始值。"""



class BodyPoseCommand(CommandTerm):
    """生成机身目标 height / pitch / roll 命令。

    采样分布为截断正态分布：
      - 均值对应「直立」状态（height=nominal, pitch=0, roll=0）
      - 标准差控制「特殊动作」出现频率
      - 截断区间防止超出物理极限
    """

    cfg: BodyPoseCommandCfg

    def __init__(self, cfg: BodyPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # command buffer: [height, pitch, roll]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.cfg.asset_cfg.resolve(self._env.scene)
        self.cfg.feet_cfg.resolve(self._env.scene)

    @property
    def _low(self) -> torch.Tensor:
        return torch.tensor(
            [self.cfg.height_range[0], self.cfg.pitch_range[0], self.cfg.roll_range[0]],
            device=self.device,
        )

    @property
    def _high(self) -> torch.Tensor:
        return torch.tensor(
            [self.cfg.height_range[1], self.cfg.pitch_range[1], self.cfg.roll_range[1]],
            device=self.device,
        )
    # ------------------------------------------------------------------
    # 必须实现的抽象方法
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"BodyPoseCommand | envs={self.num_envs} | "
            f"height~N({self.cfg.height_mean},{self.cfg.height_std}) "
            f"pitch~N({self.cfg.pitch_mean},{self.cfg.pitch_std}) "
            f"roll~N({self.cfg.roll_mean},{self.cfg.roll_std})"
        )

    @property
    def command(self) -> torch.Tensor:
        """shape: (num_envs, 3)  — [height, pitch, roll]"""
        return self._command

    @command.setter
    def command(self, value: torch.Tensor):
        self._command = value

    def _resample_command(self, env_ids: torch.Tensor):
        """对指定 env 重新采样（截断正态分布）。"""
        n = len(env_ids)
        if n == 0:
            return

        # 先采无约束正态，再 clamp 到物理区间（即截断正态近似）
        samples = (
            torch.rand(n, 3, device=self.device) * (self._high - self._low) + self._low
        )
        self._command[env_ids] = samples

    def _update_command(self):
        """每步调用：此处不需要额外逻辑（命令在 resample 时更新）。"""
        pass

    def _update_metrics(self):
        """可选：记录命令统计量用于 tensorboard。"""
        # ------------------------------------------------------------------ #
        # 读取实际 body pose
        # ------------------------------------------------------------------ #
        # 实际高度：root 在世界系下的 z 坐标
        actual_height = compute_base_height_rel_to_feet(self._env, self.cfg.asset_cfg, self.cfg.feet_cfg)  # (num_envs,)

        # 实际 pitch / roll：从四元数转欧拉角
        # euler_xyz_from_quat 返回顺序为 (roll, pitch, yaw)，单位 rad
        actual_quat = self._env.scene["robot"].data.root_quat_w                       # (num_envs, 4) — w,x,y,z
        actual_roll, actual_pitch, _ = math_utils.euler_xyz_from_quat(actual_quat)  # (num_envs,) each

        # ------------------------------------------------------------------ #
        # 误差 = 命令 - 实际
        # ------------------------------------------------------------------ #
        height_error = self._command[:, 0] - actual_height
        pitch_error  = self._command[:, 1] - actual_pitch
        roll_error   = self._command[:, 2] - actual_roll

        # MAE：反映平均跟踪精度
        self.metrics["height_error_mean"] = height_error.abs()
        self.metrics["pitch_error_mean"]  = pitch_error.abs()
        self.metrics["roll_error_mean"]   = roll_error.abs()

        # 带符号误差均值：反映系统性偏高 / 偏低趋势
        self.metrics["height_error_bias"] = height_error
        self.metrics["pitch_error_bias"]  = pitch_error
        self.metrics["roll_error_bias"]   = roll_error

    def _set_debug_vis_impl(self, debug_vis: bool):
        """创建 / 销毁可视化 marker。"""
        if debug_vis:
            # ① 倾斜圆片（绿色）：编码目标 pitch / roll
            if not hasattr(self, "_tilted_disc_visualizer"):
                tilted_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/BodyPoseCommand/body_pose_disc_tilted",
                )
                self._tilted_disc_visualizer = VisualizationMarkers(tilted_cfg)
            self._tilted_disc_visualizer.set_visibility(True)

            # ② 水平参考圆片（蓝色）：始终水平，仅表示高度
            if not hasattr(self, "_ref_disc_visualizer"):
                ref_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/BodyPoseCommand/body_pose_disc_ref",
                )
                self._ref_disc_visualizer = VisualizationMarkers(ref_cfg)
            self._ref_disc_visualizer.set_visibility(True)
        else:
            if self._tilted_disc_visualizer is not None:
                self._tilted_disc_visualizer.set_visibility(False)
            if self._ref_disc_visualizer is not None:
                self._ref_disc_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """每帧更新 marker 位置和姿态。"""
        if not self.cfg.debug_vis:
            return

        # 获取机器人基座在世界坐标系下的 XY 位置
        root_state = self._env.scene["robot"].data.root_state_w  # (N, 13)
        base_xy = root_state[:, :2]  # (N, 2)
        base_quat = root_state[:, 3:7]       # (N, 4) w, x, y, z

        target_h = self._command[:, 0]  # (N,)
        target_p = self._command[:, 1]  # (N,)  pitch
        target_r = self._command[:, 2]  # (N,)  roll

        zeros = torch.zeros(self.num_envs, device=self.device)

        ref_ground_h = self._env.scene["robot"].data.root_pos_w[:, 2] - compute_base_height_rel_to_feet(self._env, self.cfg.asset_cfg, self.cfg.feet_cfg)


        # ---------- 公共：圆片位置（两个圆片同位置）----------
        disc_pos = torch.zeros(self.num_envs, 3, device=self.device)
        disc_pos[:, :2] = base_xy
        disc_pos[:, 2] = target_h + ref_ground_h

        # ---------- 公共：基础旋转（法线朝上，圆片水平）----------
        half_pi = torch.full((self.num_envs,), -torch.pi / 2, device=self.device)
        q_base = math_utils.quat_from_euler_xyz(zeros, half_pi, zeros)  # (N, 4)

        # ---------- 公共：圆片尺寸 ----------
        disc_scale = torch.ones(self.num_envs, 3, device=self.device)
        disc_scale[:, 0] = 0.01   # 法线方向极薄
        disc_scale[:, 1] = 10.0   # 直径 Y
        disc_scale[:, 2] = 1.0    # 直径 Z

        # ---------- ① 倾斜圆片（绿色）：叠加机器人完整姿态 + 本体命令 ----------
        # 1. 从基座四元数提取欧拉角，得到每个机器人的偏航角 yaw
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat)  # 每个 (N,)
        # 注意：返回的 roll/pitch 是机器人当前的实际倾斜，但我们不需要它们，只需要 yaw

        # 2. 机器人偏航的旋转四元数 (绕世界 Z 轴旋转，将本体命令转到世界)
        q_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)   # (N, 4)

        # 3. 本体命令的四元数 (roll, pitch, yaw=0)
        q_body_cmd = math_utils.quat_from_euler_xyz(target_r, target_p, zeros)  # (N, 4)

        # 4. 组合：先本体命令，再旋转到世界坐标系 -> q_world = q_yaw * q_body_cmd
        q_target_world = math_utils.quat_mul(q_yaw, q_body_cmd)      # (N, 4)

        # 5. 最终圆片旋转：先变成水平 (q_base)，再转到目标世界姿态
        tilted_quat = math_utils.quat_mul(q_target_world, q_base)    # (N, 4)

        self._tilted_disc_visualizer.visualize(disc_pos, tilted_quat, disc_scale)

        # ---------- ② 水平参考圆片（蓝色）：始终保持水平，仅反映高度 ----------
        self._ref_disc_visualizer.visualize(disc_pos, q_base, disc_scale)


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    """BodyPoseCommand 的配置。"""

    class_type: type = BodyPoseCommand

    # ---- height（机身高度，单位 m）----
    # 正常站立高度约 0.513 m，蹲下约 0.35 m
    height_range: tuple = (0.33, 0.65) # 截断区间 [最低蹲伏, 最高]

    # ---- pitch（前后俯仰，单位 rad）----
    # 正值 = 机头抬起，负值 = 俯身
    pitch_range: tuple = (-0.35, 0.35) # ±20°

    # ---- roll（左右侧倾，单位 rad）----
    roll_range:  tuple = (-0.25, 0.25) # ±14°

    # ---- 继承自 CommandTermCfg ----
    resampling_time_range: tuple = (10.0, 10.0)  # 每 5~10 s 重采样一次
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*wheel")
    debug_vis: bool = False