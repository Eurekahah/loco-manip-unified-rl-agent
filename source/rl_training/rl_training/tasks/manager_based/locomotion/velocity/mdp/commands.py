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

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        # 小线速度指令设置为0
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


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

        self.t = torch.zeros(num_envs, device=device)
        self.pose_end_local = torch.zeros(num_envs, 7, device=device)
        self.pose_end_w = torch.zeros(num_envs, 7, device=device)
        self.pose_command_w = torch.zeros(num_envs, 7, device=device)
        self.pose_start_w = torch.zeros(num_envs,7,device=device)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_end_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)

        self.T_traj = torch.ones(num_envs, device=device)

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


    def _resample_command(self, env_ids):
        # 1. 获取当前 height-invariant 坐标系
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, env_ids)
        
        # 2. 球坐标采样新目标
        self._resample_ee_goal(env_ids)
        
        # 3. 转换到世界坐标
        self.pose_end_w[env_ids] = self.local_to_world(self.pose_end_local[env_ids], origin_pos, quat_yaw)
        
        # 4. 重置插值计时
        self.t[env_ids] = 0.0
        self.T_traj[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.T_traj)
        self.pose_start_w[env_ids, :3] = self.robot.data.body_pos_w[env_ids, self.body_idx]
        self.pose_start_w[env_ids, 3:] = self.robot.data.body_quat_w[env_ids, self.body_idx]


    def _update_command(self):
        # 每步更新插值目标
        alpha = (self.t / self.T_traj).clamp(0, 1).unsqueeze(-1).expand_as(self.ee_start_sphere)  # (N, 3)

        # 获取当前 EE 位姿（世界坐标）
        ee_pos_w  = self.robot.data.body_pos_w[:, self.body_idx]   # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]  # (N, 4)

        # 获取当前坐标系（实时跟随 yaw）
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(
            self._env, torch.arange(self._env.num_envs)
        )
        # p_end_w = self.local_to_world(self.pose_end_local, origin_pos, quat_yaw)  # (N, 7)

        # 对球坐标进行插值后转换到笛卡尔坐标
        pos_interp_local = sphere2cart(torch.lerp(self.ee_start_sphere,self.ee_end_sphere,alpha))
        # 位置：线性插值
        # pos_interp = (1 - alpha) * self.pose_start_w[:,:3] + alpha * self.pose_end_w[..., :3]  # (N, 3)
        # self.pose_command_w = torch.cat([pos_interp, quat_interp], dim=-1)  # (N, 7)
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, torch.arange(self._env.num_envs))
        pos_interp_w = math_utils.quat_apply(quat_yaw, pos_interp_local) + origin_pos  # (N, 3)
        self.pose_command_w = torch.cat([pos_interp_w, self.pose_end_w[..., 3:]], dim=-1)

        self.t += self._env.step_dt

        # 检查是否需要重采样（到达终点或自碰撞）
        done_mask = self.t >= self.T_traj
        if done_mask.any():
            self._resample(done_mask.nonzero(as_tuple=False).flatten())

    def _update_metrics(self):

        pos_error, rot_error = math_utils.compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],   # 标量idx -> (N, 3)
            self.robot.data.body_quat_w[:, self.body_idx],  # 标量idx -> (N, 4)
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        检查 pose_start_w → pose_end_w 路径是否与 AABB 碰撞盒或地面相交。
        返回: collision_mask (len(env_ids),)  True = 碰撞，需要重采样
        """
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, env_ids)

        # 球坐标插值（在球坐标空间做 lerp）
        # ee_start_sphere / ee_end_sphere shape: (N, 3)，假设为 (r, θ, φ)
        t = self.collision_check_t  # (T,)
        
        # 球坐标空间线性插值: (1, N, 3) + (T, 1, 1) * delta → (T, N, 3)
        sphere_start = self.ee_start_sphere[env_ids]  # (N, 3)
        sphere_end   = self.ee_end_sphere[env_ids]    # (N, 3)
        
        path_sphere = sphere_start.unsqueeze(0) + t[:, None, None] * (
            sphere_end - sphere_start
        ).unsqueeze(0)  # (T, N, 3)

        # 转换到笛卡尔坐标: 先 reshape 成 (T*N, 3)，转换后再 reshape 回 (T, N, 3)
        T, N, _ = path_sphere.shape
        path_cart_local = sphere2cart(path_sphere.reshape(T * N, 3)).reshape(T, N, 3)  # (T, N, 3)

        # start 需要转到世界坐标（local → world），end 已是世界坐标
        # 对路径上所有点做 local → world 变换
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
        arm_base_link_idx = env.scene["robot"].data.body_names.index(self.cfg.arm_base_link_name)
        arm_base_link_pos_w = env.scene["robot"].data.body_pos_w[env_ids, arm_base_link_idx]
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
        SO(3)中均匀采样EE目标位姿
        """
        # ---------- 姿态（欧拉角） ----------
        n = len(env_ids)
        r_roll  = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.o_roll)
        r_pitch = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.o_pitch)
        r_yaw   = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.o_yaw)
        self.ee_end_orn_quat[env_ids] = math_utils.quat_from_euler_xyz(r_roll, r_pitch, r_yaw)  # (N, 4) wxyz

    def _resample_ee_goal_sphere(self, env_ids):
        """
        在球坐标系中均匀采样末端执行器目标
        """
        # ---------- 位置（球坐标系） ----------
        n = len(env_ids)
        self.ee_end_sphere[env_ids] = torch.stack([
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_l),
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_pitch),
            torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.p_yaw),
        ], dim=-1)  # (N, 3)


    
    def _resample_ee_goal(self, env_ids):
        init_env_ids = env_ids.clone()
        self._resample_ee_goal_orn(env_ids)
        # self.ee_start_sphere[env_ids] = self.ee_end_sphere[env_ids].clone() # 此处应该改为当前ee的球坐标位姿
        # 获取当前 EE 世界坐标
        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]  # (N, 3)

        # 转回 local 坐标系
        origin_pos, quat_yaw = self.get_height_invariant_base_frame(self._env, env_ids)
        quat_yaw_inv = math_utils.quat_conjugate(quat_yaw)
        ee_pos_local = math_utils.quat_apply(quat_yaw_inv, ee_pos_w - origin_pos)  # (N, 3)

        # 转为球坐标
        self.ee_start_sphere[env_ids] = cart2sphere(ee_pos_local)
        for i in range(self.cfg.max_resample_attempts):
            self._resample_ee_goal_sphere(env_ids)
            collision_mask = self.collision_check(env_ids)
            env_ids = env_ids[collision_mask]
            if len(env_ids) == 0:
                break
        
        self.ee_end_cart[init_env_ids,:] = sphere2cart(self.ee_end_sphere[init_env_ids,:])
        self.pose_end_local[init_env_ids] = torch.cat(
            [self.ee_end_cart[init_env_ids], self.ee_end_orn_quat[init_env_ids]], dim=-1
        )  # (N, 7)

    def local_to_world(self, pose_local, origin_pos, quat_yaw):
        """
        将 height-invariant 坐标系中的目标位姿 (N, 7) 转换到世界坐标系。

        pose_local: (N, 7)  前3位为位置，后4位为四元数 wxyz
        origin_pos: (N, 3)
        quat_yaw:   (N, 4)  仅含 yaw 的四元数
        """
        pos_local  = pose_local[..., :3]   # (N, 3)
        quat_local = pose_local[..., 3:]   # (N, 4)

        # 位置：先用 yaw 旋转，再加原点偏移
        pos_world = math_utils.quat_apply(quat_yaw, pos_local) + origin_pos  # (N, 3)

        # 姿态：world_quat = quat_yaw ⊗ quat_local
        quat_world = math_utils.quat_mul(quat_yaw, quat_local)  # (N, 4)

        pose_world = torch.cat([pos_world, quat_world], dim=-1)  # (N, 7)
        return pose_world
    
    @property
    def command(self) -> torch.Tensor:
        """返回当前命令（实现抽象属性方法）"""
        return self.pose_command_w
    
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
        # 在 env 或 task 的 _post_physics_step / _get_observations 中
        robot = self.robot

        # 获取所有关节的应用力矩
        applied_torques = robot.data.applied_torque  # shape: (num_envs, num_joints)

        # 找到6个关节的索引
        joint_indices, joint_names = robot.find_joints("arm_joint[1-6]")

        # 方式二：逐个打印，更清晰
        for idx, name in zip(joint_indices, joint_names):
            print(f"  {name}: {applied_torques[:, idx]}")
    
    

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

        T_traj: tuple[float, float] = MISSING    # 插值间隔采样范围

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
