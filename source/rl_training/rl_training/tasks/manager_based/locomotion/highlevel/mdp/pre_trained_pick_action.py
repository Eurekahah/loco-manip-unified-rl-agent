# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPickAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPickActionCfg
    """The configuration of the action term."""

    leg_joint_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]

    hipx_joint_names = [
        "fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint",
    ]

    hipy_joint_names = [
        "fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint",
    ]

    knee_joint_names = [
        "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
    ]

    arm_joint_names = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6",  
    ]

    gripper_joint_names = [
        "arm_joint7", "arm_joint8",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names

    def __init__(self, cfg: PreTrainedPickActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # 分别初始化三个 low level action term
        self._joint_pos_action_term: ActionTerm = cfg.low_level_leg_actions.class_type(
            cfg.low_level_leg_actions, env
        )
        self._wheel_vel_action_term: ActionTerm = cfg.low_level_wheel_actions.class_type(
            cfg.low_level_wheel_actions, env
        )
        self._ee_ik_action_term: ActionTerm = cfg.low_level_ee_actions.class_type(
            cfg.low_level_ee_actions, env
        )

        # 各自的动作维度
        self._joint_pos_dim = self._joint_pos_action_term.action_dim
        self._wheel_vel_dim = self._wheel_vel_action_term.action_dim
        self._ee_ik_dim = self._ee_ik_action_term.action_dim
        

        self.low_level_leg_actions = torch.zeros(
            self.num_envs, self._joint_pos_dim, device=self.device
        )
        self.low_level_wheel_actions = torch.zeros(
            self.num_envs, self._wheel_vel_dim, device=self.device
        )
        self.low_level_ee_actions = torch.zeros(
            self.num_envs, self._ee_ik_dim, device=self.device
        )

        self._joint_pos_action_term.scale = {".*_hipx_joint": 0.125, '^(?!.*_hipx_joint)(?!.*arm_joint).*': 0.25}
        self._wheel_vel_action_term.scale = 20.0
        self._joint_pos_action_term.clip = {".*": (-100.0, 100.0)}
        self._wheel_vel_action_term.clip = {".*": (-100.0, 100.0)}
        self._joint_pos_action_term.joint_names = self.leg_joint_names 
        self._wheel_vel_action_term.joint_names = self.wheel_joint_names

        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_mask = env.episode_length_buf == 0
                self.low_level_leg_actions[reset_mask, :] = 0
                self.low_level_wheel_actions[reset_mask, :] = 0
                self.low_level_ee_actions[reset_mask, :] = 0
                self._raw_actions[reset_mask, :] = 0
            # 拼接两个 action term 的输出，供 low-level obs 使用
            return torch.cat([self.low_level_leg_actions, self.low_level_wheel_actions, self.low_level_ee_actions], dim=-1)

        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()

        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions[:, :3]
        cfg.low_level_observations.velocity_commands.params = dict()

        cfg.low_level_observations.ee_goal.func = lambda dummy_env: self._raw_actions[:, 3:10]
        cfg.low_level_observations.ee_goal.params = dict()

        cfg.low_level_observations.joint_pos.func = mdp.joint_pos_rel_without_wheel
        cfg.low_level_observations.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        # cfg.low_level_observations.base_lin_vel.scale = 2.0
        cfg.low_level_observations.base_ang_vel.scale = 0.25
        cfg.low_level_observations.joint_pos.scale = 1.0
        cfg.low_level_observations.joint_vel.scale = 0.05
        cfg.low_level_observations.base_lin_vel = None
        cfg.low_level_observations.height_scan = None
        cfg.low_level_observations.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        cfg.low_level_observations.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # 在 __init__ 末尾添加，提前缓存引用避免每步查找
        self._ee_command_term = env.command_manager.get_term(cfg.ee_command_name)

        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0

        # ── 增量模式：缓存上一时刻的目标位姿（world 系） ──────────────────────
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat_w[:, 0] = 1.0  # 初始化为单位四元数 (w=1)
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 直接用 find_bodies 查，不需要 SceneEntityCfg 和 resolve
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        self._delta_pos_w = torch.zeros_like(self._target_pos_w)
        self._delta_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self._delta_action = torch.zeros(self.num_envs, 4, device=self.device)  # (delta_pos_w, delta_yaw)
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 11   # base_velocity(3) + ee_pose(7): [vx, vy, wz, x, y, z, qw, qx, qy, qz, delta_scale]
        # return 10   # base_velocity(3) + ee_pose(7): [vx, vy, wz, x, y, z, qw, qx, qy, qz]
                    # 此处根据low-level policy的输入维度进行设置。当前设置为10维，包含3维的底盘速度和7维的末端执行器位姿（位置+四元数）。
        # return 7    # base_velocity(3) + ee_pos(3) + yaw(1)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """
    def tanh_scale(self, x, lo, hi):
        # 将无界输入映射到 [lo, hi]，全程可微
        return lo + (hi - lo) * (torch.tanh(x) * 0.5 + 0.5)
    
    def range_to_scale_offset(self, lo: float, hi: float):
        """将 [lo, hi] 范围转换为 scale 和 offset"""
        scale = (hi - lo) / 2.0
        offset = (hi + lo) / 2.0
        return scale, offset
    
    def _reset_target_to_current_ee(self, env_ids: torch.Tensor | None = None):
        """将目标位姿重置为当前 EE 实际位姿（world 系）。"""
        # 取当前 EE body 的 world 位姿
        ee_pos_w  = self.robot.data.body_pos_w[:, self._ee_body_idx, :]   # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]  # (N, 4)
        if env_ids is None:
            self._target_pos_w[:]  = ee_pos_w
            self._target_quat_w[:] = ee_quat_w
            self._target_initialized[:] = True
        else:
            self._target_pos_w[env_ids]  = ee_pos_w[env_ids]
            self._target_quat_w[env_ids] = ee_quat_w[env_ids]
            self._target_initialized[env_ids] = True
    
    # def process_actions(self, actions: torch.Tensor):
    #     self._raw_actions[:] = actions
    #     r = self.cfg.low_level_command_ranges
    #     # ── 1. 底盘速度：tanh + scale/offset ──────────────────────────
    #     for i, (lo, hi) in enumerate([
    #         (r.lin_vel_x[0], r.lin_vel_x[1]),
    #         (r.lin_vel_y[0], r.lin_vel_y[1]),
    #         (r.ang_vel_z[0], r.ang_vel_z[1]),
    #     ]):
    #         scale, offset = self.range_to_scale_offset(lo, hi)
    #         self._raw_actions[:, i] = torch.tanh(actions[:, i]) * scale + offset
    #     # ── 2. 底盘线速度死区 ──────────────────────────────────────────
    #     lin_vel_norm = torch.norm(self._raw_actions[:, 0:2], p=2, dim=-1, keepdim=True)
    #     self._raw_actions[:, 0:2] = torch.where(
    #         lin_vel_norm < 0.2,
    #         torch.zeros_like(self._raw_actions[:, 0:2]),
    #         self._raw_actions[:, 0:2]
    #     )
    #     # ── 3. ee_pos：tanh + scale/offset，base 坐标系 ────────────────
    #     # print(f"Raw ee_pos commands before scaling(base): {actions[:, 3:6]}")
    #     for i, (lo, hi) in enumerate([
    #         (r.ee_pos_x[0], r.ee_pos_x[1]),   # col 3
    #         (r.ee_pos_y[0], r.ee_pos_y[1]),   # col 4
    #         (r.ee_pos_z[0], r.ee_pos_z[1]),   # col 5
    #     ]):
    #         scale, offset = self.range_to_scale_offset(lo, hi)
    #         self._raw_actions[:, 3 + i] = torch.tanh(actions[:, 3 + i]) * scale + offset
    #     # print(f"Raw ee_pos commands after scaling(base): {self._raw_actions[:, 3:6]}")
    #     # ── 4. 四元数：归一化，不做 scale ─────────────────────────────
    #     # ── 4. 构造固定朝下 + yaw 的四元数 ─────────────────────────────
    #     yaw = actions[:, 6]
    #     # 可选：限制范围（防止抖动）
    #     yaw = torch.tanh(yaw) * 0.5* torch.pi  # [-pi, pi]
    #     zeros = torch.zeros_like(yaw)
    #     # Rz(yaw)
    #     quat_z = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    #     # Rx(pi) → 朝下
    #     quat_down = math_utils.quat_from_euler_xyz(
    #         torch.full_like(yaw, torch.pi),  # roll = pi
    #         zeros,
    #         zeros,
    #     )
    #     # 最终姿态：Rz * Rx
    #     quat = math_utils.quat_mul(quat_z, quat_down)
    #     self._raw_actions[:, 6:10] = quat
    #     # ── 5. base → world 坐标变换 ───────────────────────────────────
    #     root_pos_w  = self.robot.data.root_pos_w
    #     root_quat_w = self.robot.data.root_quat_w
    #     target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
    #         root_pos_w, root_quat_w,
    #         self._raw_actions[:, 3:6],
    #         self._raw_actions[:, 6:10],
    #     )
    #     self._raw_actions[:, 3:6]  = target_pos_w
    #     self._raw_actions[:, 6:10] = target_quat_w

    # def process_actions(self, actions: torch.Tensor):
    #     self._raw_actions[:] = actions
    #     r = self.cfg.low_level_command_ranges

    #     # ── 1. 底盘速度：tanh + scale/offset ──────────────────────────
    #     for i, (lo, hi) in enumerate([
    #         (r.lin_vel_x[0], r.lin_vel_x[1]),
    #         (r.lin_vel_y[0], r.lin_vel_y[1]),
    #         (r.ang_vel_z[0], r.ang_vel_z[1]),
    #     ]):
    #         scale, offset = self.range_to_scale_offset(lo, hi)
    #         self._raw_actions[:, i] = torch.tanh(actions[:, i]) * scale + offset

    #     # ── 2. 底盘线速度死区 ──────────────────────────────────────────
    #     lin_vel_norm = torch.norm(self._raw_actions[:, 0:2], p=2, dim=-1, keepdim=True)
    #     self._raw_actions[:, 0:2] = torch.where(
    #         lin_vel_norm < 0.2,
    #         torch.zeros_like(self._raw_actions[:, 0:2]),
    #         self._raw_actions[:, 0:2]
    #     )

    #     # ── 3. EE 目标位置：直接使用 object 的 world 坐标 ─────────────
    #     # 忽略 policy 输出的 actions[:, 3:6]，从 scene 中取物体位置
    #     offset = torch.tensor([0.0, 0.0, 0.1], device=self.device)  # x, y, z 偏置（米）
    #     object_pos_w = self._env.scene["object"].data.root_pos_w  # (N, 3)
    #     self._raw_actions[:, 3:6] = object_pos_w + offset

    #     # ── 4. 构造固定朝下 + yaw 的四元数 ────────────────────────────
    #     yaw = actions[:, 6]
    #     yaw = torch.tanh(yaw) * 0.5 * torch.pi  # [-0.5π, 0.5π]
    #     zeros = torch.zeros_like(yaw)

    #     # Rz(yaw)
    #     quat_z = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    #     # Rx(pi) → 朝下
    #     quat_down = math_utils.quat_from_euler_xyz(
    #         torch.full_like(yaw, torch.pi),
    #         zeros,
    #         zeros,
    #     )
    #     # 最终姿态：Rz * Rx
    #     quat = math_utils.quat_mul(quat_z, quat_down)
    #     self._raw_actions[:, 6:10] = quat

    #     # ── 5. 位置已是 world 系，只需将四元数也保持 world 系 ──────────
    #     # EE 姿态：将 quat（当前以 base 系构造的偏航）转到 world 系
    #     root_quat_w = self.robot.data.root_quat_w
    #     # 仅旋转姿态：world_quat = root_quat_w * local_quat
    #     target_quat_w = math_utils.quat_mul(root_quat_w, quat)
    #     self._raw_actions[:, 6:10] = target_quat_w

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        r = self.cfg.low_level_command_ranges

        delta_scale = torch.sigmoid(actions[:, 10])  # [0,1] 之间的缩放因子
        # print(f"Delta scale from action: {delta_scale}")

        self._raw_actions[:,10] = delta_scale  # 将 delta_scale 也放入 raw_actions，供 reward term 使用
        # ── 1. 底盘速度：tanh + scale/offset（保持不变）─────────────────
        for i, (lo, hi) in enumerate([
            (r.lin_vel_x[0], r.lin_vel_x[1]),
            (r.lin_vel_y[0], r.lin_vel_y[1]),
            (r.ang_vel_z[0], r.ang_vel_z[1]),
        ]):
            scale, offset = self.range_to_scale_offset(lo, hi)
            self._raw_actions[:, i] = torch.tanh(actions[:, i]) * scale + offset

        # ── 2. 底盘线速度死区（保持不变）────────────────────────────────
        lin_vel_norm = torch.norm(self._raw_actions[:, 0:2], p=2, dim=-1, keepdim=True)
        self._raw_actions[:, 0:2] = torch.where(
            lin_vel_norm < 0.2,
            torch.zeros_like(self._raw_actions[:, 0:2]),
            self._raw_actions[:, 0:2]
        )

        # ── 3. 未初始化的 env 先重置目标到当前 EE 位姿 ──────────────────
        uninit_ids = (~self._target_initialized).nonzero(as_tuple=False).squeeze(-1)
        if uninit_ids.numel() > 0:
            self._reset_target_to_current_ee(uninit_ids)

        # ── 4. 计算位置增量 Δpos（world 系，tanh 锁幅）──────────────────
        # actions[:, 3:6] 被解释为 base 系下的位置增量方向
        # 先用 tanh 压缩到 [-1,1]，再乘以最大步长（米/step）
        delta_pos_max = self.cfg.delta_pos_max   # e.g. 0.05 m per high-level step
        delta_pos_b = torch.tanh(self._raw_actions[:, 3:6]) * delta_pos_max * delta_scale.unsqueeze(-1)  # (N, 3), base 系
        # print(f"Raw ee_pos commands before scaling(base): {self._raw_actions[:, 3:6]}")
        # print(f"Delta ee_pos commands after tanh and scaling(base): {delta_pos_b}")

        # base → world：只旋转方向，不平移（增量是方向向量）
        root_quat_w = self.robot.data.root_quat_w  # (N,4)
        delta_pos_w = math_utils.quat_apply(root_quat_w, delta_pos_b)  # (N,3) world系增量

        # combine_frame_transforms 会加 zeros_pos，结果即为旋转后的增量向量

        # ── 5. 叠加位置增量，并 clamp 到工作空间（world 系 AABB）────────
        new_pos_w = self._target_pos_w + delta_pos_w

        self._target_pos_w[:] = new_pos_w


        # ── 6. 计算姿态增量 Δyaw，叠加到当前目标四元数 ──────────────────
        delta_rot_max_rpy = torch.tensor(
        [self.cfg.delta_roll_max, self.cfg.delta_pitch_max, self.cfg.delta_yaw_max],
            device=actions.device
        )  # e.g. 0.1 rad per high-level step
        delta_rpy = torch.tanh(actions[:, 6:9]) * delta_rot_max_rpy.unsqueeze(0)  # (N, 3)
        # (N,)

        delta_roll  = delta_rpy[:, 0]
        delta_pitch = delta_rpy[:, 1]
        delta_yaw   = delta_rpy[:, 2]
        self._delta_pos_w = delta_pos_w.clone()
        self._delta_rpy    = delta_rpy.clone()
        self._delta_action = torch.cat([
            delta_pos_w,
            delta_rpy, 
        ], dim=-1)  # (N,6)
        zeros = torch.zeros_like(delta_yaw)

        # 构造增量旋转四元数（绕 world Z 轴）
        delta_quat = math_utils.quat_from_euler_xyz(delta_roll, delta_pitch, delta_yaw)  # (N, 4)

        # 叠加到上一时刻目标四元数：q_new = delta_q * q_old
        new_quat_w = math_utils.quat_mul(delta_quat, self._target_quat_w)
        new_quat_w = torch.nn.functional.normalize(new_quat_w, p=2, dim=-1)
        self._target_quat_w[:] = new_quat_w

        # ── 7. 写入 _raw_actions（供 apply_actions 读取）────────────────
        self._raw_actions[:, 3:6]  = self._target_pos_w
        self._raw_actions[:, 6:10] = self._target_quat_w

        # 位置直接保留 step 3 写入的 world 坐标，无需再变换
        # ==================== DEBUG: 边界框转换到世界坐标 ====================
        # num_envs = root_pos_w.shape[0]
        # device = root_pos_w.device

        # # 构造 base 系下的 8 个角点（AABB 包围盒）
        # # x: [r.ee_pos_x[0], r.ee_pos_x[1]]
        # # y: [r.ee_pos_y[0], r.ee_pos_y[1]]
        # # z: [r.ee_pos_z[0], r.ee_pos_z[1]]
        # corners_b = torch.tensor([
        #     [r.ee_pos_x[0], r.ee_pos_y[0], r.ee_pos_z[0]],
        #     [r.ee_pos_x[0], r.ee_pos_y[0], r.ee_pos_z[1]],
        #     [r.ee_pos_x[0], r.ee_pos_y[1], r.ee_pos_z[0]],
        #     [r.ee_pos_x[0], r.ee_pos_y[1], r.ee_pos_z[1]],
        #     [r.ee_pos_x[1], r.ee_pos_y[0], r.ee_pos_z[0]],
        #     [r.ee_pos_x[1], r.ee_pos_y[0], r.ee_pos_z[1]],
        #     [r.ee_pos_x[1], r.ee_pos_y[1], r.ee_pos_z[0]],
        #     [r.ee_pos_x[1], r.ee_pos_y[1], r.ee_pos_z[1]],
        # ], dtype=torch.float32, device=device)  # shape: (8, 3)

        # # 取第 0 个 env 的 root pose 做 debug（如需全部 env 可循环）
        # root_pos_w_0  = root_pos_w[0:1]   # (1, 3)
        # root_quat_w_0 = root_quat_w[0:1]  # (1, 4)

        # # 将 8 个角点逐一转换到世界坐标
        # corners_w_list = []
        # for i in range(8):
        #     corner_b_i = corners_b[i:i+1]  # (1, 3)
        #     # identity quat: 角点只是点，不携带方向，用单位四元数
        #     identity_quat = torch.zeros(1, 4, device=device)
        #     identity_quat[:, 0] = 1.0  # w=1
        #     corner_w_i, _ = math_utils.combine_frame_transforms(
        #         root_pos_w_0, root_quat_w_0, corner_b_i, identity_quat
        #     )
        #     corners_w_list.append(corner_w_i)

        # corners_w = torch.cat(corners_w_list, dim=0)  # (8, 3)

        # # 计算世界系下的 AABB（轴对齐包围盒）
        # bbox_min_w = corners_w.min(dim=0).values
        # bbox_max_w = corners_w.max(dim=0).values

        # print("=" * 60)
        # print(f"[DEBUG] Env-0 root_pos_w      : {root_pos_w_0.squeeze().tolist()}")
        # print(f"[DEBUG] EE bbox in BASE frame :")
        # print(f"         x: [{r.ee_pos_x[0]:.3f}, {r.ee_pos_x[1]:.3f}]")
        # print(f"         y: [{r.ee_pos_y[0]:.3f}, {r.ee_pos_y[1]:.3f}]")
        # print(f"         z: [{r.ee_pos_z[0]:.3f}, {r.ee_pos_z[1]:.3f}]")
        # print(f"[DEBUG] EE bbox in WORLD frame (env-0, AABB after rotation):")
        # print(f"         x: [{bbox_min_w[0]:.3f}, {bbox_max_w[0]:.3f}]")
        # print(f"         y: [{bbox_min_w[1]:.3f}, {bbox_max_w[1]:.3f}]")
        # print(f"         z: [{bbox_min_w[2]:.3f}, {bbox_max_w[2]:.3f}]")
        # print(f"[DEBUG] EE target_pos_w (env-0): {target_pos_w[0].tolist()}")
        # print(f"[DEBUG] All 8 corners in world frame:")
        # for i, c in enumerate(corners_w):
        #     print(f"         corner[{i}]: {c.tolist()}")
        # print("=" * 60)
        # # =====================================================================

    def apply_actions(self):

        # ── episode reset 时重置增量目标位姿 ────────────────────────────
        if hasattr(self._env, "episode_length_buf"):
            reset_ids = (self._env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self._target_initialized[reset_ids] = False  # 标记为未初始化，下一步重置

        
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # policy 输出切分给3个 action term
            policy_output = self.policy(low_level_obs)
            self.low_level_leg_actions[:] = policy_output[:, :self._joint_pos_dim]
            self.low_level_wheel_actions[:] = policy_output[:, self._joint_pos_dim:self._joint_pos_dim + self._wheel_vel_dim]
            self.low_level_ee_actions[:] = policy_output[:, self._joint_pos_dim + self._wheel_vel_dim:self._joint_pos_dim + self._wheel_vel_dim + self._ee_ik_dim]
            # 在 apply_actions 里写入 command 之前
            # target_pos  = self._raw_actions[:, 3:6]    # (num_envs, 3)
            # target_quat = self._raw_actions[:, 6:10]   # (num_envs, 4)  qw, qx, qy, qz

            # # 归一化，防止非单位四元数导致坐标轴歪斜
            # target_quat = torch.nn.functional.normalize(target_quat, p=2, dim=-1)

            # self._ee_command_term.pose_command_w[:, 0:3] = target_pos
            # self._ee_command_term.pose_command_w[:, 3:7] = target_quat
            self._ee_command_term.pose_command_w[:] = self._raw_actions[:, 3:10] # 更新 CommandManager 中的 ee_pose 命令，供 IK controller 使用

            self._joint_pos_action_term.process_actions(self.low_level_leg_actions)
            self._wheel_vel_action_term.process_actions(self.low_level_wheel_actions)
            self._ee_ik_action_term.process_actions(self.low_level_ee_actions)
            self._counter = 0

        self._joint_pos_action_term.apply_actions()
        self._wheel_vel_action_term.apply_actions()
        self._ee_ik_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- 速度目标（绿色箭头）
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- 当前速度（蓝色箭头）
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "ee_goal_visualizer"):
                # EE 目标位姿（红色箭头，沿X轴指示朝向）
                from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/ee_goal"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.ee_goal_visualizer = VisualizationMarkers(marker_cfg)

            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
            self.ee_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)
            if hasattr(self, "ee_goal_visualizer"):
                self.ee_goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # ── base velocity 可视化（原有逻辑不变）──────────────────────
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.raw_actions[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        # ── ee_pose 目标可视化 ✅ ──────────────────────────────────────
        # raw_actions[:, 3:10] = [x, y, z, qw, qx, qy, qz]
        ee_goal_pos  = self.raw_actions[:, 3:6]   # (N, 3)
        ee_goal_quat = self.raw_actions[:, 6:10]  # (N, 4) wxyz

        # 四元数全零时（reset后还没收到命令）跳过可视化，避免除零
        valid_mask = torch.norm(ee_goal_quat, dim=-1) > 0.1
        if valid_mask.any():
            # 归一化四元数防止marker变形
            ee_goal_quat_norm = torch.nn.functional.normalize(ee_goal_quat, dim=-1)
            # marker scale 固定，不随命令变化
            ee_marker_scale = torch.tensor(
                [[0.3, 0.3, 0.3]], device=self.device
            ).expand(self.num_envs, -1)
            self.ee_goal_visualizer.visualize(ee_goal_pos, ee_goal_quat_norm, ee_marker_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedPickActionCfg(ActionTermCfg):
    """Configuration for pre-trained pick action term.

    See :class:`PreTrainedPickAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPickAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_leg_actions: ActionTermCfg = MISSING
    """Low level leg action configuration."""
    low_level_wheel_actions: ActionTermCfg = MISSING
    """Low level wheel action configuration."""
    low_level_ee_actions: ActionTermCfg = MISSING
    """Low level end-effector action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    ee_command_name: str = "ee_pose"
    """The command name in CommandManager that this action term outputs to. Should correspond to a command in CommandsCfg."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    delta_pos_max: float = 0.05
    """每个高层 step EE 位置增量的最大幅度（米），tanh 后乘以此值。"""
    
    delta_yaw_max: float = 0.1
    """每个高层 step EE yaw 增量的最大幅度（弧度），tanh 后乘以此值。"""
    delta_roll_max: float = 0.1
    """每个高层 step EE roll 增量的最大幅度（弧度），tanh 后乘以此值。"""
    delta_pitch_max: float = 0.1
    """每个高层 step EE pitch 增量的最大幅度（弧度），tanh 后乘以此值。"""

    ee_body_name: str = "arm_link6"

    ee_pos_world_x: tuple[float, float] = (0.0, 3.0)
    ee_pos_world_y: tuple[float, float] = (-2.0, 2.0)
    ee_pos_world_z: tuple[float, float] = (0.3, 1.5)  # ⚠️ 关键：z 不能低于地面
    @configclass
    class LowLevelCommandRanges:
        # base_velocity ranges，对应 CommandsCfg.base_velocity.ranges
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        # ee_pose ranges，对应 CommandsCfg.ee_pose 的 command 输出空间
        # command 输出是世界坐标系下的 [x, y, z, qw, qx, qy, qz]
        # 四元数各分量天然在 [-1, 1]，位置范围根据实际场景设置
        ee_pos_x: tuple[float, float] = (0.3, 0.8)
        ee_pos_y: tuple[float, float] = (-0.4, 0.4)
        ee_pos_z: tuple[float, float] = (-0.1, 0.2)
        ee_quat_w: tuple[float, float] = (-1.0, 1.0)
        ee_quat_x: tuple[float, float] = (-1.0, 1.0)
        ee_quat_y: tuple[float, float] = (-1.0, 1.0)
        ee_quat_z: tuple[float, float] = (-1.0, 1.0)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()
