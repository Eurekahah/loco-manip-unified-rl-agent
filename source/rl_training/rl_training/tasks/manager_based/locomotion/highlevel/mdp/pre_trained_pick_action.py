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

        cfg.low_level_observations.ee_goal.func = lambda dummy_env: self._raw_actions[:, 3:]
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
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 10   # base_velocity(3) + ee_pose(7): [vx, vy, wz, x, y, z, qw, qx, qy, qz]
                    # 此处根据low-level policy的输入维度进行设置。当前设置为10维，包含3维的底盘速度和7维的末端执行器位姿（位置+四元数）。

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        r = self.cfg.low_level_command_ranges

        # clip base_velocity commands
        self._raw_actions[:, 0].clamp_(r.lin_vel_x[0], r.lin_vel_x[1])
        self._raw_actions[:, 1].clamp_(r.lin_vel_y[0], r.lin_vel_y[1])
        self._raw_actions[:, 2].clamp_(r.ang_vel_z[0], r.ang_vel_z[1])

        # clip ee_pose commands
        self._raw_actions[:, 3].clamp_(r.ee_pos_x[0],   r.ee_pos_x[1])
        self._raw_actions[:, 4].clamp_(r.ee_pos_y[0],   r.ee_pos_y[1])
        self._raw_actions[:, 5].clamp_(r.ee_pos_z[0],   r.ee_pos_z[1])
        # self._raw_actions[:, 6].clamp_(r.ee_quat_w[0],  r.ee_quat_w[1])
        # self._raw_actions[:, 7].clamp_(r.ee_quat_x[0],  r.ee_quat_x[1])
        # self._raw_actions[:, 8].clamp_(r.ee_quat_y[0],  r.ee_quat_y[1])
        # self._raw_actions[:, 9].clamp_(r.ee_quat_z[0],  r.ee_quat_z[1])

        # 四元数：不做 clamp，直接归一化
        quat = self._raw_actions[:, 6:10]
        self._raw_actions[:, 6:10] = torch.nn.functional.normalize(quat, p=2, dim=-1)
        target_pos_b = self._raw_actions[:, 3:6]
        target_quat_b = self._raw_actions[:, 6:10]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # 将 ee_pose command 从 base 坐标系转换到 world 坐标系
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, target_pos_b, target_quat_b 
        )
        self._raw_actions[:, 3:6] = target_pos_w
        self._raw_actions[:, 6:10] = target_quat_w

    def apply_actions(self):
        
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
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
    @configclass
    class LowLevelCommandRanges:
        # base_velocity ranges，对应 CommandsCfg.base_velocity.ranges
        lin_vel_x: tuple[float, float] = (-0.2, 0.2)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        # ee_pose ranges，对应 CommandsCfg.ee_pose 的 command 输出空间
        # command 输出是世界坐标系下的 [x, y, z, qw, qx, qy, qz]
        # 四元数各分量天然在 [-1, 1]，位置范围根据实际场景设置
        ee_pos_x: tuple[float, float] = (0.0, 1.0)
        ee_pos_y: tuple[float, float] = (-0.3, 0.3)
        ee_pos_z: tuple[float, float] = (0.3, 0.65)
        ee_quat_w: tuple[float, float] = (-1.0, 1.0)
        ee_quat_x: tuple[float, float] = (-1.0, 1.0)
        ee_quat_y: tuple[float, float] = (-1.0, 1.0)
        ee_quat_z: tuple[float, float] = (-1.0, 1.0)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()
