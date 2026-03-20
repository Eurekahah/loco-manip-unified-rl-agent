# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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




class PreTrainedNavAction(ActionTerm):
    r"""Pre-trained policy action term (chassis only).

    This action term infers a pre-trained policy and applies the corresponding low-level actions
    to the robot chassis (legs + wheels). The arm-related joints are removed.

    The raw actions correspond to the base velocity commands for the pre-trained policy:
        [vx, vy, wz]  —  3-dimensional chassis velocity command.
    """

    cfg: PreTrainedNavActionCfg
    """The configuration of the action term."""

    # ── joint name definitions ──────────────────────────────────────────────
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

    # 底盘观测用到的全部关节（腿 + 轮），不含机械臂
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names

    def __init__(self, cfg: PreTrainedNavActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ── load low-level policy ───────────────────────────────────────────
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # ── instantiate low-level action terms ─────────────────────────────
        self._joint_pos_action_term: ActionTerm = cfg.low_level_leg_actions.class_type(
            cfg.low_level_leg_actions, env
        )
        self._wheel_vel_action_term: ActionTerm = cfg.low_level_wheel_actions.class_type(
            cfg.low_level_wheel_actions, env
        )

        self._joint_pos_dim = self._joint_pos_action_term.action_dim
        self._wheel_vel_dim = self._wheel_vel_action_term.action_dim

        self.low_level_leg_actions = torch.zeros(
            self.num_envs, self._joint_pos_dim, device=self.device
        )
        self.low_level_wheel_actions = torch.zeros(
            self.num_envs, self._wheel_vel_dim, device=self.device
        )
        self._joint_pos_action_term.scale = {".*_hipx_joint": 0.125, '^(?!.*_hipx_joint)(?!.*arm_joint).*': 0.25}
        self._wheel_vel_action_term.scale = 20.0
        self._joint_pos_action_term.clip = {".*": (-100.0, 100.0)}
        self._wheel_vel_action_term.clip = {".*": (-100.0, 100.0)}
        self._joint_pos_action_term.joint_names = self.leg_joint_names
        self._wheel_vel_action_term.joint_names = self.wheel_joint_names


        # ── wire observation lambdas ────────────────────────────────────────
        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_mask = env.episode_length_buf == 0
                self.low_level_leg_actions[reset_mask, :] = 0
                self.low_level_wheel_actions[reset_mask, :] = 0
                self._raw_actions[reset_mask, :] = 0
            # 拼接两个 action term 的输出供 low-level obs 使用
            return torch.cat(
                [self.low_level_leg_actions, self.low_level_wheel_actions], dim=-1
            )

        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()

        cfg.low_level_observations.velocity_commands.func = (
            lambda dummy_env: self._raw_actions[:, :3]
        )
        cfg.low_level_observations.velocity_commands.params = dict()

        # joint_pos 观测只保留底盘关节，屏蔽轮子
        cfg.low_level_observations.joint_pos.func = mdp.joint_pos_rel_without_wheel
        cfg.low_level_observations.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        cfg.low_level_observations.base_ang_vel.scale = 0.25
        cfg.low_level_observations.joint_pos.scale = 1.0
        cfg.low_level_observations.joint_vel.scale = 0.05
        # 不使用线速度观测和高度扫描
        cfg.low_level_observations.base_lin_vel = None
        cfg.low_level_observations.height_scan = None
        cfg.low_level_observations.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        cfg.low_level_observations.joint_vel.params["asset_cfg"].joint_names = self.joint_names


        self._low_level_obs_manager = ObservationManager(
            {"ll_policy": cfg.low_level_observations}, env
        )
        self._counter = 0

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        # 底盘速度命令：[vx, vy, wz]
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    # ── operations ──────────────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        r = self.cfg.low_level_command_ranges

        # clip 底盘速度命令
        self._raw_actions[:, 0].clamp_(r.lin_vel_x[0], r.lin_vel_x[1])
        self._raw_actions[:, 1].clamp_(r.lin_vel_y[0], r.lin_vel_y[1])
        self._raw_actions[:, 2].clamp_(r.ang_vel_z[0], r.ang_vel_z[1])

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # policy 输出切分给腿部和轮子两个 action term
            policy_output = self.policy(low_level_obs)
            self.low_level_leg_actions[:] = policy_output[:, : self._joint_pos_dim]
            self.low_level_wheel_actions[:] = policy_output[
                :, self._joint_pos_dim : self._joint_pos_dim + self._wheel_vel_dim
            ]

            self._joint_pos_action_term.process_actions(self.low_level_leg_actions)
            self._wheel_vel_action_term.process_actions(self.low_level_wheel_actions)
            self._counter = 0

        self._joint_pos_action_term.apply_actions()
        self._wheel_vel_action_term.apply_actions()
        self._counter += 1

    # ── debug visualization ──────────────────────────────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                # 速度目标（绿色箭头）
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # 当前速度（蓝色箭头）
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)

            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

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

    # ── internal helpers ─────────────────────────────────────────────────────

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedNavActionCfg(ActionTermCfg):
    """Configuration for pre-trained navigation action term (chassis only).

    See :class:`PreTrainedNavAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedNavAction
    """Class of the action term."""

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low-level policy (.pt file)."""
    low_level_decimation: int = 4
    """Decimation factor for the low-level action term."""
    low_level_leg_actions: ActionTermCfg = MISSING
    """Low-level leg joint position action configuration."""
    low_level_wheel_actions: ActionTermCfg = MISSING
    """Low-level wheel velocity action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low-level observation group configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information."""

    @configclass
    class LowLevelCommandRanges:
        """Clipping ranges for the 3-DoF chassis velocity command."""
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()