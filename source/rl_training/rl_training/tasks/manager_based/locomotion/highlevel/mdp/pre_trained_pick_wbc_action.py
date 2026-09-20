# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
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
from rl_training.tasks.manager_based.locomotion.highlevel.mdp.low_level_replay import (
    build_low_level_obs_manager,
    build_low_level_observation_group,
    build_history_window,
    check_low_level_action_cfgs,
    expected_policy_obs_dim,
    load_low_level_policy,
    push_ee_target_to_ik,
    resolve_layout,
    run_low_level_policy,
    verify_low_level_layout,
)



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPickWBCAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPickWBCActionCfg
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
        "gripper_joint1", "gripper_joint2",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names

    def __init__(self, cfg: PreTrainedPickWBCActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        # 统一的加载 + 明确的报错（清单 ⑦）：见 low_level_replay.load_low_level_policy
        self.policy = load_low_level_policy(cfg.policy_path, env, tag=type(self).__name__)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)     # [vx, vy, wz,  Δx, Δy, Δz,  Δr, Δp, Δy,      Δbody_height, Δbody_pitch, Δbody_roll]
        self._ll_command = torch.zeros(self.num_envs, self.action_dim + 1, device=self.device)  # [vx, vy, wz,  x, y, z,     qw, qx, qy, qz,  body_height, body_pitch, body_roll]
        # 同一命令的世界系副本（EE 目标的世界坐标/四元数），供需要世界系的奖励项使用；
        # _ll_command 本身统一为 root 系（见 ll_command_world() 的说明）
        self._ll_command_w = torch.zeros_like(self._ll_command)

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

        # ── 低层 replay 布局（清单 ④⑤⑥⑯）──────────────────────────────
        # 同 PreTrainedPickAction：布局显式声明，scale/clip/关节名单以低层 action cfg
        # 为唯一来源，观测组由模板 deepcopy 生成（不再就地改传入的 cfg，
        # 也不再在运行时替换 cfg.low_level_observations）。
        self._layout = resolve_layout(
            robot=self.robot,
            low_level_obs_cfg=cfg.low_level_observations,
            low_level_leg_cfg=cfg.low_level_leg_actions,
            low_level_wheel_cfg=cfg.low_level_wheel_actions,
            declared_ee_action_dim=cfg.ee_action_dim,
            actual_ee_ik_action_dim=self._ee_ik_action_term.action_dim,
            tag=type(self).__name__,
        )
        check_low_level_action_cfgs(
            tag=type(self).__name__,
            layout=self._layout,
            leg_cfg=cfg.low_level_leg_actions,
            wheel_cfg=cfg.low_level_wheel_actions,
        )

        self._joint_pos_dim = self._layout.leg_dim
        self._wheel_vel_dim = self._layout.wheel_dim
        self._ee_ik_dim = self._layout.ee_action_dim

        self.low_level_leg_actions = torch.zeros(
            self.num_envs, self._joint_pos_dim, device=self.device
        )
        self.low_level_wheel_actions = torch.zeros(
            self.num_envs, self._wheel_vel_dim, device=self.device
        )
        self.low_level_ee_actions = torch.zeros(
            self.num_envs, self._ee_ik_dim, device=self.device
        )

        def last_action():
            # 低层 policy 训练时的 actions 观测 = 完整动作向量 [leg | wheel | ee_ik]。
            # 复位清空统一交给 LowLevelReplayState.on_tick()（原因见该类的文档字符串）。
            return torch.cat(
                [self.low_level_leg_actions, self.low_level_wheel_actions, self.low_level_ee_actions],
                dim=-1,
            )

        self._low_level_obs_cfg = build_low_level_observation_group(
            cfg.low_level_observations,
            layout=self._layout,
            actions_fn=lambda dummy_env: last_action(),
            velocity_commands_fn=lambda dummy_env: self._ll_command[:, :3],
            ee_goal_fn=lambda dummy_env: self._ll_command[:, 3:10],
            body_pose_cmd_fn=lambda dummy_env: self._ll_command[:, 10:13],
        )
        # 在 __init__ 末尾添加，提前缓存引用避免每步查找
        self._ee_command_term = env.command_manager.get_term(cfg.ee_command_name)
        self._expected_ll_obs_dim, self._policy_layout_json = expected_policy_obs_dim(
            self.policy, cfg.policy_path, tag=type(self).__name__
        )
        self._low_level_obs_manager, self._low_level_obs_cfg, self._ll_used_ee_goal = (
            build_low_level_obs_manager(
                env=env,
                obs_cfg=self._low_level_obs_cfg,
                group_name="ll_policy",
                expected_obs_dim=self._expected_ll_obs_dim,
                tag=type(self).__name__,
            )
        )
        verify_low_level_layout(
            tag=type(self).__name__,
            robot=self.robot,
            layout=self._layout,
            obs_manager=self._low_level_obs_manager,
            group_name="ll_policy",
            policy=self.policy,
            expected_obs_dim=self._expected_ll_obs_dim,
            policy_layout_json=self._policy_layout_json,
        )
        # 回放侧的低层 tick 状态：复位检测 + 低层动作缓存清零 + （history 策略时的）10 步窗口
        self._ll_replay_state = build_history_window(
            env=env,
            layout=self._layout,
            policy_layout_json=self._policy_layout_json,
            last_action_fn=last_action,
            cache_tensors=[
                self.low_level_leg_actions,
                self.low_level_wheel_actions,
                self.low_level_ee_actions,
            ],
            asset_name=cfg.asset_name,
            tag=type(self).__name__,
        )
        self._counter = 0

        # ── 增量模式：缓存上一时刻的目标位姿（world 系） ──────────────────────
        self._target_ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_ee_orn_rpy_b = torch.zeros(self.num_envs, 3, device=self.device)

        # body_pose 增量模式：缓存上一时刻的目标 body pose（height, pitch, roll）
        self._target_body_height = torch.full(
            (self.num_envs,), 0.513, device=self.device  # 初始值取0.513m
        )
        self._target_body_pitch = torch.zeros(self.num_envs, device=self.device)
        self._target_body_roll  = torch.zeros(self.num_envs, device=self.device)
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 直接用 find_bodies 查，不需要 SceneEntityCfg 和 resolve
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 12   # base_velocity(3) + ee_pose(7) + delta_scale(1)+: [vx, vy, wz, x, y, z, r, p, y, body_height, body_pitch, body_roll]
        # return 10   # base_velocity(3) + ee_pose(7): [vx, vy, wz, x, y, z, qw, qx, qy, qz]
                    # 此处根据low-level policy的输入维度进行设置。当前设置为10维，包含3维的底盘速度和7维的末端执行器位姿（位置+四元数）。
        # return 7    # base_velocity(3) + ee_pos(3) + yaw(1)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    
    @property
    def ll_command(self) -> torch.Tensor:
        return self._ll_command

    @property
    def ll_command_w(self) -> torch.Tensor:
        """``ll_command`` 的世界系副本（``[vx,vy,wz, ee_pos_w(3), ee_quat_w(4), h,p,r]``）。"""
        return self._ll_command_w

    """
    Operations.
    """
    
    def range_to_scale_offset(self, lo: float, hi: float):
        """将 [lo, hi] 范围转换为 scale 和 offset"""
        scale = (hi - lo) / 2.0
        offset = (hi + lo) / 2.0
        return scale, offset
    
    def _reset_target_to_current_ee(self, env_ids: torch.Tensor | None = None):
        """将目标位姿重置为当前 EE 实际位姿（world 系）。"""
        ee_pos_w  = self.robot.data.body_pos_w[:, self._ee_body_idx, :]   # (N, 3)
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx, :]  # (N, 4)
        ee_roll, ee_pitch, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat_w)

        # 转到 body 系
        root_quat_w = self.robot.data.root_quat_w
        root_quat_inv = math_utils.quat_conjugate(root_quat_w)
        ee_quat_b = math_utils.quat_mul(root_quat_inv, ee_quat_w)
        ee_roll_b, ee_pitch_b, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)
        ee_pos_b = math_utils.quat_apply(root_quat_inv, ee_pos_w - self.robot.data.root_pos_w)

        # 读取当前机身实际 pose 用于重置
        default_height = 0.513
        default_pitch  = 0.0
        default_roll   = 0.0 

        if env_ids is None:
            self._target_initialized[:] = True

            self._target_ee_orn_rpy_b[:] = torch.stack([ee_roll_b, ee_pitch_b, ee_yaw_b], dim=-1)
            self._target_ee_pos_b[:] = ee_pos_b 
            self._target_body_height[:] = default_height
            self._target_body_pitch[:]  = default_pitch
            self._target_body_roll[:]   = default_roll
        else:
            self._target_initialized[env_ids] = True

            self._target_ee_orn_rpy_b[env_ids] = torch.stack(
                [ee_roll_b[env_ids], ee_pitch_b[env_ids], ee_yaw_b[env_ids]], dim=-1)
            self._target_ee_pos_b[env_ids] = ee_pos_b[env_ids]
            self._target_body_height[env_ids] = default_height
            self._target_body_pitch[env_ids]  = default_pitch
            self._target_body_roll[env_ids]   = default_roll

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        r = self.cfg.low_level_command_ranges

        # ── 1. 底盘速度：tanh + scale/offset ─────────────────────────────
        for i, (lo, hi) in enumerate([
            (r.lin_vel_x[0], r.lin_vel_x[1]),
            (r.lin_vel_y[0], r.lin_vel_y[1]),
            (r.ang_vel_z[0], r.ang_vel_z[1]),
        ]):
            scale, offset = self.range_to_scale_offset(lo, hi)
            self._ll_command[:, i] = torch.tanh(actions[:, i]) * scale + offset

        # lin_vel_norm = torch.norm(self._ll_command[:, 0:2], p=2, dim=-1, keepdim=True)
        # self._ll_command[:, 0:2] = torch.where(
        #     lin_vel_norm < 0.2,
        #     torch.zeros_like(self._ll_command[:, 0:2]),
        #     self._ll_command[:, 0:2]
        # )
        # ── 3. 未初始化的 env 先重置目标到当前 EE 位姿 ───────────────────
        uninit_ids = (~self._target_initialized).nonzero(as_tuple=False).squeeze(-1)
        if uninit_ids.numel() > 0:
            self._reset_target_to_current_ee(uninit_ids)
        
        # ── 4. 计算EE位置增量 Δpos（body系，tanh 锁幅, 最终输出world系）───────────────────
        delta_ee_pos_b = torch.tanh(self._raw_actions[:, 3:6]) * self.cfg.delta_ee_pos_max
        self._target_ee_pos_b = self._target_ee_pos_b + delta_ee_pos_b
        self._target_ee_pos_b[:, 0] = torch.clamp(self._target_ee_pos_b[:, 0], r.ee_pos_x[0], r.ee_pos_x[1])
        self._target_ee_pos_b[:, 1] = torch.clamp(self._target_ee_pos_b[:, 1], r.ee_pos_y[0], r.ee_pos_y[1])
        self._target_ee_pos_b[:, 2] = torch.clamp(self._target_ee_pos_b[:, 2], r.ee_pos_z[0], r.ee_pos_z[1])
        
        
        root_quat_w = self.robot.data.root_quat_w  # (N, 4)
        root_pos_w  = self.robot.data.root_pos_w
        target_pos_w = math_utils.quat_apply(root_quat_w, self._target_ee_pos_b) + root_pos_w
        target_pos_w[:, 2] = torch.clamp(target_pos_w[:, 2], min=0.0)  # 再次 clamp 确保世界坐标系下 z 不低于地面
        # 世界系副本先记下来（奖励项要拿它和物体的世界坐标比较）
        self._ll_command_w[:, 3:6] = target_pos_w

         # ── 5. 叠加 EE 姿态rpy增量 ───────────────────────────────────────────────
        delta_ee_orn_rpy_b = torch.tanh(self._raw_actions[:, 6:9]) * self.cfg.delta_ee_orn_max
        self._target_ee_orn_rpy_b = self._target_ee_orn_rpy_b + delta_ee_orn_rpy_b
        ee_quat_b = math_utils.quat_from_euler_xyz(
            self._target_ee_orn_rpy_b[:, 0],  # roll
            self._target_ee_orn_rpy_b[:, 1],  # pitch
            self._target_ee_orn_rpy_b[:, 2],  # yaw
        )
        target_quat_w = math_utils.quat_mul(root_quat_w, ee_quat_b)
        self._ll_command_w[:, 6:10] = target_quat_w

        # -- 6. 叠加机体姿态height， pitch，roll增量
        delta_height = torch.tanh(actions[:, 9]) * self.cfg.delta_body_height_max  # (N,)
        delta_pitch  = torch.tanh(actions[:, 10]) * self.cfg.delta_body_pitch_max   # (N,)
        delta_roll   = torch.tanh(actions[:, 11]) * self.cfg.delta_body_roll_max  # (N,)

        # 累积到目标上，并 clamp 到合法范围
        self._target_body_height = torch.clamp(
            self._target_body_height + delta_height,
            self.cfg.low_level_command_ranges.target_height[0],
            self.cfg.low_level_command_ranges.target_height[1],
        )
        self._target_body_pitch = torch.clamp(
            self._target_body_pitch + delta_pitch,
            self.cfg.low_level_command_ranges.target_pitch[0],
            self.cfg.low_level_command_ranges.target_pitch[1],
        )
        self._target_body_roll = torch.clamp(
            self._target_body_roll + delta_roll,
            self.cfg.low_level_command_ranges.target_roll[0],
            self.cfg.low_level_command_ranges.target_roll[1],
        )

        self._ll_command[:, 10] = self._target_body_height
        self._ll_command[:, 11] = self._target_body_pitch
        self._ll_command[:, 12] = self._target_body_roll

        # ── 7. 把 EE 目标统一成 root 系（规范形式，供低层 obs 的 ee_goal 与 IK 使用），
        #       世界系副本保留在 _ll_command_w 里给奖励项用 ────────────────
        #   必要性：低层训练时 ee_goal 观测来自 HeightInvariantEECommand.pose_command_b
        #   （root 系），回放侧若塞世界系值，低层 policy 收到的输入与训练分布不符。
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        self._ll_command[:, 3:6] = target_pos_b
        self._ll_command[:, 6:10] = target_quat_b
        self._ll_command_w[:, 0:3] = self._ll_command[:, 0:3]
        self._ll_command_w[:, 10:13] = self._ll_command[:, 10:13]  # 机身姿态与坐标系无关
        # print(self.ll_command)
        

    def apply_actions(self):

        # ── episode reset 时重置增量目标位姿 ────────────────────────────
        if hasattr(self._env, "episode_length_buf"):
            reset_ids = (self._env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self._target_initialized[reset_ids] = False  # 标记为未初始化，下一步重置

        
        if self._counter % self.cfg.low_level_decimation == 0:
            # 低层 tick：先复位处理 + 推 history 帧，再算观测/跑策略（顺序不能反）
            history_flat = self._ll_replay_state.on_tick()
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # policy 输出切分给3个 action term
            policy_output = run_low_level_policy(self.policy, low_level_obs, history_flat)
            leg, wheel, ee = self._layout.split(policy_output)
            self.low_level_leg_actions[:] = leg
            self.low_level_wheel_actions[:] = wheel
            self.low_level_ee_actions[:] = ee
            # 把高层目标写给 IK：IK 读的是 command_manager.get_command("ee_pose")
            # == HeightInvariantEECommand.pose_command_b（root 系目标）。
            # 之前写的是 pose_command_w —— 那个字段只被父类 _update_metrics/debug vis
            # 使用，写进去等于没写（清单 ②）。
            push_ee_target_to_ik(
                self._ee_command_term, self._ll_command[:, 3:10], tag=type(self).__name__
            )

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
            self.ll_command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        # ── ee_pose 目标可视化 ✅ ──────────────────────────────────────
        ee_goal_pos  = self.ll_command[:, 3:6]   # (N, 3)
        ee_goal_quat = self.ll_command[:, 6:10]  # (N, 4) wxyz

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
class PreTrainedPickWBCActionCfg(ActionTermCfg):
    """Configuration for pre-trained pick action term.

    See :class:`PreTrainedPickWBCAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPickWBCAction
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
    ee_action_dim: int = -1
    """低层 checkpoint 动作输出里 IK 槽位的数量；``-1`` = 从低层 cfg 推导（L2）。
    旧 checkpoint 训练时该值为 7（见 ``low_level_replay`` 与 ``PreTrainedPickActionCfg``）。"""
    ee_command_name: str = "ee_pose"
    """The command name in CommandManager that this action term outputs to. Should correspond to a command in CommandsCfg."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""

    delta_ee_pos_max: float = 0.05
    """每个高层 step EE 位置增量的最大幅度（米），tanh 后乘以此值。"""
    
    delta_ee_orn_max: float = 0.06
    """每个高层 step EE 欧拉角姿态增量的最大幅度（rad），tanh 后乘以此值。"""

    delta_yaw_max: float = 0.06
    """每个高层 step EE yaw 增量的最大幅度（弧度），tanh 后乘以此值。"""
    delta_roll_max: float = 0.06
    """每个高层 step EE roll 增量的最大幅度（弧度），tanh 后乘以此值。"""
    delta_pitch_max: float = 0.06
    """每个高层 step EE pitch 增量的最大幅度（弧度），tanh 后乘以此值。"""

    delta_body_height_max: float = 0.05
    """每个高层 step body height 增量的最大幅度（米），tanh 后乘以此值。"""

    delta_body_pitch_max: float = 0.1
    """每个高层 step body pitch 增量的最大幅度（弧度），tanh 后乘以此值。"""

    delta_body_roll_max: float = 0.05
    """每个高层 step body roll 增量的最大幅度（弧度），tanh 后乘以此值。"""

    ee_body_name: str = "gripper_base"
    @configclass
    class LowLevelCommandRanges:
        # base_velocity ranges，对应 CommandsCfg.base_velocity.ranges
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        # ee_pose ranges，对应 CommandsCfg.ee_pose 的 command 输出空间
        # command 输出是世界坐标系下的 [x, y, z, qw, qx, qy, qz]
        # 四元数各分量天然在 [-1, 1]，位置范围根据实际场景设置
        ee_pos_x: tuple[float, float] = (0.4, 0.8)  # (0.3, 0.8) 
        ee_pos_y: tuple[float, float] = (-0.4, 0.4)
        ee_pos_z: tuple[float, float] = (-0.6, 0.6)
        ee_pitch: tuple[float, float] = (-math.pi/2, 0.0 )  # 限制在朝下到朝前
        # target_height: 机器狗期望站立高度（米），参考低层训练时的正常高度
        target_height: tuple[float, float] = (0.33, 0.55)  # 与低层 body_pose.height_range 上界保持一致（实测 0.60 够不到，见 docs/review/bad_orientation_analysis_zh.md §5F）
        # target_pitch: 机身期望俯仰角（弧度），正值抬头
        target_pitch: tuple[float, float] = (-0.35, 0.35)  # (-0.35, 0.35)
        # target_roll: 机身期望侧倾角（弧度），正值右倾
        target_roll: tuple[float, float] = (-0.25, 0.25) # (-0.25, 0.25)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()
