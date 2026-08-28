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
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_wbc_cfg import WBCObservationsCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import compute_base_height_rel_to_feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TeleopLLAction(ActionTerm):
    r"""Pre-trained policy action term.

    raw_actions 直接对应 ll_command，每维都是绝对目标值（无增量累积）：
        [vx, vy, wz,  ee_x_b, ee_y_b, ee_z_b,  ee_roll_b, ee_pitch_b, ee_yaw_b,  body_height, body_pitch, body_roll]

    处理流程：
      1. tanh + range mapping → 将原始动作映射到目标物理范围
      2. body系EE位置 → world系（旋转变换）
      3. body系EE欧拉角 → world系四元数
      4. 写入 ll_command（shape: [N, 13]，含 vx/vy/wz + pos_w(3) + quat_w(4) + h/p/r(3)）
    """

    cfg: TeleopLLActionCfg

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
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    gripper_joint_names = [
        "gripper_joint1", "gripper_joint2",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names

    def __init__(self, cfg: TeleopLLActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # raw_actions: 12维绝对目标值
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # ll_command: [vx, vy, wz, x_w, y_w, z_w, qw, qx, qy, qz, height, pitch, roll]
        self._ll_command = torch.zeros(self.num_envs, 13, device=self.device)

        # 初始化三个 low-level action term
        self._joint_pos_action_term: ActionTerm = cfg.low_level_leg_actions.class_type(
            cfg.low_level_leg_actions, env
        )
        self._wheel_vel_action_term: ActionTerm = cfg.low_level_wheel_actions.class_type(
            cfg.low_level_wheel_actions, env
        )
        self._ee_ik_action_term: ActionTerm = cfg.low_level_ee_actions.class_type(
            cfg.low_level_ee_actions, env
        )

        self._joint_pos_dim = self._joint_pos_action_term.action_dim
        self._wheel_vel_dim = self._wheel_vel_action_term.action_dim
        self._ee_ik_dim     = self._ee_ik_action_term.action_dim

        self.low_level_leg_actions   = torch.zeros(self.num_envs, self._joint_pos_dim, device=self.device)
        self.low_level_wheel_actions = torch.zeros(self.num_envs, self._wheel_vel_dim,  device=self.device)
        self.low_level_ee_actions    = torch.zeros(self.num_envs, self._ee_ik_dim,      device=self.device)

        self._joint_pos_action_term.scale = {".*_hipx_joint": 0.125, '^(?!.*_hipx_joint)(?!.*arm_joint).*': 0.25}
        self._wheel_vel_action_term.scale = 5.0
        self._joint_pos_action_term.clip  = {".*": (-100.0, 100.0)}
        self._wheel_vel_action_term.clip  = {".*": (-100.0, 100.0)}
        self._joint_pos_action_term.joint_names = self.leg_joint_names
        self._wheel_vel_action_term.joint_names = self.wheel_joint_names

        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_mask = env.episode_length_buf == 0
                self.low_level_leg_actions[reset_mask, :]   = 0
                self.low_level_wheel_actions[reset_mask, :] = 0
                self.low_level_ee_actions[reset_mask, :]    = 0
            return torch.cat(
                [self.low_level_leg_actions,
                 self.low_level_wheel_actions,
                 self.low_level_ee_actions], dim=-1
            )

        wbc_obs_cfg = WBCObservationsCfg()
        cfg.low_level_observations = wbc_obs_cfg.policy

        cfg.low_level_observations.actions.func   = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()

        cfg.low_level_observations.velocity_commands.func   = lambda dummy_env: self._ll_command[:, :3]
        cfg.low_level_observations.velocity_commands.params = dict()

        cfg.low_level_observations.ee_goal.func   = lambda dummy_env: self._ll_command[:, 3:10]
        cfg.low_level_observations.ee_goal.params = dict()

        cfg.low_level_observations.body_pose_cmd.func   = lambda dummy_env: self._ll_command[:, 10:13]
        cfg.low_level_observations.body_pose_cmd.params = dict()

        cfg.low_level_observations.joint_pos.func = mdp.joint_pos_rel_without_wheel
        cfg.low_level_observations.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=False
        )

        cfg.low_level_observations.base_ang_vel.scale = 0.25
        cfg.low_level_observations.joint_pos.scale    = 1.0
        cfg.low_level_observations.joint_vel.scale    = 0.05
        cfg.low_level_observations.base_lin_vel       = None
        cfg.low_level_observations.height_scan        = None
        cfg.low_level_observations.joint_pos.params["asset_cfg"].joint_names    = self.joint_names
        cfg.low_level_observations.joint_vel.params["asset_cfg"].joint_names    = self.joint_names
        cfg.low_level_observations.joint_vel.params["asset_cfg"].preserve_order = True

        self._ee_command_term = env.command_manager.get_term(cfg.ee_command_name)

        cfg.low_level_observations.enable_corruption = False
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0

        # default EE 位姿缓存（body系），首次使用时 / reset 时填充
        self._default_ee_pos_b: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self._default_ee_quat_b: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self._default_ee_quat_b[:, 0] = 1.0  # 单位四元数初始化，防止未初始化时出现非法值
        self._default_ee_pose_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]
        # default 身体姿态缓存（height, pitch, roll），reset 时恢复到初始值
        self._default_body_pose: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self._default_body_pose[:, 0] = 0.51  # height 默认 0.51，pitch/roll 默认 0
        
        self.arm_link_ids = self.robot.find_bodies(["arm.*"])[0]
    

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        # [vx, vy, wz,  ee_x_b, ee_y_b, ee_z_b,  ee_roll_b, ee_pitch_b, ee_yaw_b,  body_height, body_pitch, body_roll]
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def ll_command(self) -> torch.Tensor:
        return self._ll_command

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tanh_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """将标量/向量 x 经 tanh 映射到 [lo, hi]。"""
        scale  = (hi - lo) / 2.0
        offset = (hi + lo) / 2.0
        return torch.tanh(x) * scale + offset
    
    @staticmethod
    def _clamp_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """线性裁剪到 [lo, hi]，不做 tanh 非线性压缩。"""
        # 角度应该是normalize到[-pi, pi]，位置应该是直接裁剪到范围
        return torch.clamp(x, min=lo, max=hi)

    def _capture_default_ee_pose(self, env_ids: torch.Tensor):
        """记录指定 env 在 body 系下的当前 EE 位姿，作为后续增量命令的基准。"""
        if env_ids.numel() == 0:
            return

        root_quat_w = self.robot.data.root_quat_w[env_ids]            # (n, 4)
        root_pos_w  = self.robot.data.root_pos_w[env_ids]              # (n, 3)
        ee_pos_w    = self.robot.data.body_pos_w[env_ids, self._ee_body_idx]   # (n, 3)
        ee_quat_w   = self.robot.data.body_quat_w[env_ids, self._ee_body_idx]  # (n, 4)

        root_quat_inv = math_utils.quat_inv(root_quat_w)
        ee_pos_b  = math_utils.quat_apply(root_quat_inv, ee_pos_w - root_pos_w)
        ee_quat_b = math_utils.quat_mul(root_quat_inv, ee_quat_w)

        self._default_ee_pos_b[env_ids]  = ee_pos_b
        self._default_ee_quat_b[env_ids] = ee_quat_b
        self._default_ee_pose_initialized[env_ids] = True

    def recalibrate(self, env_ids: torch.Tensor | None = None):
        """把标定基准重锚到机器人当前状态（EE 位姿 + 机身高度/pitch/roll）。

        ``absolute_commands=True`` 时，每次按 B（重新标定）或环境 reset 后调用：
        之后"手柄偏移 = 0"时，目标正好等于机器人当前位姿，手柄偏移精确映射为目标偏移，
        不再做滚动累加。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device)
        if env_ids.numel() == 0:
            return

        # EE 基准（body 系位置 + 四元数）
        self._capture_default_ee_pose(env_ids)

        # 机身基准：height 与 WBC 奖励同约定（相对足端高度），pitch/roll 取 root 欧拉角
        feet_cfg = SceneEntityCfg("robot", body_names=".*wheel")
        feet_cfg.resolve(self._env.scene)
        height = compute_base_height_rel_to_feet(
            self._env, SceneEntityCfg("robot"), feet_cfg
        )[env_ids]
        roll, pitch, _ = math_utils.euler_xyz_from_quat(
            self.robot.data.root_quat_w[env_ids]
        )
        self._default_body_pose[env_ids, 0] = height
        self._default_body_pose[env_ids, 1] = pitch
        self._default_body_pose[env_ids, 2] = roll

    def _reset_default_body_pose(self, env_ids: torch.Tensor):
        """reset 时把身体姿态默认值恢复为初始值 (0.51, 0, 0)。"""
        if env_ids.numel() == 0:
            return
        self._default_body_pose[env_ids, 0] = 0.51
        self._default_body_pose[env_ids, 1] = 0.0
        self._default_body_pose[env_ids, 2] = 0.0
    # ------------------------------------------------------------------
    # process_actions  （核心：直接映射，无累积）
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        r = self.cfg.low_level_command_ranges

        uninit_ids = (~self._default_ee_pose_initialized).nonzero(as_tuple=False).squeeze(-1)
        if uninit_ids.numel() > 0:
            self._capture_default_ee_pose(uninit_ids)

        # ── 1. 底盘速度（绝对量） ──────────────
        self._ll_command[:, 0] = self._clamp_range(actions[:, 0], r.lin_vel_x[0], r.lin_vel_x[1])
        self._ll_command[:, 1] = self._clamp_range(actions[:, 1], r.lin_vel_y[0], r.lin_vel_y[1])
        self._ll_command[:, 2] = self._clamp_range(actions[:, 2], r.ang_vel_z[0], r.ang_vel_z[1])

        # ── 4(提前). 机体姿态：先叠加增量/偏移，再对叠加后的绝对值 clamp ──────
        delta_body_pose = torch.stack([
            actions[:, 9],
            actions[:, 10],
            actions[:, 11],
        ], dim=-1)  # (N, 3) 增量/偏移：height, pitch, roll

        body_pose_raw = self._default_body_pose + delta_body_pose

        body_pose = torch.stack([
            self._clamp_range(body_pose_raw[:, 0], r.target_height[0], r.target_height[1]),
            self._clamp_range(body_pose_raw[:, 1], r.target_pitch[0],  r.target_pitch[1]),
            self._clamp_range(body_pose_raw[:, 2], r.target_roll[0],   r.target_roll[1]),
        ], dim=-1)

        # ── 2. EE 位置：先叠加增量，再对叠加后的绝对位置 clamp ──────
        delta_pos_b = torch.stack([
            actions[:, 3],
            actions[:, 4],
            actions[:, 5],
        ], dim=-1)  # 原始增量，先不 clamp

        ee_pos_b_raw = self._default_ee_pos_b + delta_pos_b

        # 用"本步"的机身目标高度反解 EE 在 body 系下的动态可达区间
        current_body_height = body_pose[:, 0]  # (N,)
 
        # 把“离地间隙 / 相对身体最大可达高度”反解到 body 系下的动态区间
        ee_z_lo_dyn = r.ee_ground_clearance - current_body_height
        ee_z_hi_dyn = r.ee_max_reach_above_ground - current_body_height
 
        # 再与固定的 body 系区间 r.ee_pos_z 取交集，得到最终上下界
        ee_z_lo = torch.maximum(
            torch.full_like(current_body_height, r.ee_pos_z[0]), ee_z_lo_dyn
        )
        ee_z_hi = torch.minimum(
            torch.full_like(current_body_height, r.ee_pos_z[1]), ee_z_hi_dyn
        )
        # 极端情况下（身体高度超出常规范围）可能出现 lo > hi，排序兜底避免异常/报错
        ee_z_lo, ee_z_hi = torch.minimum(ee_z_lo, ee_z_hi), torch.maximum(ee_z_lo, ee_z_hi)
 

        ee_pos_b = torch.stack([
            self._clamp_range(ee_pos_b_raw[:, 0], r.ee_pos_x[0], r.ee_pos_x[1]),
            self._clamp_range(ee_pos_b_raw[:, 1], r.ee_pos_y[0], r.ee_pos_y[1]),
            torch.clamp(ee_pos_b_raw[:, 2], min=ee_z_lo, max=ee_z_hi),
        ], dim=-1)  # (N, 3) 对结果 clamp，保证真正意义上的工作空间限制
        self._ll_command[:, 3:6] = ee_pos_b

        # ── 3. EE 姿态：四元数乘法叠加（相对标定基准/上一步 default）──
        ee_droll_b  = -actions[:, 6]
        ee_dpitch_b = actions[:, 7]
        ee_dyaw_b   = actions[:, 8]

        delta_quat_b = math_utils.quat_from_euler_xyz(ee_droll_b, ee_dpitch_b, ee_dyaw_b)
        ee_quat_b = math_utils.quat_mul(self._default_ee_quat_b, delta_quat_b)
        self._ll_command[:, 6:10] = ee_quat_b

        self._ll_command[:, 10:13] = body_pose

        # ── 5. 基准更新策略 ──
        #   absolute_commands=True（VR 遥操）：_default_* 是"标定基准"，只在
        #   recalibrate()/reset 时更新，本步不滚动 → 手柄静止时目标恒定，
        #   微小偏移不会被积分放大到极限。
        #   absolute_commands=False（键盘遥操）：保持"相对上一步"的滚动积分语义。
        if not self.cfg.absolute_commands:
            self._default_ee_pos_b  = ee_pos_b
            self._default_ee_quat_b = ee_quat_b
            self._default_body_pose = body_pose
        
    # ------------------------------------------------------------------
    # apply_actions
    # ------------------------------------------------------------------

    def apply_actions(self):
        # episode reset 时清空 low-level actions（无需重置增量缓存）
        if hasattr(self._env, "episode_length_buf"):
            reset_ids = (self._env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.low_level_leg_actions[reset_ids, :]   = 0
                self.low_level_wheel_actions[reset_ids, :] = 0
                self.low_level_ee_actions[reset_ids, :]    = 0
                if self.cfg.absolute_commands:
                    # VR 绝对目标语义：reset 后把标定基准重锚到初始位姿
                    self.recalibrate(reset_ids)
                else:
                    # 键盘增量语义：保持原有重置行为
                    self._capture_default_ee_pose(reset_ids)
                    self._reset_default_body_pose(reset_ids)

        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            policy_output = self.policy(low_level_obs)
            self.low_level_leg_actions[:]   = policy_output[:, :self._joint_pos_dim]
            self.low_level_wheel_actions[:] = policy_output[:, self._joint_pos_dim:self._joint_pos_dim + self._wheel_vel_dim]
            self.low_level_ee_actions[:]    = policy_output[:, self._joint_pos_dim + self._wheel_vel_dim:self._joint_pos_dim + self._wheel_vel_dim + self._ee_ik_dim]

            # self._ee_command_term.pose_command_w[:] = self._ll_command[:, 3:10]
            self._ee_command_term.pose_command_b[:] = self._ll_command[:, 3:10]

            self._joint_pos_action_term.process_actions(self.low_level_leg_actions)
            self._wheel_vel_action_term.process_actions(self.low_level_wheel_actions)
            self._ee_ik_action_term.process_actions(self.low_level_ee_actions)
            self._counter = 0

        self._joint_pos_action_term.apply_actions()
        self._wheel_vel_action_term.apply_actions()
        self._ee_ik_action_term.apply_actions()
        self._counter += 1

    # ------------------------------------------------------------------
    # Debug visualization（与原版相同，不赘述）
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "ee_goal_visualizer"):
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
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.ll_command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        ee_goal_pos  = self.ll_command[:, 3:6]
        ee_goal_quat = self.ll_command[:, 6:10]
        valid_mask = torch.norm(ee_goal_quat, dim=-1) > 0.1
        if valid_mask.any():
            ee_goal_quat_norm = torch.nn.functional.normalize(ee_goal_quat, dim=-1)
            ee_marker_scale = torch.tensor([[0.3, 0.3, 0.3]], device=self.device).expand(self.num_envs, -1)
            self.ee_goal_visualizer.visualize(ee_goal_pos, ee_goal_quat_norm, ee_marker_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class TeleopLLActionCfg(ActionTermCfg):
    """Configuration for pre-trained pick WBC action term (absolute-command version)."""

    class_type: type[ActionTerm] = TeleopLLAction

    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_decimation: int = 4
    low_level_leg_actions: ActionTermCfg = MISSING
    low_level_wheel_actions: ActionTermCfg = MISSING
    low_level_ee_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    ee_command_name: str = "ee_pose"
    debug_vis: bool = False
    ee_body_name: str = "gripper_base"
    absolute_commands: bool = False
    """True：把 raw_actions 当作"相对标定基准的偏移"，目标 = 标定基准 + 偏移，不滚动累加（VR 遥操用）。
    False：保持"相对上一步"的滚动积分增量语义（键盘遥操用）。"""

    # delta_* 字段全部移除

    @configclass
    class LowLevelCommandRanges:
        lin_vel_x: tuple[float, float] = (-5.0,  5.0)
        lin_vel_y: tuple[float, float] = (-1.0,  1.0)
        ang_vel_z: tuple[float, float] = (-1.0,  1.0)
        ee_pos_x:  tuple[float, float] = ( 0.2,  0.8)
        ee_pos_y:  tuple[float, float] = (-0.4,  0.4)
        ee_pos_z:  tuple[float, float] = (-0.6,  0.6)
        #   ee_ground_clearance: 允许 EE 距离脚下地面的最小高度（避免蹭地）
        #   ee_max_reach_above_ground: 允许 EE 距离脚下地面的最大高度（机械臂实际可达上限）
        ee_ground_clearance: float = 0.135
        ee_max_reach_above_ground: float = 1.2
        ee_pitch:  tuple[float, float] = (-math.pi / 2, 0.0)
        target_height: tuple[float, float] = (0.33, 0.6)
        target_pitch:  tuple[float, float] = (-0.35, 0.35)
        target_roll:   tuple[float, float] = (-0.25, 0.25)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()
