# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 遥操作环境（history 低层策略回放）。

结构与 ``Isaac-M20-Piper-Teleop-History-v0`` 相同：

* 高层 13 维动作 = ``TeleopLLAction``（12 维 chassis / EE / body）+ 1 维夹爪；
* ``TeleopLLAction`` 把高层动作翻译成低层命令，跑导出的 history 低层策略；
* 低层策略：``logs/rsl_rl/history_anchor_as2w_piperx/.../exported_deploy/policy.pt``
  （policy_obs 83 + history 10x70 -> action 16）。

与 M20 的差异只在机器人常量：

====================  =============  =================
项                     AS2W           M20
====================  =============  =================
足端 body 正则         ``.*_foot``    ``.*wheel``
轮半径                 0.0774 m       0.09 m
默认站高               0.4466 m       0.51 m
机身高度范围           0.33-0.46 m    0.33-0.55 m
夹爪行程               +/-0.05 rad    +/-0.035 rad
====================  =============  =================

这些值全部来自 ``as2w_piperx/profile.py`` 或资产实测，不在本文件重复硬编码。
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
import rl_training.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from rl_training.tasks.manager_based.locomotion.highlevel.mdp.low_level_replay import (
    resolve_policy_path,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.env_cfg import (
    AS2WPiperXFlatEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)


_PROFILE = AS2W_PIPERX_PROFILE

# 低层 checkpoint 的**部署态**导出（含 history encoder）。
# 环境变量覆盖：RL_TRAINING_LOW_LEVEL_POLICY_AS2W_TELEOP_HISTORY=<abs/path/policy.pt>
_LOW_LEVEL_TELEOP_POLICY = resolve_policy_path(
    "logs/rsl_rl/history_anchor_as2w_piperx/2026-11-26_21-45-28_remote4096_20k/"
    "exported_deploy/policy.pt",
    key="as2w_teleop_history",
)

# 低层 cfg 只作为「动作项 / 观测模板 / 命令项」的唯一来源，不继承 M20 配置。
_low_level_env_cfg = AS2WPiperXFlatEnvCfg()


def _as2w_teleop_command_ranges() -> mdp.TeleopLLActionCfg.LowLevelCommandRanges:
    """遥操高层命令的物理范围（body 系 EE + 机身姿态）。

    EE 的可达范围用 AS2W 实测工作空间（``profile.ee_workspace``，以 arm_base_link
    为原点的球坐标）换算到 root 系后的保守包络；``ee_ground_clearance`` /
    ``ee_max_reach_above_ground`` 再按当前机身高度做动态裁剪。
    """
    return mdp.TeleopLLActionCfg.LowLevelCommandRanges(
        lin_vel_x=(-5.0, 5.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        ee_pos_x=(-0.05, 0.65),
        ee_pos_y=(-0.50, 0.50),
        ee_pos_z=(-0.55, 0.70),
        ee_ground_clearance=0.10,
        ee_max_reach_above_ground=1.05,
        ee_pitch=(-math.pi / 2.0, 0.0),
        # 默认站高 0.4466 m 略高于训练命令上界 0.45，留 1 cm 余量避免 reset 首帧
        # 就被 clamp 到上界、导致「按 C 升不起来」。
        target_height=(_PROFILE.base_height_range[0], 0.46),
        target_pitch=(-0.35, 0.35),
        target_roll=(-0.25, 0.25),
    )


@configclass
class AS2WPiperXTeleopActionsCfg:
    """13 维高层动作：12 维 TeleopLLAction + 1 维夹爪。"""

    pre_trained_pick_action = mdp.TeleopLLActionCfg(
        asset_name="robot",
        policy_path=_LOW_LEVEL_TELEOP_POLICY,
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_ee_actions=_low_level_env_cfg.actions.ee_ik,
        low_level_observations=_low_level_env_cfg.observations.policy,
        low_level_command_ranges=_as2w_teleop_command_ranges(),
        ee_body_name=_PROFILE.body_groups.ee,
        feet_body_names=_PROFILE.foot_body_regex,
        feet_radius=_PROFILE.wheel_radius,
        default_body_height=_PROFILE.default_base_height,
        debug_vis=False,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_PROFILE.joint_groups.gripper),
        # AS2W Piper X 夹爪行程：joint1 [0, 0.05] / joint2 [-0.05, 0]。
        open_command_expr={"gripper_joint1": 0.05, "gripper_joint2": -0.05},
        close_command_expr={"gripper_joint1": 0.0, "gripper_joint2": 0.0},
    )


@configclass
class AS2WPiperXTeleopObservationsCfg:
    """遥操不训练，只保留最小观测组，避免无意义的噪声/历史计算。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=velocity_mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=velocity_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=velocity_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)
            },
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=velocity_mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class AS2WPiperXTeleopRewardsCfg:
    """遥操不训练；留空即可（ManagerBasedRLEnv 允许空奖励组）。"""


@configclass
class AS2WPiperXTeleopEventsCfg:
    """只保留复位事件，不做随机化/推挤（遥操要可重复、可重锚）。"""

    reset_base = EventTerm(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=velocity_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class AS2WPiperXTeleopTerminationsCfg:
    """摔倒/过低时复位；超时由遥操脚本关闭。"""

    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    bad_orientation_2 = DoneTerm(
        func=velocity_mdp.bad_orientation_2,
        params={"limit_angle": 0.8},
    )
    root_height_below_minimum = DoneTerm(
        func=velocity_mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": _PROFILE.termination_height,
        },
    )


@configclass
class AS2WPiperXTeleopEnvCfg(AS2WPiperXFlatEnvCfg):
    """AS2W-PiperX 遥操任务（history 低层策略）。"""

    actions: AS2WPiperXTeleopActionsCfg = AS2WPiperXTeleopActionsCfg()
    observations: AS2WPiperXTeleopObservationsCfg = AS2WPiperXTeleopObservationsCfg()
    rewards: AS2WPiperXTeleopRewardsCfg = AS2WPiperXTeleopRewardsCfg()
    events: AS2WPiperXTeleopEventsCfg = AS2WPiperXTeleopEventsCfg()
    terminations: AS2WPiperXTeleopTerminationsCfg = AS2WPiperXTeleopTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 遥操：高层与低层同频（50 Hz 决策），去掉 10x 高层延迟。
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # 遥操不推进课程；PLAY/训练脚本也不会误用课程项。
        self.curriculum = None