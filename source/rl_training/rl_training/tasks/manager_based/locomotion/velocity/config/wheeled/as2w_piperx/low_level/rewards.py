# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层奖励组（**只定义启用的 23 项**，权重显式）。

不定义任何零权重项（没有 ``arm_joint_*`` / EE 跟踪奖励），也不调用
``disable_zero_weight_rewards()``。参数基准见交接单 §9.5，机器人相关常量
（轮半径 / 接触阈值 / 默认站高 / 关节分组 / 镜像对）全部来自 ``profile.py``。
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
    leg_joint_cfg,
    wheel_body_cfg,
    wheel_joint_cfg,
)


_PROFILE = AS2W_PIPERX_PROFILE


def _hip_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(_PROFILE.joint_groups.hip), preserve_order=True)


def _thigh_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(_PROFILE.joint_groups.thigh), preserve_order=True)


def _calf_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(_PROFILE.joint_groups.calf), preserve_order=True)


def _joint_pos_penalty_cfg(joint_cfg: SceneEntityCfg) -> dict:
    return {
        "command_name": "base_velocity",
        "pose_command_name": "body_pose",
        "asset_cfg": joint_cfg,
        "stand_still_scale": 5.0,
        "velocity_threshold": 0.5,
        "command_threshold": 0.1,
        "default_height": _PROFILE.default_base_height,
    }


@configclass
class RewardsCfg:
    """低层奖励（23 项，Anchor 口径）。"""

    # -- 任务 -- #
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(2.0)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    upward = RewTerm(func=mdp.upward, weight=0.08)
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_PROFILE.foot_body_regex),
            "command_name": "base_velocity",
        },
    )

    # -- 机身姿态跟踪 -- #
    body_height_tracking = RewTerm(
        func=mdp.body_height_tracking,
        weight=0.001,
        params={
            "command_name": "body_pose",
            "std": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": wheel_body_cfg(),
            "feet_radius": _PROFILE.wheel_radius,
        },
    )
    body_pitch_tracking = RewTerm(
        func=mdp.body_pitch_tracking,
        weight=0.001,
        params={"command_name": "body_pose", "std": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    body_roll_tracking = RewTerm(
        func=mdp.body_roll_tracking,
        weight=0.001,
        params={"command_name": "body_pose", "std": 0.04, "asset_cfg": SceneEntityCfg("robot")},
    )

    # -- 静止伫立（零速命令）-- #
    stand_still_vel = RewTerm(
        func=mdp.stand_still_vel_l2,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "yaw_weight": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stand_still_wheel_vel = RewTerm(
        func=mdp.stand_still_wheel_vel_l2,
        weight=-5e-5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": wheel_joint_cfg(),
        },
    )

    # -- 关节姿态（按体型分组的 stand-still 缩放）-- #
    hip_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_wbc, weight=-0.4, params=_joint_pos_penalty_cfg(_hip_cfg())
    )
    thigh_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_wbc, weight=-0.1, params=_joint_pos_penalty_cfg(_thigh_cfg())
    )
    calf_joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_wbc, weight=-0.1, params=_joint_pos_penalty_cfg(_calf_cfg())
    )

    # -- 左右镜像（只做 FL<->FR、RL<->RR）-- #
    joint_mirror = RewTerm(
        func=mdp.joint_mirror_signed,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": _PROFILE.mirror_joints(),
            "mirror_signs": _PROFILE.mirror_signs(),
            "joint_kind_patterns": _PROFILE.kind_patterns_dict(),
        },
    )

    # -- 平滑 / 能耗 / 限位 -- #
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.5e-5, params={"asset_cfg": leg_joint_cfg()}
    )
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2e-7, params={"asset_cfg": leg_joint_cfg()})
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1e-7, params={"asset_cfg": wheel_joint_cfg()}
    )
    joint_power = RewTerm(func=mdp.joint_power, weight=-2e-5, params={"asset_cfg": leg_joint_cfg()})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": leg_joint_cfg()}
    )

    # -- 接触 -- #
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_PROFILE.non_wheel_body_regex),
            "threshold": 1.0,
        },
    )
    undesired_base_link_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_PROFILE.body_groups.base),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_PROFILE.foot_body_regex),
            "threshold": _PROFILE.contact_force_threshold,
        },
    )
