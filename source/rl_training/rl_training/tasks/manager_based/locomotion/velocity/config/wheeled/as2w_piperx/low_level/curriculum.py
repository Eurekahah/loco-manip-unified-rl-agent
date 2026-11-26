# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层课程组（交接单 §9.8，共 14 项）。

* 不定义 Anchor 里为 ``None`` 的三项（``terrain_levels`` / ``command_levels`` /
  ``body_pose_height_range_s2``）；
* ``ee_goal_stages`` 的三阶段区间按 AS2W 实测锚点（l=0.436 / pitch=-0.122）重算；
* 所有函数通过 ``mdp`` 命名空间引用，机器人常量来自 ``profile.py``。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)


_PROFILE = AS2W_PIPERX_PROFILE

# EE 三阶段（p_l / p_pitch 按 AS2W 可达区间重算；姿态与 yaw 沿用 Anchor 阶梯）。
_EE_STAGE_1 = {
    "p_l": (0.39, 0.48),
    "p_pitch": (-0.35, 0.05),
    "p_yaw": (-0.45, 0.45),
    "o_roll": (-0.09, 0.09),
    "o_pitch": (-0.09, 0.09),
    "o_yaw": (0.0, 0.0),
}
_EE_STAGE_2 = {
    "p_l": (0.34, 0.50),
    "p_pitch": (-0.55, 0.15),
    "p_yaw": (-0.85, 0.85),
    "o_roll": (-0.175, 0.175),
    "o_pitch": (-0.175, 0.175),
    "o_yaw": (-0.6, 0.6),
}
_EE_STAGE_3 = {
    "p_l": _PROFILE.ee_workspace[0],
    "p_pitch": _PROFILE.ee_workspace[1],
    "p_yaw": _PROFILE.ee_workspace[2],
    "o_roll": (-math.pi / 8.0, math.pi / 8.0),
    "o_pitch": (-math.pi / 8.0, math.pi / 8.0),
    "o_yaw": (-math.pi, math.pi),
}


def _override(address: str, value, num_steps: int) -> CurrTerm:
    return CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": address,
            "modify_fn": mdp.override_value,
            "modify_params": {"value": value, "num_steps": num_steps},
        },
    )


def _ramp_weight(term_name: str, weight: float, num_steps: int) -> CurrTerm:
    return CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": term_name, "weight": weight, "num_steps": num_steps},
    )


@configclass
class CurriculumCfg:
    """机身姿态 / 速度 / EE / 扰动 / 静止比例课程。"""

    # -- 机身姿态区间 -- #
    body_pose_pitch_range_s3 = _override("commands.body_pose.pitch_range", (-0.35, 0.35), 50000)
    body_pose_roll_range_s3 = _override("commands.body_pose.roll_range", (-0.25, 0.25), 50000)

    # -- 机身姿态奖励权重量级 -- #
    body_height_rew_s2 = _ramp_weight("body_height_tracking", 0.8, 25000)
    body_pitch_rew_s3 = _ramp_weight("body_pitch_tracking", 0.8, 50000)
    body_roll_rew_s3 = _ramp_weight("body_roll_tracking", 0.8, 50000)

    # -- 前向速度区间 -- #
    base_velocity_lin_vel_x_s4 = _override("commands.base_velocity.ranges.lin_vel_x", (-2.0, 2.0), 75000)
    base_velocity_lin_vel_x_s5 = _override("commands.base_velocity.ranges.lin_vel_x", (-3.0, 3.0), 100000)
    base_velocity_lin_vel_x_s6 = _override("commands.base_velocity.ranges.lin_vel_x", (-4.0, 4.0), 125000)
    base_velocity_lin_vel_x_s7 = _override("commands.base_velocity.ranges.lin_vel_x", (-5.0, 5.0), 150000)

    # -- EE 目标三阶段 -- #
    ee_goal_stages = CurrTerm(
        func=mdp.apply_range_stages,
        params={
            "command_name": "ee_pose",
            "stages": [
                {"num_steps": 25000, "ranges": _EE_STAGE_1},
                {"num_steps": 50000, "ranges": _EE_STAGE_2},
                {"num_steps": 75000, "ranges": _EE_STAGE_3},
            ],
        },
    )

    # -- 扰动放大 -- #
    disturbance_ramp = CurrTerm(
        func=mdp.apply_event_scale,
        params={
            "num_steps": 50000,
            "start_scale": 0.2,
            "spec": [
                {"term": "randomize_push_robot", "param": "velocity_range",
                 "base": {"x": (-2.0, 2.0), "y": (-1.0, 1.0), "yaw": (-0.52, 0.52)}},
                {"term": "randomize_apply_external_force_torque", "param": "force_range",
                 "base": (-10.0, 10.0)},
                {"term": "randomize_apply_external_force_torque", "param": "torque_range",
                 "base": (-10.0, 10.0)},
            ],
        },
    )

    # -- 静止比例 -- #
    standing_env_ratio_ramp = CurrTerm(
        func=mdp.ramp_command_param,
        params={
            "term_name": "base_velocity",
            "param": "rel_standing_envs",
            "start": 0.02,
            "end": 0.15,
            "num_steps": 25000,
        },
    )

    # -- 静止惩罚权重爬升 -- #
    stand_still_vel_ramp = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "stand_still_vel",
            "start_weight": -0.2,
            "end_weight": -2.0,
            "num_steps": 25000,
        },
    )
    stand_still_wheel_vel_ramp = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "stand_still_wheel_vel",
            "start_weight": -5e-5,
            "end_weight": -5e-4,
            "num_steps": 25000,
        },
    )