# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层动作项（总 16 维）。

* ``joint_pos``：12 个非轮腿关节位置目标（scale: hip 0.125 / thigh 0.25 / calf 0.25）；
* ``joint_vel``：4 个 ``*_foot`` 轮关节速度目标（scale 5.0）；
* ``ee_ik``   ：机械臂 IK，由 ``ee_pose`` 命令直接驱动，**动作维度 0**。

12 + 4 + 0 = 16，与 Anchor（M20）逐位一致。机器人常量（关节名单 / 缩放）只从
``profile.py`` 读取。
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)


_PROFILE = AS2W_PIPERX_PROFILE


def _leg_scale_dict() -> dict[str, float]:
    """腿关节缩放字典：键是**完整正则**（IsaacLab 用 ``re.fullmatch`` 匹配关节名）。"""
    return {f".*_{kind}": float(scale) for kind, scale in _PROFILE.leg_action_scale}


@configclass
class ActionsCfg:
    """动作项：12 腿位置 + 4 轮速度 + IK 臂（0 维）。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_PROFILE.joint_groups.legs),
        scale=_leg_scale_dict(),
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(_PROFILE.joint_groups.wheels),
        scale=float(_PROFILE.wheel_action_scale),
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-100.0, 100.0)},
    )

    # IK 由 ``ee_pose`` 命令驱动，不占 policy 的 action 维度（action_dim == 0）。
    ee_ik = mdp.CommandDrivenIKActionCfg(
        asset_name="robot",
        joint_names=list(_PROFILE.joint_groups.arm),
        body_name=_PROFILE.body_groups.ee,
        command_name="ee_pose",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        scale=1.0,
        debug_vis=False,
    )
