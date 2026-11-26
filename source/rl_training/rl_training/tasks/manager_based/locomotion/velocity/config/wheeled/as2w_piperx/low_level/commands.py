# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层命令组：``base_velocity`` / ``ee_pose`` / ``body_pose``。

* 不含 ``arm_weight``（明确属 legacy，D-06）；
* ``base_velocity`` **完全不写**分层采样字段（``rel_zero_cmd_envs`` 等保持默认 0）；
* EE 球坐标锚点 / 碰撞盒 / ``sampled_height`` 全部来自 Phase 0 实测（``profile.py``）。
"""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets.deeprobotics import AS2W_PIPERX_URDF_PATH
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
    wheel_body_cfg,
)


_PROFILE = AS2W_PIPERX_PROFILE


@configclass
class CommandsCfg:
    """命令项：底盘速度 / 末端位姿 / 机身姿态。"""

    # ---- 底盘全向速度 ----
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=False,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    # ---- 末端位姿（高度不变系球坐标）----
    ee_pose = mdp.HeightInvariantEECommandCfg(
        asset_name="robot",
        body_name=_PROFILE.body_groups.ee,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        sampled_height=_PROFILE.ee_sampled_height,
        arm_base_link_name=_PROFILE.arm_base_link_name,
        ranges=mdp.HeightInvariantEECommandCfg.Ranges(
            p_l=(_PROFILE.ee_anchor.p_l, _PROFILE.ee_anchor.p_l),
            p_pitch=(_PROFILE.ee_anchor.p_pitch, _PROFILE.ee_anchor.p_pitch),
            p_yaw=(_PROFILE.ee_anchor.p_yaw, _PROFILE.ee_anchor.p_yaw),
            o_roll=(0.0, 0.0),
            o_pitch=(0.0, 0.0),
            o_yaw=(0.0, 0.0),
            T_traj=(1.0, 3.0),
            orn_cone_min_scale=0.1,
        ),
        # 高度不变系局部坐标：原点世界 z = sampled_height，机体在其下方 ⇒ z 为负（D-13 / DEF-066）。
        collision_lower_limits=list(_PROFILE.ee_collision_lower),
        collision_upper_limits=list(_PROFILE.ee_collision_upper),
        underground_limit=0.05,
        num_collision_check_samples=10,
        max_resample_attempts=10,
        target_blend_pos=1.0,
        target_blend_orn=1.0,
        # 启用 FK 可达性栅格（DEF-049/054），AS2W 臂较小更易饱和，第一版即开。
        urdf_path=AS2W_PIPERX_URDF_PATH,
        ee_link_name=_PROFILE.ee_link_name,
    )

    # ---- 机身姿态（高度 / pitch / roll）----
    body_pose = mdp.BodyPoseCommandCfg(
        resampling_time_range=(10.0, 10.0),
        height_range=_PROFILE.base_height_range,
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        default_height=_PROFILE.default_base_height,
        default_pitch=0.0,
        default_roll=0.0,
        rel_default_pose_envs=0.0,
        asset_cfg=SceneEntityCfg("robot"),
        feet_cfg=wheel_body_cfg(),
        feet_radius=_PROFILE.wheel_radius,
        steady_error_clip=0.15,
        debug_vis=False,
    )