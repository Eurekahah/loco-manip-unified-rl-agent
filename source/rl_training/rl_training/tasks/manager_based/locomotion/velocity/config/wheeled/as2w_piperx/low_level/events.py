# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层事件组：startup / reset / interval 三档（交接单 §9.6）。

含启动期 ``check_policy_layout``（打印 policy/critic/history/privileged/action 布局）。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)


_PROFILE = AS2W_PIPERX_PROFILE


@configclass
class EventCfg:
    """随机化 / 自检 / 扰动。"""

    # ---- startup ---- #
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.35, 1.5),
            "dynamic_friction_range": (0.35, 1.5),
            "restitution_range": (0.0, 0.7),
            "num_buckets": 1024,
        },
    )
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=r"^(?!.*base_link).*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_end_effector = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_PROFILE.body_groups.ee),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_PROFILE.body_groups.base),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_PROFILE.body_groups.base),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )
    check_policy_layout = EventTerm(
        func=mdp.check_policy_layout,
        mode="startup",
        params={"group_name": "policy"},
    )

    # ---- reset ---- #
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_PROFILE.body_groups.base),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # ---- interval ---- #
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-2.0, 2.0), "y": (-1.0, 1.0), "yaw": (-0.52, 0.52)}},
    )