# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层观测组：policy 83 / critic 86 / history 700 / privileged 54。

维度与 Anchor（M20）**逐位一致**（AS2W 关节数同为 24），见交接单 §9.3：

================  ==========================================================
组                 组成
================  ==========================================================
policy (83)       ang_vel(3) + gravity(3) + vel_cmd(3) + joint_pos(24)
                  + joint_vel(24) + actions(16) + ee_goal(7) + body_pose_cmd(3)
critic (86)       base_lin_vel(3) + policy 的全部 8 项（不含噪声 / latent）
history (700)     history_single_step_obs 70 维 × 10 步
privileged (54)   privileged_env_params_v4
================  ==========================================================

只有 policy 开 ``enable_corruption``。
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
    all_joint_cfg,
    wheel_joint_cfg,
)


_PROFILE = AS2W_PIPERX_PROFILE


@configclass
class ObservationsCfg:
    """四个观测组。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测（83 维，带噪声）。"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 宽度仍是 24（只把轮列置零，不删列）。
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={"asset_cfg": all_joint_cfg(), "wheel_asset_cfg": wheel_joint_cfg()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": all_joint_cfg()},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        ee_goal = ObsTerm(
            func=mdp.ee_goal_local,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-3.0, 3.0),
            scale=1.0,
        )
        body_pose_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """值函数观测（86 维，无噪声；特权 54 维由 runner 拼在后面）。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={"asset_cfg": all_joint_cfg(), "wheel_asset_cfg": wheel_joint_cfg()},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": all_joint_cfg()},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        ee_goal = ObsTerm(
            func=mdp.ee_goal_local,
            params={"command_name": "ee_pose"},
            clip=(-3.0, 3.0),
            scale=1.0,
        )
        body_pose_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        """Adaptation 历史窗口：70 维 × 10 步 = 700。"""

        history_obs = ObsTerm(
            func=mdp.history_single_step_obs,
            history_length=10,
            flatten_history_dim=True,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """教师特权向量（54 维，全通道 O(1)）。"""

        env_params = ObsTerm(
            func=mdp.privileged_env_params_v4,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_body_name": _PROFILE.body_groups.base,
                "gripper_body_name": _PROFILE.body_groups.ee,
                "foot_body_regex": _PROFILE.foot_body_regex,
                "included_actuators": tuple(_PROFILE.included_actuators),
                "wheel_joint_regex": _PROFILE.wheel_joint_regex,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    history: HistoryCfg = HistoryCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()