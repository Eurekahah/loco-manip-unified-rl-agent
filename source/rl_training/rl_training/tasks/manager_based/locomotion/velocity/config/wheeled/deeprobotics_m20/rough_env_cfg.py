# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    CurriculumCfg,
    CommandsCfg,
)
from isaaclab.controllers import DifferentialIKControllerCfg
##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG  # isort: skip
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_PIPER_CFG


@configclass
class DeeproboticsM20ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=20.0, use_default_offset=True, clip=None, preserve_order=True
    )

    # EE使用IK进行移动
    # ee_ik = mdp.DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["arm_joint[1-6]"],  # 只包含机械臂关节
    #     body_name="arm_link6",     # 末端link名
    #     debug_vis=True,
    #     controller=DifferentialIKControllerCfg(
    #         command_type="pose",       # "pose"=位姿, "position"=仅位置
    #         use_relative_mode=False,   # False=绝对位姿, True=相对增量
    #         ik_method="dls",           # "dls"(阻尼最小二乘) 或 "pinv"(伪逆)
    #     ),
    #     scale=1.0,
    # )

    # ee_ik = mdp.DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=["arm_joint[1-6]"],  # 只包含机械臂关节
    #     body_name="arm_link6",     # 末端link名
    #     debug_vis=True,
    #     controller=DifferentialIKControllerCfg(
    #         command_type="pose",       # "pose"=位姿, "position"=仅位置
    #         use_relative_mode=True,   # False=绝对位姿, True=相对增量
    #         ik_method="dls",           # "dls"(阻尼最小二乘) 或 "pinv"(伪逆)
    #     ),
    #     scale=0.1,
    # )

    # IK由command直接驱动，不占policy的action_dim
    ee_ik = mdp.CommandDrivenIKActionCfg(
        asset_name="robot",
        joint_names=["arm_joint[1-6]"],
        body_name="arm_link6",
        command_name="ee_pose",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,  # 接收绝对位姿
            ik_method="dls",
            ik_params={"lambda_val": 0.01}
        ),
        scale=1.0,
    )

    # gripper_action = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["arm_joint7", "arm_joint8"],
    #     open_command_expr={"arm_joint7": 0.04, "arm_joint8": 0.04},
    #     close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
    # )


@configclass
class DeeproboticsM20RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class DeeproboticsM20RewardsWithArmsCfg(DeeproboticsM20RewardsCfg):
    """Reward terms for the MDP with arm-related rewards."""
    # EE
    # 机械臂关节偏离默认收纳姿态的惩罚（不执行任务时）
    arm_joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_joint.*"),
        },
    )

    # 1. 位置跟踪（密集）
    arm_ee_pos_tracking = RewTerm(
        func=mdp.ee_position_tracking,
        weight=2.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "arm_link6",
            "std": 0.15,
            "arm_weight_command_name": "arm_weight", 
        },
    )

    # 2. 姿态跟踪（密集）
    arm_ee_ori_tracking = RewTerm(
        func=mdp.ee_orientation_tracking,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "arm_link6",
            "std": 0.5,
            "arm_weight_command_name": "arm_weight", 
        },
    )

    # 3. 到达目标稀疏奖励
    arm_ee_goal_reached = RewTerm(
        func=mdp.ee_goal_reached,
        weight=5.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "arm_link6",
            "pos_threshold": 0.05,
            "angle_threshold": 0.2,
            "arm_weight_command_name": "arm_weight", 
        },
    )


    # 5. 关节力矩惩罚
    arm_joint_torque = RewTerm(
        func=mdp.arm_joint_torque_penalty,
        weight=0.0,  # -1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": "arm_weight", 
        },
    )

    # 6. 关节速度惩罚
    arm_joint_vel = RewTerm(
        func=mdp.arm_joint_velocity_penalty,
        weight=0.0,  # -1e-3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": "arm_weight", 
        },
    )

    # 7. 关节加速度惩罚（可选）
    arm_joint_acc = RewTerm(
        func=mdp.arm_joint_acceleration_penalty,
        weight=0.0,  # -1e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": "arm_weight", 
        },
    )

@configclass
class DeeproboticsM20CurriculumsCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""
    arm_weight_curriculum = CurrTerm(
        func=mdp.advance_arm_weight,
        params={
            "max_iterations": 5000,       # 和 RunnerCfg 保持一致
            "num_steps_per_env": 24,
            "ramp_start_frac": 0.0,
            "ramp_end_frac": 0.5,         # iteration=2500 时 max_weight=1.0
            "max_target": 1.0,
            "min_target": 0.8,
            "min_start_frac": 0.5,        # max_weight>0.5 后才推 min
            "initial_max_weight": 0.0,
            "initial_min_weight": 0.0,
        }
    )

@configclass
class DeeproboticsM20CommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""

    # 基座角速度和线速度
    # 被注释的部分是全向运动命令采样
    # base_velocity = mdp.UniformThresholdVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )
    # 训练全身运控时，只有x方向的前向速度
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.9), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

    # EE位姿
    ee_pose = mdp.HeightInvariantEECommandCfg(
        asset_name="robot",
        body_name="arm_link6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        sampled_height=0.6,  # 采样坐标系的固定高度
        arm_base_link_name="arm_base",  # 采样坐标系xy位置
        ranges=mdp.HeightInvariantEECommandCfg.Ranges(
            # 球坐标位置采样范围
            p_l= (0.4, 0.7),           # 半径 l
            p_pitch= (-1, 2*math.pi/5),   # pitch p
            p_yaw = (-3*math.pi/5, 3*math.pi/5),     # yaw y
            # 姿态采样范围
            o_roll = (-math.pi / 4, math.pi / 4),
            o_pitch =(-math.pi / 4, math.pi / 4),
            o_yaw = (-math.pi, math.pi),
            # 插值时间间隔采样范围
            T_traj = (1.0, 3.0),
            T_hold = (0.5, 2.0)
        ),
    )

    arm_weight = mdp.ArmWeightCommandCfg(
        resampling_time_range=(10.0, 10.0),
        init_max_weight=0.0,
    )



@configclass
class DeeproboticsM20RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: DeeproboticsM20ActionsCfg = DeeproboticsM20ActionsCfg()
    rewards: DeeproboticsM20RewardsWithArmsCfg = DeeproboticsM20RewardsWithArmsCfg()
    curriculum: DeeproboticsM20CurriculumsCfg = DeeproboticsM20CurriculumsCfg()
    commands: DeeproboticsM20CommandsCfg = DeeproboticsM20CommandsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_wheel"

    # fmt: off
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
        "arm_joint7", "arm_joint8",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # self.scene.robot = DEEPROBOTICS_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = DEEPROBOTICS_M20_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hipx_joint": 0.125, '^(?!.*_hipx_joint)(?!.*arm_joint).*': 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names 
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
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
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.16)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.35, 1.5]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.35, 1.5]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.7]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -10.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.50
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.1
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.03
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_base_link_contacts.weight = -3.0
        self.rewards.undesired_base_link_contacts.params["sensor_cfg"].body_names = [self.base_link_name]

        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0 # 1.8
        self.rewards.track_ang_vel_z_exp.weight = 1.0 # 1.2

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        
        self.rewards.upward.weight = 0.08

        # Arms


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsM20RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.9)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)