# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg, ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20RoughEnvCfg  # noqa: F401, F403
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_PIPER_CFG

LOW_LEVEL_ENV_CFG = DeeproboticsM20RoughEnvCfg()

    
    

@configclass
class HighLevelSceneCfg(MySceneCfg):
    """高层环境的场景配置，继承自低层环境，添加了相机和激光雷达传感器。"""
    
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.055),   # 初始位置（相对于 env origin）
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 桌子（无物理交互的静态物体用 AssetBaseCfg）
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )
    # ---------------------------------------------------------------
    # 前视 RGB-D 相机，挂载在机械臂的camera_link 上
    # ---------------------------------------------------------------
    arm_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.1,                      # 10 Hz
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,               # 焦距
            focus_distance=400.0,            # 对焦距离
            f_stop=0.0,                      # 光圈值，0.0表示为理想针孔模型
            horizontal_aperture=20.955,      # 水平视场，单位为度，根据焦距和传感器尺寸计算得出
            clipping_range=(0.1, 20.0),      # 近裁剪面和远裁剪面，单位为米
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),              # 相机位于机械臂前端
            rot=(1.0, 0.0, 0.0, 0.0),         # 四元数(w,x,y,z)
            convention="ros",
        ),
    )

    # ---------------------------------------------------------------
    # 激光雷达，使用 RayCaster + LidarPatternCfg 实现
    # 模拟 16 线机械旋转式 LiDAR（水平 360°，垂直 ±15°）
    # ---------------------------------------------------------------
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.3),                # base 上方 0.3 m
        ),
        attach_yaw_only=True,                   # 只跟随偏航角，模拟真实旋转式 LiDAR
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.4,
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        max_distance=20.0,
    )

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"D:\\nvidia-isaac-sim\\loco-manip-unified-rl-agent\\logs\\rsl_rl\\deeprobotics_m20_flat\\2026-03-04_23-02-58\\exported\\policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_wheel_actions=LOW_LEVEL_ENV_CFG.actions.joint_vel,
        low_level_ee_actions=LOW_LEVEL_ENV_CFG.actions.ee_ik,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 基座线速度
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 基座角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 投影重力，用于姿态感知
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 所有关节位置
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 所有关节速度
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # 上一次行动
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # EE在世界系下的位姿
        ee_pose_w = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # arm_camera_embedding = ObsTerm(
        #     func=mdp.camera_feature_embedding,
        #     params={
        #         "sensor_name":  "arm_camera",
        #         "data_type":    "rgb",
        #         "encoder_name": "dinov3_small",   # 换成 "dinov3_base" / "clip_vit" 等
        #         "image_size":   (224, 224),
        #     },
        # )

        # lidar_embedding   = ObsTerm(
        #     func=mdp.lidar_feature_embedding,
        #     params={
        #         "sensor_name":  "lidar",
        #         "encoder_name": "mini_pointnet",  # 换成 "my_custom_lidar_encoder" 等
        #         "max_points":   1024,
        #         "max_range":    20.0,
        #         "min_range":    0.1,
        #     },
        # )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-4.0)
    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.5,
    #     params={"std": 2.0, "command_name": "pose_command"},
    # )
    # position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.5,
    #     params={"std": 0.2, "command_name": "pose_command"},
    # )
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-0.2,
    #     params={"command_name": "pose_command"},
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HighLevelEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the high level environment."""

    # environment settings
    scene: HighLevelSceneCfg = HighLevelSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()


    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
    
    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


class HighLevelEnvCfg_PLAY(HighLevelEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
