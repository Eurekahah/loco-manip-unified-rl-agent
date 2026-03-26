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
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns, TiledCameraCfg
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

    # 全局唯一的仓库背景，固定路径不带 {ENV_REGEX_NS}
    warehouse: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # dex_cube
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",  
            scale=(1.0, 1.0, 1.0),          # 缩小到半尺寸，更轻巧
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.455),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 桌子（无物理交互的静态物体用 AssetBaseCfg）
    # table: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.8, 1.2, 0.05),          # 桌面尺寸 (x, y, z)
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.6, 0.4, 0.2),  # 木色
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(2.0, 0.0, 0.5),   # 桌面中心高度 0.5m
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )
    table = AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    # ---------------------------------------------------------------
    # 前视 RGB-D 相机，挂载在机械臂的camera_link 上
    # ---------------------------------------------------------------
    arm_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/abase",
        update_period=0.1,                      # 10 Hz
        height=120,
        width=160,
        data_types=["rgb", "depth"],
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,               # 焦距
            focus_distance=400.0,            # 对焦距离
            f_stop=0.0,                      # 光圈值，0.0表示为理想针孔模型
            horizontal_aperture=20.955,      # 水平视场，单位为度，根据焦距和传感器尺寸计算得出
            clipping_range=(0.1, 3.0),      # 近裁剪面和远裁剪面，单位为米
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),              # 相机位于机械臂前端
            rot=(0.5, -0.5, 0.5, -0.5),         # 四元数(w,x,y,z)
            convention="ros",
        ),
    )

    nav_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/bbase",
        update_period=0.1,                      # 10 Hz
        height=120,
        width=160,
        data_types=["rgb", "depth"],
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,               # 焦距
            focus_distance=400.0,            # 对焦距离
            f_stop=0.0,                      # 光圈值，0.0表示为理想针孔模型
            horizontal_aperture=20.955,      # 水平视场，单位为度，根据焦距和传感器尺寸计算得出
            clipping_range=(0.1, 20.0),      # 近裁剪面和远裁剪面，单位为米
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),              # 相机位于base_link前端
            rot=(0.5, -0.5, 0.5, -0.5),       # 四元数(w,x,y,z)
            convention="ros",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.2, 0.2),  "yaw": (-0.393, 0.393)},
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
                "z": (0.155, 0.155),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""
    pass
    # pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
    #     asset_name="robot",
    #     policy_path=f"D:\\nvidia-isaac-sim\\loco-manip-unified-rl-agent\\logs\\rsl_rl\\deeprobotics_m20_flat\\2026-03-04_23-02-58\\exported\\policy.pt",
    #     low_level_decimation=4,
    #     low_level_leg_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
    #     low_level_wheel_actions=LOW_LEVEL_ENV_CFG.actions.joint_vel,
    #     low_level_ee_actions=LOW_LEVEL_ENV_CFG.actions.ee_ik,
    #     low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    # )


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
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   # 拼成一个向量送入 MLP

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   # 拼成一个向量送入 MLP

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-4.0)

    # 惩罚：碰撞（撞桌腿、撞物体）
    # collision_penalty = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 5.0},
    # )

    # 惩罚：action rate，防止指令抖动
    # batch_size也可以调小一点
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
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
    pass
    # pose_command = mdp.UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=False,
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=False,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"),"minimum_height": 0.3},
    )

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HighLevelEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the high level environment."""

    # environment settings
    scene: HighLevelSceneCfg = HighLevelSceneCfg(num_envs=16, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    foot_link_name = ".*_wheel"

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = 8.0

        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = (
        #         self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
        #     )
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
