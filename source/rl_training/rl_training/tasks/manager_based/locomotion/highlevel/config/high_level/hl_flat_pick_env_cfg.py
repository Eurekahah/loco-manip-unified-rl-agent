import math
from isaaclab.utils import configclass
from .high_level_flat_env_cfg import HighLevelFlatEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import TiledCameraCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from rl_training.tasks.manager_based.locomotion.highlevel.mdp.encoder import make_cnn_model_zoo_cfg
import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
# from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import ObservationsCfg as LowLevelObsCfg
# from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20ActionsCfg as LowLevelActCfg

from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ObservationsCfg as HighLevelObservationsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ActionsCfg as HighLevelActionsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import TerminationsCfg as HighLevelTerminationsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import CommandsCfg as HighLevelCommandsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import RewardsCfg as HighLevelRewardsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import EventCfg as HighLevelEventCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import DeeproboticsM20FlatEnvCfg as LOW_LEVEL_ENV_CFG
import isaaclab.sim as sim_utils

from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import HeightInvariantEECommandCfg

_low_level_env_cfg = LOW_LEVEL_ENV_CFG()

@configclass
class HLFlatPickActionsCfg(HighLevelActionsCfg):
    pre_trained_pick_action: mdp.PreTrainedPickActionCfg = mdp.PreTrainedPickActionCfg(
        asset_name="robot",
        # policy_path=f"logs/rsl_rl/deeprobotics_m20_flat/2026-03-18_18-06-34/exported/policy.pt",
        policy_path=f"logs/rsl_rl/deeprobotics_m20_flat/2026-04-21_00-02-23/exported/policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_ee_actions=_low_level_env_cfg.actions.ee_ik,
        low_level_observations=_low_level_env_cfg.observations.policy,
        debug_vis=False,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint7", "arm_joint8"],
        open_command_expr={"arm_joint7": 0.04, "arm_joint8": -0.04},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
    )

@configclass
class HLFlatPickWBCActionsCfg(HLFlatPickActionsCfg):
    pre_trained_pick_action: mdp.PreTrainedPickWBCActionCfg = mdp.PreTrainedPickWBCActionCfg(
        asset_name="robot",
        policy_path=f"logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-04-21_21-12-48/exported/policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_ee_actions=_low_level_env_cfg.actions.ee_ik,
        low_level_observations=_low_level_env_cfg.observations.policy,
        debug_vis=False,
    )


@configclass
class PolicyCfg(HighLevelObservationsCfg.PolicyCfg):
    # 使用预训练视觉编码器
    arm_camera_embedding = ObsTerm(
        func=mdp.image_features,
        params={
            "sensor_cfg":    SceneEntityCfg("arm_camera"),
            "data_type":     "rgb",
            "model_zoo_cfg": None,
            "model_name":    "resnet18",
        },
    )
    nav_camera_embedding = ObsTerm(
        func=mdp.image_features,
        params={
            "sensor_cfg":    SceneEntityCfg("nav_camera"),
            "data_type":     "rgb",
            "model_zoo_cfg": None,
            "model_name":    "resnet18",
        },
    )
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True   # 拼成一个向量送入 MLP

@configclass
class CriticCfg(HighLevelObservationsCfg.CriticCfg):
    # 目标物体相对机器人的位姿
    target_object_rel_pose = ObsTerm(
        func=mdp.object_pose_in_robot_root_frame,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg":  SceneEntityCfg("robot"),
        },
    )
    target_object_rel_pose_ee = ObsTerm(
        func=mdp.object_pose_in_ee_frame,
        params={
            "object_cfg":   SceneEntityCfg("object"),
            "robot_cfg":    SceneEntityCfg("robot"),
            "ee_link_name": "arm_link6",   # 可按需改
            "ee_offset_z":  0.135,
        },
    )
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class TeacherCfg(HighLevelObservationsCfg.PolicyCfg):
    # 目标物体相对机器人的位姿
    target_object_rel_pose = ObsTerm(
        func=mdp.object_pose_in_robot_root_frame,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg":  SceneEntityCfg("robot"),
        },
    )
    target_object_rel_pose_ee = ObsTerm(
        func=mdp.object_pose_in_ee_frame,
        params={
            "object_cfg":   SceneEntityCfg("object"),
            "robot_cfg":    SceneEntityCfg("robot"),
            "ee_link_name": "arm_link6",   # 可按需改
            "ee_offset_z":  0.135,
        },
    )
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
@configclass
class HLFlatPickObservationsCfg(HighLevelObservationsCfg):
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class HLFlatPickTeacherObservationsCfg(HighLevelObservationsCfg):
    critic: CriticCfg = CriticCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class HLFlatPickTeacherWithCameraObservationsCfg(HighLevelObservationsCfg):
    @configclass
    class CameraTeacherCfg(TeacherCfg):
        arm_camera_embedding = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg":    SceneEntityCfg("arm_camera"),
                "data_type":     "rgb",
                "model_zoo_cfg": None,
                "model_name":    "resnet18",
            },
        )
        side_camera_embedding = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg":    SceneEntityCfg("side_camera"),
                "data_type":     "rgb",
                "model_zoo_cfg": None,
                "model_name":    "resnet18",
            },
        )
        target_object_rel_pose = ObsTerm(
            func=mdp.object_pose_in_robot_root_frame,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg":  SceneEntityCfg("robot"),
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class CameraCriticCfg(CriticCfg):
        arm_camera_embedding = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg":    SceneEntityCfg("arm_camera"),
                "data_type":     "rgb",
                "model_zoo_cfg": None,
                "model_name":    "resnet18",
            },
        )
        side_camera_embedding = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg":    SceneEntityCfg("side_camera"),
                "data_type":     "rgb",
                "model_zoo_cfg": None,
                "model_name":    "resnet18",
            },
        )
        target_object_rel_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg":  SceneEntityCfg("robot"),
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    critic: CameraCriticCfg = CameraCriticCfg()
    teacher: CameraTeacherCfg = CameraTeacherCfg()


@configclass
class HLFlatPickRewardsCfg(HighLevelRewardsCfg):
    """
    抓取任务奖励配置
    - 机器人类型：移动底盘 + 机械臂
    - 末端执行器：arm_link6
    - 夹爪：arm_link7, arm_link8
    - 目标：抓取桌面物体并抬起
    """

    # =========================================================
    # 阶段一：底盘接近物体（导航层，沿用原逻辑）
    # =========================================================

    # 整体接近：机器人基座靠近物体
    approach_object = RewTerm(
        func=mdp.distance_to_target_reward_shift,
        weight=2.0,                          # 略降权重，让位给 EE 精确接近
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "lam": 2.0,                        # λ 越大，收敛越快（建议 3.0–5.0）
        },
    )

    # 底盘朝向物体
    heading_to_object = RewTerm(
        func=mdp.heading_to_target_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "std": 0.3,                        # 角度奖励的σ，单位：弧度，约17度
        },
    )

    # 靠近后减速，稳定底盘（沿用原逻辑）
    # slow_near_target = RewTerm(
    #     func=mdp.slow_down_near_target_reward,
    #     weight=0.01,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "target_cfg": SceneEntityCfg("object"),
    #         "distance_threshold": 1.0,
    #         "vel_max": 0.5,
    #         "penalty_scale": 1.0,
    #     },
    # )

    forward_velocity_penalty = RewTerm(
        func=mdp.forward_velocity_penalty,
        weight=-0.5,
        params={"action_name": "pre_trained_pick_action"},
        )

    lateral_velocity_penalty = RewTerm(
        func=mdp.lateral_velocity_penalty,
        weight=-0.5,
        params={"action_name": "pre_trained_pick_action"},
    )

    angular_velocity_penalty = RewTerm(
        func=mdp.angular_velocity_penalty,
        weight=-0.2,
        params={"action_name": "pre_trained_pick_action"},
    )

    arm_posture_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link[1-6]"),
        },
    )

    # =========================================================
    # 阶段二：末端执行器精确接近物体
    # =========================================================

    cmd_pos_to_object = RewTerm(
        func=mdp.cmd_pos_to_object_reward,
        weight=4.0,
        params={
            "action_term_name": "pre_trained_pick_action",
            "object_cfg":       SceneEntityCfg("object"),
            "pos_sigma":        0.7,   # 单位：米，70cm内奖励显著上升
            "use_shaped":       True,
        },
    )

    cmd_pos_to_object_fine_grained = RewTerm(
        func=mdp.cmd_pos_to_object_reward,
        weight=4.0,
        params={
            "action_term_name": "pre_trained_pick_action",
            "object_cfg":       SceneEntityCfg("object"),
            "pos_sigma":        0.2,   # 单位：米，20cm内奖励显著上升
            "use_shaped":       False,
        },
    )

    # 核心密集奖励：arm_link6（EE）到物体距离，高斯核塑形
    reach_object_ee = RewTerm(
        func=mdp.object_ee_distance,
        weight=5.0,                          # 权重高于底盘接近，引导手臂精细运动
        params={
            "std": 0.3,                      # 高斯核宽度，越小精度要求越高
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
        },
    )

    delta_action_penalty_near_goal = RewTerm(
        func=mdp.delta_action_l2_near_target,
        weight=-2.0,
        params={
            "action_name": "pre_trained_pick_action",
            "object_cfg": SceneEntityCfg("object"),
            "distance_threshold": 0.2,
        },
    )

    ee_velocity_penalty = RewTerm(
        func=mdp.ee_velocity_l2,
        weight=-0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
        },
    )

    # =========================================================
    # 阶段三：夹爪对准物体
    # =========================================================

    # 夹爪朝向对准：arm_link7/8 两指到物体的距离之和最小化
    gripper_alignment_symmetric = RewTerm(
        func=mdp.object_ee_symmetric_alignment,
        weight=20.0,
        params={
            "std": 0.05,
            "min_finger_dist": 0.04,   # 新增，根据夹爪实际尺寸设置
                                    # 物体直径约多少就设多少，例如物体直径4cm
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg_finger1": SceneEntityCfg("robot", body_names="arm_link7"),
            "ee_frame_cfg_finger2": SceneEntityCfg("robot", body_names="arm_link8"),
        },
    )


    # 夹爪同步接触奖励：arm_link7/8 两指同时接触物体时给予奖励
    grasp_contact_symmetric = RewTerm(
        func=mdp.gripper_contact_symmetric_grasp,
        weight=20.0,  
        params={
            "threshold": 0.5,
            "sensor_cfg_finger1": SceneEntityCfg("arm_link7_contact_forces", body_names="arm_link7"),
            "sensor_cfg_finger2": SceneEntityCfg("arm_link8_contact_forces", body_names="arm_link8"),
            "ee_frame_cfg_finger1": SceneEntityCfg("robot", body_names="arm_link7"),
            "ee_frame_cfg_finger2": SceneEntityCfg("robot", body_names="arm_link8"),

        },
    )

    delta_scale_reward = RewTerm(
        func=mdp.reward_delta_scale,
        weight=3.0,
        params={
            "action_name": "pre_trained_pick_action",
            "object_cfg": SceneEntityCfg("object"),
            "d_max": 0.2,  
        }
    )
    # =========================================================
    # 阶段四：抬起物体（稀疏高奖励）
    # =========================================================

    # 稀疏成功奖励：物体高度超过阈值即触发
    lift_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=30.0,                         # 最高权重，作为最终目标信号
        params={
            "minimal_height": 0.04,          # 离桌面 4cm 算抬起
            "object_cfg": SceneEntityCfg("object"),
        },
    )


    # =========================================================
    # 全程惩罚项
    # =========================================================


    # 手臂主体非预期碰撞（排除夹爪，夹爪需要接触物体）
    undesired_arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "arm_contact_forces",
                body_names=["arm_link[1-6]", "camera_link"],
            ),
            "threshold": 1.0,
        },
    )
    undesired_body_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link","(f|h)(l|r)_(hip(x|y)|knee)"],
            ),
            "threshold": 1.0,
        },
    )

    joint_pos_limit_penalty = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    joint_vel_limit_penalty = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={
            "soft_ratio": 0.9,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    

    # 关节速度惩罚：防止手臂抖动、过激运动
    # joint_vel_penalty = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.005,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )



@configclass
class HLFlatPickTerminationsCfg(HighLevelTerminationsCfg):
    object_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "height_threshold": 0.2,  # 物体世界位姿高度小于0.2m算掉落
        },  
    )

    action_target_too_far = DoneTerm(
        func=mdp.action_target_too_far,
        params={
            "action_term_name": "pre_trained_pick_action",
            "ee_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
            "distance_threshold": 1.0,
        },
    )

    
    # hold_object = DoneTerm(
    #     func=mdp.object_held_for_duration,  # 替换为新函数
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "minimal_height": 0.04,
    #         "hold_duration": 5.0,           # 持续举起 5 秒终止
    #     },
    # )

@configclass
class HLFlatPickTerminationsCfg_PLAY(HLFlatPickTerminationsCfg):
    lift_object = DoneTerm(
        func=mdp.object_held_for_duration,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "minimal_height": 0.04,
            "hold_duration": 1.0,           # 持续举起 1 秒终止
        },
    )

@configclass
class HLFlatPickCommandCfg(HighLevelCommandsCfg):
     ee_pose = HeightInvariantEECommandCfg(
        asset_name="robot",
        body_name="arm_link6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        sampled_height=0.6,  # 采样坐标系的固定高度
        arm_base_link_name="arm_base",  # 采样坐标系xy位置
        ranges=HeightInvariantEECommandCfg.Ranges(
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
    
@configclass
class HLFlatPickEventCfg(HighLevelEventCfg):
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.9, 1.0), "y": (-0.6, 0.6),  "yaw": (-0.393, 0.393)},
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

    reset_arm_default = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), # 1.0 表示缩放到默认关节位置的 100%
            "velocity_range": (0.0, 0.0), # 速度重置为 0
        },
    ) 

    # reset_table_height = EventTerm(
    #     func=mdp.reset_xform_z_only,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "pose_range": {
    #             "z": (-0.7, 0.0),  # 只写 z 就够了
    #         },
    #     },
    # )

@configclass
class HLFlatSideCameraSceneCfg(HighLevelSceneCfg):
    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/side_camera",
        update_period=0.1,                      # 10 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,               # 焦距
            focus_distance=400.0,            # 对焦距离
            f_stop=0.0,                      # 光圈值，0.0表示为理想针孔模型
            horizontal_aperture=20.955,      # 水平视场，单位为度，根据焦距和传感器尺寸计算得出
            clipping_range=(0.1, 10.0),      # 近裁剪面和远裁剪面，单位为米
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 2.0),              # 相机位于桌面正上方
            rot=(0.0, 1.0, 0.0, 0.0),         # 四元数(w,x,y,z)
            convention="ros",
        ),
    )

# def override_value(env, env_ids, data, value, num_steps):
#     print(f"Curriculum step: {env.common_step_counter}, overriding value to {value} for envs {env_ids}")
#     if env.common_step_counter > num_steps:
#         return value
#     return mdp.modify_term_cfg.NO_CHANGE

# @configclass
# class HLFlatPickCurriculumCfg:
#     ee_pos_delta_max = CurrTerm(
#         func=mdp.modify_term_cfg, 
#         params={
#             "address": "actions.pre_trained_pick_action.delta_pos_max",  # note: `_manager.cfg` is omitted
#             "modify_fn": override_value,
#             "modify_params": {"value": 0.05, "num_steps": 10},
#         }
#     )



@configclass
class HLFlatPickEnvCfg(HighLevelFlatEnvCfg):
    actions: HLFlatPickActionsCfg = HLFlatPickActionsCfg()
    observations: HLFlatPickObservationsCfg = HLFlatPickObservationsCfg()
    rewards: HLFlatPickRewardsCfg = HLFlatPickRewardsCfg()
    terminations: HLFlatPickTerminationsCfg = HLFlatPickTerminationsCfg()
    commands: HLFlatPickCommandCfg = HLFlatPickCommandCfg()
    events: HLFlatPickEventCfg = HLFlatPickEventCfg()
    # curriculum: HLFlatPickCurriculumCfg = HLFlatPickCurriculumCfg()
    gripper_link_names = "arm_link[7-8]"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HLFlatPickEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class HLFlatPickTeacherEnvCfg(HLFlatPickEnvCfg):
    observations: HLFlatPickTeacherObservationsCfg = HLFlatPickTeacherObservationsCfg()
    def __post_init__(self):
        super().__post_init__()

        self.scene.arm_camera = None
        self.scene.nav_camera = None
        self.scene.warehouse = None
        if self.__class__.__name__ == "HLFlatPickTeacherEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class HLFlatPickWBCTeacherEnvCfg(HLFlatPickTeacherEnvCfg):
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    def __post_init__(self):
        super().__post_init__()

        if self.__class__.__name__ == "HLFlatPickWBCTeacherEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class HLFlatPickTeacherEnvCfg_PLAY(HLFlatPickTeacherEnvCfg):
    terminations: HLFlatPickTerminationsCfg_PLAY = HLFlatPickTerminationsCfg_PLAY()
    def __post_init__(self):
        super().__post_init__()

        if self.__class__.__name__ == "HLFlatPickTeacherEnvCfg_PLAY":
            self.disable_zero_weight_rewards()

@configclass
class HLFlatPickTeacherWithCameraEnvCfg(HLFlatPickEnvCfg):
    observations: HLFlatPickTeacherWithCameraObservationsCfg = HLFlatPickTeacherWithCameraObservationsCfg()
    scene: HLFlatSideCameraSceneCfg = HLFlatSideCameraSceneCfg(num_envs=16, env_spacing=5.0)
    def __post_init__(self):
        super().__post_init__()

        self.scene.warehouse = None
        self.terminations.base_contact = None  # 取消底盘碰撞终止条件
        if self.__class__.__name__ == "HLFlatPickTeacherWithCameraEnvCfg":
            self.disable_zero_weight_rewards()