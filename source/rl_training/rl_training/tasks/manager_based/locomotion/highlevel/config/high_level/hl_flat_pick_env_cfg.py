import math
from isaaclab.utils import configclass
from .high_level_flat_env_cfg import HighLevelFlatEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup

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
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import DeeproboticsM20FlatEnvCfg as LOW_LEVEL_ENV_CFG

from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import HeightInvariantEECommandCfg

_low_level_env_cfg = LOW_LEVEL_ENV_CFG()

@configclass
class HLFlatPickActionsCfg(HighLevelActionsCfg):
    pre_trained_pick_action: mdp.PreTrainedPickActionCfg = mdp.PreTrainedPickActionCfg(
        asset_name="robot",
        policy_path=f"logs/rsl_rl/deeprobotics_m20_flat/2026-03-18_18-06-34/exported/policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_ee_actions=_low_level_env_cfg.actions.ee_ik,
        low_level_observations=_low_level_env_cfg.observations.policy,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint7", "arm_joint8"],
        open_command_expr={"arm_joint7": 0.04, "arm_joint8": 0.04},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
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
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True   # 拼成一个向量送入 MLP

@configclass
class CriticCfg(HighLevelObservationsCfg.CriticCfg):
    # 目标物体相对机器人的位置
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

@configclass
class TeacherCfg(HighLevelObservationsCfg.PolicyCfg):
    # 目标物体相对机器人的位置
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
        func=mdp.distance_to_target_reward,
        weight=0.25,                          # 略降权重，让位给 EE 精确接近
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
        },
    )

    # 底盘朝向物体
    heading_to_object = RewTerm(
        func=mdp.heading_to_target_reward,
        weight=0.25,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
        },
    )

    # 靠近后减速，稳定底盘（沿用原逻辑）
    slow_near_target = RewTerm(
        func=mdp.slow_down_near_target_reward,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "distance_threshold": 1.0,
            "vel_max": 0.5,
            "penalty_scale": 1.0,
        },
    )

    # =========================================================
    # 阶段二：末端执行器精确接近物体
    # =========================================================

    cmd_pose_to_object = RewTerm(
        func=mdp.cmd_pos_to_object_reward,
        weight=1.5,
        params={
            "action_term_name": "pre_trained_pick_action",
            "object_cfg":       SceneEntityCfg("object"),
            "pos_sigma":        0.1,   # 单位：米，10cm内奖励显著上升
        },
    )

    # 核心密集奖励：arm_link6（EE）到物体距离，高斯核塑形
    reach_object_ee = RewTerm(
        func=mdp.object_ee_distance,
        weight=3.0,                          # 权重高于底盘接近，引导手臂精细运动
        params={
            "std": 0.1,                      # 高斯核宽度，越小精度要求越高
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
        },
    )

    # =========================================================
    # 阶段三：夹爪对准物体
    # =========================================================

    # 夹爪朝向对准：arm_link7/8 两指到物体的距离之和最小化
    # 鼓励物体处于两指正中间（对称抓取）
    gripper_alignment = RewTerm(
        func=mdp.object_ee_distance,
        weight=2.0,
        params={
            "std": 0.05,                     # 更小的核，要求更精确的对准
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link7"),
        },
    )

    gripper_alignment_finger2 = RewTerm(
        func=mdp.object_ee_distance,
        weight=2.0,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link8"),
        },
    )

    # 夹爪接触物体奖励：两指均应有接触力
    grasp_contact_finger1 = RewTerm(
        func=mdp.gripper_object_contact,         # 正权重 → 鼓励夹爪接触
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "arm_contact_forces",
                body_names="arm_link7",
            ),
            "threshold": 0.5,               # 最小接触力阈值（N），低于此不奖励
        },
    )

    grasp_contact_finger2 = RewTerm(
        func=mdp.gripper_object_contact,         # 正权重 → 鼓励夹爪接触
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "arm_contact_forces",
                body_names="arm_link8",
            ),
            "threshold": 0.5,
        },
    )

    # =========================================================
    # 阶段四：抬起物体（稀疏高奖励）
    # =========================================================

    # 稀疏成功奖励：物体高度超过阈值即触发
    lift_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=10.0,                         # 最高权重，作为最终目标信号
        params={
            "minimal_height": 0.04,          # 离桌面 4cm 算抬起
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # 抬起后 EE 与物体高度保持同步（防止松手后继续乱动）
    ee_object_height_sync = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.5,
        params={
            "std": 0.08,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
        },
    )

    # =========================================================
    # 全程惩罚项
    # =========================================================

    # 手臂主体非预期碰撞（排除夹爪，夹爪需要接触物体）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "arm_contact_forces",
                body_names=["arm_link1", "arm_link2", "arm_link3",
                            "arm_link4", "arm_link5", "arm_link6"],
            ),
            "threshold": 5.0,
        },
    )

    # 关节速度惩罚：防止手臂抖动、过激运动
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )



@configclass
class HLFlatPickTerminationsCfg(HighLevelTerminationsCfg):
    
    pass

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
            "pose_range": {"x": (1.2, 1.3), "y": (-0.1, 0.1),  "yaw": (-0.393, 0.393)},
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

@configclass
class HLFlatPickEnvCfg(HighLevelFlatEnvCfg):
    actions: HLFlatPickActionsCfg = HLFlatPickActionsCfg()
    observations: HLFlatPickObservationsCfg = HLFlatPickObservationsCfg()
    rewards: HLFlatPickRewardsCfg = HLFlatPickRewardsCfg()
    terminations: HLFlatPickTerminationsCfg = HLFlatPickTerminationsCfg()
    commands: HLFlatPickCommandCfg = HLFlatPickCommandCfg()
    events: HLFlatPickEventCfg = HLFlatPickEventCfg()
    gripper_link_names = "arm_link[7-8]"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.nav_camera = None
        
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name})(?!.*{self.gripper_link_names}).*"]
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HLFlatPickEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class HLFlatPickTeacherEnvCfg(HLFlatPickEnvCfg):
    observations: HLFlatPickTeacherObservationsCfg = HLFlatPickTeacherObservationsCfg()
    def __post_init__(self):
        super().__post_init__()

        self.scene.arm_camera = None
        self.scene.warehouse = None
        if self.__class__.__name__ == "HLFlatPickTeacherEnvCfg":
            self.disable_zero_weight_rewards()