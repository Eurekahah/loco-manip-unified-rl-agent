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
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import ObservationsCfg as LowLevelObsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import NavigationActionsCfg as LowLevelActCfg

from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ObservationsCfg as HighLevelObservationsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ActionsCfg as HighLevelActionsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import TerminationsCfg as HighLevelTerminationsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import RewardsCfg as HighLevelRewardsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import DeeproboticsM20FlatNavEnvCfg as LOW_LEVEL_ENV_CFG

_low_level_env_cfg = LOW_LEVEL_ENV_CFG()

@configclass
class HLFlatNavActionsCfg(HighLevelActionsCfg):
    pre_trained_nav_action: mdp.PreTrainedNavActionCfg = mdp.PreTrainedNavActionCfg(
        asset_name="robot",
        policy_path=f"logs\\rsl_rl\\deeprobotics_m20_nav_flat\\2026-03-20_22-13-56\\exported\\policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_observations=_low_level_env_cfg.observations.policy,
    )

@configclass
class PolicyCfg(HighLevelObservationsCfg.PolicyCfg):
    # 使用预训练视觉编码器
    nav_camera_embedding = ObsTerm(
        func=mdp.image_features,
        params={
            "sensor_cfg":    SceneEntityCfg("nav_camera"),
            "data_type":     "rgb",
            "model_zoo_cfg": None,
            "model_name":    "resnet18",
        },
    )
    # 直接将image展平作为输入
    # nav_camera_embedding = ObsTerm(
    #     func=mdp.image,
    #     params={
    #         "sensor_cfg":    SceneEntityCfg("nav_camera"),
    #         "data_type":     "rgb",
    #     },
    # )
    # nav_camera_embedding = ObsTerm(
    #     func=mdp.image_features,
    #     params={
    #         "sensor_cfg":    SceneEntityCfg("nav_camera"),
    #         "data_type":     "rgb",
    #         "model_zoo_cfg": make_cnn_model_zoo_cfg(),
    #         "model_name":    "nav_resnet18_unfrozen",
    #     },
    # )
    # 桌子相对机器人的位置（在机器人body坐标系下）
    # target_table_rel_pos = ObsTerm(
    #     func=mdp.object_position_in_robot_root_frame,  # 或自定义 mdp 函数
    #     params={
    #         "object_cfg": SceneEntityCfg("table"),
    #         "robot_cfg":  SceneEntityCfg("robot"),
    #     },
    # )

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
        self.concatenate_terms = True   # 拼成一个向量送入 MLP

@configclass
class ImageObsCfg(ObsGroup):
    # 使用预训练视觉编码器
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
        self.concatenate_terms = True  

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
class HLFlatNavObservationsCfg(HighLevelObservationsCfg):
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    images: ImageObsCfg = ImageObsCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class HLFlatNavTeacherObservationsCfg(HighLevelObservationsCfg):
    critic: CriticCfg = CriticCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class HLFlatNavRewardsCfg(HighLevelRewardsCfg):

    # 核心：越靠近目标得分越高（以距离倒数或负距离）
    approach_object = RewTerm(
        func=mdp.distance_to_target_reward,   
        weight=2.0,
        params={"robot_cfg": SceneEntityCfg("robot"),"target_cfg": SceneEntityCfg("object")},
    )

    # 朝向奖励：机器人朝向桌子方向
    heading_to_object = RewTerm(
        func=mdp.heading_to_target_reward,
        weight=0.5,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("object")},
    )

    # 稀疏：成功到达桌边
    reach_object_success = RewTerm(
        func=mdp.is_terminated_term,   # 配合 TerminationCfg 使用
        weight=10.0,
        params={"term_keys": "reach_object"},
    )

    reach_object_high_vel = RewTerm(
        func=mdp.is_terminated_term,   # 配合 TerminationCfg 使用
        weight=7.0,
        params={"term_keys": "reach_object_high_vel"},
    )

    # 到达目标附近时，速度越小奖励越高，鼓励稳健停靠
    slow_near_target = RewTerm(
        func=mdp.slow_down_near_target_reward,
        weight=1.5,                                  # 适当权重，不要盖过 approach
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "distance_threshold": 0.6,              # 与 approach_object 保持一致
            "vel_max": 0.5,                          # 超过 0.5m/s 则无奖励
            "penalty_scale": 1.0,                     # 超速惩罚强度
        },
    )

    

@configclass
class HLFlatNavTerminationsCfg(HighLevelTerminationsCfg):
    # 成功：到达物体附近足够近
    reach_object = DoneTerm(
        func=mdp.reached_target,   # 自定义，检查 dist < threshold
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "threshold": 0.4,
            "vel_threshold": 0.2,   # 只有在速度足够慢时才算成功到达
        },
    )

    reach_object_high_vel = DoneTerm(
        func=mdp.reached_target,   # 自定义，检查 dist < threshold
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "threshold": 0.4,
            "vel_threshold": 0.5,   # 到达目标但速度过快，给予较小奖励但不算完全成功
        },
    )


    

@configclass
class HLFlatNavEnvCfg(HighLevelFlatEnvCfg):
    actions: HLFlatNavActionsCfg = HLFlatNavActionsCfg()
    observations: HLFlatNavObservationsCfg = HLFlatNavObservationsCfg()
    rewards: HLFlatNavRewardsCfg = HLFlatNavRewardsCfg()
    terminations: HLFlatNavTerminationsCfg = HLFlatNavTerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.arm_camera = None
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HLFlatNavEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class HLFlatNavTeacherEnvCfg(HLFlatNavEnvCfg):
    observations: HLFlatNavTeacherObservationsCfg = HLFlatNavTeacherObservationsCfg()
    def __post_init__(self):
        super().__post_init__()

        self.scene.nav_camera = None
        self.scene.warehouse = None
        if self.__class__.__name__ == "HLFlatNavTeacherEnvCfg":
            self.disable_zero_weight_rewards()