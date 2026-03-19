from isaaclab.utils import configclass
from .high_level_flat_env_cfg import HighLevelFlatEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from rl_training.tasks.manager_based.locomotion.highlevel.mdp.encoder import make_cnn_model_zoo_cfg
import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import ObservationsCfg as LowLevelObsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import NavigationActionsCfg as LowLevelActCfg

from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ObservationsCfg as HighLevelObservationsCfg
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import ActionsCfg as HighLevelActionsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_nav_cfg import DeeproboticsM20FlatNavEnvCfg as LOW_LEVEL_ENV_CFG

_low_level_env_cfg = LOW_LEVEL_ENV_CFG()

@configclass
class HLFlatNavActionsCfg(HighLevelActionsCfg):
    pre_trained_nav_action: mdp.PreTrainedNavActionCfg = mdp.PreTrainedNavActionCfg(
        asset_name="robot",
        policy_path=f"D:\\nvidia-isaac-sim\\loco-manip-unified-rl-agent\\logs\\rsl_rl\\deeprobotics_m20_nav_flat\\2026-03-18_23-06-52\\exported\\policy.pt",
        low_level_decimation=4,
        low_level_leg_actions=_low_level_env_cfg.actions.joint_pos,
        low_level_wheel_actions=_low_level_env_cfg.actions.joint_vel,
        low_level_observations=_low_level_env_cfg.observations.policy,
    )

@configclass
class HLFlatNavObservationsCfg(HighLevelObservationsCfg):
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
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   # 拼成一个向量送入 MLP
    policy: PolicyCfg = PolicyCfg()



@configclass
class HLFlatNavEnvCfg(HighLevelFlatEnvCfg):
    actions: HLFlatNavActionsCfg = HLFlatNavActionsCfg()
    observations: HLFlatNavObservationsCfg = HLFlatNavObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "HLFlatNavEnvCfg":
            self.disable_zero_weight_rewards()