
from isaaclab.utils import configclass
import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_wbc_cfg import FlatEnvWBCConfig as LOW_LEVEL_ENV_CFG
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.hl_flat_pick_env_cfg import (
    HLFlatPickActionsCfg,
    HLFlatPickTeacherEnvCfg,
    )


_low_level_env_cfg = LOW_LEVEL_ENV_CFG()


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
class HLFlatPickWBCTeacherEnvCfg(HLFlatPickTeacherEnvCfg):
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    def __post_init__(self):
        super().__post_init__()

        if self.__class__.__name__ == "HLFlatPickWBCTeacherEnvCfg":
            self.disable_zero_weight_rewards()

