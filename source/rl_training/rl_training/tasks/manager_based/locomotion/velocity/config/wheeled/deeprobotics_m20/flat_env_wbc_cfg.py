from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20CommandsCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as DeeproboticsM20ObservationsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20RewardsCfg
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

'''
全身控制（WBC）配置：
- 任务目标：在平坦环境中，机器人需要同时控制底盘速度和机身姿态（高度、俯仰、横滚），以实现更自然和稳定的运动。
- 主要挑战：需要在保持底盘速度的同时，调整机身姿态以适应不同的运动需求，例如加速时稍微降低机身高度，转弯时适当倾斜等。
去除rewards中的机械臂相关奖励项，新增机身姿态跟踪奖励项，鼓励机器人在执行底盘速度命令的同时，保持合理的机身姿态。
'''

@configclass
class WBCCommandsCfg(DeeproboticsM20CommandsCfg):
    """全身控制（WBC）命令集。

    继承父类：
      - base_velocity : 底盘全向速度 (v_x, v_y, omega_z)
      - ee_pose       : 末端执行器目标位姿

    新增：
      - body_pose     : 机身目标 height / pitch / roll（截断正态分布）
    """

    body_pose: mdp.BodyPoseCommandCfg = mdp.BodyPoseCommandCfg(
        # ---- height：正常站立为主，偶尔蹲下 ----
        height_mean=0.55,
        height_std=0.03,
        height_range=(0.33, 0.60),

        # ---- pitch：通常保持水平，偶尔俯身 ----
        pitch_mean=0.0,
        pitch_std=0.08,
        pitch_range=(-0.35, 0.35),

        # ---- roll：通常保持水平，偶尔侧身 ----
        roll_mean=0.0,
        roll_std=0.06,
        roll_range=(-0.25, 0.25),

        resampling_time_range=(5.0, 10.0),
    )

@configclass
class WBCObservationsCfg(DeeproboticsM20ObservationsCfg):
    """全身控制（WBC）观测配置。

    继承父类：
      - base_observation : 基础观测
    """
    @configclass
    class PolicyCfg(DeeproboticsM20ObservationsCfg.PolicyCfg):
        body_pose_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},  # 对应 cfg 中的属性名
        )

    @configclass
    class CriticCfg(DeeproboticsM20ObservationsCfg.CriticCfg):
        body_pose_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},  # 对应 cfg 中的属性名
        )
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class WBCRewardsCfg(DeeproboticsM20RewardsCfg):
    """全身控制（WBC）奖励配置。

    继承父类：
      - base_rewards : 基础奖励
    """
    # ---- 机身高度跟踪 ----
    body_height_tracking = RewTerm(
        func=mdp.body_height_tracking,          # 或 mdp.body_height_tracking
        weight=2.0,
        params={
            "command_name": "body_pose",
            "std": 0.05,                    # 误差容忍度（m），越小越严格
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 pitch 跟踪 ----
    body_pitch_tracking = RewTerm(
        func=mdp.body_pitch_tracking,
        weight=1.0,
        params={
            "command_name": "body_pose",
            "std": 0.1,                     # 误差容忍度（rad），约 5.7°
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 roll 跟踪 ----
    body_roll_tracking = RewTerm(
        func=mdp.body_roll_tracking,
        weight=1.0,
        params={
            "command_name": "body_pose",
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    
@configclass
class FlatEnvWBCConfig(DeeproboticsM20FlatEnvCfg):
    commands: WBCCommandsCfg = WBCCommandsCfg()
    observations: WBCObservationsCfg = WBCObservationsCfg()
    rewards: WBCRewardsCfg = WBCRewardsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_height_l2.weight = 0.0  # 关闭原有的高度奖励，改用新的 body_height_tracking
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "FlatEnvWBCConfig":
            self.disable_zero_weight_rewards()