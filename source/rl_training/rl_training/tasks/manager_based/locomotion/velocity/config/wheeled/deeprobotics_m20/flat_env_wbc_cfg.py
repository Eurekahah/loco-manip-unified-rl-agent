from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20CommandsCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as DeeproboticsM20ObservationsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20RewardsCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20CurriculumsCfg
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
      - body_pose     : 机身目标 height / pitch / roll
    """

    body_pose: mdp.BodyPoseCommandCfg = mdp.BodyPoseCommandCfg(
        # ---- height：正常站立为主，偶尔蹲下 ----
        height_range=(0.33, 0.60),
        # ---- pitch：通常保持水平，偶尔俯身 ----
        # mean=0.0°, std≈4.6°, range=(-20.1°, 20.1°)
        pitch_range=(-0.35, 0.35),
        # ---- roll：通常保持水平，偶尔侧身 ----
        # mean=0.0°, std≈3.4°, range=(-14.3°, 14.3°)
        roll_range=(-0.25, 0.25),
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
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
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(DeeproboticsM20ObservationsCfg.CriticCfg):
        body_pose_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},  # 对应 cfg 中的属性名
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
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
        weight=0.001,
        params={
            "command_name": "body_pose",
            "std": 0.04,                    # 误差容忍度（m），越小越严格
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 pitch 跟踪 ----
    body_pitch_tracking = RewTerm(
        func=mdp.body_pitch_tracking,
        weight=0.001,
        params={
            "command_name": "body_pose",
            "std": 0.05,                     # 误差容忍度（rad），约 5.7°
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 roll 跟踪 ----
    body_roll_tracking = RewTerm(
        func=mdp.body_roll_tracking,
        weight=0.001,
        params={
            "command_name": "body_pose",
            "std": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class WBCCurriculumCfg(DeeproboticsM20CurriculumsCfg):
    """WBC 课程配置。

    每个属性是一个 CurriculumTermCfg，对应一个课程函数。
    Isaac Lab 在每个 episode 结束后调用这些函数。
    """

    # ── Stage 2：1M步后开放 height 范围 ──────────────────────────
    body_pose_height_range_s2: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.height_range",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (0.33, 0.60),
                "num_steps": 25_000,
            },
        },
    )

    # ── Stage 3：2M步后开放 pitch/roll 范围 ──────────────────────
    body_pose_pitch_range_s3: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.pitch_range",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-0.35, 0.35),
                "num_steps": 50_000,
            },
        },
    )
    body_pose_roll_range_s3: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.roll_range",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-0.25, 0.25),
                "num_steps": 50_000,
            },
        },
    )

    # ── Stage 2：1M步后提升 height 奖励权重 ──────────────────────
    body_height_rew_s2: CurrTerm = CurrTerm(
        func=mdp.modify_reward_weight,   # ← 直接用官方类
        params={
            "term_name": "body_height_tracking",
            "weight":    0.8,
            "num_steps": 25_000,
        },
    )

    # ── Stage 3：2M步后提升 pitch/roll 奖励权重 ──────────────────
    body_pitch_rew_s3: CurrTerm = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "body_pitch_tracking",
            "weight":    0.8,
            "num_steps": 50_000,
        },
    )
    body_roll_rew_s3: CurrTerm = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "body_roll_tracking",
            "weight":    0.8,
            "num_steps": 50_000,
        },
    )


    
@configclass
class FlatEnvWBCConfig(DeeproboticsM20FlatEnvCfg):
    commands: WBCCommandsCfg = WBCCommandsCfg()
    observations: WBCObservationsCfg = WBCObservationsCfg()
    rewards: WBCRewardsCfg = WBCRewardsCfg()
    curriculum: WBCCurriculumCfg = WBCCurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_height_l2.weight = 0.0  # 关闭原有的高度奖励，改用新的 body_height_tracking
        self.rewards.lin_vel_z_l2.weight = 0.0      # 降低底盘 z 轴速度惩罚
        self.rewards.ang_vel_xy_l2.weight = 0.0     # 关闭水平面角速度惩罚
        self.rewards.stand_still.weight = 0.0      # 关闭站立不动奖励

        self.rewards.hipx_joint_pos_penalty.func = mdp.joint_pos_penalty_wbc
        self.rewards.hipx_joint_pos_penalty.params["pose_command_name"] = "body_pose"
        self.rewards.hipy_joint_pos_penalty.func = mdp.joint_pos_penalty_wbc
        self.rewards.hipy_joint_pos_penalty.params["pose_command_name"] = "body_pose"
        self.rewards.knee_joint_pos_penalty.func = mdp.joint_pos_penalty_wbc
        self.rewards.knee_joint_pos_penalty.params["pose_command_name"] = "body_pose"
        
        self.commands.base_velocity.ranges.lin_vel_x = (-5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.rewards.body_height_tracking.weight = 0.8
        # self.rewards.body_pitch_tracking.weight = 0.8
        # self.rewards.body_roll_tracking.weight = 0.0
        self.commands.body_pose.height_range = (0.513, 0.513)  # Stage 1 初始值
        self.commands.body_pose.pitch_range  = (0.0, 0.0)
        self.commands.body_pose.roll_range   = (0.0, 0.0)
        
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "FlatEnvWBCConfig":
            self.disable_zero_weight_rewards()

class FlatEnvWBCConfig_PLAY(FlatEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (1.0, 1.0)
        self.commands.body_pose.height_range = (0.513, 0.513)
        self.commands.body_pose.pitch_range = (0.35, 0.35)
        self.commands.body_pose.roll_range = (0.0, 0.0)
        if self.__class__.__name__ == "FlatEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()