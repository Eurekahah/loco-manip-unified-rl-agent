from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
        height_std=0.05,
        height_range=(0.33, 0.60),
        # ---- pitch：通常保持水平，偶尔俯身 ----
        # mean=0.0°, std≈4.6°, range=(-20.1°, 20.1°)
        pitch_mean=0.0,
        pitch_std=0.08,
        pitch_range=(-0.35, 0.35),
        # ---- roll：通常保持水平，偶尔侧身 ----
        # mean=0.0°, std≈3.4°, range=(-14.3°, 14.3°)
        roll_mean=0.0,
        roll_std=0.06,
        roll_range=(-0.25, 0.25),
        resampling_time_range=(5.0, 10.0),
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
        weight=1.0,
        params={
            "command_name": "body_pose",
            "std": 0.04,                    # 误差容忍度（m），越小越严格
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 pitch 跟踪 ----
    body_pitch_tracking = RewTerm(
        func=mdp.body_pitch_tracking,
        weight=0.8,
        params={
            "command_name": "body_pose",
            "std": 0.05,                     # 误差容忍度（rad），约 5.7°
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ---- 机身 roll 跟踪 ----
    body_roll_tracking = RewTerm(
        func=mdp.body_roll_tracking,
        weight=0.8,
        params={
            "command_name": "body_pose",
            "std": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

TOTAL_STEPS = 5_000 * 24 * 4096   # ≈ 491_520_000

# 四段切分点（按25%, 50%, 75%, 100%）
S1 = int(TOTAL_STEPS * 0.25)   # ~122_880_000
S2 = int(TOTAL_STEPS * 0.50)   # ~245_760_000
S3 = int(TOTAL_STEPS * 0.75)   # ~368_640_000
S4 = TOTAL_STEPS               # ~491_520_000

# height std 四阶段：0.05 → 0.07 → 0.10 → 0.13 → 0.09*（range=0.27, 均匀≈range/√12≈0.078，取略大）
# pitch std 四阶段：0.08 → 0.12 → 0.17 → 0.20 （range=0.70, 均匀≈0.202）
# roll  std 四阶段：0.06 → 0.09 → 0.13 → 0.14 （range=0.50, 均匀≈0.144）

def step_curriculum(env, env_ids, old_value, thresholds_values, num_steps=None):
    """
    thresholds_values: list of (threshold, value) 按从小到大排列
    超过对应 threshold 就切换到对应 value，否则 NO_CHANGE
    """
    counter = env.common_step_counter
    result = mdp.modify_term_cfg.NO_CHANGE
    for threshold, value in thresholds_values:
        if counter > threshold:
            result = value
    return result

@configclass
class WBCCurriculumCfg:
    """全身控制（WBC）课程配置 —— 四阶段 std 放大。"""

    # ------------------------------------------------------------------ #
    # height_std:  0.05 → 0.07 → 0.10 → 0.13
    # range=(0.33, 0.60), span=0.27
    # 均匀分布等效 std = 0.27/√12 ≈ 0.078  →  末段取 0.13 使尾部充分覆盖
    # ------------------------------------------------------------------ #
    body_pose_height_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.height_std",
            "modify_fn": step_curriculum,
            "modify_params": {
                "thresholds_values": [
                    (S1, 0.07),
                    (S2, 0.10),
                    (S3, 0.13),
                    # S4 段不再变化，保持 0.13
                ],
            },
        },
    )

    # ------------------------------------------------------------------ #
    # pitch_std:  0.08 → 0.12 → 0.17 → 0.20
    # range=(-0.35, 0.35), span=0.70
    # 均匀分布等效 std = 0.70/√12 ≈ 0.202  →  末段取 0.20 贴近均匀
    # ------------------------------------------------------------------ #
    body_pose_pitch_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.pitch_std",
            "modify_fn": step_curriculum,
            "modify_params": {
                "thresholds_values": [
                    (S1, 0.12),
                    (S2, 0.17),
                    (S3, 0.20),
                ],
            },
        },
    )

    # ------------------------------------------------------------------ #
    # roll_std:   0.06 → 0.09 → 0.12 → 0.14
    # range=(-0.25, 0.25), span=0.50
    # 均匀分布等效 std = 0.50/√12 ≈ 0.144  →  末段取 0.14 贴近均匀
    # ------------------------------------------------------------------ #
    body_pose_roll_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.body_pose.roll_std",
            "modify_fn": step_curriculum,
            "modify_params": {
                "thresholds_values": [
                    (S1, 0.09),
                    (S2, 0.12),
                    (S3, 0.14),
                ],
            },
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
        self.rewards.lin_vel_z_l2.weight = -0.5      # 降低底盘 z 轴速度惩罚
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "FlatEnvWBCConfig":
            self.disable_zero_weight_rewards()