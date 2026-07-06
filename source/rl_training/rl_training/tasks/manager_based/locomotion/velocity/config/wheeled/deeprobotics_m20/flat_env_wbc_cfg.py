from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
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
        asset_cfg= SceneEntityCfg("robot"),
        feet_cfg= SceneEntityCfg("robot", body_names=".*wheel"),
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
    
    @configclass
    class HistoryCfg(ObsGroup):
        """Adaptation module (history encoder) 的输入：最近 history_length 步的 [状态, 上一步动作] 拼接序列。

        论文里状态窗口和动作窗口是错位一格的 (s_{t-10:t-1}, a_{t-11:t-2})，这里按你的要求统一用
        history_length=50 的对齐窗口，简化实现；如果以后要还原论文的错位窗口，需要分别为状态和动作
        维护两个不同长度/偏移的 buffer，目前先不处理。
        """
        history_obs = ObsTerm(
            func=mdp.history_single_step_obs,
            history_length=10,          # 对应 ActorCriticHistory 里的 history_length 参数，两边必须一致
            flatten_history_dim=True,   # 关键: 输出 (history_length * single_step_dim,)，而不是 (history_length, single_step_dim)
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            # 部署时机载传感器读数本身就有噪声，这里是否加 Unoise 取决于你想不想让 adaptation module
            # 在训练时就适应噪声输入；如果想加噪声，要在 history_single_step_obs 内部手动加，
            # 因为 ObsTerm 的 noise 字段是在单个 ObsTerm 输出整段历史之后才生效的，不会按时间步分别加噪。
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # 对应 randomize_rigid_body_mass_base（base_link, add）
        base_extra_payload = ObsTerm(
            func=mdp.privileged_base_extra_payload,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        )
        # 对应 randomize_rigid_body_mass（非base_link, scale）
        end_effector_payload = ObsTerm(
            func=mdp.privileged_end_effector_payload,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")},
        )
        # 对应 randomize_com_positions（base_link）
        # base_com_offset = ObsTerm(
        #     func=mdp.privileged_base_com_offset,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
        # )
        # 对应 randomize_rigid_body_inertia
        inertia_scale = ObsTerm(
            func=mdp.privileged_rigid_body_inertia,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        # 对应 randomize_actuator_gains
        gain_scale = ObsTerm(
            func=mdp.privileged_joint_gain_scale,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        # 对应 randomize_rigid_body_material 的 静摩擦、动摩擦、恢复系数
        material_properties = ObsTerm(
            func=mdp.privileged_material_properties,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*wheel"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    history: HistoryCfg = HistoryCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

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
            "feet_cfg": SceneEntityCfg("robot", body_names=".*wheel"),  # 足端的 body_names 正则表达式
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

    # ── Stage 4：75k步后 v_x 范围升级到 (-2, 2) ──────────────────
    base_velocity_lin_vel_x_s4: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-2.0, 2.0),
                "num_steps": 75_000,
            },
        },
    )
    # ── Stage 5：100k步后 v_x 范围升级到 (-3, 3) ─────────────────
    base_velocity_lin_vel_x_s5: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-3.0, 3.0),
                "num_steps": 100_000,
            },
        },
    )
    # ── Stage 6：125k步后 v_x 范围升级到 (-4, 4) ─────────────────
    base_velocity_lin_vel_x_s6: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-4.0, 4.0),
                "num_steps": 125_000,
            },
        },
    )
    # ── Stage 7：150k步后 v_x 范围升级到 (-5, 5) ─────────────────
    base_velocity_lin_vel_x_s7: CurrTerm = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_value,
            "modify_params": {
                "value": (-5.0, 5.0),
                "num_steps": 150_000,
            },
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
        
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
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
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import DeeproboticsM20RoughEnvCfg

@configclass
class RoughEnvWBCConfig(DeeproboticsM20RoughEnvCfg):
    
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

        self.terminations.root_height_below_minimum = None # pyramid_stairs_inv地形存在高度低于0m的部分，删除根据高度判断终止的条件
        # self.scene.terrain.terrain_generator=mdp.ALL_TERRAINS_CFG
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.rewards.body_height_tracking.weight = 0.8
        # self.rewards.body_pitch_tracking.weight = 0.8
        # self.rewards.body_roll_tracking.weight = 0.0
        self.commands.body_pose.height_range = (0.513, 0.513)  # Stage 1 初始值
        self.commands.body_pose.pitch_range  = (0.0, 0.0)
        self.commands.body_pose.roll_range   = (0.0, 0.0)
        
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RoughEnvWBCConfig":
            self.disable_zero_weight_rewards()

class FlatEnvWBCConfig_PLAY(FlatEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.commands.base_velocity.ranges.lin_vel_x = (-5.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.body_pose.height_range = (0.33, 0.6)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        
        if self.__class__.__name__ == "FlatEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()

class RoughEnvWBCConfig_PLAY(RoughEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.body_pose.height_range = (0.33, 0.6)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        if self.__class__.__name__ == "RoughEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()

class RoughWOStairsEnvWBCConfig(RoughEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = mdp.NONE_STAIRS_TERRAINS_CFG
        self.curriculum.base_velocity_lin_vel_x_s4 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {
                    "value": (-2.0, 2.0),
                    "num_steps": 75_000,
                },
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s5 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {
                    "value": (-3.0, 3.0),
                    "num_steps": 100_000,
                },
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s6 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {
                    "value": (-4.0, 4.0),
                    "num_steps": 125_000,
                },
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s7 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {
                    "value": (-5.0, 5.0),
                    "num_steps": 150_000,
                },
            },
        )
        if self.__class__.__name__ == "RoughWOStairsEnvWBCConfig":
            self.disable_zero_weight_rewards()

class RoughWOStairsEnvWBCConfig_PLAY(RoughWOStairsEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.body_pose.height_range = (0.33, 0.6)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        if self.__class__.__name__ == "RoughWOStairsEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()