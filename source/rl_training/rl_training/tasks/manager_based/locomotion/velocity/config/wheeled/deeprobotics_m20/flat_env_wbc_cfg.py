import math

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
        # 上界 0.60 → 0.55：实测（probe_root_height_termination.py）机器人在
        # (0.51,0.55] 桶已经系统性偏低 +0.019 m、(0.55,0.60] 桶偏低 +0.043~0.091 m，
        # 即 0.60 根本够不到；而"够不到还硬拉"的那部分姿态恰好是摔倒率最高的桶
        # （root_z<0.30 的比例 5.6%~11.1%，其余桶 0~5%）。
        # 下界保留 0.33：实测能蹲到 0.32~0.34（该桶偏差 −0.006~−0.016 m，不是够不到）。
        height_range=(0.33, 0.55),
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
        # 稳态高度误差：裁剪 ±0.15 m 后统计（避免塌陷瞬间把 body_pose/height_error_bias 拉偏，
        # 实测：塌陷时单次误差 ~0.35 m，而稳态只有 2~3 cm。见 known_issues #20）
        steady_error_clip=0.15,
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
        history_length=10 的对齐窗口，简化实现；
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_base")},
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
                "value": (0.33, 0.55),
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

    # ── EE 目标课程 s0→s3：先把机械臂锁在**低位**，再逐步放开 ────────────────
    #
    # 背景（docs/review/DEFECT_LOG_zh.md DEF-006）：机械臂目标从"默认位姿"开始
    # 逐步放开。**但实测推翻了"锁在默认位姿"这个锚点**（`probe_root_height_termination.py`，
    # 512 envs × 20 s，19999 iter 的策略）：
    #
    #   | s0 锚点 | `root_z<0.30` 的 20s 触发率 |
    #   |---|---|
    #   | 臂跟随全范围目标（= 无课程） | 25.8% |
    #   | 锁在**默认位姿**（举起：EE 在采样平面之上 0.32 m） | **55.5%**（更差！） |
    #   | 锁在**低位锚点**（任务工作空间中心 r=0.41、仰角 −0.08 rad） | **1.0%** |
    #
    # 原因：默认位姿是"把臂举起来"，重心高、更衣倒；而任务采样的目标中心是"前伸低位"。
    # 所以 s0 锁的是**低位锚点**，s1/s2 以它为中心逐步放宽，s3 = 原有完整分布。
    #
    # 实现：一个 `mdp.apply_range_stages` 课程项负责 6 个 ranges 字段的阶段推进
    # （幂等；s0 的取值写在 `FlatEnvWBCConfig.__post_init__` / `RoughEnvWBCConfig.__post_init__` 里）。
    # `HeightInvariantEECommandCfg.target_blend_pos/_orn` 机制保留（默认 1.0 = 直接用采样目标），
    # 它是"按比例释放"的可选旋钮，本课程只用区间阶梯。
    ee_goal_stages: CurrTerm = CurrTerm(
        func=mdp.apply_range_stages,
        params={
            "command_name": "ee_pose",
            "stages": [
                # s1（25k 步）：围绕低位锚点的小范围移动；姿态仍锁 0
                {
                    "num_steps": 25_000,
                    "ranges": {
                        "p_l": (0.36, 0.47),
                        "p_pitch": (-0.35, 0.20),
                        "p_yaw": (-0.45, 0.45),
                        "o_roll": (-0.09, 0.09),
                        "o_pitch": (-0.09, 0.09),
                        "o_yaw": (0.0, 0.0),
                    },
                },
                # s2（50k 步）：位置再放宽 + 姿态 ±10° 量级（o_yaw 放到 ±0.6）
                {
                    "num_steps": 50_000,
                    "ranges": {
                        "p_l": (0.33, 0.50),
                        "p_pitch": (-0.55, 0.40),
                        "p_yaw": (-0.85, 0.85),
                        "o_roll": (-0.175, 0.175),
                        "o_pitch": (-0.175, 0.175),
                        "o_yaw": (-0.6, 0.6),
                    },
                },
                # s3（75k 步）：完整任务（= `DeeproboticsM20CommandsCfg.ee_pose` 的原分布）
                {
                    "num_steps": 75_000,
                    "ranges": {
                        "p_l": (0.30, 0.52),
                        "p_pitch": (-0.7853981633974483, 0.6283185307179586),
                        "p_yaw": (-1.2566370614359172, 1.2566370614359172),
                        "o_roll": (-0.39269908169872414, 0.39269908169872414),
                        "o_pitch": (-0.39269908169872414, 0.39269908169872414),
                        "o_yaw": (-3.141592653589793, 3.141592653589793),
                    },
                },
            ],
        },
    )

    # ── 扰动课程：push / 外力从 30% 线性放大到 100%（25k 步内）────────────────
    # 实测：第 0 步就全量开启 push（每 10~15 s、±0.5 m/s）与 reset 外力（±10 N / ±10 N·m）时，
    # `root_z<0.30` 的 20s 触发率 22.9% → 25.8%（多 3 个百分点），而且早期"一被推就趴窝"
    # 会被记成高度终止。早段压低扰动，让底盘先把平衡学会。
    disturbance_ramp: CurrTerm = CurrTerm(
        func=mdp.apply_event_scale,
        params={
            "num_steps": 25_000,
            "start_scale": 0.3,
            "spec": [
                {"term": "randomize_push_robot", "param": "velocity_range",
                 "base": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
                {"term": "randomize_apply_external_force_torque", "param": "force_range",
                 "base": (-10.0, 10.0)},
                {"term": "randomize_apply_external_force_torque", "param": "torque_range",
                 "base": (-10.0, 10.0)},
            ],
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
        # 本次训练**保留** ee_goal 观测（不使用 "去掉 ee_goal" 的版本）。
        # 代价：低层 policy 观测宽度 76 -> 83，之前在不含 ee_goal 下训出的
        # checkpoint 都不能再复用，必须重训。
        # self.observations.policy.ee_goal = None
        # self.observations.critic.ee_goal = None
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

        # ── EE 目标课程 Stage 0：**低位锚点** ────────────────────────────────
        # 目标锁死在工作空间中心（r=0.41 m、仰角 −0.08 rad、方位 0、姿态 o_*=0）：
        # 机械臂一次性放到"前伸低位"，之后在 s0 期间不再移动（重心低、不扰动底盘）。
        # ⚠️ 不要锁在**默认位姿**（那是"举起"姿态，实测让 root_z<0.30 的 20s 触发率
        # 从 25.8% 涨到 55.5%；锁低位只有 1.0%）—— 见 WBCCurriculumCfg.ee_goal_stages 的注释。
        self.commands.ee_pose.ranges.p_l = (0.41, 0.41)
        self.commands.ee_pose.ranges.p_pitch = (-0.08, -0.08)
        self.commands.ee_pose.ranges.p_yaw = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_yaw = (0.0, 0.0)

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
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
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

        # EE 目标课程 Stage 0（同 FlatEnvWBCConfig：锁在**低位锚点**，臂不动且重心低）
        self.commands.ee_pose.ranges.p_l = (0.41, 0.41)
        self.commands.ee_pose.ranges.p_pitch = (-0.08, -0.08)
        self.commands.ee_pose.ranges.p_yaw = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.o_yaw = (0.0, 0.0)

        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RoughEnvWBCConfig":
            self.disable_zero_weight_rewards()

@configclass
class FlatEnvWBCConfig_PLAY(FlatEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        # PLAY 直接给完整任务（不做 EE 目标课程 / 扰动课程）：关掉课程项 + 放开 EE 区间
        self.curriculum.ee_goal_stages = None
        self.curriculum.disturbance_ramp = None
        self.commands.ee_pose.ranges.p_l = (0.30, 0.52)
        self.commands.ee_pose.ranges.p_pitch = (-math.pi / 4, math.pi / 5)
        self.commands.ee_pose.ranges.p_yaw = (-2 * math.pi / 5, 2 * math.pi / 5)
        self.commands.ee_pose.ranges.o_roll = (-math.pi / 8, math.pi / 8)
        self.commands.ee_pose.ranges.o_pitch = (-math.pi / 8, math.pi / 8)
        self.commands.ee_pose.ranges.o_yaw = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.body_pose.height_range = (0.33, 0.55)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        
        if self.__class__.__name__ == "FlatEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()
@configclass
class RoughEnvWBCConfig_PLAY(RoughEnvWBCConfig):
    def __post_init__(self):
        super().__post_init__()
        # self.curriculum.body_pose_cmd_schedule = None
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        # PLAY 直接给完整任务（不做 EE 目标课程 / 扰动课程）
        self.curriculum.ee_goal_stages = None
        self.curriculum.disturbance_ramp = None
        self.commands.ee_pose.ranges.p_l = (0.30, 0.52)
        self.commands.ee_pose.ranges.p_pitch = (-math.pi / 4, math.pi / 5)
        self.commands.ee_pose.ranges.p_yaw = (-2 * math.pi / 5, 2 * math.pi / 5)
        self.commands.ee_pose.ranges.o_roll = (-math.pi / 8, math.pi / 8)
        self.commands.ee_pose.ranges.o_pitch = (-math.pi / 8, math.pi / 8)
        self.commands.ee_pose.ranges.o_yaw = (-math.pi, math.pi)
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.body_pose.height_range = (0.33, 0.55)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        if self.__class__.__name__ == "RoughEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()
@configclass
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
        # self.disable_rewards(keep=[])
        # self.disable_curriculum(keep=[])
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # self.commands.ee_pose = None
        # self.observations.policy.ee_goal = None
        # self.observations.critic.ee_goal = None
        # self.actions.ee_ik = None
        
    def disable_curriculum(self, keep: list[str] | None = None, exclude: list[str] | None = None):
        """
        禁用CurrTerm用于逐项profiling。

        Args:
            keep: 只保留这些名字的curriculum term，其余全部设为None。
                传 [] 则禁用全部curriculum term。
                传 None 则不按keep过滤(配合exclude使用，或者两者都不传=只按weight==0原逻辑禁用)。
            exclude: 禁用这些名字的curriculum term，其余保留。
                    keep和exclude不要同时传，keep优先级更高。
        """
        all_names = [
            attr for attr in dir(self.curriculum)
            if not attr.startswith("__") and not callable(getattr(self.curriculum, attr))
        ]

        disabled = []
        for name in all_names:
            curriculum_attr = getattr(self.curriculum, name)
            if curriculum_attr is None:
                continue

            should_disable = False
            if keep is not None:
                should_disable = name not in keep
            elif exclude is not None:
                should_disable = name in exclude
            else:
                should_disable = False  # 默认不禁用

            if should_disable:
                setattr(self.curriculum, name, None)
                disabled.append(name)

        print(f"[disable_curriculum] Disabled {len(disabled)} curriculum terms: {disabled}")
        return disabled

    def disable_rewards(self, keep: list[str] | None = None, exclude: list[str] | None = None):
        """
        禁用RewTerm用于逐项profiling。

        Args:
            keep: 只保留这些名字的reward term，其余全部设为None。
                传 [] 则禁用全部reward term。
                传 None 则不按keep过滤(配合exclude使用，或者两者都不传=只按weight==0原逻辑禁用)。
            exclude: 禁用这些名字的reward term，其余保留。
                    keep和exclude不要同时传，keep优先级更高。
        """
        all_names = [
            attr for attr in dir(self.rewards)
            if not attr.startswith("__") and not callable(getattr(self.rewards, attr))
        ]

        disabled = []
        for name in all_names:
            reward_attr = getattr(self.rewards, name)
            if reward_attr is None:
                continue

            should_disable = False
            if keep is not None:
                should_disable = name not in keep
            elif exclude is not None:
                should_disable = name in exclude
            else:
                should_disable = reward_attr.weight == 0

            if should_disable:
                setattr(self.rewards, name, None)
                disabled.append(name)

        print(f"[disable_rewards] Disabled {len(disabled)} reward terms: {disabled}")
        return disabled
@configclass
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
        self.commands.body_pose.height_range = (0.33, 0.55)
        self.commands.body_pose.pitch_range = (-0.35, 0.35)
        self.commands.body_pose.roll_range = (-0.25, 0.25)
        if self.__class__.__name__ == "RoughWOStairsEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()
