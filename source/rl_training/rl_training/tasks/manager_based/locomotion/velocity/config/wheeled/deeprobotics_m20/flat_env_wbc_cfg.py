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

    # ---- 静止伫立（零速命令）专项（DEF-026）--------------------------------
    # 旧配置里所有"站立"相关的项都是关闭状态：
    #   * `stand_still`（关节偏离默认姿态）/ `stand_still_without_cmd` → weight 0；
    #   * `wheel_vel_penalty`（零速时的轮速惩罚）→ weight 0；
    #   * `lin_vel_xy_l2_with_ang_z_command` → 未启用；
    #   * `joint_pos_penalty_wbc`（hipx/hipy/knee）虽有 -0.4/-0.1/-0.1，但它的
    #     `is_truly_still` 要求 `body_vel < 0.5` —— 一旦真的漂起来，这条惩罚自己就关了
    #     （鸡生蛋：越漂越没惩罚）。
    # 结果：命令 (0,0,0) 时底盘以 ~0.148 m/s 前向漂移（eval_fixed_command.py 实测）。
    # 这里补两项**只按命令门控**的惩罚；权重由课程从下面的初值爬到终值：
    #   stand_still_vel        -0.2  → -2.0
    #   stand_still_wheel_vel  -0.00005 → -0.0005
    # 权重量级怎么定的（**实测标定**，两步，别凭感觉改）：
    #
    # 【第一步：先估】RewardManager 返回 `Σ term_value·weight·dt`，而 `Episode_Reward/*`
    # 记的是**每秒速率**（episode 积分 / max_episode_length_s）。按"命令 (0,0,0) 时
    # 回报/秒 = 1.75、漂移 |v|≈0.15 m/s"估：weight=-8.0 ⇒ 静止 env 上约 -0.18/s（~10%）。
    #
    # 【第二步：用第一次 A/B 的实测把估值打脸】1500-iter run
    # `2026-09-29_18-30-52_stand_still_fix`（权重 -8.0/-0.01）在 iter 1234 实测
    # `Episode_Reward/stand_still_vel = -0.2603/s`、`stand_still_wheel_vel = -0.2230/s`
    # ⇒ **两项合计 -0.483/s，而同一步全 batch 的 Σ Episode_Reward 只有 +0.396/s**，
    # 也就是惩罚量级 = 总回报的 122%（折算到"静止 env"上 ≈ -3.2/s，是它们正回报的 2 倍）。
    # 反解出真实量级：静止 env 的 E|v|² ≈ 0.217（|v|≈0.47 m/s）、E[Σω²] ≈ 149（ω≈6.1 rad/s）。
    # 后果（同一次 A/B 的固定命令 eval，与**同代**基线 `2026-09-20_00-50-31/model_1500`
    # 对比，见 DONE_zh.md 第七节）：命令 (0,0,0) 的漂移确实降了（0.1153→0.0886），
    # 但摔倒率从 0.178 涨到 **0.708**，终止构成几乎全是 `bad_orientation_2`
    # （473 次 vs 76 次）—— 策略学会"把轮子冻住"，而轮式倒立摆靠轮子微动平衡 ⇒ 翻倒。
    #
    # 【最终取值】按"两项合计 ≈ 总回报的 10~15%"重新定标：
    #   stand_still_vel       -2.0    ⇒ 约 -0.065/s（~16%）
    #   stand_still_wheel_vel -0.0005 ⇒ 约 -0.011/s（~3%）
    # 分工：底盘速度项直接惩罚用户看到的"静止仍有前向速度"（且它是净速度，不干扰平衡用的
    # 微小往复）；轮速项只留一个很小的"别空转"信号。**注意这两组权重的对照实测见 DONE 第七节。**
    # 初值必须**非零**：`disable_zero_weight_rewards()` 会把 weight==0 的项置 None，
    # 之后课程再去 `get_term_cfg` 就会抛 ValueError。
    stand_still_vel = RewTerm(
        func=mdp.stand_still_vel_l2,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "yaw_weight": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            # B3 / DEF-061：**坡面上关掉静止惩罚**（上坡"停住轮子"= 往下滑 ⇒ 振荡）。
            # 用轮心高度差估"脚下坡度"，`slope_gate_rad` = 判为坡面的阈值（rad）。
            # ⚠️ 2026-10-09：**默认暂设 0（= 关闭门控，复现旧行为）** —— 它的验证 run
            # `cloud_slopefix10k`（多地形 10k）还在跑，结论未定；合并到 main 时不该把
            # 未验证的默认改动带进去。等 run 出来（若上坡指标明显改善）再单独提一条改回
            # 0.06（或按结果调阈值）。临时开启：hydra
            # `env.rewards.stand_still_vel_l2.params.slope_gate_rad=0.06`。
            "slope_gate_rad": 0.0,
            "feet_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
        },
    )
    stand_still_wheel_vel = RewTerm(
        func=mdp.stand_still_wheel_vel_l2,
        weight=-0.00005,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=None),  # 具体名单在 deeprobotics_m20 的 rough cfg 里填
            "slope_gate_rad": 0.0,   # B3 / DEF-061：同上，默认暂关（等 cloud_slopefix10k 验证）
            "feet_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
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
    # 实测（旧幅度）：第 0 步就全量开启 push（每 10~15 s、±0.5 m/s）与 reset 外力
    # （±10 N / ±10 N·m）时，`root_z<0.30` 的 20s 触发率 22.9% → 25.8%（多 3 个百分点），
    # 而且早期"一被推就趴窝"会被记成高度终止。早段压低扰动，让底盘先把平衡学会。
    #
    # 2026-09-29 按需求 4 **加强扰动**（EventCfg.randomize_push_robot：间隔 5~10 s、
    # x ±2.0 / y ±1.0 m/s、yaw ±0.52 rad/s），同时把这条课程拉长到 50k 步
    # （≈2083 iter）、起点压到 0.2×：全量扰动本身已经难了，再不给缓冲会前段就学不动。
    # `base` 必须与 EventCfg 里的**终态幅度**逐位一致（课程按它算绝对值，幂等）。
    disturbance_ramp: CurrTerm = CurrTerm(
        func=mdp.apply_event_scale,
        params={
            "num_steps": 50_000,
            "start_scale": 0.2,
            # B1/DEF-059 实验旋钮：爬到 peak_scale（>1 = 训练扰动比评测口径更强）并保持。
            # 1.0 = 旧行为（爬到 1.0×）。`cloud_push15_20k` 用 hydra 覆盖成 1.5 试。
            "peak_scale": 1.0,
            # 0 = 用 num_steps（注意：这里不能用 None —— hydra 覆盖 None 字段会按 NoneType
            # 校验，连 100 这种整数都传不进来，见 DEF-040 §5 的同类坑）
            "peak_steps": 0,
            "spec": [
                {"term": "randomize_push_robot", "param": "velocity_range",
                 "base": {"x": (-2.0, 2.0), "y": (-1.0, 1.0), "yaw": (-0.52, 0.52)}},
                {"term": "randomize_apply_external_force_torque", "param": "force_range",
                 "base": (-10.0, 10.0)},
                {"term": "randomize_apply_external_force_torque", "param": "torque_range",
                 "base": (-10.0, 10.0)},
            ],
        },
    )

    # ── 静止伫立课程（DEF-026）─────────────────────────────────────────────
    # 1) 站姿占比：2% → 15%（25k 步 ≈ 1042 iter）。策略见不到零命令就学不会站住。
    # 2) 两项静止惩罚的权重爬升；`start_*` 与 WBCRewardsCfg 里的初值一致。
    standing_env_ratio_ramp: CurrTerm = CurrTerm(
        func=mdp.ramp_command_param,
        params={
            "term_name": "base_velocity",
            "param": "rel_standing_envs",
            "start": 0.02,
            "end": 0.15,
            "num_steps": 25_000,
        },
    )
    stand_still_vel_ramp: CurrTerm = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "stand_still_vel",
            "start_weight": -0.2,
            "end_weight": -2.0,
            "num_steps": 25_000,
        },
    )
    stand_still_wheel_vel_ramp: CurrTerm = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "stand_still_wheel_vel",
            "start_weight": -0.00005,
            "end_weight": -0.0005,
            "num_steps": 25_000,
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
        # 静止伫立：PLAY 直接用终态（占比 0.15 + 满权重），不做爬升
        self.curriculum.standing_env_ratio_ramp = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.rewards.stand_still_vel.weight = -2.0
        self.rewards.stand_still_wheel_vel.weight = -0.0005
        
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
        # 静止伫立：PLAY 直接用终态（占比 0.15 + 满权重），不做爬升
        self.curriculum.standing_env_ratio_ramp = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.rewards.stand_still_vel.weight = -2.0
        self.rewards.stand_still_wheel_vel.weight = -0.0005
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
        # 静止伫立：PLAY 直接用终态（占比 0.15 + 满权重），不做爬升
        self.curriculum.standing_env_ratio_ramp = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.rewards.stand_still_vel.weight = -2.0
        self.rewards.stand_still_wheel_vel.weight = -0.0005
        if self.__class__.__name__ == "RoughWOStairsEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()


# ============================================================================
# 消融（ablation）：把"加强扰动 / 静止伫立惩罚 / 镜像符号修正"三项改动**单独**打开，
# 用于判断各自的贡献（2026-09-30 加，配合云端 2×2 对照实验）。
# ----------------------------------------------------------------------------
# 背景：完整包（DEFECT_LOG_zh.md DEF-026~028）在 10000 iter 的中途验收里
# "静止漂移 / 步态对称"都变好了，但**训练期**的 `root_height_below_minimum`
# 从旧代码的 0.088 涨到 0.28（DEF-026 §4 / DONE 第七节 §5）。要拆开归因，需要两条对照：
#   A) PushOnly   = 只加强扰动（奖励与镜像都回"旧行为"）
#   B) RewardOnly = 只改奖励（静止惩罚 + 镜像修正），扰动保持旧值
# 加上已有的"旧代码 @10k"（`2026-09-20_00-50-31/model_10000.pt`）与
# "完整包 @10k"（`2026-09-30_00-09-25_cloud_soft20k/model_10000.pt`），
# 正好凑成 {旧/新 扰动} × {旧/新 奖励} 的 **2×2**，且四条都在 10000 iter / 4096 envs /
# seed 42 上，可直接用 `eval_fixed_command.py` + `summarize_run.py` 对比。
# ⚠️ 这两条**只是消融实验**，不要当成"可选配置"日常使用；正式配置 = 三项全开。
# ============================================================================


def _apply_ablation(self, *, new_stand_still: bool, new_mirror: bool, new_push: bool) -> None:
    """按开关把对应改动**退回旧行为**（幂等；默认三项全开时不改动任何东西）。"""
    if not new_stand_still:
        # 静止伫立专项：两项惩罚 + 三条课程全部撤掉，站姿占比回 0.02（= 旧口径）
        self.rewards.stand_still_vel = None
        self.rewards.stand_still_wheel_vel = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None
        self.curriculum.standing_env_ratio_ramp = None
        self.commands.base_velocity.rel_standing_envs = 0.02
    if not new_mirror:
        # 镜像惩罚退回旧实现：无符号约定的 `joint_mirror`，只留对角 2 对、权重 −0.03
        self.rewards.joint_mirror.func = mdp.joint_mirror
        self.rewards.joint_mirror.weight = -0.03
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
        self.rewards.joint_mirror.params.pop("mirror_signs", None)
    if not new_push:
        # 扰动退回旧口径：每 10~15 s、±0.5/±0.5 m/s、无 yaw；课程也回到 30%→100% / 25k 步
        legacy = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.events.randomize_push_robot.interval_range_s = (10.0, 15.0)
        self.events.randomize_push_robot.params["velocity_range"] = dict(legacy)
        ramp = getattr(self.curriculum, "disturbance_ramp", None)
        if ramp is not None:
            ramp.params["num_steps"] = 25_000
            ramp.params["start_scale"] = 0.3
            for item in ramp.params["spec"]:
                if item["term"] == "randomize_push_robot":
                    item["base"] = dict(legacy)


@configclass
class AblPushOnlyEnvWBCConfig(FlatEnvWBCConfig):
    """消融 A：只加强扰动（静止惩罚与镜像符号都回旧行为）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_ablation(self, new_stand_still=False, new_mirror=False, new_push=True)
        if self.__class__.__name__ == "AblPushOnlyEnvWBCConfig":
            self.disable_zero_weight_rewards()


@configclass
class AblRewardOnlyEnvWBCConfig(FlatEnvWBCConfig):
    """消融 B：只改奖励（静止惩罚 + 镜像符号修正），扰动保持旧值。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_ablation(self, new_stand_still=True, new_mirror=True, new_push=False)
        if self.__class__.__name__ == "AblRewardOnlyEnvWBCConfig":
            self.disable_zero_weight_rewards()


@configclass
class AblLegacyAllEnvWBCConfig(FlatEnvWBCConfig):
    """消融 C（2026-10-02 新增）：三项改动**全退**（静止奖励 / 镜像 / 扰动都回旧行为）。

    用途：`AblPushOnly` 与 `AblRewardOnly` 共同包含、但**未被 2×2 控制**的那部分改动
    （最典型的是 `HeightInvariantEECommand.reset()` 即 DEF-034 §2 的 ⑫ 修复：
    复位首帧的 EE 目标不再是全 0 位姿）。本变体 = "main 行为 + ⑫ 等工程修复"，
    与云端旧代码 run（`2026-09-20_00-50-31`，同 seed / 同 10k）对照，就能单独量出 ⑫ 的影响。

    注意：本变体与 `main` 的差别**不止 ⑫** —— 分支上还有若干与训练无关的改动
    （删死代码 ⑧、加报错提示 ⑩、加启动打印 ⑱、`setup.py` 打包、视觉编码器本地权重、
    `vr_extented` 调试开关等），它们对低层训练是惰性的，但严格说这条轴测的是
    "⑫ + 这些惰性改动"。见 `docs/review/DEFECT_LOG_zh.md` DEF-040 §3。
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_ablation(self, new_stand_still=False, new_mirror=False, new_push=False)
        if self.__class__.__name__ == "AblLegacyAllEnvWBCConfig":
            self.disable_zero_weight_rewards()


# ============================================================================
# 多地形（随机粗糙 + 正/反斜坡 + 平地）—— 需求 3，2026-09-29 新增
# ----------------------------------------------------------------------------
# 直接对标已有的两个多地形任务：
#   * `Rough-History-Adaptation-Deeprobotics-M20-v0`          → RoughEnvWBCConfig
#     （用 IsaacLab 官方 `ROUGH_TERRAINS_CFG`：含楼梯 / boxes / rails / pit 等，噪声 0.02~0.10）
#   * `Rough-WO-Stairs-History-Adaptation-Deeprobotics-M20-v0` → RoughWOStairsEnvWBCConfig
#     （`NONE_STAIRS_TERRAINS_CFG`：粗糙 0.35 + 上下坡 0.25/0.25 + 平地 0.15，噪声 0.02~0.10）
# 本类 = RoughWOStairs 的"温和噪声"版：地形组成只要 **随机粗糙 / 上坡 / 下坡 / 平地**，
# 且随机粗糙的噪声降到 **0.01~0.05**（见 `mdp.ROUGH_SLOPES_FLAT_TERRAINS_CFG`）。
# 其余（命令、奖励、课程、观测、history 窗口）全部沿用 RoughEnvWBCConfig，
# 因此也自动继承本轮的两处修复：静止伫立专项（DEF-026）与镜像符号修正（DEF-027）。
# ============================================================================
@configclass
class RoughSlopesEnvWBCConfig(RoughEnvWBCConfig):
    """多地形（随机粗糙 + 正反斜坡 + 平地，噪声 0.01~0.05）的 history-adaptation 版本。"""

    def __post_init__(self):
        super().__post_init__()
        # 地形：粗糙 0.40（噪声 0.01~0.05）+ 上坡 0.25 + 下坡 0.25 + 平地 0.10
        self.scene.terrain.terrain_generator = mdp.ROUGH_SLOPES_FLAT_TERRAINS_CFG
        # 地形课程：与 `LocomotionVelocityRoughEnvCfg.__post_init__` 里"按 curriculum 开关
        # 决定 terrain_generator.curriculum"的写法对齐 —— 本配置保留 `terrain_levels`
        # （RoughEnvWBCConfig 没关它），所以这里显式打开，让难度随成绩爬升。
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 5

        # 多地形上把 v_x 课程重新打开（RoughEnvWBCConfig 里被置 None 了）。
        #
        # ⚠️ 2026-10-08（B4 / DEF-044）：**这里直接落 SlowVx 的时间表**（四个台阶
        # 150k / 200k / 250k / 300k 环境步）——原来它们是 75k/100k/125k/150k，
        # 实测"地形等级中段爬到 5.8 之后在末段回落到 3.6"（DEF-040 §2），
        # 根因是**命令难度涨得比地形课程快**；台阶各推迟一倍后地形等级不再回落
        # （末 1000 `terrain_levels` 3.706(−0.09/1k) → **4.74(+0.095/1k)**、
        # `bad_orientation_2` −51%、回报 +33%，DEF-044）。
        # 所以把 SlowVx 配方作为**默认**：`Rough-Slopes-*-M20-v0` 现在就是当初
        # 验证过的那个配方；`RoughSlopesSlowVxEnvWBCConfig` 保留为**别名**（不再二次推迟）。
        # `-play-` 变体不受影响（PLAY 把四个台阶全置 None，直接用终态 ±5）。
        self.curriculum.base_velocity_lin_vel_x_s4 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {"value": (-2.0, 2.0), "num_steps": 150_000},
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s5 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {"value": (-3.0, 3.0), "num_steps": 200_000},
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s6 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {"value": (-4.0, 4.0), "num_steps": 250_000},
            },
        )
        self.curriculum.base_velocity_lin_vel_x_s7 = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "commands.base_velocity.ranges.lin_vel_x",
                "modify_fn": mdp.override_value,
                "modify_params": {"value": (-5.0, 5.0), "num_steps": 300_000},
            },
        )
        if self.__class__.__name__ == "RoughSlopesEnvWBCConfig":
            self.disable_zero_weight_rewards()


@configclass
class RoughSlopesEnvWBCConfig_PLAY(RoughSlopesEnvWBCConfig):
    """PLAY：直接给完整任务（关课程 + 放开 EE 区间 + 终态静止权重）。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.body_pose_height_range_s2 = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.curriculum.ee_goal_stages = None
        self.curriculum.disturbance_ramp = None
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        self.curriculum.standing_env_ratio_ramp = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None

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

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.rewards.stand_still_vel.weight = -2.0
        self.rewards.stand_still_wheel_vel.weight = -0.0005
        if self.__class__.__name__ == "RoughSlopesEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()


# ============================================================================
# SlowVx：v_x 命令课程台阶 ×2（2026-10-02 提出 → **2026-10-08 落成默认**，B4/DEF-044）
# ----------------------------------------------------------------------------
# 背景：`cloud_roughslopes20k` 的地形等级在中段爬到峰值 5.80（满分 9），后 1/3 回落到 3.6。
# 后 1/3 恰好在放开 v_x 命令课程（±2→±5 m/s，s4~s7）⇒ 根因是"命令难度涨得比地形课程快"。
# **验证结论（DEF-044）**：台阶各推迟一倍后地形等级不再回落
# （末 1000 `terrain_levels` 3.706（−0.09/1k）→ **4.74（+0.095/1k）**、`bad_orientation_2` −51%、
# 回报 +33%）⇒ 从 2026-10-08 起，这四档时间表**直接写在 `RoughSlopesEnvWBCConfig` 里**，
# 也就是说 `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` 现在就是验证过的那个配方；
# `SLOW_VX_FACTOR` / `RoughSlopesSlowVxEnvWBCConfig` 保留为**别名**（任务名继续可用，不再二次推迟）。
# `-play-` 变体不另开：PLAY 本来就关掉 v_x 课程（把所有台阶置 None），
# 所以验收直接用 `Rough-Slopes-History-Adaptation-Deeprobotics-M20-play-v0` 即可。
# ============================================================================
SLOW_VX_FACTOR: int = 2
"""v_x 命令课程台阶的推迟倍数（历史常量：该配方 2026-10-08 已落成
`RoughSlopesEnvWBCConfig` 的默认值，`RoughSlopesSlowVxEnvWBCConfig` 只是别名，不再乘它）。"""


@configclass
class RoughSlopesSlowVxEnvWBCConfig(RoughSlopesEnvWBCConfig):
    """**别名**：SlowVx 配方（v_x 台阶 ×2）从 2026-10-08 起已是 `RoughSlopesEnvWBCConfig` 的默认。

    保留这个类只是为了不破坏 `--task=Rough-Slopes-SlowVx-…-v0` 这个既有名字
    （B4：配方落默认、老 task 名继续可用），因此这里**不再二次推迟**。
    """

    def __post_init__(self):
        super().__post_init__()
        if self.__class__.__name__ == "RoughSlopesSlowVxEnvWBCConfig":
            self.disable_zero_weight_rewards()


@configclass
class RoughSlopesSlowVxEnvWBCConfig_PLAY(RoughSlopesEnvWBCConfig_PLAY):
    """PLAY：完整难度（v_x 课程本来就全关，所以与 `RoughSlopesEnvWBCConfig_PLAY` 等价）。

    单独建一个类只是为了 `--task=Rough-Slopes-SlowVx-…-play-v0` 这个名字好用
    （`play.py` / `policy_report.py` 的 `-play-` 变体习惯）。
    """

    def __post_init__(self):
        super().__post_init__()
        if self.__class__.__name__ == "RoughSlopesSlowVxEnvWBCConfig_PLAY":
            self.disable_zero_weight_rewards()
