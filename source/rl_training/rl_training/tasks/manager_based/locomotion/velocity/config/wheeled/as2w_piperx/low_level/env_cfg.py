# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 平地低层 RL 环境（训练 / PLAY）。

设计约束（交接单 §4 / §6.3 / D-09）：

* **不继承** ``velocity_env_cfg.LocomotionVelocityRoughEnvCfg``，也**不 import**
  ``deeprobotics_m20/*``；直接继承 ``ManagerBasedRLEnvCfg``，五个 manager 组各自来自
  ``low_level/*``，场景来自 ``scene_cfg.AS2WPiperXSceneCfg``。
* ``__post_init__`` 只做**环境级**设置（``sim`` / ``decimation`` / ``episode_length_s`` /
  ``viewer``），不写任何奖励 / 命令 / 课程数值（它们都在各自的 cfg 文件里）。
* 物理参数（求解器迭代次数、patch 上限、``bounce_threshold_velocity``、接触传感器周期）
  复刻 Anchor（``params/env.yaml``）；这些是 ``sim.*`` 设置，不属于 MDP 数值。
* **禁止**任何"按类名分支"的写法（PLAY 用独立子类覆盖，而非在 ``__post_init__`` 里判断类名）。
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.actions import (
    ActionsCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.commands import (
    CommandsCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.curriculum import (
    CurriculumCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.events import (
    EventCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.observations import (
    ObservationsCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.rewards import (
    RewardsCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.low_level.terminations import (
    TerminationsCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.scene_cfg import (
    AS2WPiperXSceneCfg,
)


_PROFILE = AS2W_PIPERX_PROFILE


@configclass
class AS2WPiperXFlatEnvCfg(ManagerBasedRLEnvCfg):
    """AS2W-PiperX 平地低层速度 / EE / 机身姿态跟踪环境（Anchor 口径）。"""

    # ---- 场景（平地 + 机器人 + 灯光 + 接触传感器）----
    scene: AS2WPiperXSceneCfg = AS2WPiperXSceneCfg(num_envs=4096, env_spacing=2.5)

    # ---- manager 组 ----
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # ---- 时序 ----
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # ---- 物理（复刻 Anchor params/env.yaml）----
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 1
        self.scene.contact_forces.update_period = self.sim.dt

        # ---- 视角（仅 GUI 生效）----
        self.viewer.eye = (1.70, 1.35, 1.05)
        self.viewer.lookat = (0.03, 0.0, 0.50)


@configclass
class AS2WPiperXFlatEnvCfg_PLAY(AS2WPiperXFlatEnvCfg):
    """PLAY：不做课程，直接给完整任务；命令/EE 区间放开到工作空间。

    按 M20 ``FlatEnvWBCConfig_PLAY`` 的做法：关掉 EE / 扰动 / 速度 / 姿态课程，
    底盘速度命令置零（演示以站立 + EE 目标为主），静止惩罚权重取训练终值。
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- 关闭所有课程项（PLAY 不再推进）----
        self.curriculum.ee_goal_stages = None
        self.curriculum.disturbance_ramp = None
        self.curriculum.body_pose_pitch_range_s3 = None
        self.curriculum.body_pose_roll_range_s3 = None
        self.curriculum.base_velocity_lin_vel_x_s4 = None
        self.curriculum.base_velocity_lin_vel_x_s5 = None
        self.curriculum.base_velocity_lin_vel_x_s6 = None
        self.curriculum.base_velocity_lin_vel_x_s7 = None
        self.curriculum.standing_env_ratio_ramp = None
        self.curriculum.stand_still_vel_ramp = None
        self.curriculum.stand_still_wheel_vel_ramp = None

        # ---- EE 区间：放开到实测工作空间 + Stage 3 姿态 ----
        self.commands.ee_pose.ranges.p_l = _PROFILE.ee_workspace[0]
        self.commands.ee_pose.ranges.p_pitch = _PROFILE.ee_workspace[1]
        self.commands.ee_pose.ranges.p_yaw = _PROFILE.ee_workspace[2]
        self.commands.ee_pose.ranges.o_roll = (-math.pi / 8.0, math.pi / 8.0)
        self.commands.ee_pose.ranges.o_pitch = (-math.pi / 8.0, math.pi / 8.0)
        self.commands.ee_pose.ranges.o_yaw = (-math.pi, math.pi)

        # ---- 底盘速度命令置零（M20 PLAY 口径）----
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # ---- 机身姿态：高度放开到实测区间，pitch/roll 保持默认 ----
        self.commands.body_pose.height_range = _PROFILE.base_height_range
        self.commands.body_pose.pitch_range = (0.35, 0.35)
        self.commands.body_pose.roll_range = (0.0, 0.0)

        # ---- 静止惩罚取训练终值 ----
        self.rewards.stand_still_vel.weight = -2.0
        self.rewards.stand_still_wheel_vel.weight = -5e-4
