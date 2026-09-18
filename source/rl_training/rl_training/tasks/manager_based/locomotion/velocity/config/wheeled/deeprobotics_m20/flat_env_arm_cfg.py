"""M20 平地场景 + **策略直接控制机械臂（关节空间）** 的隔离环境配置。

与 :class:`DeeproboticsM20FlatEnvCfg` 的区别（只在本文件内生效，不影响其它 env）：

1. 动作：去掉 ``ee_ik``（CommandDrivenIKAction 会和策略的臂关节目标抢同一批关节），
   新增 ``arm_joint_pos``（6 维），策略直接输出 6 个臂关节的位置目标增量。
   于是动作维度 = 腿 12 + 轮 4 + 臂 6 = 22。
2. 臂奖励：不再经过 ``ArmWeightCommand`` 门控（原实现里 ``arm_weight`` 恒为 ~0，
   导致所有臂奖励实际是死的），改为直接用官方 reward 权重 + 线性课程逐步加入。
3. 跟踪目标仍然是 ``ee_pose`` 命令（root 系位姿，与 IK 版本、与策略观测里的
   ``ee_goal`` 完全一致），由 mdp.arm_rewards 计算奖励。

注意：本 env 的动作/观测布局与其它 M20 env 不同（动作 22 维、policy 观测 86 维），
因此 checkpoint 不通用。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import DeeproboticsM20FlatEnvCfg
from .rough_env_cfg import DeeproboticsM20ActionsCfg, DeeproboticsM20CurriculumsCfg, DeeproboticsM20RewardsWithArmsCfg

##
# 调参提示
##
# arm_joint_pos.scale = 每步允许的臂关节目标增量（rad）。
#   Piper 臂关节速度上限 3.0 rad/s，控制周期 0.02 s → 单步物理上限约 0.06 rad；
#   0.1 略高于该上限（会被驱动器饱和限制），是"学得快但不会飞"的折中，可按训练效果调小/调大。
_ARM_ACTION_SCALE = 0.1
# 臂奖励从 start_frac 线性升到 1.0 所用的步数（policy step，非 iteration）。
_ARM_RAMP_STEPS = 20_000
_ARM_REWARD_START_FRAC = 0.1

# 臂关节正则项的权重按"零动作稳态实测值"标定（见下），目标是每步贡献 ≲ 0.25
# （对比：arm_ee_pos_tracking 满值 2.0；实测零动作稳态 value 见括号）。
#   arm_joint_torque  value ≈ 2.4e4  (sum τ²，含 gripper 质量随机化后的重力矩)
#   arm_joint_vel     value ≈ 5.5e1
#   arm_joint_acc     value ≈ 8.7e6  ← legacy 注释里的 -1e-5 会给 -87/step，直接压死训练信号
_ARM_TORQUE_WEIGHT = -1e-5
_ARM_VEL_WEIGHT = -1e-3
_ARM_ACC_WEIGHT = -1e-8


@configclass
class ArmJointActionsCfg(DeeproboticsM20ActionsCfg):
    """腿=关节位置、轮=关节速度、臂=关节位置；不再有 IK 动作项。"""

    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint[1-6]"],
        scale=_ARM_ACTION_SCALE,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class ArmEnvRewardsCfg(DeeproboticsM20RewardsWithArmsCfg):
    """把臂奖励从 ``arm_weight`` 门控里解出来（weight 直接生效，由课程控制大小）。

    与 legacy 版本的差异只在 ``arm_weight_command_name=None`` 与权重取值上；
    奖励函数本体（含坐标系修复）在 ``mdp/arm_rewards.py``。
    """

    # 1. EE 位置跟踪（root 系，密集）
    arm_ee_pos_tracking = RewTerm(
        func=mdp.ee_position_tracking,
        weight=2.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "gripper_base",
            "std": 0.15,
            "arm_weight_command_name": None,
        },
    )

    # 2. EE 姿态跟踪（相对旋转角，密集）
    arm_ee_ori_tracking = RewTerm(
        func=mdp.ee_orientation_tracking,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "gripper_base",
            "std": 0.5,
            "arm_weight_command_name": None,
        },
    )

    # 3. 到达目标（稀疏）
    arm_ee_goal_reached = RewTerm(
        func=mdp.ee_goal_reached,
        weight=5.0,
        params={
            "command_name": "ee_pose",
            "ee_frame_name": "gripper_base",
            "pos_threshold": 0.05,
            "angle_threshold": 0.2,
            "arm_weight_command_name": None,
        },
    )

    # 4. 臂关节正则项（legacy 里权重为 0，这里给出可用的默认值）
    arm_joint_torque = RewTerm(
        func=mdp.arm_joint_torque_penalty,
        weight=_ARM_TORQUE_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": None,
        },
    )
    arm_joint_vel = RewTerm(
        func=mdp.arm_joint_velocity_penalty,
        weight=_ARM_VEL_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": None,
        },
    )
    arm_joint_acc = RewTerm(
        func=mdp.arm_joint_acceleration_penalty,
        weight=_ARM_ACC_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*"),
            "arm_weight_command_name": None,
        },
    )


@configclass
class ArmEnvCurriculumCfg(DeeproboticsM20CurriculumsCfg):
    """线性加入臂奖励（替代原 ``ArmWeightCommand`` + ``advance_arm_weight`` 实现）。

    实现见 ``mdp.curriculums.ramp_reward_weight``；curriculum 在每次 episode reset 时被调用，
    所以实际是"每 ~1000 step 更新一次权重"的阶梯，对 20k step 的爬升足够平滑。
    """

    arm_pos_ramp = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "arm_ee_pos_tracking",
            "start_weight": 2.0 * _ARM_REWARD_START_FRAC,
            "end_weight": 2.0,
            "num_steps": _ARM_RAMP_STEPS,
        },
    )
    arm_ori_ramp = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "arm_ee_ori_tracking",
            "start_weight": 1.0 * _ARM_REWARD_START_FRAC,
            "end_weight": 1.0,
            "num_steps": _ARM_RAMP_STEPS,
        },
    )
    arm_goal_ramp = CurrTerm(
        func=mdp.ramp_reward_weight,
        params={
            "term_name": "arm_ee_goal_reached",
            "start_weight": 5.0 * _ARM_REWARD_START_FRAC,
            "end_weight": 5.0,
            "num_steps": _ARM_RAMP_STEPS,
        },
    )


@configclass
class DeeproboticsM20ArmEnvCfg(DeeproboticsM20FlatEnvCfg):
    """平地 + 策略直接控制机械臂（关节空间）。"""

    actions: ArmJointActionsCfg = ArmJointActionsCfg()
    rewards: ArmEnvRewardsCfg = ArmEnvRewardsCfg()
    curriculum: ArmEnvCurriculumCfg = ArmEnvCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # 1) 去掉 IK 动作项：臂由策略的 arm_joint_pos 直接控制
        self.actions.ee_ik = None

        # 2) 不再需要 arm_weight 命令（奖励已解耦）
        self.commands.arm_weight = None

        # 3) 统一清理 weight==0 的奖励项。
        #    父类的 disable_zero_weight_rewards() 有 `self.__class__.__name__ == "...FlatEnvCfg"` 的守卫，
        #    子类不会自动执行；不清理的话，像 feet_air_time_variance 这种带 body_names="" 占位的项
        #    会在 RewardManager 解析正则时直接抛 "Not all regular expressions are matched"。
        #    （arm_joint_deviation_l1 权重为 0，也会在这里被置 None —— 它和"跟踪 EE 目标"是冲突的。）
        self.disable_zero_weight_rewards()


@configclass
class DeeproboticsM20ArmEnvCfg_PLAY(DeeproboticsM20ArmEnvCfg):
    """回放用：直接用最终权重，去掉臂奖励爬升课程。"""

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.arm_pos_ramp = None
        self.curriculum.arm_ori_ramp = None
        self.curriculum.arm_goal_ramp = None
        self.rewards.arm_ee_pos_tracking.weight = 2.0
        self.rewards.arm_ee_ori_tracking.weight = 1.0
        self.rewards.arm_ee_goal_reached.weight = 5.0
