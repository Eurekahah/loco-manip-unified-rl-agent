# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg ,ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Lite3/Lite3_usd/Lite3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.8,
            ".*Knee_joint": 1.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
    },
)

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_usd/M20.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=5,
        ),
    },
)


DEEPROBOTICS_M20_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/usd/M20_assemble.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/usd/M20_adjusted.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20_Piper_own/usd/M20_Piper_own.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*wheel_joint": 0.0,
            # 机械臂旋转关节
            "arm_joint1": 0.0,       # limit: [-2.618, 2.618]  ✓ 安全
            "arm_joint2": 0.5,       # limit: [0, 3.14]        ⚠️ 边界改为0.1
            "arm_joint3": -0.5,      # limit: [-2.697, 0]      ⚠️ 边界改为-0.1
            "arm_joint4": 0.0,       # limit: [-1.832, 1.832]  ✓ 安全
            "arm_joint5": 0.0,       # limit: [-1.22, 1.22]    ✓ 安全
            "arm_joint6": 0.0,       # limit: [-3.14, 3.14]    ✓ 安全
            # 夹爪prismatic关节
            "gripper_joint1": 0.0,       # limit: [0, 0.05]        ✓ 夹爪闭合
            "gripper_joint2": 0.0,       # limit: [-0.05, 0]       ✓ 夹爪闭合
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=5,
        ),
        "piper_arm": DelayedPDActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit=100.0,       # 根据 Piper 实际力矩限制填写
            velocity_limit=3.0,     # rad/s
            # ⚠️ 标定记录（2026-09-19，见 docs/review/bad_orientation_analysis_zh.md）：
            # 下面注释里的 "DelayedPD 在 60~100（stiffness） / 0~20（damping）" 是当时定的目标区间，
            # 而现在的 300/20 **超出该区间 3~5 倍**。实测（同一份已训低层策略、同一任务）：
            #   软臂（=7 月那代 Implicit 40/8 的效果）→ 臂关节速度 RMS 0.95 rad/s，倾角越限比例 0.0002
            #   硬臂（当前 300/20）              → 臂关节速度 RMS 1.74 rad/s，倾角越限比例 0.0006
            # 即"臂刚度 ≈ 它把多少扰动传给底盘"。仍然偏"硬"的话，可选方案是把 300 降回 60~100，
            # 或者做"刚度课程"（前段 40/8，训练中后期线性升到 300/20；实现方式：
            # 在 curriculum 里改 robot.actuators["piper_arm"].stiffness/damping 后调用
            # robot.write_joint_stiffness_to_sim(...) / write_joint_damping_to_sim(...)）。
            # 目前**没有**启用刚度课程：先靠 EE 目标课程（WBCCurriculumCfg 的 ee_goal_*_blend_*
            # 与 HeightInvariantEECommandCfg.target_blend_*）把"早期一动臂就终止"这一条解决掉。
            stiffness=300.0, # 20
            damping=20, # 0.1
            friction=0.01,
            armature=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "piper_gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint[1-2]"],
            effort_limit=10.0,       # 根据 Piper 实际力矩限制填写
            velocity_limit=1.0,     # rad/s
            stiffness=4000.0, # 20
            damping=200.0, # 0.1
            friction=0.01,
            armature=0.01,
            min_delay=0,
            max_delay=5,
        ),
        # "piper_arm": ImplicitActuatorCfg(
        #     joint_names_expr=["arm_joint[1-6]"],
        #     effort_limit_sim=100.0,       # 力矩限制（仿真）
        #     velocity_limit_sim=3.0,     # rad/s
        #     stiffness=40.0, # 20  # 在ImplicitActuatorCfg不生效，用DelayedPDActuatorCfg在60-100
        #     damping=8.0, # 0.1  # 用DelayedPDActuatorCfg在0-20
        #     friction=0.01,
        #     armature=0.01,
        #     # min_delay=0,
        #     # max_delay=4,
        # ),


        # # 新增：夹爪（如果是位置控制）
        # "piper_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["gripper_joint[1-2]"],
        #     effort_limit_sim=100.0,
        #     velocity_limit_sim=1.0,
        #     stiffness=4000.0,
        #     damping=200.0,
        #     friction=0.0,
        #     armature=0.0,
        #     # min_delay=0,
        #     # max_delay=5,
        # ),
    },
)

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_usd/Piper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # 机械臂旋转关节
            "arm_joint1": 0.0,       # limit: [-2.618, 2.618]  ✓ 安全
            "arm_joint2": 0.0,       # limit: [0, 3.14]        ⚠️ 边界改为0.1
            "arm_joint3": 0.0,      # limit: [-2.697, 0]      ⚠️ 边界改为-0.1
            "arm_joint4": 0.0,       # limit: [-1.832, 1.832]  ✓ 安全
            "arm_joint5": 0.0,       # limit: [-1.22, 1.22]    ✓ 安全
            "arm_joint6": 0.0,       # limit: [-3.14, 3.14]    ✓ 安全
            # 夹爪prismatic关节
            "gripper_joint1": 0.0,       # limit: [0, 0.05]        ✓ 夹爪闭合
            "gripper_joint2": 0.0,       # limit: [-0.05, 0]       ✓ 夹爪闭合
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit=100.0,       # 根据 Piper 实际力矩限制填写
            velocity_limit=3.0,     # rad/s
            stiffness=400.0, # 20
            damping=80.0, # 0.1
            friction=0.01,
            armature=0.01,
            # min_delay=0,
            # max_delay=0,
        ),


        # 新增：夹爪（如果是位置控制）
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint[1-2]"],
            effort_limit=100.0,
            velocity_limit=1.0,
            stiffness=4000.0,
            damping=200.0,
            friction=0.0,
            armature=0.0,
            # min_delay=0,
            # max_delay=5,
        ),
    },
)
