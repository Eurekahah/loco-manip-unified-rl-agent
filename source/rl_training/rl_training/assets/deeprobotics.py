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
            # ⚠️ 2026-10-08（B2-⑤ / DEF-064）：stiffness 保持 300，**damping 20 → 8**。
            # 开环定目标探针（`probe_arm_pd.py`，同一串已知可达的 EE 目标 + 训练好的站立策略）
            # 实测（有效增益 ≈ 配置值的 1.07~1.13 倍）：
            #   300/20（现状，有效 322/22.5）→ 逐档 |tau| 均 53.4 N·m、**饱和 23.5%**、
            #        静止档 47.6 N·m / |qd| p99 5.0、4 档平均位置误差 4.14 cm
            #   300/12 → 34.3 N·m、饱和 0%、2.71 cm
            #   300/8  → **22.5 N·m、饱和 0%、2.19 cm**（本次采用）
            #   150/8  → 21.0 N·m、饱和 0%、**2.89 cm**（再降刚度只会让跟踪变差）
            # 另：`max_delay` 0~5 → 0~0 几乎无变化 ⇒ 颤振**不是**延迟造成的，是阻尼项太大
            # （|qd| 到 5 rad/s 时 20×5 = 100 N·m = 满限幅）。
            # 注意：臂仍会贴 `joint_vel_limits`（joint1~5 是 5.0 rad/s，cfg 里写 3.0）
            # ⇒ 速度那条要另想办法（B2-⑤ 的 IK 限速/力矩层），不是 PD 能解决的。
            stiffness=300.0,
            damping=8,
            friction=0.01,
            armature=0.01,
            min_delay=0,
            max_delay=5,
        ),
        "piper_gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint[1-2]"],
            effort_limit=10.0,       # 根据 Piper 实际力矩限制填写
            velocity_limit=1.0,     # rad/s
            # ⚠️ 2026-10-08（B2-④ / DEF-058）：原值 4000 / 200 实测**长期顶满 10 N·m**
            # （饱和 88%/84%）。扫描结论（64 envs × 900 步，`logs/smoke/cloud_batch2.log`）：
            #   k4000 c200 → 饱和 88/84%、|tau|均 9.2 N·m（现状）
            #   k4000 c40  → 饱和 47/44%、|tau|均 6.0
            #   k4000 c12.6（临界阻尼）→ 饱和 6/3%、|tau|均 2.1
            #   k1000 c20  → 饱和 12/12%、|tau|均 2.8
            #   k286  c5   → 饱和 **0/0%**、|tau|均 **0.5**、|qd| p99 0.5 rad/s（限幅 1.0）
            # 取"行程匹配"刚度：夹爪行程 0.035 rad，满行程误差刚好给到限幅 10 N·m
            # ⇒ 夹持力上限不变（仍能顶到 10 N·m），但**不再长期饱和**、速度也在限幅内。
            # 阻尼取 ζ≈1.5（armature=0.01 下 c_crit≈3.4，取 5）。
            # 保守替代（只想改阻尼）：k=4000 / c=12.6，饱和也能从 88% 降到 6%。
            stiffness=286.0,
            damping=5.0,
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
