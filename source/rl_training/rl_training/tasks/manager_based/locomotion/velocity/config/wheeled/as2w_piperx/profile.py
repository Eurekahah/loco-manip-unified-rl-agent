# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 的**唯一机器人常量来源**。

设计约束（交接单 §6.1）：

* ``@dataclass(frozen=True)``，只存 ``str / float / tuple / Mapping``；
* **不存 SceneEntityCfg** —— ``SceneEntityCfg.resolve()`` 会原地改写对象，模块级共享会被污染。
  需要 cfg 的地方一律走本文件末尾的工厂函数，每次新建。

数值来源见 ``logs/probe/as2w/final_off/report.md``（Phase 0 实测）：

============================  ===========  ===============================================
项                            实测值       说明
============================  ===========  ===============================================
站高（base_link 世界 z）       0.4466 m     200 步零动作，std < 1e-5
轮心世界 z = 等效轮半径        0.0774 m     四个 ``*_foot`` body 的均值
base 到轮心垂直距离            0.3692 m     用于交叉校验
arm_base_link 世界 z           0.5264 m     ``sampled_height`` 取这个值
EE 锚点（球坐标）              0.436 / -0.122 / 0    arm_base 系中位数（低重心、前伸）
最低可行蹲伏（FK，foot_x 约束） 0.246 m      动态位置保持到 0.332 m 仍稳定
镜像符号                      hip=-1, thigh=+1, calf=+1   FK 扫 θ ∈ [-0.3, 0.3]
self-collision                OFF          开启后站高被动抬高 2.2 cm、力矩峰值 5.6→60 N·m
============================  ===========  ===============================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from isaaclab.managers import SceneEntityCfg


# --------------------------------------------------------------------------- #
# 分组
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JointGroups:
    """关节分组（``legs`` 是 **12 个非轮腿关节**，轮在 ``wheels``）。"""

    legs: tuple[str, ...]
    wheels: tuple[str, ...]
    arm: tuple[str, ...]
    gripper: tuple[str, ...]
    hip: tuple[str, ...]
    thigh: tuple[str, ...]
    calf: tuple[str, ...]

    @property
    def leg_with_wheel(self) -> tuple[str, ...]:
        return self.legs + self.wheels

    @property
    def all(self) -> tuple[str, ...]:
        """按“组”拼接（**不是** articulation 原生顺序）。

        观测里用 ``joint_names=[".*"]`` + ``preserve_order=True`` 拿原生顺序；
        只有动作项需要显式分组顺序时才用这里的列表。
        """
        return self.legs + self.wheels + self.arm + self.gripper

    @property
    def num_joints(self) -> int:
        return len(self.all)


@dataclass(frozen=True)
class BodyGroups:
    base: str
    arm_base: str
    ee: str
    wheels: tuple[str, ...]
    wheel_regex: str
    non_wheel_regex: str


@dataclass(frozen=True)
class MirrorPair:
    """一对需要镜像的关节组：``a`` / ``b`` 是**匹配 3 个关节的正则**。"""

    a: str
    b: str
    signs: tuple[tuple[str, float], ...]

    def sign_dict(self) -> dict[str, float]:
        return dict(self.signs)


@dataclass(frozen=True)
class EEAnchor:
    """Stage 0 的固定 EE 目标（高度不变系球坐标）。"""

    p_l: float
    p_pitch: float
    p_yaw: float


@dataclass(frozen=True)
class WheeledArmProfile:
    """一台“轮式底盘 + 机械臂”机器人的全部常量。"""

    name: str
    joint_groups: JointGroups
    body_groups: BodyGroups

    # ---- actuator / 命名 ----
    actuator_names: tuple[str, ...]
    included_actuators: tuple[str, ...]
    wheel_joint_regex: str
    foot_body_regex: str
    non_wheel_body_regex: str

    # ---- 几何 / 高度 ----
    wheel_radius: float
    default_base_height: float
    base_height_range: tuple[float, float]
    termination_height: float

    # ---- 动作 ----
    leg_action_scale: tuple[tuple[str, float], ...]
    wheel_action_scale: float

    # ---- 镜像 ----
    mirror_pairs: tuple[MirrorPair, ...]
    joint_kind_patterns: tuple[tuple[str, str], ...]

    # ---- EE ----
    ee_anchor: EEAnchor
    ee_sampled_height: float
    ee_workspace: tuple[
        tuple[float, float],  # p_l
        tuple[float, float],  # p_pitch
        tuple[float, float],  # p_yaw
    ]
    ee_collision_lower: tuple[float, float, float]
    ee_collision_upper: tuple[float, float, float]
    ee_urdf_path: str | None
    ee_link_name: str
    arm_base_link_name: str

    # ---- 其他 ----
    contact_force_threshold: float
    self_collision: bool
    default_joint_pos: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    # ---- 便捷视图 ----
    def action_scale_dict(self) -> dict[str, float]:
        return dict(self.leg_action_scale)

    def kind_patterns_dict(self) -> dict[str, str]:
        return dict(self.joint_kind_patterns)

    def default_joint_pos_dict(self) -> dict[str, float]:
        return dict(self.default_joint_pos)

    def mirror_joints(self) -> list[list[str]]:
        return [[p.a, p.b] for p in self.mirror_pairs]

    def mirror_signs(self) -> list[dict[str, float]]:
        return [p.sign_dict() for p in self.mirror_pairs]


# --------------------------------------------------------------------------- #
# AS2W-PiperX 实例
# --------------------------------------------------------------------------- #
_LEGS = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)
_WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
_ARM = tuple(f"arm_joint{i}" for i in range(1, 7))
_GRIPPER = ("gripper_joint1", "gripper_joint2")

AS2W_PIPERX_PROFILE = WheeledArmProfile(
    name="AS2W-PiperX",
    joint_groups=JointGroups(
        legs=_LEGS,
        wheels=_WHEELS,
        arm=_ARM,
        gripper=_GRIPPER,
        hip=tuple(f"{leg}_hip" for leg in ("FL", "FR", "RL", "RR")),
        thigh=tuple(f"{leg}_thigh" for leg in ("FL", "FR", "RL", "RR")),
        calf=tuple(f"{leg}_calf" for leg in ("FL", "FR", "RL", "RR")),
    ),
    body_groups=BodyGroups(
        base="base_link",
        arm_base="arm_base_link",
        ee="gripper_base",
        wheels=_WHEELS,
        wheel_regex=r".*_foot",
        non_wheel_regex=r"^(?!.*_foot).*",
    ),
    actuator_names=("hip_thigh", "calf", "wheel", "piper_arm", "piper_gripper"),
    included_actuators=("hip_thigh", "calf", "wheel", "piper_arm", "piper_gripper"),
    # `*_foot` 既是关节名也是 body 名：这里两处语义分开，别混。
    wheel_joint_regex=r".*_foot",
    foot_body_regex=r".*_foot",
    non_wheel_body_regex=r"^(?!.*_foot).*",
    wheel_radius=0.0774,
    default_base_height=0.4466,
    base_height_range=(0.33, 0.45),
    termination_height=0.28,
    leg_action_scale=(("hip", 0.125), ("thigh", 0.25), ("calf", 0.25)),
    wheel_action_scale=5.0,
    mirror_pairs=(
        MirrorPair(
            a=r"FL_(hip|thigh|calf)",
            b=r"FR_(hip|thigh|calf)",
            signs=(("hip", -1.0), ("thigh", 1.0), ("calf", 1.0)),
        ),
        MirrorPair(
            a=r"RL_(hip|thigh|calf)",
            b=r"RR_(hip|thigh|calf)",
            signs=(("hip", -1.0), ("thigh", 1.0), ("calf", 1.0)),
        ),
    ),
    joint_kind_patterns=(("hip", r"_hip$"), ("thigh", r"_thigh$"), ("calf", r"_calf$")),
    ee_anchor=EEAnchor(p_l=0.436, p_pitch=-0.122, p_yaw=0.0),
    ee_sampled_height=0.5264,
    ee_workspace=(
        (0.30, 0.52),
        (-math.pi / 4.0, math.pi / 5.0),
        (-2.0 * math.pi / 5.0, 2.0 * math.pi / 5.0),
    ),
    # 高度不变系局部坐标；原点世界 z = ee_sampled_height，机体在下方 ⇒ z 为负。
    ee_collision_lower=(-0.32, -0.28, -0.54),
    ee_collision_upper=(0.34, 0.28, 0.05),
    ee_urdf_path=None,  # 在工厂函数里填绝对路径，避免 profile 里存绝对路径依赖
    ee_link_name="gripper_base",
    arm_base_link_name="arm_base_link",
    contact_force_threshold=120.0,
    self_collision=False,
    default_joint_pos=(
        ("FL_hip", 0.10), ("RL_hip", 0.10), ("FR_hip", -0.10), ("RR_hip", -0.10),
        ("FL_thigh", 0.44), ("FR_thigh", 0.44), ("RL_thigh", 0.44), ("RR_thigh", 0.44),
        ("FL_calf", -0.88), ("FR_calf", -0.88), ("RL_calf", -0.88), ("RR_calf", -0.88),
        ("FL_foot", 0.0), ("FR_foot", 0.0), ("RL_foot", 0.0), ("RR_foot", 0.0),
        ("arm_joint1", 0.0), ("arm_joint2", 0.5), ("arm_joint3", -0.5),
        ("arm_joint4", 0.0), ("arm_joint5", 0.0), ("arm_joint6", 0.0),
        ("gripper_joint1", 0.0), ("gripper_joint2", 0.0),
    ),
)


# --------------------------------------------------------------------------- #
# SceneEntityCfg 工厂（每次新建，避免 resolve() 原地改写）
# --------------------------------------------------------------------------- #
def leg_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(AS2W_PIPERX_PROFILE.joint_groups.legs), preserve_order=True)


def wheel_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(AS2W_PIPERX_PROFILE.joint_groups.wheels), preserve_order=True)


def all_joint_cfg() -> SceneEntityCfg:
    # `.*` + preserve_order 才是 articulation 原生顺序（与 joint_pos_rel_without_wheel 的前提一致）。
    return SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)


def wheel_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=AS2W_PIPERX_PROFILE.body_groups.wheel_regex)


def non_wheel_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=AS2W_PIPERX_PROFILE.body_groups.non_wheel_regex)


def base_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=AS2W_PIPERX_PROFILE.body_groups.base)


def ee_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=AS2W_PIPERX_PROFILE.body_groups.ee)