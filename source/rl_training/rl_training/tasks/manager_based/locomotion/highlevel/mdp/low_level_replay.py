# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""高层 -> 低层预训练策略的「回放（replay）」公共设施。

背景
----
高层 action term（``PreTrainedPickAction`` / ``PreTrainedPickWBCAction`` /
``TeleopLLAction`` ...）都要做同一件事：构造低层 policy 的观测 -> 跑低层 policy ->
把输出路由给低层 action term。这段逻辑以前在 6 个文件里各抄了一份，导致：

* 低层动作 ``scale`` / ``clip`` / 关节名单被硬编码在 6 处（清单 ④）；
* 低层观测覆写散落在各处，且是**就地修改传入的 cfg**（清单 ⑥）；
* 引用不存在的观测项（清单 ⑨）；
* 低层 policy 的观测/动作布局一旦变化，没有任何地方会报错，只会在
  ``self.policy(obs)`` 处抛出难懂的 matmul 形状错误（清单 ⑯）。

本模块把上面这些收敛成单一来源 + 启动期校验。

关于布局（L1 / L2）
------------------
低层 policy 的观测/动作布局由**产生该 checkpoint 的那次低层训练**决定：

* **L1（当前）**：显式声明 —— :func:`default_layout` 固定 ``ee_action_dim=7``。
  旧 checkpoint 的 ``actions`` 观测里包含 7 维 IK 槽位（当时 ``ee_ik`` 还是普通的
  ``DifferentialInverseKinematicsAction``）。后来 IK 改成由 CommandManager 直接驱动
  （``CommandDrivenIKAction.action_dim == 0``），这 7 维不再被任何低层 action term
  消费，但**旧 checkpoint 的观测仍然包含它们**，所以回放时必须照原样喂回去。
* **L2（后续）**：:func:`layout_from_low_level_cfg` 从低层 env cfg 推导布局，
  并与 checkpoint 的 ``actor.0.weight.shape`` 严格校验，不匹配直接报错。
  切换到 L2 需要先用当前布局重训低层 policy。
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.envs.mdp import joint_vel_rel

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.managers import ObservationManager


# ---------------------------------------------------------------------------
# 低层机器人的关节分组（唯一来源）
# ---------------------------------------------------------------------------

LEG_JOINT_NAMES: tuple[str, ...] = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
)

WHEEL_JOINT_NAMES: tuple[str, ...] = (
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)

ARM_JOINT_NAMES: tuple[str, ...] = (
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
)

GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "gripper_joint1", "gripper_joint2",
)


@dataclass(frozen=True)
class LowLevelActionLayout:
    """低层 policy 的动作分块 + 观测关节列顺序。"""

    policy_joint_names: tuple[str, ...] = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES + ARM_JOINT_NAMES
    """低层 policy 观测里 ``joint_pos`` / ``joint_vel`` 的关节列顺序（22 维）。"""

    leg_joint_names: tuple[str, ...] = LEG_JOINT_NAMES
    """低层 ``joint_pos`` 动作 term 驱动的关节（12 维）。"""

    wheel_joint_names: tuple[str, ...] = WHEEL_JOINT_NAMES
    """低层 ``joint_vel`` 动作 term 驱动的关节（4 维）。"""

    ee_action_dim: int = 7
    """checkpoint 动作输出里「IK 槽位」的数量。

    旧 checkpoint 训练时 ``ee_ik`` 是普通 IK action term（7 维，进 policy 动作空间），
    因此 ``actions`` 观测宽度是 12 + 4 + 7 = 23。现在 IK 由 CommandManager 驱动，
    这 7 维不再被任何低层 action term 消费，但**必须照样出现在 ``actions`` 观测里**，
    否则低层 policy 收到的观测与训练时不一致。
    """

    # -- 派生量 ------------------------------------------------------------

    @property
    def leg_dim(self) -> int:
        return len(self.leg_joint_names)

    @property
    def wheel_dim(self) -> int:
        return len(self.wheel_joint_names)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        driven = set(self.leg_joint_names) | set(self.wheel_joint_names)
        return tuple(n for n in self.policy_joint_names if n not in driven)

    @property
    def total_action_dim(self) -> int:
        """低层 policy 的动作输出维度（= ``actions`` 观测的宽度）。"""
        return self.leg_dim + self.wheel_dim + self.ee_action_dim

    @property
    def wheel_columns(self) -> tuple[int, ...]:
        """轮关节在 ``policy_joint_names`` 列空间里的下标。

        ``joint_pos`` 观测要把轮关节置零，必须用**列**下标而不是 articulation 的
        原生 joint id —— 两者只有在列序恰好等于原生序时才相同（见 known_issues #1）。
        """
        wheel = set(self.wheel_joint_names)
        return tuple(i for i, name in enumerate(self.policy_joint_names) if name in wheel)

    def split(self, policy_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把低层 policy 的输出切成 ``(leg, wheel, ee_ik)`` 三块。"""
        leg_end = self.leg_dim
        wheel_end = leg_end + self.wheel_dim
        return (
            policy_output[:, :leg_end],
            policy_output[:, leg_end:wheel_end],
            policy_output[:, wheel_end:wheel_end + self.ee_action_dim],
        )


def default_layout(*, ee_action_dim: int = 7) -> LowLevelActionLayout:
    """L1：显式声明的布局（对应现有 checkpoint）。"""
    return LowLevelActionLayout(ee_action_dim=ee_action_dim)


def layout_from_low_level_cfg(env_cfg, *, ee_action_dim: int | None = None) -> LowLevelActionLayout:
    """L2：从低层 env cfg 推导布局。

    关节名单从产生 checkpoint 的那份低层 cfg 读取，因此低层 cfg 是唯一来源，
    回放侧不再手抄。
    """
    leg_names = tuple(getattr(env_cfg, "leg_joint_names", LEG_JOINT_NAMES))
    wheel_names = tuple(getattr(env_cfg, "wheel_joint_names", WHEEL_JOINT_NAMES))
    arm_names = tuple(getattr(env_cfg, "arm_joint_names", ARM_JOINT_NAMES))
    if ee_action_dim is None:
        ee_term = getattr(getattr(env_cfg, "actions", None), "ee_ik", None)
        ee_action_dim = 7 if ee_term is None else int(getattr(ee_term, "action_dim", 7) or 7)
    return LowLevelActionLayout(
        policy_joint_names=leg_names + wheel_names + arm_names,
        leg_joint_names=leg_names,
        wheel_joint_names=wheel_names,
        ee_action_dim=int(ee_action_dim),
    )


def _joint_names_of(asset_cfg) -> str | list[str]:
    names = getattr(asset_cfg, "joint_names", None)
    if names is None:
        raise ValueError("低层观测的 asset_cfg.joint_names 为 None，无法推导关节列顺序。")
    return names


def resolve_layout(
    *,
    robot: "Articulation",
    low_level_obs_cfg: ObservationGroupCfg,
    low_level_leg_cfg,
    low_level_wheel_cfg,
    declared_ee_action_dim: int,
    actual_ee_ik_action_dim: int,
    tag: str,
) -> LowLevelActionLayout:
    """构造回放布局。

    * ``declared_ee_action_dim < 0`` -> **L2（默认）**：关节列顺序、轮关节名单、
      IK 槽位数全部从「低层 cfg + 机器人」推导，不手抄任何东西。
    * ``declared_ee_action_dim >= 0`` -> **L1**：沿用显式声明的分组与 IK 槽位数
      （用于那些"当时的低层 cfg 与现在不同"的旧 checkpoint）。
    """
    if declared_ee_action_dim >= 0:
        layout = default_layout(ee_action_dim=declared_ee_action_dim)
        # L1 也校验一遍：显式声明的关节名单必须真的存在于机器人上
        robot.find_joints(list(layout.policy_joint_names), preserve_order=True)
        return layout

    # ---- L2：从低层 cfg 推导 ----
    pos_asset_cfg = low_level_obs_cfg.joint_pos.params.get("asset_cfg")
    vel_asset_cfg = low_level_obs_cfg.joint_vel.params.get("asset_cfg")
    if pos_asset_cfg is None or vel_asset_cfg is None:
        raise ValueError(
            f"[{tag}] 低层观测的 joint_pos/joint_vel 没有 asset_cfg，无法推导布局。"
        )
    _, pos_names = robot.find_joints(_joint_names_of(pos_asset_cfg), preserve_order=True)
    _, vel_names = robot.find_joints(_joint_names_of(vel_asset_cfg), preserve_order=True)
    if list(pos_names) != list(vel_names):
        raise RuntimeError(
            f"[{tag}] 低层观测的 joint_pos 与 joint_vel 关节列顺序不一致，需要分别声明：\n"
            f"  joint_pos: {list(pos_names)}\n"
            f"  joint_vel: {list(vel_names)}"
        )

    wheel_names = tuple(getattr(low_level_wheel_cfg, "joint_names", ()) or ())
    if not wheel_names:
        raise ValueError(f"[{tag}] 低层 joint_vel 动作 cfg 没有 joint_names，无法确定轮关节。")
    leg_names = tuple(getattr(low_level_leg_cfg, "joint_names", ()) or ())

    layout = LowLevelActionLayout(
        policy_joint_names=tuple(pos_names),
        leg_joint_names=leg_names,
        wheel_joint_names=wheel_names,
        ee_action_dim=int(actual_ee_ik_action_dim),
    )
    return layout


# ---------------------------------------------------------------------------
# 低层观测
# ---------------------------------------------------------------------------


def joint_pos_rel_without_wheel_columns(
    env, asset_cfg: SceneEntityCfg, wheel_columns: Sequence[int]
) -> torch.Tensor:
    """``joint_pos_rel``（去掉默认位姿），并把指定的**列**置零。

    语义与 ``velocity.mdp.joint_pos_rel_without_wheel`` 相同，区别在于按列置零：
    原函数用 ``wheel_asset_cfg.joint_ids``（articulation 原生 id）直接索引
    ``asset_cfg.joint_ids`` 重排后的列，只有在「列序 == 原生序」时才正确。
    回放侧 ``policy_joint_names`` 的顺序是 leg -> wheel -> arm，并不是原生序
    （原生序是 fl_hipx, fr_hipx, hl_hipx, hr_hipx, arm_joint1, fl_hipy, ...），
    因此必须显式按列下标置零。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos_rel = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    joint_pos_rel[:, list(wheel_columns)] = 0.0
    return joint_pos_rel


def check_low_level_action_cfgs(
    *,
    tag: str,
    layout: LowLevelActionLayout,
    leg_cfg,
    wheel_cfg,
) -> None:
    """校验低层动作 cfg（scale / clip / 关节名单的唯一来源）与布局一致。

    回放侧不再手抄 scale / clip / joint_names：低层 action cfg 怎么说，
    低层 action term 就怎么做（``JointAction.__init__`` 会把 cfg 里的
    ``scale`` / ``clip`` / ``joint_names`` 编译成内部张量，回放侧事后赋值
    ``term.scale = ...`` 只是新增了一个不起作用的实例属性）。
    """
    problems = []
    leg_names = tuple(getattr(leg_cfg, "joint_names", ()) or ())
    wheel_names = tuple(getattr(wheel_cfg, "joint_names", ()) or ())
    if leg_names != layout.leg_joint_names:
        problems.append(f"joint_pos.joint_names {leg_names} != 布局 {layout.leg_joint_names}")
    if wheel_names != layout.wheel_joint_names:
        problems.append(f"joint_vel.joint_names {wheel_names} != 布局 {layout.wheel_joint_names}")
    if problems:
        raise RuntimeError(
            f"[{tag}] 低层动作 cfg 与 replay 布局不一致：\n  - " + "\n  - ".join(problems)
        )
    print(
        f"[ll-replay:{tag}] 低层动作 cfg: joint_pos.scale={getattr(leg_cfg, 'scale', None)} "
        f"joint_vel.scale={getattr(wheel_cfg, 'scale', None)} "
        f"clip={getattr(leg_cfg, 'clip', None)}"
    )


def build_low_level_observation_group(
    base_group: ObservationGroupCfg,
    *,
    layout: LowLevelActionLayout,
    actions_fn,
    velocity_commands_fn,
    ee_goal_fn,
    body_pose_cmd_fn=None,
    base_ang_vel_scale: float = 0.25,
    joint_pos_scale: float = 1.0,
    joint_vel_scale: float = 0.05,
) -> ObservationGroupCfg:
    """基于模板生成**独立副本**的低层观测组（不再就地改传入的 cfg）。

    模板只提供「有哪些观测项、顺序如何」；具体内容（哪些量来自高层命令、
    哪些量来自机器人本体）由这里统一覆写。
    """
    group = copy.deepcopy(base_group)

    def _override(term_name: str, func, *, required: bool = True) -> None:
        if func is None:
            # 该观测项在本次低层 layout 里不参与（例如低层 policy 不含 ee_goal）
            return
        term = getattr(group, term_name, None)
        if term is None or term == "MISSING":
            if not required:
                return
            raise AttributeError(
                f"低层观测模板里没有 '{term_name}' 项，无法覆写。模板现有项："
                f"{[k for k in group.__dict__ if not k.startswith('_')]}"
            )
        term.func = func
        term.params = {}

    _override("actions", actions_fn)
    _override("velocity_commands", velocity_commands_fn)
    # ee_goal / body_pose_cmd 是否存在于低层观测由模板决定（例如低层 cfg 可能把
    # ee_goal 置 None，或者用 flat 模板时压根没有 body_pose_cmd）
    _override("ee_goal", ee_goal_fn, required=False)
    _override("body_pose_cmd", body_pose_cmd_fn, required=False)

    # 本体量：列顺序显式给出，且用「按列置零」的版本避免索引空间混用
    group.joint_pos.func = joint_pos_rel_without_wheel_columns
    group.joint_pos.params = {
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(layout.policy_joint_names), preserve_order=True
        ),
        "wheel_columns": list(layout.wheel_columns),
    }
    group.joint_vel.func = joint_vel_rel
    group.joint_vel.params = {
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=list(layout.policy_joint_names), preserve_order=True
        ),
    }

    group.base_ang_vel.scale = base_ang_vel_scale
    group.joint_pos.scale = joint_pos_scale
    group.joint_vel.scale = joint_vel_scale
    if getattr(group, "base_lin_vel", None) is not None:
        group.base_lin_vel = None
    if getattr(group, "height_scan", None) is not None:
        group.height_scan = None
    group.enable_corruption = False
    return group


def verify_wheel_columns(robot: "Articulation", layout: LowLevelActionLayout, *, tag: str) -> None:
    """校验轮关节的「列下标」确实指向轮关节（known_issues #1 的索引空间陷阱）。"""
    native_ids, native_names = robot.find_joints(
        list(layout.policy_joint_names), preserve_order=True
    )
    if list(native_names) != list(layout.policy_joint_names):
        raise RuntimeError(
            f"[{tag}] 低层观测关节顺序与布局不一致：\n"
            f"  期望 {list(layout.policy_joint_names)}\n"
            f"  实际 {list(native_names)}"
        )
    wheel_native_ids, _ = robot.find_joints(
        list(layout.wheel_joint_names), preserve_order=False
    )
    wheel_columns_from_robot = tuple(native_ids.index(i) for i in wheel_native_ids)
    if wheel_columns_from_robot != layout.wheel_columns:
        raise RuntimeError(
            f"[{tag}] 轮关节列下标不一致：布局声明 {layout.wheel_columns}，"
            f"机器人解析结果 {wheel_columns_from_robot}。"
            f"（原生 joint 顺序：{robot.joint_names}）"
        )


# ---------------------------------------------------------------------------
# 启动期校验（清单 ⑯）
# ---------------------------------------------------------------------------


def checkpoint_dims(policy) -> tuple[int, int]:
    """返回低层 checkpoint 的 ``(观测维度, 动作维度)``。

    ``export_policy_as_jit`` 导出的模块带 ``.actor`` 子模块；而一个"纯 MLP"的导出
    可能把 actor 放在顶层，所以这里两种结构都兼容。
    """
    root = getattr(policy, "actor", policy)
    linears = [m for m in root.modules() if hasattr(m, "weight")]
    if not linears:
        raise RuntimeError("低层 policy 里找不到带 weight 的层，无法校验布局。")
    return int(linears[0].weight.shape[1]), int(linears[-1].weight.shape[0])


def ll_command_world(action_term) -> torch.Tensor:
    """取 action term 的「世界系」高层命令（``[vx,vy,wz, ee_pos_w(3), ee_quat_w(4)]``）。

    为什么要这个 helper
    -------------------
    按（已确认的）O1 方案，高层 action term 的 ``ll_command`` 统一存 **root 系**
    （低层观测 ``ee_goal`` 与 IK 都要 root 系）。而奖励项里判断"命令目标离物体有多远"
    需要 **世界系**（`object.data.root_pos_w` 是世界坐标）——直接拿 root 系去比会算错。

    所以：

    * 已迁移的 term（``PreTrainedPickAction`` / ``PreTrainedPickWBCAction`` /
      ``TeleopLLAction``）额外提供 ``ll_command_w``，本函数优先用它；
    * 尚未迁移的 term（``PreTrainedNavAction`` / ``VLAPickAction`` /
      ``PreTrainedPolicyAction``，见清单 ⑤ 的 R1 步骤）连 ``ll_command`` 都没有 ——
      这里给出明确报错，而不是让上游抛一个难懂的 AttributeError。
    """
    world = getattr(action_term, "ll_command_w", None)
    if world is not None:
        return world
    root = getattr(action_term, "ll_command", None)
    if root is not None:
        # 还没区分 root/world 的旧实现：历史上它们的 ll_command[:, 3:6] 就是世界系
        return root
    raise RuntimeError(
        f"{type(action_term).__name__} 没有 ll_command / ll_command_w，"
        "无法给奖励项提供世界系命令。"
        "（该 action term 尚未迁移到 low_level_replay 的统一接口，见清单 ⑤ 的 R1 步骤）"
    )


def read_policy_layout(policy_path: str) -> dict | None:
    """读取导出产物旁边的 ``policy_layout.json``（由 export_deploy_policy.py 写出）。

    这份 json 是「该 checkpoint 到底要什么输入」的权威描述：对
    ``ActorCriticHistory`` 这类 actor 输入含 latent 的策略，光看
    ``actor.0.weight`` 是**读不出观测维度**的。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(policy_path)), "policy_layout.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expected_policy_obs_dim(policy, policy_path: str, *, tag: str) -> tuple[int, dict | None]:
    """低层 policy 期望的**观测**维度 + 解析到的 layout json。"""
    layout_json = read_policy_layout(policy_path)
    if layout_json is not None:
        if layout_json.get("kind") == "history":
            raise NotImplementedError(
                f"[{tag}] 该低层 checkpoint 是带 history encoder 的 ROA 策略"
                f"（policy_obs={layout_json.get('policy_obs_dim')}, "
                f"history={layout_json.get('history_length')} x "
                f"{layout_json.get('history_single_step_dim')}）。\n"
                "高层 replay 侧还没实现 history 窗口回放，见 "
                "docs/review/history_low_level_policy_todo.md。\n"
                "临时替代：用不带 history encoder 的普通 ActorCritic 低层策略。"
            )
        return int(layout_json["policy_obs_dim"]), layout_json
    obs_dim, _ = checkpoint_dims(policy)
    return obs_dim, None


def build_low_level_obs_manager(
    *,
    env,
    obs_cfg: ObservationGroupCfg,
    group_name: str,
    expected_obs_dim: int,
    tag: str,
):
    """建低层观测组，并让它与 checkpoint 期望的观测维度**严格一致**。

    低层 cfg 里 ``ee_goal`` 可能被置 None（省掉 7 维），也可能保留。这里按 checkpoint
    的实际维度取舍：先按模板建；若多出来的宽度正好是一个 ``ee_goal``，就把它去掉重建；
    否则直接报错（不做任何猜测性的维度拼凑）。

    Returns:
        ``(obs_manager, obs_cfg, used_ee_goal)``
    """
    from isaaclab.managers import ObservationManager

    manager = ObservationManager({group_name: obs_cfg}, env)
    dim = int(manager.group_obs_dim[group_name][0])
    if dim == expected_obs_dim:
        return manager, obs_cfg, getattr(obs_cfg, "ee_goal", None) is not None

    if getattr(obs_cfg, "ee_goal", None) is not None:
        terms = manager.active_terms[group_name]
        dims = manager.group_obs_term_dim[group_name]
        ee_dim = int(math.prod(dims[terms.index("ee_goal")]))
        if dim - ee_dim == expected_obs_dim:
            obs_cfg.ee_goal = None
            manager = ObservationManager({group_name: obs_cfg}, env)
            return manager, obs_cfg, False

    raise RuntimeError(
        f"[{tag}] 低层观测维度对不上：回放构造出 {dim}，checkpoint 期望 {expected_obs_dim}。"
        " 观测项："
        + ", ".join(
            f"{n}{tuple(d)}"
            for n, d in zip(
                manager.active_terms[group_name], manager.group_obs_term_dim[group_name]
            )
        )
        + "\n（没有做任何自动加减维度：请确认低层 cfg 与产生该 checkpoint 的训练一致，"
        "或用 export_deploy_policy.py 重新导出以生成 policy_layout.json。）"
    )


def verify_low_level_layout(
    *,
    tag: str,
    robot: "Articulation",
    layout: LowLevelActionLayout,
    obs_manager: "ObservationManager",
    group_name: str,
    policy,
    expected_obs_dim: int | None = None,
) -> int:
    """打印并校验「回放布局 == checkpoint 布局」。

    任何一条不满足都直接报错，而不是等到训练跑偏。
    """
    policy_obs_dim, policy_action_dim = checkpoint_dims(policy)
    expected_obs = policy_obs_dim if expected_obs_dim is None else int(expected_obs_dim)
    group_dim = obs_manager.group_obs_dim[group_name]
    actual_obs = int(group_dim[0]) if isinstance(group_dim, tuple) else int(group_dim)

    verify_wheel_columns(robot, layout, tag=tag)

    print(
        f"[ll-replay:{tag}] 低层 policy action_dim={policy_action_dim}, "
        f"低层 obs 维度={actual_obs} (checkpoint 期望 {expected_obs})"
    )
    print(
        f"[ll-replay:{tag}] 动作分块 leg={layout.leg_dim} wheel={layout.wheel_dim} "
        f"ee_ik={layout.ee_action_dim} -> total={layout.total_action_dim}"
    )
    print(f"[ll-replay:{tag}] 轮关节列下标={list(layout.wheel_columns)}")
    terms = obs_manager.active_terms[group_name]
    dims = obs_manager.group_obs_term_dim[group_name]
    print(
        f"[ll-replay:{tag}] '{group_name}' 观测项: "
        + ", ".join(f"{n}{tuple(d)}" for n, d in zip(terms, dims))
    )

    problems = []
    if policy_action_dim != layout.total_action_dim:
        problems.append(
            f"低层 policy 动作维度 {policy_action_dim} != 布局声明的 {layout.total_action_dim}"
        )
    if actual_obs != expected_obs:
        problems.append(
            f"回放构造的低层 obs 维度 {actual_obs} != checkpoint 期望 {expected_obs}"
        )
    if problems:
        raise RuntimeError(
            f"[{tag}] 低层 replay 布局与 checkpoint 不匹配：\n  - "
            + "\n  - ".join(problems)
            + "\n提示：布局由产生 checkpoint 的那次低层训练决定；若低层已重训，"
              "请改用 layout_from_low_level_cfg() 并更新低层 checkpoint 路径。"
        )
    return actual_obs
