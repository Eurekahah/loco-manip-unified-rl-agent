# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import observations as base_mdp  # noqa: F401, F403
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)

    ⚠️ 索引空间（known_issues #1 / #16）：``wheel_asset_cfg.joint_ids`` 是 articulation 的
    **原生 joint id**，而 ``joint_pos_rel`` 的列是 ``asset_cfg.joint_ids`` 重排后的顺序 ——
    两者只有在"列序 == 原生序"（即调用方传 ``joint_names=[".*"]`` 且 ``preserve_order``
    不改变顺序）时才等价。这里把这个前提显式断言出来，避免又出现"清错了关节"
    （实测过：会清掉 ``hr_wheel_joint`` + ``arm_joint1/2/3``，放行 ``fl/fr/hl_wheel``）。

    如果你的观测列本来就是重排过的（例如 leg→wheel→arm），请改用
    ``highlevel/mdp/low_level_replay.py::joint_pos_rel_without_wheel_columns()``。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = len(asset.data.joint_names)

    def _as_index_list(ids) -> list[int]:
        """把 SceneEntityCfg.joint_ids（可能是 slice / list / tensor）统一成列下标列表。"""
        if isinstance(ids, slice):
            return list(range(num_joints))[ids]
        return [int(i) for i in ids]

    if asset_cfg.joint_ids is None or wheel_asset_cfg.joint_ids is None:
        raise RuntimeError(
            "joint_pos_rel_without_wheel 需要已解析的 SceneEntityCfg（joint_ids 不能为 None）："
            f"asset_cfg.joint_ids={asset_cfg.joint_ids}, wheel_asset_cfg.joint_ids={wheel_asset_cfg.joint_ids}"
        )
    asset_ids = _as_index_list(asset_cfg.joint_ids)
    wheel_ids = _as_index_list(wheel_asset_cfg.joint_ids)
    # "列序 == 原生序" 的精确条件：对每个要清零的列 c，它的列映射必须是 c 本身。
    bad_cols = [c for c in wheel_ids if c >= len(asset_ids) or asset_ids[c] != c]
    if bad_cols:
        raise RuntimeError(
            "joint_pos_rel_without_wheel 的索引空间不一致（known_issues #1）："
            f"wheel_asset_cfg.joint_ids={wheel_ids} 里的列 {bad_cols} 并不对应同名关节，"
            f"当前 joint_pos 的列映射是 {asset_ids}。\n"
            "  也就是说调用方的 joint_pos 列序不是 articulation 原生序（例如 leg→wheel→arm）；"
            "请改用 joint_pos_rel_without_wheel_columns()（按**列下标**置零）。"
        )
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def check_policy_layout(
    env: ManagerBasedRLEnv,
    env_ids,  # startup 事件不会用到（EventManager 对签名要求"env, env_ids, ..."）
    group_name: str = "policy",
    actions_term: str = "actions",
    raise_on_mismatch: bool = True,
) -> None:
    """启动期打印低层 policy 的观测/动作布局，并断言"actions 观测宽度 == 动作总维度"。

    known_issues ⑯（低层侧）：``mdp.last_action`` 观测的宽度就是
    ``action_manager.total_action_dim``，所以**任何** action term 维度变化都会改变 policy
    观测布局、静默让旧 checkpoint 失效（历史上 ``ee_ik`` 从 7 维变 0 维就是这么废掉一批
    checkpoint 的）。在 env 创建时把布局打印出来 + 断言这条不变量，比等到加载 checkpoint
    时 matmul 报 "shapes cannot be multiplied" 好定位得多。

    用法：在 EventCfg 里挂一个 ``mode="startup"`` 的事件项（见 ``EventCfg.check_policy_layout``）。
    """
    obs_mgr = env.observation_manager
    act_mgr = env.action_manager
    total_action_dim = int(act_mgr.total_action_dim)

    def _numel(shape) -> int:
        n = 1
        for s in shape:
            n *= int(s)
        return n

    print("[layout-check] 低层 policy 布局（known_issues ⑯）:")
    for gname in obs_mgr.active_terms.keys():
        names = obs_mgr.active_terms[gname]
        dims = obs_mgr.group_obs_term_dim[gname]
        total = _numel(tuple(obs_mgr.group_obs_dim[gname]))
        print(f"  - 观测组 '{gname}'：{total} 维 = "
              + " + ".join(f"{n}{_numel(d)}" for n, d in zip(names, dims)))
    for tname, term in act_mgr._terms.items():  # noqa: SLF001 - 只用于打印
        print(f"  - 动作项 '{tname}'：{term.action_dim} 维")
    print(f"  - 动作总维度 = {total_action_dim}")

    # ── known_issues ⑱：接触传感器的 body 顺序 ≠ articulation body 顺序 ─────────────
    # `ContactSensor.data.net_forces_w[:, i]` 的 i 是**传感器自己的行序**，而
    # `Articulation.data.body_*` 用 articulation 行序 —— 两套不一样。历史上"臂/夹爪
    # 持续 90 N"的误判就是这么来的（见 DEFECT_LOG_zh.md 第三节）。这里启动期把两个顺序
    # 都打出来，并检查"传感器里的 body 都能在 articulation 里找到名字"（否则没法按名字归因）。
    sensors = getattr(env.scene, "sensors", None)
    contact = sensors.get("contact_forces") if isinstance(sensors, dict) else None
    if contact is not None and hasattr(contact, "body_names"):
        sensor_bodies = list(contact.body_names)
        artic_bodies = list(env.scene["robot"].body_names)
        same_order = sensor_bodies == artic_bodies
        print(
            f"[layout-check] 接触传感器 body 数 {len(sensor_bodies)}；与 articulation "
            f"（{len(artic_bodies)} 个 body）{'顺序一致' if same_order else '**顺序不同**'}"
            f"（known_issues ⑱：归因一律按名字，别按下标）"
        )
        missing = [b for b in sensor_bodies if b not in artic_bodies]
        if missing:
            print(f"[layout-check] ⚠️ 传感器里有 articulation 不存在的 body 名：{missing[:5]}"
                  "（按名字归因会失败，请核对 contact_forces.prim_path）")
        else:
            idx = {b: i for i, b in enumerate(sensor_bodies)}
            head = ", ".join(f"{b}#{idx[b]}" for b in sensor_bodies[:6])
            print(f"[layout-check] 传感器行序（前 6）：{head} …")

    problems = []
    if group_name in obs_mgr.active_terms:
        names = list(obs_mgr.active_terms[group_name])
        if actions_term in names:
            width = _numel(tuple(obs_mgr.group_obs_term_dim[group_name][names.index(actions_term)]))
            if width != total_action_dim:
                problems.append(
                    f"观测 '{group_name}.{actions_term}' 宽度 {width} != 动作总维度 {total_action_dim}"
                )
    if problems:
        msg = (
            "[layout-check] 观测/动作布局不一致（这会让旧 checkpoint 静默失效）：\n  - "
            + "\n  - ".join(problems)
        )
        if raise_on_mismatch:
            raise RuntimeError(msg)
        print(msg)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

# mdp/observations.py

def ee_goal_pos_local(
        env: ManagerBasedRLEnv,
        command_name: str,
    ) -> torch.Tensor:
    """返回 local frame 下的 EE 目标位置 (N, 3)"""
    command_term = env.command_manager.get_term(command_name)
    return command_term.command_local[:, :3]

# 用 6D rotation representation（前两列）更稳定
def ee_goal_orn_local_6d(env, command_name):
    command_term = env.command_manager.get_term(command_name)
    quat = command_term.command_local[:, 3:]  # (N, 4) wxyz
    rot_mat = math_utils.matrix_from_quat(quat)  # (N, 3, 3)
    return rot_mat[:, :, :2].reshape(-1, 6)       # (N, 6)，取前两列展平

def ee_goal_local(env, command_name):
    command_term = env.command_manager.get_term(command_name)
    return command_term.command_local  # (N, 7) pos + quat


def history_single_step_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adaptation module 单个时间步的输入: base state + arm state + leg state + 上一步18维关节目标位置。

    顺序需要和你认为的 "single_step_dim" 保持一致，方便排查维度问题:
      [base_ang_vel(3), projected_gravity(3),
       joint_pos(N), joint_vel(N), last_action(18)]
    """
    base_ang_vel = base_mdp.base_ang_vel(env, asset_cfg)
    projected_gravity = base_mdp.projected_gravity(env, asset_cfg)
    joint_pos = base_mdp.joint_pos_rel(env, asset_cfg)
    joint_vel = base_mdp.joint_vel_rel(env, asset_cfg)
    last_action = base_mdp.last_action(env)
    return torch.cat(
        [base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action],
        dim=-1,
    )


def history_single_step_obs_full(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """V2：历史窗口的每一帧 = **actor 自己的观测**（顺序与 ``WBCObservationsCfg.PolicyCfg`` 一致）。

    为什么这么改：参考实现（MarkFzp/Deep-Whole-Body-Control、Ericonaldo/visual_wholebody）里历史窗口
    是"最近 N 帧 proprio"，而它们的 proprio 向量**含速度指令**（vwb config ``num_proprio``
    末尾那几组就是 commands/gait phase；actor 的 ``infer_hist_latent`` 取
    ``obs[:, -num_hist*num_prop:]``，与 actor 输入同序同宽）。

    旧版 ``history_single_step_obs`` 只有 [ang_vel, gravity, joint_pos, joint_vel, last_action]，
    **没有指令**——学生就分不清"我被外力推了"和"我被命令加速了"，而这两者在历史里长得一样、
    对"辨识环境参数"的含义却完全不同。

    顺序（与 PolicyCfg 一致）::

        [base_ang_vel(3), projected_gravity(3), velocity_commands(3),
         joint_pos(N), joint_vel(N), last_action(A), ee_goal(7), body_pose_cmd(3)]
    """
    base_ang_vel = base_mdp.base_ang_vel(env, asset_cfg)
    projected_gravity = base_mdp.projected_gravity(env, asset_cfg)
    velocity_commands = base_mdp.generated_commands(env, command_name="base_velocity")
    joint_pos = base_mdp.joint_pos_rel(env, asset_cfg)
    joint_vel = base_mdp.joint_vel_rel(env, asset_cfg)
    last_action = base_mdp.last_action(env)
    ee_goal = ee_goal_local(env, command_name="ee_pose")
    body_pose_cmd = base_mdp.generated_commands(env, command_name="body_pose")
    return torch.cat(
        [
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            last_action,
            ee_goal,
            body_pose_cmd,
        ],
        dim=-1,
    )


def history_zero_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """无历史对照臂（P1-7）：返回与 `history_single_step_obs_full` **同宽但全零**的单步观测。

    用途：在"适配通路活下来的那套配置"上把历史信息彻底拿掉，直接回答"历史/adaptation 到底有
    没有用"。与"删掉 history 项 / 换成纯 MLP 基线"相比，这里保持张量宽度、编码器结构、参数量、
    优化器状态全部不变，**唯一的变量就是历史里有没有信息** ⇒ 更干净的同构对照。

    实现上先算真实观测再 `zeros_like`：形状永远跟着 `PolicyCfg` 走，不会因为观测项增删而错位。
    """
    return torch.zeros_like(history_single_step_obs_full(env, asset_cfg))


def history_zero_obs_legacy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """同 `history_zero_obs`，但宽度跟**旧版**单步历史（70 维：无指令）一致。

    用途：给"旧 70 维历史 + V4 特权组"的对照臂做**同宽**置零（例如 V4/Anchor 用的
    `WBCObservationsV4Cfg` 历史就是 700 维）。这样 Anchor vs NoHist 的唯一差异仍是"信息有无"，
    编码器输入宽度不做额外变化。
    """
    return torch.zeros_like(history_single_step_obs(env, asset_cfg))


"""
特权信息观测项(lazy-init cached version)

关键点:
- IsaacLab 的 manager 构建顺序是 ObservationManager 先于 EventManager 的 "startup" 模式事件。
  也就是说,如果在 ObsTerm 的 __init__ 里就去查 root_physx_view,拿到的还是随机化之前的默认值,
  是错的。
- 解决办法:__init__ 只保存 asset/body_ids 等引用,不做任何物理查询;真正的查询延迟到
  第一次被调用(__call__)时才执行 —— 那时候 startup 随机化事件肯定已经跑完了。查询结果
  缓存进 self.buf,之后每次 __call__ 都是直接返回缓存,开销为零。
- reset 语义：只有"对应的随机化在 reset 模式"的项才需要在 episode reset 时刷新，由 ObsTerm
  参数 ``update_on_reset`` 控制（默认 False）。IsaacLab 在 ``_reset_idx`` 里先
  ``event_manager.apply(mode="reset")`` 再 ``observation_manager.reset()``，所以在 ``reset()``
  里重查物理量拿到的一定是随机化之后的新值。
  当前 EventCfg 中只有 ``randomize_actuator_gains`` 是 reset 模式 → 只有
  ``privileged_joint_gain_scale`` 默认开启刷新；其余项对应 startup 随机化，保持"只取一次"。
"""


class _PrivilegedCachedTerm(ManagerTermBase):
    """特权观测的公共缓存逻辑（子类只需实现 _compute）。"""

    #: 该 ObsTerm 对应的随机化是否在 episode reset 时重新采样（EventTerm mode="reset"）。
    #: 子类可覆盖；也可以在 ObsTerm 参数里传 ``update_on_reset=True/False`` 单独覆盖。
    default_update_on_reset: bool = False

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 注意：ObsTerm.params 会被 manager 原样透传给 __call__（见 ObservationManager._prepare_terms），
        # 所以这个开关必须 pop 掉，否则 __call__ 会收到意外关键字参数而报 TypeError。
        override = cfg.params.pop("update_on_reset", None)
        self.update_on_reset: bool = self.default_update_on_reset if override is None else bool(override)
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        # manager 构建期第一次 __call__ 之前 buf 还不存在，交给正常路径去填
        if self.update_on_reset and self.buf is not None:
            self._compute(env_ids=env_ids)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # 缓存闸门（2026-10-16 加固）：只要仿真还没真正 step 过（common_step_counter == 0），
        # 每次调用都重算。
        #   旧写法是 `count < 2`，它隐含假设"startup 随机化之前的调用次数恰好 = 1"
        #   （= ObservationManager._prepare_terms 量维度时那一次）。这个假设在真探针里验证过是对的，
        #   但它是**隐式契约**：IsaacLab 换版本、或哪个 startup 事件提前读一次 obs，缓存就会被
        #   填成"随机化之前"的常数（这时教师输入跨环境完全相同 ⇒ 必然塌成常数，且不会报错）。
        #   改成用显式仿真步计数做闸门后：构造期 + 第一次 compute（都在 counter==0）都会重算，
        #   仿真开跑（counter ≥ 1）之后才真正走缓存。开销只在启动时多几次物理查询。
        if self.buf is None or int(getattr(env, "common_step_counter", 0)) == 0:
            self._compute()
            self.count += 1
        return self.buf


class privileged_base_extra_payload(_PrivilegedCachedTerm):
    """基座额外负载 (kg),相对默认质量的偏移量。对应 randomize_rigid_body_mass(add)（startup）。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        current_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self._env.device)
        default_mass = self.asset.data.default_mass[:, self.body_id].to(self._env.device)
        value = (current_mass - default_mass).unsqueeze(-1)
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value[env_ids]


class privileged_end_effector_payload(_PrivilegedCachedTerm):
    """末端负载 (kg)。对应 randomize_rigid_body_mass(scale)（startup）。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0]

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        current_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self._env.device)
        default_mass = self.asset.data.default_mass[:, self.body_id].to(self._env.device)
        value = (current_mass - default_mass).unsqueeze(-1)
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value[env_ids]


class privileged_rigid_body_inertia(_PrivilegedCachedTerm):
    """指定 body 的惯量偏移(对角项均值)。对应 randomize_rigid_body_inertia（startup）。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        current_inertia = self.asset.root_physx_view.get_inertias()[:, self.body_ids].to(self._env.device)
        default_inertia = self.asset.data.default_inertia[:, self.body_ids].to(self._env.device)
        value = (current_inertia - default_inertia).mean(dim=-1)
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value[env_ids]


class privileged_base_com_offset(_PrivilegedCachedTerm):
    """基座质心偏移 (3,)。对应 randomize_com_positions（startup）。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        current_com = self.asset.root_physx_view.get_coms()[:, self.body_id, :3].to(self._env.device)
        default_com = self.asset.data.default_com[:, self.body_id, :3].to(self._env.device)
        value = current_com - default_com
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value[env_ids]


class privileged_material_properties(_PrivilegedCachedTerm):
    """脚部 PhysX 材质特权信息:静摩擦、动摩擦、恢复系数。对应 randomize_rigid_body_material。

    该随机化是 startup 模式 → 默认不在 reset 时刷新。

    Returns:
        torch.Tensor: shape [num_envs, num_feet * 3],按
            [foot0_static, ..., footN_static,
             foot0_dynamic, ..., footN_dynamic,
             foot0_restitution, ..., footN_restitution] 排列。
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.foot_body_ids = asset_cfg.body_ids

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        materials = self.asset.root_physx_view.get_material_properties().to(self._env.device)  # [num_envs, num_bodies, 3]
        static_friction = materials[:, self.foot_body_ids, 0]
        dynamic_friction = materials[:, self.foot_body_ids, 1]
        restitution = materials[:, self.foot_body_ids, 2]
        value = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value[env_ids]


class privileged_joint_gain_scale(_PrivilegedCachedTerm):
    """关节 PD 增益缩放系数(stiffness/damping 相对默认值的比例)。

    对应 ``randomize_actuator_gains``，该 EventTerm 是 **reset 模式** → 默认在每个 episode
    reset 后刷新缓存（``update_on_reset`` 默认 True）；多次 reset 只更新被 reset 的那几行。
    如果把它改回 startup 模式，可在 ObsTerm 参数里传 ``update_on_reset=False`` 省掉这次查询。
    """

    default_update_on_reset: bool = True

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.included_actuators = {"joint", "wheel", "piper_arm", "piper_gripper"}  # 轮子和夹爪可按需去掉

    @staticmethod
    def _scale(current: torch.Tensor, default) -> torch.Tensor:
        if isinstance(default, float):
            return current / max(default, 1e-8)
        return current / current.new_tensor(default).clamp_min(1e-8)

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        scales = []
        for actuator_name, actuator in self.asset.actuators.items():
            if actuator_name not in self.included_actuators:
                continue
            k_scale = self._scale(actuator.stiffness, actuator.cfg.stiffness)
            d_scale = self._scale(actuator.damping, actuator.cfg.damping)
            if env_ids is not None:
                k_scale = k_scale[env_ids]
                d_scale = d_scale[env_ids]
            scales.append(k_scale)
            scales.append(d_scale)
        value = torch.cat(scales, dim=-1)
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value


# ══════════════════════════════════════════════════════════════════════════════════════
# V4 特权向量（2026-10-16）：对齐参考实现（MarkFzp/Deep-Whole-Body-Control、Ericonaldo/
# visual_wholebody）的"小而齐"特权向量设计，并修掉旧版（89 维）的三个结构性问题。
#
# 参考实现的特权向量（直接读代码得到）：
#   * DWL `widowGo1.py:986` → `cat(mass_params(5), friction(1), motor_strength−1(18))`
#     其中 `mass_params = [base 附加质量, base COM_xyz, gripper 附加质量]`（同文件 `:431`）；
#   * vwb 低层 `manip_loco.py:237` → 同样三段，motor_strength 只取 12 个腿关节。
#   ⇒ 24 / 18 维，**没有惯量、没有恢复系数**，但 **COM 在向量里**。
#
# 我们旧版 89 维的三个问题（真 env 探针实测，见 scripts/.../probe_privileged_obs.py）：
#   1) 27 维惯量 Δ 是 kg·m² 量级（跨环境 σ≈4.4e-4），比摩擦/负载小 ~1300 倍 ⇒ 在教师 MLP 的
#      第一层里贡献可忽略，等价于"没接进去"；
#   2) 48 维 gain 里有 4 维结构性恒 0（轮子 actuator stiffness=0 ⇒ 0/1e-8=0），是死维；
#   3) 12 维逐轮摩擦里 4 维是恢复系数，真正影响接触的静/动摩擦只占 8 维，且**臂/夹爪/腿杆的
#      摩擦完全没被观测**（而 `randomize_rigid_body_material` 的 body_names 是 `.*`，逐 shape 独立采样）。
#   另外把 `base_com_offset` 注释掉了 —— 参考实现恰恰把 COM 放进特权向量。
#
# V4 的做法：**保留 5 个通道、全部按各自域随机化的实际范围线性映射到 [-1,1]**，
# 范围从 `env.cfg.events` 现读（不写死常量），因此改 DR 范围不会再让输入量级失衡。
# 拼接顺序（总 54 维）::
#
#     [0:1]    base 附加质量（add）                       1
#     [1:4]    base COM 偏移 xyz（add）                   3
#     [4:5]    gripper 质量缩放（scale, 以 DR 中点为 0）   1
#     [5:8]    "足"（轮）摩擦均值 static/dynamic/restitution  3
#     [8:10]   其余 23 个连杆摩擦均值 static/dynamic       2
#     [10:54]  关节增益比：20 个非轮关节的 k 比 + 全部 24 关节的 d 比  44
#
# 注：`randomize_rigid_body_inertia`（全 body 惯量 scale 0.85~1.15）**故意不进向量**（对齐参考
# 实现：DWL/vwb 都不放惯量）；如果后续要覆盖它，应该加"按 body 组聚合的相对 scale"（4 维），
# 而不是旧版的 27 维绝对 Δ。
# ══════════════════════════════════════════════════════════════════════════════════════

#: 域随机化范围的回退值（只在读不到 `env.cfg.events.<term>.params[<key>]` 时使用）
_V4_DR_FALLBACK: dict[str, tuple[float, float]] = {
    "base_add": (-1.0, 3.0),
    "com_x": (-0.03, 0.03),
    "com_y": (-0.03, 0.03),
    "com_z": (-0.02, 0.02),
    "gripper_scale": (1.0, 3.0),
    "static": (0.35, 1.5),
    "dynamic": (0.35, 1.5),
    "restitution": (0.0, 0.7),
    "gain": (0.85, 1.15),
}


def _to_unit(x: torch.Tensor, rng: tuple[float, float]) -> torch.Tensor:
    """把落在 `rng=[lo,hi]` 里的量线性映射到 [-1,1]（DR 中点 → 0）。"""
    lo, hi = float(rng[0]), float(rng[1])
    mid = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-8)
    return (x - mid) / half


def _dr_range(env: ManagerBasedRLEnv, term_name: str, key: str,
              fallback: tuple[float, float], axis: str | None = None) -> tuple[float, float]:
    """从 `env.cfg.events.<term_name>.params[key]` 读真实随机化范围（拿不到就回退）。

    这样"归一化尺度"永远跟着 DR 配置走，不会因为有人改了 range 就让输入量级失衡。
    `axis` 用于 `com_range={"x":(...), "y":(...)}` 这种字典参数。
    """
    try:
        term = getattr(env.cfg.events, term_name)
        raw = term.params[key]
        if axis is not None:
            raw = raw[axis]
        rng = (float(raw[0]), float(raw[1]))
        if not rng[1] > rng[0]:
            raise ValueError(f"范围退化: {rng}")
        return rng
    except Exception as exc:  # noqa: BLE001 - 读不到就用回退值，但要吼一声
        print(
            f"[privileged-v4] WARN: 读不到 events.{term_name}.params[{key!r}"
            + (f"][{axis!r}]" if axis else "]")
            + f"（{type(exc).__name__}: {exc}），改用回退范围 {fallback}"
        )
        return fallback


class privileged_env_params_v4(_PrivilegedCachedTerm):
    """V4 特权向量（54 维，全部 O(1)）——见文件末尾的通道表。

    与旧版逐项 term 的区别：
      * 单一 term、单一缓存，拼接口径在这里唯一确定（不会被 cfg 里的增删顺序改掉）；
      * 通道值都映射到 [-1,1]，教师 MLP 第一层看到的输入量级一致；
      * 摩擦用**逐 body 的 shape 区间**索引（修复旧 term 直接用 body_id 当 shape 下标的问题），
        并聚合成"足"与"其余连杆"两组均值；
      * 丢掉 4 个结构性恒 0 的轮子 stiffness 维。
    """

    # 增益是 reset 模式随机化 ⇒ 该项需要在 episode reset 时刷新
    default_update_on_reset: bool = True

    #: 通道切片（供日志/探针按块统计）
    block_slices: dict[str, tuple[int, int]] = {
        "base_payload": (0, 1),
        "base_com": (1, 4),
        "ee_payload": (4, 5),
        "foot_friction": (5, 8),
        "link_friction": (8, 10),
        "joint_gain": (10, 54),
    }

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 注意：ObsTerm.params 会被原样透传给 __call__，所以这里必须把自定义参数都 pop 掉
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.base_body_name = cfg.params.pop("base_body_name", "base_link")
        self.gripper_body_name = cfg.params.pop("gripper_body_name", "gripper_base")
        self.foot_body_regex = cfg.params.pop("foot_body_regex", ".*wheel")
        self.included_actuators = set(
            cfg.params.pop("included_actuators", ("joint", "wheel", "piper_arm", "piper_gripper"))
        )

        import re as _re

        self.base_id = self.asset.body_names.index(self.base_body_name)
        self.gripper_id = self.asset.body_names.index(self.gripper_body_name)
        foot_re = _re.compile(self.foot_body_regex)
        self.foot_body_ids = [i for i, n in enumerate(self.asset.body_names) if foot_re.fullmatch(n)]
        self.other_body_ids = [i for i in range(len(self.asset.body_names)) if i not in self.foot_body_ids]
        # 轮关节：用关节名里的 "wheel" 判定（body 名是 `.*_wheel`、关节名是 `*_wheel_joint`，
        # 两个正则不能混用）。数量对不上就直接报错，避免静默产生错位的通道。
        self.wheel_joint_names = {jn for jn in self.asset.joint_names if "wheel" in jn}
        if len(self.wheel_joint_names) != len(self.foot_body_ids):
            raise ValueError(
                f"轮关节数({len(self.wheel_joint_names)}) 与轮 body 数({len(self.foot_body_ids)}) 不一致，"
                f"privileged_env_params_v4 的通道切片会错位。请检查 foot_body_regex / 运行器命名。"
            )

        # ---- 归一化范围：全部从 EventCfg 现读 ------------------------------------ #
        r = _V4_DR_FALLBACK
        self.rng = {
            "base_add": _dr_range(env, "randomize_rigid_body_mass_base", "mass_distribution_params", r["base_add"]),
            "com_x": _dr_range(env, "randomize_com_positions", "com_range", r["com_x"], axis="x"),
            "com_y": _dr_range(env, "randomize_com_positions", "com_range", r["com_y"], axis="y"),
            "com_z": _dr_range(env, "randomize_com_positions", "com_range", r["com_z"], axis="z"),
            "gripper_scale": _dr_range(env, "randomize_rigid_body_mass_end_effector",
                                       "mass_distribution_params", r["gripper_scale"]),
            "static": _dr_range(env, "randomize_rigid_body_material", "static_friction_range", r["static"]),
            "dynamic": _dr_range(env, "randomize_rigid_body_material", "dynamic_friction_range", r["dynamic"]),
            "restitution": _dr_range(env, "randomize_rigid_body_material", "restitution_range", r["restitution"]),
            "gain": _dr_range(env, "randomize_actuator_gains", "stiffness_distribution_params", r["gain"]),
        }

        # ---- 每个 body 占多少个 shape（用于把 [N, S, 3] 的材质张量按 body 切片）---- #
        self._num_shapes_per_body = self._resolve_num_shapes_per_body()
        #: COM 基准（第一次 _compute 时抓，那时还没做 startup 随机化）
        self._com_ref: torch.Tensor | None = None

        # ---- 关节 → (k 比, d 比) 的列 ------------------------------------------- #
        self.k_col: dict[str, int] = {}
        self.d_col: dict[str, int] = {}
        for name, actuator in self.asset.actuators.items():
            if name not in self.included_actuators:
                continue
            for j, jn in enumerate(actuator.joint_names):
                self.k_col[jn] = j
                self.d_col[jn] = j

        #: 走教师 latent 的关节顺序：[20 个非轮关节的 k] + [24 个关节的 d]
        self.pos_joint_names = [jn for jn in self.asset.joint_names if jn not in self.wheel_joint_names]
        self.out_dim = 1 + 3 + 1 + 3 + 2 + len(self.pos_joint_names) + len(self.asset.joint_names)
        print(
            f"[privileged-v4] 通道 = base{self.rng['base_add']} COM{self.rng['com_x']} "
            f"gripper_scale{self.rng['gripper_scale']} friction({self.rng['static']}/{self.rng['dynamic']}/"
            f"{self.rng['restitution']}) gain{self.rng['gain']} | "
            f"foot_bodies={self.foot_body_ids} 轮关节={sorted(self.wheel_joint_names)} | "
            f"输出 {self.out_dim} 维（k:{len(self.pos_joint_names)} + d:{len(self.asset.joint_names)}）"
        )

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        base_body_name: str = "base_link",
        gripper_body_name: str = "gripper_base",
        foot_body_regex: str = ".*wheel",
        included_actuators: tuple[str, ...] = ("joint", "wheel", "piper_arm", "piper_gripper"),
    ) -> torch.Tensor:
        """缓存版 __call__。

        注意：这组可选参数的**唯一作用是让 ObservationManager 的静态参数校验通过** —— 该校验
        发生在类实例化之前（`manager_base._resolve_common_term_cfg` 拿 `cls.__call__` 的签名
        和 `term_cfg.params` 做集合比较），所以自定义参数必须出现在签名里；实例化之后
        `__init__` 会把它们从 `cfg.params` 里 pop 掉，实际生效的是 `__init__` 存下来的属性。
        """
        return super().__call__(env, asset_cfg)

    def _resolve_num_shapes_per_body(self) -> list[int] | None:
        """每个 body 有几个碰撞 shape（IsaacLab 自己的 `randomize_rigid_body_material` 也是这么做的）。"""
        try:
            counts = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)
                counts.append(int(link_view.max_shapes))
            total = int(self.asset.root_physx_view.max_shapes)
            if sum(counts) != total:
                raise ValueError(f"shape 数对不上：sum(per_body)={sum(counts)} vs max_shapes={total}")
            return counts
        except Exception as exc:  # noqa: BLE001
            print(f"[privileged-v4] WARN: 无法解析 per-body shape 数（{type(exc).__name__}: {exc}）")
            return None

    @staticmethod
    def _scale(current: torch.Tensor, default) -> torch.Tensor:
        if isinstance(default, float):
            return current / max(default, 1e-8)
        return current / current.new_tensor(default).clamp_min(1e-8)

    def _friction_group_means(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (足部 3 维材质均值, 其余连杆 2 维静/动摩擦均值)。"""
        mats = self.asset.root_physx_view.get_material_properties().to(self._env.device)  # [N, S, 3]
        if self._num_shapes_per_body is None:
            # 退化路径：拿不到 shape↔body 映射时，两侧都用全局均值（会打 WARN 提示）
            foot = mats.mean(dim=1)
            other = mats.mean(dim=1)[:, :2]
            return foot, other
        starts = [0]
        for c in self._num_shapes_per_body:
            starts.append(starts[-1] + c)

        def _sel(body_ids: list[int]) -> torch.Tensor:
            idx = [s for b in body_ids for s in range(starts[b], starts[b + 1])]
            if not idx:
                return mats.new_zeros(mats.shape[0], 3)
            return mats[:, idx, :].mean(dim=1)

        return _sel(self.foot_body_ids), _sel(self.other_body_ids)[:, :2]

    def _compute(self, env_ids: torch.Tensor | slice | None = None):
        device = self._env.device
        sel = slice(None) if env_ids is None else env_ids

        masses = self.asset.root_physx_view.get_masses().to(device)
        default_mass = self.asset.data.default_mass.to(device)
        coms = self.asset.root_physx_view.get_coms().to(device)
        # IsaacLab 的 `ArticulationData` 没有 `default_com` 缓冲（只有 default_mass/default_inertia），
        # 所以 COM 的"默认值"只能自己取：本 term 的**第一次** `_compute()` 一定发生在 startup
        # 随机化之前（= ObservationManager 量维度那一次），此刻的 coms 就是默认值，缓存下来当基准。
        # 若时序哪天变了，这个通道会全 0 —— 启动日志里的"参考值 vs 当前值"那两行会直接暴露，
        # TB 的 `Latent/priv_input_spread` 也会掉到 0。
        if self._com_ref is None:
            self._com_ref = coms.clone()
            spread = self._com_ref[:, self.base_id, :3].std(dim=0, unbiased=False).max().item()
            print(
                f"[privileged-v4] COM 基准（取第一次调用=随机化前）：base_link[env0]="
                f"{[round(v, 5) for v in self._com_ref[0, self.base_id, :3].tolist()]}"
                f" 跨环境 std={spread:.2e}（应≈0；若非 0 说明基准取晚了）"
            )
        default_com = self._com_ref

        # [0:1] base 附加质量（add）
        base_payload = _to_unit(masses[:, self.base_id] - default_mass[:, self.base_id],
                                self.rng["base_add"]).unsqueeze(-1)
        # [1:4] base COM 偏移（add，逐轴归一化）
        d_com = coms[:, self.base_id, :3] - default_com[:, self.base_id, :3]
        base_com = torch.stack(
            [
                _to_unit(d_com[:, 0], self.rng["com_x"]),
                _to_unit(d_com[:, 1], self.rng["com_y"]),
                _to_unit(d_com[:, 2], self.rng["com_z"]),
            ],
            dim=-1,
        )
        # [4:5] gripper 质量缩放（scale，DR 中点为 0）
        ee_ratio = masses[:, self.gripper_id] / default_mass[:, self.gripper_id].clamp_min(1e-8)
        ee_payload = _to_unit(ee_ratio, self.rng["gripper_scale"]).unsqueeze(-1)
        # [5:8] 足部材质均值 + [8:10] 其余连杆摩擦均值
        foot, other = self._friction_group_means()
        foot_friction = torch.stack(
            [
                _to_unit(foot[:, 0], self.rng["static"]),
                _to_unit(foot[:, 1], self.rng["dynamic"]),
                _to_unit(foot[:, 2], self.rng["restitution"]),
            ],
            dim=-1,
        )
        link_friction = torch.stack(
            [
                _to_unit(other[:, 0], self.rng["static"]),
                _to_unit(other[:, 1], self.rng["dynamic"]),
            ],
            dim=-1,
        )
        # [10:54] 关节增益比：非轮关节 k + 全部关节 d（丢掉结构性恒 0 的轮子 stiffness）
        gain_cols = []
        for name, actuator in self.asset.actuators.items():
            if name not in self.included_actuators:
                continue
            gain_cols.append((actuator, self._scale(actuator.stiffness, actuator.cfg.stiffness),
                              self._scale(actuator.damping, actuator.cfg.damping)))
        k_ratio, d_ratio = {}, {}
        for actuator, ks, ds in gain_cols:
            for j, jn in enumerate(actuator.joint_names):
                k_ratio[jn] = ks[:, j]
                d_ratio[jn] = ds[:, j]
        gain_parts = [_to_unit(k_ratio[jn], self.rng["gain"]) for jn in self.pos_joint_names]
        gain_parts += [_to_unit(d_ratio[jn], self.rng["gain"]) for jn in self.asset.joint_names]
        joint_gain = torch.stack(gain_parts, dim=-1)

        value = torch.cat([base_payload, base_com, ee_payload, foot_friction, link_friction, joint_gain], dim=-1)
        value = value[sel]
        if self.buf is None or env_ids is None:
            self.buf = value
        else:
            self.buf[env_ids] = value
