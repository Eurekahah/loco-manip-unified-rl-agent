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
        if self.count < 2 or self.buf is None:
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
