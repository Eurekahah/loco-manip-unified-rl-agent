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
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


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
