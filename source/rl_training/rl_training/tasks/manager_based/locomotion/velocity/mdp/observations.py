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
- reset(env_ids) 保留接口、留空。如果以后把某个随机化改成 "reset" mode(每个 episode
  都重新随机化),把对应类 reset() 里注释掉的查询逻辑取消注释即可。
"""


class privileged_base_extra_payload(ManagerTermBase):
    """基座额外负载 (kg),相对默认质量的偏移量。对应 randomize_rigid_body_mass(add)。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        current_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self._env.device)
        default_mass = self.asset.data.default_mass[:, self.body_id].to(self._env.device)
        self.buf = (current_mass - default_mass).unsqueeze(-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf


class privileged_end_effector_payload(ManagerTermBase):
    """末端负载 (kg)。对应 randomize_rigid_body_mass(scale)。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0]
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        current_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self._env.device)
        default_mass = self.asset.data.default_mass[:, self.body_id].to(self._env.device)
        self.buf = (current_mass - default_mass).unsqueeze(-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf


class privileged_rigid_body_inertia(ManagerTermBase):
    """指定 body 的惯量偏移(对角项均值)。对应 randomize_rigid_body_inertia。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        current_inertia = self.asset.root_physx_view.get_inertias()[:, self.body_ids].to(self._env.device)
        default_inertia = self.asset.data.default_inertia[:, self.body_ids].to(self._env.device)
        self.buf = (current_inertia - default_inertia).mean(dim=-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf


class privileged_base_com_offset(ManagerTermBase):
    """基座质心偏移 (3,)。对应 randomize_com_positions。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        current_com = self.asset.root_physx_view.get_coms()[:, self.body_id, :3].to(self._env.device)
        default_com = self.asset.data.default_com[:, self.body_id, :3].to(self._env.device)
        self.buf = current_com - default_com

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf


class privileged_material_properties(ManagerTermBase):
    """脚部 PhysX 材质特权信息:静摩擦、动摩擦、恢复系数。对应 randomize_rigid_body_material。

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
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        materials = self.asset.root_physx_view.get_material_properties().to(self._env.device)  # [num_envs, num_bodies, 3]
        static_friction = materials[:, self.foot_body_ids, 0]
        dynamic_friction = materials[:, self.foot_body_ids, 1]
        restitution = materials[:, self.foot_body_ids, 2]
        self.buf = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf


class privileged_joint_gain_scale(ManagerTermBase):
    """关节 PD 增益缩放系数(stiffness/damping 相对默认值的比例)。对应 randomize_actuator_gains。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.included_actuators = {"joint", "wheel", "piper_arm", "piper_gripper"}  # 轮子和夹爪可按需去掉
        self.buf: torch.Tensor | None = None
        self.count = 0

    def _compute(self):
        scales = []
        for actuator_name, actuator in self.asset.actuators.items():
            if actuator_name not in self.included_actuators:
                continue
            current_k = actuator.stiffness
            current_d = actuator.damping
            default_k = actuator.cfg.stiffness
            default_d = actuator.cfg.damping
            k_scale = (
                current_k / max(default_k, 1e-8)
                if isinstance(default_k, float)
                else current_k / current_k.new_tensor(default_k).clamp_min(1e-8)
            )
            d_scale = (
                current_d / max(default_d, 1e-8)
                if isinstance(default_d, float)
                else current_d / current_d.new_tensor(default_d).clamp_min(1e-8)
            )
            scales.append(k_scale)
            scales.append(d_scale)
        self.buf = torch.cat(scales, dim=-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        pass

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        if self.count < 2 or self.buf is None:
            self._compute()
            self.count += 1
        return self.buf