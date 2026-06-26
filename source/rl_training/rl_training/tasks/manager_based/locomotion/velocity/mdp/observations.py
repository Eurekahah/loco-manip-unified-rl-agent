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
      [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
       joint_pos(N), joint_vel(N), last_action(18)]
    """
    base_lin_vel = base_mdp.base_lin_vel(env, asset_cfg)
    base_ang_vel = base_mdp.base_ang_vel(env, asset_cfg)
    projected_gravity = base_mdp.projected_gravity(env, asset_cfg)
    joint_pos = base_mdp.joint_pos_rel(env, asset_cfg)
    joint_vel = base_mdp.joint_vel_rel(env, asset_cfg)
    last_action = base_mdp.last_action(env)
    return torch.cat(
        [base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, last_action],
        dim=-1,
    )

def privileged_base_extra_payload(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基座额外负载 (kg)，相对默认质量的偏移量。

    假设你在 EventCfg 里用 `events.randomize_rigid_body_mass` 对 base link 做了质量随机化，
    这里直接从 physx view 里读当前质量、减去 asset 的默认质量得到偏移。
    """
    asset = env.scene[asset_cfg.name]
    base_body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0
    current_mass = asset.root_physx_view.get_masses()[:, base_body_id].to(env.device)
    default_mass = asset.data.default_mass[:, base_body_id].to(env.device)
    return (current_mass - default_mass).unsqueeze(-1)


def privileged_end_effector_payload(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """末端负载 (kg)，同上，但取末端执行器 link 的 body_id。"""
    asset = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]  # 调用时传入末端 link 对应的 SceneEntityCfg(body_names=...)
    current_mass = asset.root_physx_view.get_masses()[:, ee_body_id].to(env.device)
    default_mass = asset.data.default_mass[:, ee_body_id].to(env.device)
    return (current_mass - default_mass).unsqueeze(-1)

def privileged_rigid_body_inertia(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指定 body 的惯量缩放系数，对应 randomize_rigid_body_inertia。"""
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    current_inertia = asset.root_physx_view.get_inertias()[:, body_ids].to(env.device)
    default_inertia = asset.data.default_inertia[:, body_ids].to(env.device)
    # 取对角项的均值缩放比例作为简化标量特征，也可保留完整张量
    return (current_inertia - default_inertia).mean(dim=-1)

def privileged_base_com_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基座质心偏移 (3,)，相对默认 COM 的偏移向量。"""
    asset = env.scene[asset_cfg.name]
    base_body_id = asset_cfg.body_ids[0] if asset_cfg.body_ids is not None else 0
    current_com = asset.root_physx_view.get_coms()[:, base_body_id, :3].to(env.device)
    default_com = asset.data.default_com[:, base_body_id, :3].to(env.device)
    return current_com - default_com

def privileged_friction_coefficient(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """地面摩擦系数。

    假设你用 `events.randomize_rigid_body_material` 对脚部/地面做了摩擦随机化，这里从
    physx material properties 里读静摩擦系数 (index 0: static friction)。如果你的随机化是对
    terrain 而不是 robot body 做的，需要换成对应 terrain asset 的读取方式。
    """
    asset = env.scene[asset_cfg.name]
    foot_body_ids = asset_cfg.body_ids  # 传入脚部 link 对应的 body_ids
    materials = asset.root_physx_view.get_material_properties().to(env.device)  # [num_envs, num_bodies, 3]
    static_friction = materials[:, foot_body_ids, 0]
    return static_friction.mean(dim=-1, keepdim=True)  # 多只脚取平均，或者保留逐脚维度自行决定

def privileged_restitution_coefficient(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """脚部恢复系数 (restitution)，对应 randomize_rigid_body_material 里的 restitution_range。"""
    asset = env.scene[asset_cfg.name]
    foot_body_ids = asset_cfg.body_ids
    materials = asset.root_physx_view.get_material_properties().to(env.device)  # [N, num_bodies, 3]
    restitution = materials[:, foot_body_ids, 2]  # index 2: restitution
    return restitution.mean(dim=-1, keepdim=True)


def privileged_joint_gain_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
) -> torch.Tensor:
    """指定关节子集（如手臂/腿部）的 stiffness、damping 缩放系数，对应 randomize_actuator_gains。

    不依赖任何自定义 buffer，直接读 articulation 当前的关节增益与默认值做比值。
    具体属性名请按你所用 IsaacLab 版本核实（不同版本可能是
    asset.data.joint_stiffness / asset.actuators[name].stiffness 等）。
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    current_stiffness = asset.data.joint_stiffness[:, joint_ids].to(env.device)
    default_stiffness = asset.data.default_joint_stiffness[:, joint_ids].to(env.device)
    current_damping = asset.data.joint_damping[:, joint_ids].to(env.device)
    default_damping = asset.data.default_joint_damping[:, joint_ids].to(env.device)

    stiffness_scale = current_stiffness / default_stiffness.clamp_min(1e-8)
    damping_scale = current_damping / default_damping.clamp_min(1e-8)
    return torch.cat([stiffness_scale, damping_scale], dim=-1)