# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific rewards for the two-cylinder pick-and-place scene."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import compute_base_height_rel_to_feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_height_for_place_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    low_height: float = 0.40,
    high_height: float = 0.52,
    lift_transition_margin: float = 0.10,
    tolerance: float = 0.07,
) -> torch.Tensor:
    """Reward lowering at the source and raising the body for the high target.

    The desired body height switches once the object has been lifted near the
    high platform. This makes the vertical body motion useful without forcing a
    fixed posture throughout the whole episode.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]

    feet_cfg = SceneEntityCfg("robot", body_names=".*wheel")
    feet_cfg.resolve(env.scene)
    height = compute_base_height_rel_to_feet(env, asset_cfg, feet_cfg)

    target_height = target.data.root_pos_w[:, 2]
    lifted_for_high_place = obj.data.root_pos_w[:, 2] > (target_height - lift_transition_margin)
    desired_height = torch.where(
        lifted_for_high_place,
        torch.full_like(height, high_height),
        torch.full_like(height, low_height),
    )
    return torch.exp(-torch.square((height - desired_height) / max(tolerance, 1e-6)))


def _lift_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    lift_height: float,
) -> dict[str, torch.Tensor]:
    """Track per-episode lift state shared by the staged rewards."""
    obj = env.scene[object_cfg.name]
    current_height = obj.data.root_pos_w[:, 2]

    state = getattr(env, "_two_cylinder_lift_state", None)
    if state is None:
        state = {
            "init_height": current_height.clone(),
            "lifted_once": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "pickup_awarded": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        setattr(env, "_two_cylinder_lift_state", state)

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        state["init_height"][just_reset] = current_height[just_reset].clone()
        state["lifted_once"][just_reset] = False
        state["pickup_awarded"][just_reset] = False

    state["lifted_once"] |= (current_height - state["init_height"]) > lift_height
    return state


def pick_up_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_height: float = 0.04,
    ee_distance_threshold: float = 0.12,
    gripper_closed_threshold: float = 0.03,
) -> torch.Tensor:
    """One-time reward for lifting the object with the gripper closed."""
    state = _lift_state(env, object_cfg, lift_height)

    obj = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_ids, _ = robot.find_bodies("gripper_base")
    gripper_ids, _ = robot.find_joints(("gripper_joint1", "gripper_joint2"))

    ee_pos_w = robot.data.body_pos_w[:, ee_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_distance = torch.linalg.vector_norm(ee_pos_w - obj_pos_w, dim=-1)
    gripper_closed = torch.all(
        torch.abs(robot.data.joint_pos[:, gripper_ids]) < gripper_closed_threshold,
        dim=-1,
    )

    qualifies = (
        state["lifted_once"]
        & gripper_closed
        & (ee_distance < ee_distance_threshold)
        & (~state["pickup_awarded"])
    )
    state["pickup_awarded"] |= qualifies
    return qualifies.float()


def object_to_target_reward_gated(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_pad"),
    lift_height: float = 0.04,
    sensitivity: float = 6.0,
) -> torch.Tensor:
    """Reward object-to-target progress only after the object has been lifted."""
    state = _lift_state(env, object_cfg, lift_height)
    lifted = state["lifted_once"]

    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    curr_dist = torch.linalg.vector_norm(
        obj.data.root_pos_w[:, :3] - target.data.root_pos_w[:, :3],
        dim=-1,
    )

    key = "_two_cylinder_min_object_target_dist"
    if key not in env.extras:
        env.extras[key] = torch.full((env.num_envs,), float("inf"), device=env.device)
    min_dist = env.extras[key]

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        min_dist[just_reset] = float("inf")

    first_lifted = lifted & torch.isinf(min_dist)
    min_dist[first_lifted] = curr_dist[first_lifted]

    progress = (min_dist - curr_dist).clamp(min=0.0)
    env.extras[key] = torch.where(
        lifted,
        torch.minimum(min_dist, curr_dist),
        min_dist,
    )
    return torch.tanh(sensitivity * progress) * lifted.float()


def standpick_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.60,
    forward_speed_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward stopping near the object before closing the gripper."""
    obj = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    delta_xy = obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    base_distance = torch.linalg.vector_norm(delta_xy, dim=-1)
    forward_speed = torch.abs(robot.data.root_lin_vel_b[:, 0])
    return (
        (base_distance < distance_threshold)
        & (forward_speed < forward_speed_threshold)
    ).float()


def ee_object_progress_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensitivity: float = 20.0,
) -> torch.Tensor:
    """Reward new minimum EE-to-object distance."""
    obj = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_ids, _ = robot.find_bodies("gripper_base")
    ee_pos_w = robot.data.body_pos_w[:, ee_ids[0], :]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    curr_dist = torch.linalg.vector_norm(ee_pos_w - obj_pos_w, dim=-1)

    key = "_two_cylinder_min_ee_object_dist"
    if key not in env.extras:
        env.extras[key] = torch.full((env.num_envs,), float("inf"), device=env.device)
    min_dist = env.extras[key]

    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        min_dist[just_reset] = float("inf")

    progress = (min_dist - curr_dist).clamp(min=0.0)
    env.extras[key] = torch.minimum(min_dist, curr_dist)
    return torch.tanh(sensitivity * progress)
