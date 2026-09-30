# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .utils import compute_base_height_rel_to_feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2) / 2 + torch.exp(-lin_vel_error / 0.1) / 2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def body_height_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "body_pose",
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*wheel"),
) -> torch.Tensor:
    """机身高度跟踪奖励（高斯核，误差越小奖励越高）。"""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 3)
    target_h = cmd[:, 0]

    actual_h = compute_base_height_rel_to_feet(env, asset_cfg, feet_cfg)  # z 高度
    error = (actual_h - target_h).pow(2)
    return torch.exp(-error / (2 * std ** 2))


def body_pitch_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "body_pose",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身 pitch 跟踪奖励。"""

    from isaaclab.utils.math import euler_xyz_from_quat
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 3)
    target_p = cmd[:, 1]

    quat = asset.data.root_quat_w  # (N, 4) w,x,y,z
    _, actual_p, _ = euler_xyz_from_quat(quat)  # roll, pitch, yaw

    error = (actual_p - target_p).pow(2)
    return torch.exp(-error / (2 * std ** 2))


def body_roll_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "body_pose",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身 roll 跟踪奖励。"""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 3)
    target_r = cmd[:, 2]

    quat = asset.data.root_quat_w  # (N, 4) w,x,y,z
    actual_r, _, _ = euler_xyz_from_quat(quat)  # roll, pitch, yaw

    error = (actual_r - target_r).pow(2)
    return torch.exp(-error / (2 * std ** 2))

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty_wbc(
    env: ManagerBasedRLEnv,
    command_name: str,
    pose_command_name: str,           # 新增
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    pose_command_threshold: float = 0.05,
    default_height: float = 0.513,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    vel_cmd  = env.command_manager.get_command(command_name)
    pose_cmd = env.command_manager.get_command(pose_command_name)

    cmd_norm  = torch.linalg.norm(vel_cmd, dim=1)
    body_vel  = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    default_pose = vel_cmd.new_tensor([default_height, 0.0, 0.0])
    pose_is_default = torch.linalg.norm(pose_cmd - default_pose, dim=1) < pose_command_threshold

    running_reward = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )

    # 只有速度小 且 姿态也是默认值，才用大scale
    is_truly_still = (
        (cmd_norm < command_threshold)
        & (body_vel < velocity_threshold)
        & pose_is_default
    )

    reward = torch.where(is_truly_still, stand_still_scale * running_reward, running_reward)
    return reward

def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ---------------------------------------------------------------------------
# `action_mirror` / `action_sync` **2026-09-30 删除**（known_issues ⑧，
# docs/review/DEFECT_LOG_zh.md DEF-034）。原因：
#   1. 两者拿 `asset.find_joints(...)`（**articulation 关节 id**）去索引
#      `env.action_manager.action`，但动作向量是按**动作项自己的列序**排的
#      （本机型 = 12 腿(fl,fr,hl,hr) + 4 轮），两套下标不一致 ⇒ 打开 weight 就会算错；
#   2. 配置里的关节名（`FR_hip_joint` / `RL_thigh_joint` …）是 Go1 风格，本机型不存在
#      （M20 用 `fl_hipx_joint` …）⇒ 一打开就抛"找不到关节"。
#   3. 动作层的左右对称意义不大：同样的目的已由**状态层**的 `joint_mirror_signed`
#      （带符号约定、4 对镜像，见 DEF-027）实现。
# ---------------------------------------------------------------------------


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# def stand_still_joint_deviation_l1(
#     env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize offsets from the default joint positions when the command is very small."""
    # command = env.command_manager.get_command(command_name)
#     # Penalize motion when command is nearly zero.
#     return joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :], dim=1) < command_threshold)

# def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions that deviate from the default one."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
#     return torch.sum(torch.abs(angle), dim=1)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身朝上奖励."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ============================================================================
# 静止伫立（零速命令）专项
# ----------------------------------------------------------------------------
# 背景：用 scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py 把命令钉在
# (0,0,0) 上滚动某 checkpoint，实测 `err_vel_xy = 0.148 m/s`
# （历史基线 run `history_adaptation/2026-09-20_00-50-31` 的 model_19999.pt）——
# 命令为零时底盘仍以约 0.15 m/s 前向漂移。原因不是标定问题，而是**没有任何一项
# 奖励在"命令≈0"时惩罚底盘残余速度**（详见 docs/review/DEFECT_LOG_zh.md DEF-026）。
# ============================================================================


def stand_still_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    yaw_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """零速命令下惩罚底盘残余线速度（xy 平面）与残余 yaw 角速度。

    与 ``track_lin_vel_xy_exp`` 的关系
    ---------------------------------
    那个项在命令为 0 时当然也有梯度，但它是双高斯核，在 ``|v| ≈ 0.15`` 处已经饱和：
    ``exp(-0.0225/2)/2 + exp(-0.0225/0.1)/2 ≈ 0.49``，把 0.15 压到 0 只多拿 ~0.01 奖励，
    信噪比太低；这里直接用二次型，梯度在 0 附近也是最大的。

    为什么门控只看**命令**、不看实测速度
    -----------------------------------
    ``joint_pos_penalty_wbc`` 里的 ``is_truly_still`` 要求 ``body_vel < 0.5``，
    也就是说"正在漂移"这个事实恰好会把"站立姿态"惩罚关掉 —— 鸡生蛋：
    越漂越没有惩罚，越没惩罚越漂。所以本项只按命令门控。

    Args:
        command_name: 速度命令项名（通常是 ``base_velocity``）。
        command_threshold: 命令分量（线速度范数 / yaw 绝对值）低于该值视为"要求静止"。
        yaw_weight: yaw 角速度项的权重（1.0 = 与线速度同权）。
        asset_cfg: 资产（取 root 的 body 系速度，与 joint 选择无关）。

    Returns:
        ``|v_xy|² · 1[|cmd_xy| < threshold] + yaw_weight · ω_z² · 1[|cmd_z| < threshold]``
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    stand_lin = (torch.linalg.norm(cmd[:, :2], dim=1) < command_threshold).float()
    stand_yaw = (torch.abs(cmd[:, 2]) < command_threshold).float()
    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    yaw_rate_sq = torch.square(asset.data.root_ang_vel_b[:, 2])
    return lin_vel_sq * stand_lin + yaw_weight * yaw_rate_sq * stand_yaw


def stand_still_wheel_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
) -> torch.Tensor:
    """零速命令下惩罚轮关节转速（"轮足原地站着不该一直转"）。

    与 ``stand_still_vel_l2`` 的分工：那个管**底盘真的走了**（净位移），
    这个管**轮子空转**（打滑 / 靠轮子抖动维持平衡）。两者都要，因为
    "命令为 0 但轮子匀速转、机体被摩擦力顶住"这种情况前者测不到。

    注意：轮式倒立摆靠轮子微动平衡，"压太狠会摔"。所以本项权重应该比
    ``stand_still_vel_l2`` 小一到两个数量级（cfg 里默认 -0.02 对 -2.0）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    is_standing = (torch.linalg.norm(cmd[:, :2], dim=1) < command_threshold).float()
    wheel_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return wheel_vel_sq * is_standing


# ============================================================================
# 关节镜像惩罚（带符号约定）
# ============================================================================

_MIRROR_JOINT_TYPE_RE = re.compile(r"_(hipx|hipy|knee)_joint$")


def joint_mirror_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    mirror_signs: list[dict[str, float]],
    gate_by_gravity: bool = True,
) -> torch.Tensor:
    """带**关节符号约定**的镜像惩罚：``Σ_pairs Σ_j (θ_a,j − s_j · θ_b,j)² / n_pairs``

    为什么必须带符号（原 ``joint_mirror`` 的 bug）
    -------------------------------------------
    原 ``joint_mirror`` 直接算 ``(θ_a − θ_b)²``，隐含假设"镜像姿态里两侧关节角**相等**"。
    对本机型的腿关节这个假设**不成立**。三份独立证据：

    1. **MJCF 关节轴**（``deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml``）：
       四条腿的 hipx 轴都是 ``(-1,0,0)``、hipy/knee 轴都是 ``(0,-1,0)`` ——
       左右腿不是"旋转副本"而是"镜像副本"，所以同一个正角度在左右腿上是**反向**的。
    2. **关节限位**：``fl_hipx ∈ (-0.436, 0.611)`` 而 ``hr_hipx ∈ (-0.611, 0.436)``
       （恰好取负）；``fl_hipy ∈ (-2.583, 2.286)`` 而 ``hl_hipy ∈ (-2.286, 2.583)``。
       只有"绕 z 轴 180° 对称"能同时解释这两条。
    3. **默认姿态**（``deeprobotics.py`` 的 ``init_state.joint_pos``）：
       ``fl_hipy = -0.6`` vs ``hl_hipy = +0.6``、``fl_knee = +1.0`` vs ``hl_knee = -1.0`` ——
       默认站姿本身就是"对角取负"的对称姿态。

    结论（本机型的三条镜像关系）：

    ==================  ========  ========  =======
    镜像对               hipx      hipy      knee
    ==================  ========  ========  =======
    左/右 (fl↔fr, hl↔hr)  −1        +1        +1
    前/后 (fl↔hl, fr↔hr)  +1        −1        −1
    对角 (fl↔hr, fr↔hl)   −1        −1        −1
    ==================  ========  ========  =======

    而 cfg 里 ``mirror_joints`` 用的正是**对角对** —— 原实现在对角对上要求
    ``θ_fl = θ_hr``，即把"正确的镜像姿态"当误差、把"两条腿往同侧掰"当最优，
    直接产生"右后腿往右前方撇"这种反向撇腿（见 DEFECT_LOG_zh.md DEF-027）。

    Args:
        mirror_joints: ``[[regex_a, regex_b], ...]``，与 ``joint_mirror`` 同格式。
        mirror_signs: 与 ``mirror_joints`` **一一对应**的符号字典，键是关节类型
            （``hipx`` / ``hipy`` / ``knee``），值是 ``+1.0`` / ``-1.0``。
            用关节名后缀判定，**不依赖** ``find_joints`` 的返回顺序。
        gate_by_gravity: 与原 ``joint_mirror`` 一致，用 ``-g_z`` 做"直立才计分"的门控。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if len(mirror_joints) != len(mirror_signs):
        raise RuntimeError(
            f"joint_mirror_signed: mirror_joints({len(mirror_joints)}) 与 "
            f"mirror_signs({len(mirror_signs)}) 数量不一致"
        )
    cache = getattr(env, "joint_mirror_signed_cache", None)
    if cache is None or env.joint_mirror_signed_cache_key != tuple(map(tuple, mirror_joints)):
        cache = []
        for (pat_a, pat_b), sign_map in zip(mirror_joints, mirror_signs):
            ids_a, names_a = asset.find_joints(pat_a)
            ids_b, names_b = asset.find_joints(pat_b)
            if len(ids_a) != len(ids_b) or len(ids_a) == 0:
                raise RuntimeError(
                    f"joint_mirror_signed: 镜像对 {pat_a!r} / {pat_b!r} 匹配到的关节数不同"
                    f"（{len(ids_a)} vs {len(ids_b)}）—— 检查正则与关节命名。"
                )
            signs = []
            for name in names_a:
                m = _MIRROR_JOINT_TYPE_RE.search(name)
                if m is None or m.group(1) not in sign_map:
                    raise RuntimeError(
                        f"joint_mirror_signed: 无法从关节名 {name!r} 推出镜像符号；"
                        f"mirror_signs 必须给 {{hipx, hipy, knee}}（收到 {sign_map}）。"
                    )
                signs.append(float(sign_map[m.group(1)]))
            cache.append((ids_a, ids_b, signs))
        env.joint_mirror_signed_cache = cache
        env.joint_mirror_signed_cache_key = tuple(map(tuple, mirror_joints))

    reward = torch.zeros(env.num_envs, device=env.device)
    for ids_a, ids_b, signs in cache:
        sign_t = torch.tensor(signs, device=env.device, dtype=asset.data.joint_pos.dtype)
        diff = asset.data.joint_pos[:, ids_a] - sign_t.unsqueeze(0) * asset.data.joint_pos[:, ids_b]
        reward += torch.sum(torch.square(diff), dim=-1)
    reward = reward / len(cache)
    if gate_by_gravity:
        reward = reward * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
