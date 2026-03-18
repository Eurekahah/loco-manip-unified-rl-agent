# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

def advance_arm_weight(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    # ---- 训练进度参数 ----
    max_iterations: int = 5000,          # 与 RunnerCfg.max_iterations 保持一致
    num_steps_per_env: int = 24,         # 与 RunnerCfg.num_steps_per_env 保持一致
    ramp_start_frac: float = 0.0,        # 从第几比例开始爬升（0 = 一开始就爬）
    ramp_end_frac: float = 0.5,          # 到第几比例时 max_weight 达到 max_target
    # ---- weight 目标值 ----
    max_target: float = 1.0,
    min_target: float = 0.8,
    min_start_frac: float = 0.3,         # max_weight 达到该比例后才开始推 min_weight
    initial_max_weight: float = 0.0,     # 训练最开始时 max_weight 的初始值
    initial_min_weight: float = 0.0,
) -> float:
    """
    按训练进度线性推进 arm_weight 的 max/min。

    进度计算：
        iteration ≈ common_step_counter / num_steps_per_env
        progress  = clamp((iteration - ramp_start) / (ramp_end - ramp_start), 0, 1)

    max_weight: initial_max_weight → max_target，在 [ramp_start_frac, ramp_end_frac] 内线性爬升
    min_weight: 当 max_weight 超过 min_start_frac 后，0 → min_target 线性爬升
    """
    cmd = env.command_manager.get_term("arm_weight")

    # ---------- 计算当前 iteration ----------
    num_envs = env.num_envs
    current_iter = env.common_step_counter / num_steps_per_env
    # ---------- 归一化进度 [0, 1] ----------
    ramp_start = ramp_start_frac * max_iterations
    ramp_end   = ramp_end_frac   * max_iterations
    if ramp_end <= ramp_start:
        progress = 1.0
    else:
        progress = float(torch.clamp(
            torch.tensor((current_iter - ramp_start) / (ramp_end - ramp_start)),
            0.0, 1.0
        ))

    # ---------- 线性插值 max_weight ----------
    new_max = initial_max_weight + progress * (max_target - initial_max_weight)
    cmd.set_max_weight(new_max)

    # ---------- 线性插值 min_weight（延迟启动）----------
    # min_weight 在 max_weight 超过 min_start_frac 之后才开始爬升
    if new_max >= min_start_frac:
        # 把 min_weight 的 progress 映射到 [min_start_frac, max_target] 区间
        min_progress = (new_max - min_start_frac) / max(max_target - min_start_frac, 1e-6)
        min_progress = float(torch.clamp(torch.tensor(min_progress), 0.0, 1.0))
        new_min = initial_min_weight + min_progress * (min_target - initial_min_weight)
        cmd.set_min_weight(new_min)

    return cmd.get_max_weight()
