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
from isaaclab.envs.mdp import modify_term_cfg  # noqa: F401, F403

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


def override_value(env, env_ids, data, value, num_steps):
    """通用的"超过 num_steps 后覆盖为 value"函数，配合 modify_term_cfg 使用。"""
    if env.common_step_counter > num_steps:
        return value
    return modify_term_cfg.NO_CHANGE  # 不触发则不写回


def ramp_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    num_steps: int,
) -> float:
    """把某个 reward term 的权重从 ``start_weight`` 线性升到 ``end_weight``。

    与官方 ``modify_reward_weight``（到达阈值后一次性跳变）不同，这里是线性爬升，
    适合"先学走路、再逐步加入手臂跟踪"这类课程。

    注意：curriculum 在 episode reset 时被调用（``common_step_counter`` 按 env step 计），
    所以实际更新频率 ≈ 1 / max_episode_length，对 2e4 步量级的爬升足够平滑。

    Args:
        term_name: reward term 名（与 RewardsCfg 里的属性名一致）。
        start_weight / end_weight: 起始与结束权重。
        num_steps: 爬升长度（env step）。
    """
    progress = min(max(env.common_step_counter / max(num_steps, 1), 0.0), 1.0)
    weight = start_weight + (end_weight - start_weight) * progress
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if abs(term_cfg.weight - weight) > 1e-12:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
    return weight


# ---------------------------------------------------------------------------
# 课程：区间阶梯 / 扰动缩放（幂等，可每步调用）
# ---------------------------------------------------------------------------


def _progress(env: ManagerBasedRLEnv, num_steps: int, start_scale: float) -> float:
    """线性进度：``common_step_counter`` 从 0 → num_steps 时返回 start_scale → 1.0。"""
    p = min(max(env.common_step_counter / max(int(num_steps), 1), 0.0), 1.0)
    return start_scale + (1.0 - start_scale) * p


def apply_range_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    stages: Sequence[dict],
) -> None:
    """按训练步数在预设的若干组 ``ranges`` 之间推进（**区间课程**）。

    ``stages`` 形如::

        [{"num_steps": 25_000, "ranges": {"p_l": (0.36, 0.47), ...}},
         {"num_steps": 50_000, "ranges": {...}},
         {"num_steps": 75_000, "ranges": {...}}]

    语义：``common_step_counter > num_steps`` 之后采用该组的取值；多组同时满足时取**最后**一组
    （即区间是单调放宽的）。只写 ``ranges`` 里出现过的字段，没写的保持 cfg 里的初始值（s0）。

    为什么不用 ``mdp.modify_term_cfg`` + 每个字段一个 term：``commands.<name>.ranges`` 有 6 个
    字段、3 个阶段就是 18 个 term，噪声太大。这里一个 term 管一组，而且**幂等**（每次都从
    stages 里写的目标值写回，不做相对修改），可以安全地每步调用。
    """
    term_cfg = env.command_manager.get_term(command_name).cfg
    active = [s for s in stages if env.common_step_counter > s["num_steps"]]
    if not active:
        return
    target = active[-1]["ranges"]
    changed = []
    for name, value in target.items():
        value = tuple(value) if isinstance(value, (list, tuple)) else value
        if getattr(term_cfg.ranges, name) != value:
            setattr(term_cfg.ranges, name, value)
            changed.append(f"{name}={value}")
    if changed:
        print(
            f"[curriculum] {command_name}.ranges 进入阶段 "
            f"(num_steps={active[-1]['num_steps']}, step={env.common_step_counter}): "
            + ", ".join(changed)
        )


def _scale_nested(obj, s: float):
    """把嵌套的数值 / 区间按比例 s 缩放（tuple/list/dict 递归）。"""
    if isinstance(obj, dict):
        return {k: _scale_nested(v, s) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return tuple(_scale_nested(v, s) for v in obj)
    return obj * s


def apply_event_scale(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    spec: Sequence[dict],
    num_steps: int,
    start_scale: float = 0.3,
) -> None:
    """把**扰动类事件**的参数从 ``start_scale`` 线性放大到 1.0（扰动课程）。

    ``spec`` 里给出"事件名 + 参数名 + **完整幅度**"，缩放始终基于完整幅度计算，
    因此幂等；例如::

        [{"term": "randomize_push_robot", "param": "velocity_range",
          "base": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
         {"term": "randomize_apply_external_force_torque", "param": "force_range",
          "base": (-10.0, 10.0)}]

    为什么需要：实测（`scripts/reinforcement_learning/rsl_rl/probe_root_height_termination.py`）
    第 0 步就全量开启的 push / 外力会让早期"一被推就趴窝"，
    而趴窝现在由 `root_height_below_minimum` 记账。
    """
    s = _progress(env, num_steps, start_scale)
    for item in spec:
        term_name = item["term"]
        # 事件可能被某个 cfg/探针关掉（置 None）—— `get_term_cfg` 对不存在的项会抛
        # ValueError，所以先查 active_terms，缺了就跳过（不要因此让训练崩掉）。
        if term_name not in env.event_manager.active_terms:
            continue
        term_cfg = env.event_manager.get_term_cfg(term_name)
        if term_cfg is None:
            continue
        new_value = _scale_nested(item["base"], s)
        if term_cfg.params.get(item["param"]) != new_value:
            term_cfg.params[item["param"]] = new_value
            env.event_manager.set_term_cfg(term_name, term_cfg)
            # 只在缩放比例有明显变化时打印（否则每步都会因为 0.03% 的变化刷屏）
            last = item.get("_last_printed_scale")
            if last is None or abs(s - last) >= 0.02:
                item["_last_printed_scale"] = s
                print(
                    f"[curriculum] 扰动 {term_name}.{item['param']} 缩放到 {s:.2f}× "
                    f"(step={env.common_step_counter}): {new_value}"
                )
