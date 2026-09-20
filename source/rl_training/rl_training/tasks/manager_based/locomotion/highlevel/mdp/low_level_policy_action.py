# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""预训练低层策略 action term 的公共基类（清单 ⑤ 的 R1 步骤）。

背景
----
``pre_trained_pick_action`` / ``pre_trained_pick_wbc_action`` / ``teleop_ll_action`` /
``pre_trained_nav_action`` / ``pre_trained_policy_action`` / ``openvla_pick_action``
这 6 个 action term 各自抄了一份：

* 载入低层 TorchScript 策略；
* 腿/轮/臂关节名单、低层 action cfg 的 scale/clip 赋值；
* 低层观测组的构造与覆写（``cfg.low_level_observations`` 就地改！）；
* ``last_action`` 闭包、复位时清低层动作缓存；
* 低层 tick 的 ``counter % low_level_decimation`` 循环、策略输出切分、路由给
  3 个低层 action term。

结果是清单 ②③④⑥⑨⑯ 的每一条都要在 6 个文件里各改一遍，而且"训练用什么、回放就用什么"
没有任何结构性保证。

:class:`LowLevelPolicyActionBase` 把这些收敛成**唯一一份**：

* 低层布局/观测/校验：复用 :mod:`low_level_replay`（``resolve_layout`` /
  ``build_low_level_observation_group`` / ``build_low_level_obs_manager`` /
  ``verify_low_level_layout``）；
* 低层 tick 状态（复位清理 + history 窗口）：``build_history_window()``；
* 调策略：``run_low_level_policy(policy, obs, history_flat)``（单/双输入自动分支）；
* ``ll_command`` / ``ll_command_w``：统一给奖励项用（清单 ①③；
  ``PreTrainedNavAction`` 以前**根本没有** ``ll_command``，奖励项一读就 AttributeError）。

子类只需要实现自己的"高层动作 -> ll_command"语义（``process_actions``）以及
``action_dim`` / ``raw_actions`` / ``processed_actions``，需要时再加 ``_on_low_level_tick()``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.highlevel.mdp.low_level_replay import (
    build_history_window,
    build_low_level_obs_manager,
    build_low_level_observation_group,
    check_low_level_action_cfgs,
    expected_policy_obs_dim,
    load_low_level_policy,
    resolve_layout,
    run_low_level_policy,
    verify_low_level_layout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LowLevelPolicyActionBase(ActionTerm):
    """高层 action term -> 预训练低层策略的公共骨架。"""

    cfg: ActionTermCfg
    """子类的 cfg（必须提供 policy_path / low_level_* 等字段，见模块文档）。"""

    #: 调试可视化用：是否打印低层动作缓存（默认关）
    print_ll_actions: bool = False

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ── 1. 载入低层策略：统一的加载 + 明确的报错（清单 ⑦）─────────────────
        self.policy = load_low_level_policy(cfg.policy_path, env, tag=type(self).__name__)

        # ── 2. 三个低层 action term（nav 只有腿+轮）────────────────────────────
        self._joint_pos_action_term: ActionTerm = cfg.low_level_leg_actions.class_type(
            cfg.low_level_leg_actions, env
        )
        self._wheel_vel_action_term: ActionTerm = cfg.low_level_wheel_actions.class_type(
            cfg.low_level_wheel_actions, env
        )
        ee_cfg = getattr(cfg, "low_level_ee_actions", None)
        self._ee_ik_action_term: ActionTerm | None = (
            None if ee_cfg is None else ee_cfg.class_type(ee_cfg, env)
        )

        # ── 3. 布局（清单 ④⑤⑥⑯）：关节名单/scale/clip/IK 槽位全部从低层 cfg 推导 ──
        self._layout = resolve_layout(
            robot=self.robot,
            low_level_obs_cfg=cfg.low_level_observations,
            low_level_leg_cfg=cfg.low_level_leg_actions,
            low_level_wheel_cfg=cfg.low_level_wheel_actions,
            declared_ee_action_dim=getattr(cfg, "ee_action_dim", -1),
            actual_ee_ik_action_dim=(
                0 if self._ee_ik_action_term is None else self._ee_ik_action_term.action_dim
            ),
            tag=type(self).__name__,
        )
        check_low_level_action_cfgs(
            tag=type(self).__name__,
            layout=self._layout,
            leg_cfg=cfg.low_level_leg_actions,
            wheel_cfg=cfg.low_level_wheel_actions,
        )
        self._joint_pos_dim = self._layout.leg_dim
        self._wheel_vel_dim = self._layout.wheel_dim
        self._ee_ik_dim = self._layout.ee_action_dim

        self.low_level_leg_actions = torch.zeros(self.num_envs, self._joint_pos_dim, device=self.device)
        self.low_level_wheel_actions = torch.zeros(self.num_envs, self._wheel_vel_dim, device=self.device)
        self.low_level_ee_actions = torch.zeros(self.num_envs, self._ee_ik_dim, device=self.device)

        # 低层动作缓存（= 训练时 ``actions`` 观测与 history 单步里的 ``last_action``）
        def last_action() -> torch.Tensor:
            parts = [self.low_level_leg_actions, self.low_level_wheel_actions]
            if self._layout.ee_action_dim > 0:
                parts.append(self.low_level_ee_actions)
            return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        self._last_action_fn = last_action

        # ── 4. 低层观测组 + 维度校验（不再就地改传入的 cfg）────────────────────
        self._ll_command = torch.zeros(
            self.num_envs, 3 + 7 + 3, device=self.device
        )  # [vx,vy,wz | ee_pos_b(3),ee_quat_b(4) | body_pose(3)]；子类按需覆写
        self._ll_command_w = torch.zeros_like(self._ll_command)

        self._low_level_obs_cfg = self._build_low_level_obs_cfg(cfg, last_action)
        self._ee_command_term = (
            None
            if getattr(cfg, "ee_command_name", None) is None
            else env.command_manager.get_term(cfg.ee_command_name)
        )
        self._expected_ll_obs_dim, self._policy_layout_json = expected_policy_obs_dim(
            self.policy, cfg.policy_path, tag=type(self).__name__
        )
        self._low_level_obs_manager, self._low_level_obs_cfg, self._ll_used_ee_goal = (
            build_low_level_obs_manager(
                env=env,
                obs_cfg=self._low_level_obs_cfg,
                group_name="ll_policy",
                expected_obs_dim=self._expected_ll_obs_dim,
                tag=type(self).__name__,
            )
        )
        verify_low_level_layout(
            tag=type(self).__name__,
            robot=self.robot,
            layout=self._layout,
            obs_manager=self._low_level_obs_manager,
            group_name="ll_policy",
            policy=self.policy,
            expected_obs_dim=self._expected_ll_obs_dim,
            policy_layout_json=self._policy_layout_json,
        )

        # ── 5. 低层 tick 状态：复位清理 + 可选 history 窗口 ────────────────────
        caches = [self.low_level_leg_actions, self.low_level_wheel_actions]
        if self._layout.ee_action_dim > 0:
            caches.append(self.low_level_ee_actions)
        self._ll_replay_state = build_history_window(
            env=env,
            layout=self._layout,
            policy_layout_json=self._policy_layout_json,
            last_action_fn=last_action,
            cache_tensors=caches,
            asset_name=cfg.asset_name,
            tag=type(self).__name__,
        )
        self._counter = 0

    # ------------------------------------------------------------------ #
    # 子类可覆盖的钩子
    # ------------------------------------------------------------------ #

    def _build_low_level_obs_cfg(self, cfg: ActionTermCfg, last_action_fn):
        """构造低层观测组；子类可覆盖以接入自己的 command 字段。"""
        return build_low_level_observation_group(
            cfg.low_level_observations,
            layout=self._layout,
            actions_fn=lambda dummy_env: last_action_fn(),
            velocity_commands_fn=lambda dummy_env: self._ll_command[:, :3],
            ee_goal_fn=lambda dummy_env: self._ll_command[:, 3:10],
            body_pose_cmd_fn=lambda dummy_env: self._ll_command[:, 10:13],
        )

    def _on_low_level_tick(self) -> None:
        """在低层 tick（跑策略之前）做额外处理，默认什么都不做。"""
        return None

    def _route_policy_output(self, policy_output: torch.Tensor) -> None:
        """把低层策略输出切分给低层 action term。"""
        leg, wheel, ee = self._layout.split(policy_output)
        self.low_level_leg_actions[:] = leg
        self.low_level_wheel_actions[:] = wheel
        if self._layout.ee_action_dim > 0:
            self.low_level_ee_actions[:] = ee

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #

    @property
    def ll_command(self) -> torch.Tensor:
        """规范形式的高层命令：``[vx,vy,wz, ee_pos_b(3), ee_quat_b(4), body_pose(3)]``（root 系）。

        奖励项与低层观测都读它（清单 ①③）。
        """
        return self._ll_command

    @property
    def ll_command_w(self) -> torch.Tensor:
        """``ll_command`` 的世界系副本（``[vx,vy,wz, ee_pos_w(3), ee_quat_w(4), h,p,r]``）。"""
        return self._ll_command_w

    def apply_actions(self):
        """低层 tick：复位处理 -> 推 history 帧 -> 算低层观测 -> 跑低层策略 -> 路由动作。"""
        if self._counter % self.cfg.low_level_decimation == 0:
            history_flat = self._ll_replay_state.on_tick()
            self._on_low_level_tick()
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            policy_output = run_low_level_policy(self.policy, low_level_obs, history_flat)
            self._route_policy_output(policy_output)

            self._joint_pos_action_term.process_actions(self.low_level_leg_actions)
            self._wheel_vel_action_term.process_actions(self.low_level_wheel_actions)
            if self._ee_ik_action_term is not None:
                self._ee_ik_action_term.process_actions(self.low_level_ee_actions)
            self._counter = 0

        self._joint_pos_action_term.apply_actions()
        self._wheel_vel_action_term.apply_actions()
        if self._ee_ik_action_term is not None:
            self._ee_ik_action_term.apply_actions()
        self._counter += 1
