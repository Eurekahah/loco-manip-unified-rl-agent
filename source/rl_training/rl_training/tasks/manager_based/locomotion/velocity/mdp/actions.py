# mdp/actions.py
from __future__ import annotations
import math
import os
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ik import (
    ProjectedIKOptions,
    make_seed_candidates,
    normalize_quat,
    orientation_error,
    projected_ik_step,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
    solve_multi_start,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RobustIKCfg:
    """Configuration for the projected/multi-start arm IK controller."""

    enabled: bool = True
    urdf_path: str | None = None
    arm_root_link_name: str = "arm_base_link"
    ee_link_name: str | None = None

    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.08
    position_weight: float = 1.0
    orientation_weight: float = 0.35

    step_gain: float = 0.45
    max_joint_step: float = 0.05
    adaptive_damping: bool = True
    damping_min: float = 0.002
    damping_max: float = 0.15
    singular_value_threshold: float = 0.06

    full_solve_interval_s: float = 1.0
    retarget_position_threshold: float = 0.15
    retarget_orientation_threshold: float = 0.50
    full_solve_iterations: int = 20
    full_step_gain: float = 0.70
    full_max_joint_step: float = 0.35
    num_restarts: int = 1
    random_seed: int = 0
    seed_travel_weight: float = 0.02

    stall_patience: int = 12
    improvement_tolerance: float = 5.0e-4
    orientation_fallback: bool = True
    joint_limit_margin: float = 0.01
    debug: bool = False


class CommandDrivenIKAction(DifferentialInverseKinematicsAction):
    """
    从 CommandManager 直接读取 root 系下的目标 EE 位姿，
    完全绕过 policy 的 action 输出来驱动机械臂 IK。

    适用于装载机械臂的轮腿式机器人（floating base），
    底盘运动由其他 action term 控制，机械臂由本 term 接管。

    command 格式: (num_envs, 7) -> [x, y, z, qw, qx, qy, qz]（绝对模式），
    或 (num_envs, 6) -> [dx, dy, dz, drx, dry, drz]（相对模式）。
    """

    cfg: CommandDrivenIKActionCfg

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self._ee_vis_markers: VisualizationMarkers | None = None
        self._command_override: torch.Tensor | None = None
        self._robust_ready = False
        self._full_ik_ready = False
        super().__init__(cfg, env)
        self._init_joint_protection()
        self._init_robust_ik()

    def process_actions(self, actions: torch.Tensor):
        if self._command_override is None:
            command = self._env.command_manager.get_command(self.cfg.command_name)
        else:
            command = self._command_override
        if command.shape[-1] != self._ik_controller.action_dim:
            mode = "relative" if self.cfg.controller.use_relative_mode else "absolute"
            raise ValueError(
                f"[{type(self).__name__}] {mode} IK 需要 {self._ik_controller.action_dim}D command，"
                f"实际收到 {command.shape[-1]}D。相对模式请通过 set_command_override() 注入 6D 位姿增量。"
            )
        # 当前末端真实位姿（root系）
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)
        if self._robust_ready:
            self._target_pos_b[:] = self._ik_controller.ee_pos_des
            self._target_quat_b[:] = normalize_quat(self._ik_controller.ee_quat_des)
            self._maybe_run_full_ik()

    def set_command_override(self, command: torch.Tensor | None) -> None:
        """覆盖 command_manager 的输入（Teleop 相对 IK 用）。``None`` 恢复默认来源。"""
        self._command_override = None if command is None else command

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._command_override = None
        if getattr(self, "_robust_ready", False):
            ids = slice(None) if env_ids is None else env_ids
            joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
            self._q_cmd[ids] = joint_pos[ids]
            self._q_goal[ids] = joint_pos[ids]
            self._q_solution[ids] = joint_pos[ids]
            self._q_last_feasible[ids] = joint_pos[ids]
            self._tracking_blocked[ids] = False
            self._orientation_relaxed[ids] = False
            self._full_solution_valid[ids] = False
            self._has_full_solve[ids] = False
            self._solve_timer[ids] = float(self.cfg.robust_ik.full_solve_interval_s)
            self._stall_count[ids] = 0
            self._last_weighted_error[ids] = float("inf")
            self._last_solve_target_pos_b[ids] = self._target_pos_b[ids]
            self._last_solve_target_quat_b[ids] = self._target_quat_b[ids]
            self._relaxed_quat_b[ids] = self._target_quat_b[ids]

    # ------------------------------------------------------------------
    # B2-②/⑤：IK 输出的"关节保护"（DEF-049）
    # ------------------------------------------------------------------
    def _init_joint_protection(self) -> None:
        """缓存臂关节的位置限位 / 速度限幅，用于给 IK 解出的关节目标做保护。

        为什么需要（DEF-049 实测）：臂是 **IK 直接驱动**的，
        `DifferentialIKController.compute` 返回的是 `joint_pos + delta`，
        既不查关节限位、也不看速度限幅 ⇒

        * 目标超出可达范围时，关节目标顶在限位外，PD 只能一路顶到力矩上限
          （实测 joint4/5/6 有 40~60% 时间贴着 100 N·m、joint5 顶限位 65%）；
        * `velocity_limit=3.0` 在 `DelayedPDActuator` 里**只参与力矩裁剪、不限速**
          （实测腕关节 |qd| p99 = 5.0 rad/s、超速时间占比 84%）。

        这里把限位/限速缓存下来，交给 `_protect_joint_target()` 用。注意
        `robot.data.joint_effort_limits` 这类张量在本资产上是 **1e9 占位值**，
        所以速度限幅优先从 **actuator 实例**读（`act.velocity_limit`）。
        """
        self._jpos_lo: torch.Tensor | None = None
        self._jpos_hi: torch.Tensor | None = None
        self._jvel_max: torch.Tensor | None = None
        self._protect_stats = {"steps": 0, "pos_clamp": 0, "vel_clamp": 0}

        n_j = self._num_protected_joints()

        # ---- 位置限位：取所有实例的**最紧**交集（限位逐实例相同，取 min/max 只是保险）----
        if not bool(getattr(self.cfg, "protect_joint_pos", True)):
            if bool(os.environ.get("RL_TRAINING_IK_PROTECT_DEBUG")):
                print("[IK protect] 位置 clamp 已被 cfg 关闭（protect_joint_pos=False）")
        else:
            try:
                lim = self._asset.data.joint_pos_limits[:, self._joint_ids, :]  # (N, J, 2)
                lo = lim[..., 0].max(dim=0).values
                hi = lim[..., 1].min(dim=0).values
                margin = float(getattr(self.cfg, "joint_limit_margin", 0.0))
                ok = torch.isfinite(lo) & torch.isfinite(hi) & (hi > lo)
                if bool(ok.all()):
                    self._jpos_lo = lo + margin
                    self._jpos_hi = hi - margin
                else:
                    print(f"[IK protect] 位置限位有 {int((~ok).sum())} 个关节不可用 ⇒ 不做位置 clamp")
            except Exception as exc:  # noqa: BLE001 - 保护项失败不该让训练起不来
                print(f"[IK protect] 位置限位读取失败（{type(exc).__name__}: {exc}）⇒ 不做位置 clamp")

        # ---- 速度限幅：cfg 显式给了就用它；**负数 = 显式关闭**；None/0 = 从 actuator 实例读 ----
        cfg_v = getattr(self.cfg, "max_joint_vel", None)
        if cfg_v is not None and float(cfg_v) < 0.0:
            self._jvel_max = None
        elif cfg_v is not None and float(cfg_v) > 0.0:
            self._jvel_max = torch.full((n_j,), float(cfg_v), device=self.device)
        else:
            self._jvel_max = self._resolve_velocity_limits(n_j)

        if bool(os.environ.get("RL_TRAINING_IK_PROTECT_DEBUG")):
            _p = "off" if self._jpos_lo is None else "on"
            _v = "off" if self._jvel_max is None else \
                "/".join(f"{float(x):.2f}" for x in self._jvel_max[:4])
            print(f"[IK protect] 位置 clamp={_p}；速度限幅={_v}（前 4 个关节）")

    def _init_robust_ik(self) -> None:
        """Initialize projected DLS state and the optional full multi-start IK chain."""
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        if not bool(r_cfg.enabled):
            if bool(r_cfg.debug):
                print("[IK robust] disabled; falling back to the stock one-step DLS")
            return

        num_envs = self.num_envs
        num_joints = self._num_protected_joints()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
        lower = limits[..., 0].max(dim=0).values
        upper = limits[..., 1].min(dim=0).values
        finite = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
        if not bool(finite.all()):
            if bool(r_cfg.debug):
                print(f"[IK robust] {int((~finite).sum())} joints have invalid limits; disabled")
            return
        margin = float(r_cfg.joint_limit_margin)
        lower = lower + margin
        upper = upper - margin
        if not bool((upper > lower).all()):
            raise ValueError("[IK robust] joint_limit_margin is too large for the configured arm")

        self._robust_joint_lower = lower.to(self.device)
        self._robust_joint_upper = upper.to(self.device)
        self._q_cmd = joint_pos.clone()
        self._q_goal = joint_pos.clone()
        self._q_solution = joint_pos.clone()
        self._q_last_feasible = joint_pos.clone()
        self._target_pos_b = torch.zeros(num_envs, 3, device=self.device)
        self._target_quat_b = torch.zeros(num_envs, 4, device=self.device)
        self._target_quat_b[:, 0] = 1.0
        self._relaxed_quat_b = self._target_quat_b.clone()
        self._last_solve_target_pos_b = self._target_pos_b.clone()
        self._last_solve_target_quat_b = self._target_quat_b.clone()
        self._has_full_solve = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._tracking_blocked = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._orientation_relaxed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._full_solution_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._solve_timer = torch.full(
            (num_envs,),
            float(r_cfg.full_solve_interval_s),
            dtype=torch.float32,
            device=self.device,
        )
        self._stall_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._last_weighted_error = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self._projected_options = self._make_projected_options()
        self._seed_generator = torch.Generator(device=self.device)
        self._seed_generator.manual_seed(int(r_cfg.random_seed))
        self._robust_stats = {
            "full_success": 0,
            "position_fallback": 0,
            "unreachable": 0,
            "stall_latched": 0,
        }

        self._chain_joint_indices = None
        if r_cfg.urdf_path:
            try:
                import pytorch_kinematics as pk

                with open(r_cfg.urdf_path, "rb") as urdf_file:
                    urdf = urdf_file.read()
                ee_link_name = r_cfg.ee_link_name or self.cfg.body_name
                chain = pk.build_serial_chain_from_urdf(
                    urdf,
                    ee_link_name,
                    root_link_name=r_cfg.arm_root_link_name,
                ).to(dtype=torch.float32, device=self.device)
                chain_names = list(chain.get_joint_parameter_names())
                action_names = self._protect_joint_names()
                missing = [name for name in chain_names if name not in action_names]
                if missing:
                    raise ValueError(f"URDF chain joints are missing from the action term: {missing}")
                if len(chain_names) != num_joints:
                    raise ValueError(
                        f"URDF chain has {len(chain_names)} joints but IK controls {num_joints}"
                    )
                self._chain_joint_indices = [action_names.index(name) for name in chain_names]
                self._fk_chain = chain
                self._fk_joint_names = chain_names
                self._chain_lower = lower[self._chain_joint_indices].contiguous()
                self._chain_upper = upper[self._chain_joint_indices].contiguous()
                self._chain_q_rest = 0.5 * (self._chain_lower + self._chain_upper)
                body_names = list(self._asset.data.body_names)
                self._arm_root_body_idx = body_names.index(r_cfg.arm_root_link_name)
                self._full_ik_ready = True
            except Exception as exc:  # noqa: BLE001 - projected DLS remains usable
                print(
                    f"[IK robust] full multi-start IK disabled "
                    f"({type(exc).__name__}: {exc}); projected DLS remains enabled"
                )

        self._robust_ready = True
        if bool(r_cfg.debug):
            print(
                f"[IK robust] projected DLS enabled; full_ik={self._full_ik_ready}, "
                f"joints={num_joints}, interval={r_cfg.full_solve_interval_s:.2f}s"
            )

    def _make_projected_options(self) -> ProjectedIKOptions:
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        return ProjectedIKOptions(
            position_tolerance=float(r_cfg.position_tolerance),
            orientation_tolerance=float(r_cfg.orientation_tolerance),
            position_weight=float(r_cfg.position_weight),
            orientation_weight=0.0
            if self.cfg.controller.command_type == "position"
            else float(r_cfg.orientation_weight),
            step_gain=float(r_cfg.step_gain),
            max_joint_step=float(r_cfg.max_joint_step),
            adaptive_damping=bool(r_cfg.adaptive_damping),
            damping_min=float(r_cfg.damping_min),
            damping_max=float(r_cfg.damping_max),
            singular_value_threshold=float(r_cfg.singular_value_threshold),
        )

    def _maybe_run_full_ik(self) -> None:
        """Retry full IK periodically or after a large target jump."""
        if not self._full_ik_ready:
            self._solve_timer += float(self._env.step_dt)
            if bool((self._solve_timer >= float(self.cfg.robust_ik.full_solve_interval_s)).any()):
                retry = self._solve_timer >= float(self.cfg.robust_ik.full_solve_interval_s)
                self._tracking_blocked[retry] = False
                self._stall_count[retry] = 0
                self._last_weighted_error[retry] = float("inf")
                self._solve_timer[retry] = 0.0
            return
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        self._solve_timer += float(self._env.step_dt)
        position_jump = torch.linalg.vector_norm(
            self._target_pos_b - self._last_solve_target_pos_b, dim=-1
        )
        orientation_jump = torch.linalg.vector_norm(
            orientation_error(self._last_solve_target_quat_b, self._target_quat_b), dim=-1
        )
        retry_due = self._tracking_blocked & (
            self._solve_timer >= float(r_cfg.full_solve_interval_s)
        )
        jumped = (~self._has_full_solve) | (
            position_jump >= float(r_cfg.retarget_position_threshold)
        ) | (orientation_jump >= float(r_cfg.retarget_orientation_threshold))
        env_ids = (retry_due | jumped).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._run_full_ik(env_ids)

    def _run_full_ik(self, env_ids: torch.Tensor) -> None:
        """Solve full pose, then position-only fallback, for the selected environments."""
        if not self._full_ik_ready or env_ids.numel() == 0:
            return
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        chain_indices = self._chain_joint_indices
        target_pos, target_quat = self._target_to_chain_root(
            env_ids, self._target_pos_b[env_ids], self._target_quat_b[env_ids]
        )
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids][env_ids][
            :, chain_indices
        ]
        previous_solution = self._q_solution[env_ids][:, chain_indices]
        seeds = make_seed_candidates(
            current_joint_pos,
            previous_solution,
            self._chain_q_rest.expand(env_ids.numel(), -1),
            self._chain_lower,
            self._chain_upper,
            int(r_cfg.num_restarts),
            self._seed_generator,
        )
        result = solve_multi_start(
            self._fk_chain,
            target_pos,
            target_quat,
            seeds,
            current_joint_pos,
            self._chain_lower,
            self._chain_upper,
            self._chain_q_rest,
            self._projected_options,
            max_iterations=int(r_cfg.full_solve_iterations),
            full_step_gain=float(r_cfg.full_step_gain),
            full_max_joint_step=float(r_cfg.full_max_joint_step),
            seed_travel_weight=float(r_cfg.seed_travel_weight),
        )

        solved_joint_pos = torch.zeros(
            (env_ids.numel(), self._num_protected_joints()),
            dtype=result.joint_pos.dtype,
            device=self.device,
        )
        solved_joint_pos[:, chain_indices] = result.joint_pos
        full_success = result.success.clone()

        fallback_success = torch.zeros_like(full_success)
        if bool(r_cfg.orientation_fallback) and bool((~full_success).any()):
            fallback_ids = (~full_success).nonzero(as_tuple=False).squeeze(-1)
            fallback = solve_multi_start(
                self._fk_chain,
                target_pos[fallback_ids],
                target_quat[fallback_ids],
                seeds[:, fallback_ids],
                current_joint_pos[fallback_ids],
                self._chain_lower,
                self._chain_upper,
                self._chain_q_rest,
                self._projected_options,
                max_iterations=int(r_cfg.full_solve_iterations),
                full_step_gain=float(r_cfg.full_step_gain),
                full_max_joint_step=float(r_cfg.full_max_joint_step),
                seed_travel_weight=float(r_cfg.seed_travel_weight),
                position_only=True,
            )
            fallback_solution = torch.zeros(
                (fallback_ids.numel(), self._num_protected_joints()),
                dtype=fallback.joint_pos.dtype,
                device=self.device,
            )
            fallback_solution[:, chain_indices] = fallback.joint_pos
            fallback_success[fallback_ids] = fallback.success
            solved_joint_pos[fallback_ids] = torch.where(
                fallback.success.unsqueeze(-1), fallback_solution, solved_joint_pos[fallback_ids]
            )

        full_success_ids = env_ids[full_success]
        fallback_ids = env_ids[fallback_success]
        failed_ids = env_ids[~(full_success | fallback_success)]
        full_solutions = solved_joint_pos[full_success]
        fallback_solutions = solved_joint_pos[fallback_success]

        if full_success_ids.numel() > 0:
            self._q_solution[full_success_ids] = full_solutions
            self._q_goal[full_success_ids] = full_solutions
            self._q_cmd[full_success_ids] = full_solutions
            self._q_last_feasible[full_success_ids] = full_solutions
            self._tracking_blocked[full_success_ids] = False
            self._orientation_relaxed[full_success_ids] = False
            self._full_solution_valid[full_success_ids] = True
            self._stall_count[full_success_ids] = 0
            self._last_weighted_error[full_success_ids] = float("inf")

        if fallback_ids.numel() > 0:
            self._q_solution[fallback_ids] = fallback_solutions
            self._q_goal[fallback_ids] = fallback_solutions
            self._q_cmd[fallback_ids] = fallback_solutions
            self._q_last_feasible[fallback_ids] = fallback_solutions
            self._tracking_blocked[fallback_ids] = False
            self._orientation_relaxed[fallback_ids] = True
            self._full_solution_valid[fallback_ids] = True
            self._stall_count[fallback_ids] = 0
            self._last_weighted_error[fallback_ids] = float("inf")
            _, relaxed_quat = self._chain_pose_to_root(
                fallback_ids, fallback_solutions[:, chain_indices]
            )
            self._relaxed_quat_b[fallback_ids] = relaxed_quat

        if failed_ids.numel() > 0:
            self._q_goal[failed_ids] = self._q_last_feasible[failed_ids]
            self._tracking_blocked[failed_ids] = True
            self._orientation_relaxed[failed_ids] = False
            self._full_solution_valid[failed_ids] = False

        self._last_solve_target_pos_b[env_ids] = self._target_pos_b[env_ids]
        self._last_solve_target_quat_b[env_ids] = self._target_quat_b[env_ids]
        self._has_full_solve[env_ids] = True
        self._solve_timer[env_ids] = 0.0
        self._robust_stats["full_success"] += int(full_success_ids.numel())
        self._robust_stats["position_fallback"] += int(fallback_ids.numel())
        self._robust_stats["unreachable"] += int(failed_ids.numel())
        if bool(r_cfg.debug):
            print(
                f"[IK robust] solve envs={env_ids.numel()} full={full_success_ids.numel()} "
                f"position_only={fallback_ids.numel()} failed={failed_ids.numel()}"
            )

    def _target_to_chain_root(
        self, env_ids: torch.Tensor, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert root-frame EE targets to the URDF chain root frame."""
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            target_pos_b,
            target_quat_b,
        )
        offset_pos = self._offset_for_envs(env_ids)
        offset_quat = self._offset_rot_for_envs(env_ids)
        body_quat_w = normalize_quat(quat_mul(target_quat_w, quat_conjugate(offset_quat)))
        body_pos_w = target_pos_w - math_utils.quat_apply(body_quat_w, offset_pos)
        arm_pos_w = self._asset.data.body_pos_w[env_ids, self._arm_root_body_idx]
        arm_quat_w = self._asset.data.body_quat_w[env_ids, self._arm_root_body_idx]
        return math_utils.subtract_frame_transforms(
            arm_pos_w, arm_quat_w, body_pos_w, body_quat_w
        )

    def _chain_pose_to_root(
        self, env_ids: torch.Tensor, chain_joint_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the EE pose in the articulation root frame for a chain configuration."""
        transform = self._fk_chain.forward_kinematics(chain_joint_pos).get_matrix()
        body_pos_ab = transform[:, :3, 3]
        body_quat_ab = quat_from_matrix(transform[:, :3, :3])
        offset_pos = self._offset_for_envs(env_ids)
        offset_quat = self._offset_rot_for_envs(env_ids)
        ee_pos_ab, ee_quat_ab = math_utils.combine_frame_transforms(
            body_pos_ab, body_quat_ab, offset_pos, offset_quat
        )
        arm_pos_b, arm_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            self._asset.data.body_pos_w[env_ids, self._arm_root_body_idx],
            self._asset.data.body_quat_w[env_ids, self._arm_root_body_idx],
        )
        return math_utils.combine_frame_transforms(
            arm_pos_b, arm_quat_b, ee_pos_ab, ee_quat_ab
        )

    def _offset_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._offset_pos.shape[0] == self.num_envs:
            return self._offset_pos[env_ids]
        return self._offset_pos.expand(env_ids.numel(), -1)

    def _offset_rot_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._offset_rot.shape[0] == self.num_envs:
            return self._offset_rot[env_ids]
        return self._offset_rot.expand(env_ids.numel(), -1)

    def _num_protected_joints(self) -> int:
        return (self._asset.num_joints if isinstance(self._joint_ids, slice)
                else len(self._joint_ids))

    def _protect_joint_names(self) -> list[str]:
        names = list(self._asset.data.joint_names)
        if isinstance(self._joint_ids, slice):
            return names[self._joint_ids]
        return [names[i] for i in self._joint_ids]

    def _resolve_velocity_limits(self, n_j: int) -> torch.Tensor | None:
        """从 actuator 实例读速度限幅（`act.velocity_limit`）；读不到 ⇒ 返回 None（不保护）。"""
        names = self._protect_joint_names()
        vals: dict[str, float] = {}
        try:
            for act in self._asset.actuators.values():
                jn = list(getattr(act, "joint_names", []) or [])
                if not jn:
                    continue
                buf = getattr(act, "velocity_limit", None)
                if buf is None and getattr(act, "cfg", None) is not None:
                    buf = getattr(act.cfg, "velocity_limit", None)
                if buf is None:
                    continue
                a = torch.as_tensor(buf, dtype=torch.float32).ravel().cpu()
                if a.numel() == 1:
                    a = a.expand(len(jn))
                elif a.numel() % len(jn) == 0:
                    # actuator 实例上的形状是 (num_envs, num_joints) ⇒ 取第 0 行
                    a = a[: len(jn)]
                if a.numel() != len(jn):
                    continue
                for n, v in zip(jn, a.tolist()):
                    vals[n] = float(v)
        except Exception:  # noqa: BLE001
            return None
        if not vals:
            return None
        arr = torch.tensor([vals.get(n, float("nan")) for n in names], dtype=torch.float32)
        ok = torch.isfinite(arr) & (arr > 0.0) & (arr < 1.0e6)
        if not bool(ok.any()):
            return None
        # 缺失的关节用 +inf ⇒ 对它们不生效（不猜数值）
        return torch.where(ok, arr, torch.full_like(arr, float("inf"))).to(self.device)

    def _protect_joint_target(self, joint_pos: torch.Tensor,
                              joint_pos_des: torch.Tensor) -> torch.Tensor:
        """把 IK 解出的关节目标夹进"可执行范围"（位置限位 + 单步变化量限速）。"""
        des = joint_pos_des
        st = self._protect_stats
        st["steps"] += 1
        if self._jpos_lo is not None:
            clamped = torch.clamp(des, self._jpos_lo, self._jpos_hi)
            st["pos_clamp"] += int((clamped != des).any(dim=-1).sum())
            des = clamped
        if self._jvel_max is not None:
            step_dt = float(getattr(self._env, "step_dt", 0.02))
            scale = float(getattr(self.cfg, "vel_limit_scale", 1.0))
            max_delta = self._jvel_max * step_dt * scale
            limited = joint_pos + torch.clamp(des - joint_pos, -max_delta, max_delta)
            st["vel_clamp"] += int((limited != des).any(dim=-1).sum())
            des = limited
        if os.environ.get("RL_TRAINING_IK_PROTECT_DEBUG") and st["steps"] % 200 == 0:
            print(f"[IK protect] step={st['steps']} 位置裁 {st['pos_clamp']} / "
                  f"限速裁 {st['vel_clamp']}")
        return des

    def apply_actions(self):
        """Compute a projected IK target, then latch it if the target is unreachable."""
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if self._robust_ready and ee_quat_curr.norm() != 0:
            joint_pos_des = self._robust_joint_target(
                ee_pos_curr, ee_quat_curr, joint_pos
            )
        elif ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        joint_pos_des = self._protect_joint_target(joint_pos, joint_pos_des)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def _robust_joint_target(
        self,
        ee_pos_curr: torch.Tensor,
        ee_quat_curr: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Run projected DLS and enforce continuity around the actual joint state."""
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        target_pos = self._target_pos_b
        target_quat = self._target_quat_b
        if bool(self._orientation_relaxed.any()):
            target_quat = torch.where(
                self._orientation_relaxed.unsqueeze(-1),
                self._relaxed_quat_b,
                target_quat,
            )
        position_only = torch.full_like(self._tracking_blocked, self.cfg.controller.command_type == "position")
        position_only |= self._orientation_relaxed
        jacobian = self._compute_frame_jacobian()

        # A single call handles mixed position-only/full-pose environments by running
        # the full-pose step and then replacing relaxed rows with a position-only step.
        step = projected_ik_step(
            ee_pos_curr,
            ee_quat_curr,
            target_pos,
            target_quat,
            jacobian,
            joint_pos,
            self._robust_joint_lower,
            self._robust_joint_upper,
            self._projected_options,
        )
        candidate = step.joint_pos
        if bool(position_only.any()):
            pos_step = projected_ik_step(
                ee_pos_curr,
                ee_quat_curr,
                target_pos,
                target_quat,
                jacobian,
                joint_pos,
                self._robust_joint_lower,
                self._robust_joint_upper,
                self._projected_options,
                position_only=True,
            )
            candidate = torch.where(position_only.unsqueeze(-1), pos_step.joint_pos, candidate)
            step = step._replace(
                position_error=step.position_error,
                orientation_error=torch.where(
                    position_only, torch.zeros_like(step.orientation_error), step.orientation_error
                ),
            )

        max_step = float(r_cfg.max_joint_step)
        commanded = self._q_cmd + torch.clamp(candidate - self._q_cmd, -max_step, max_step)
        # Keep the internally accumulated target close to the measured state. This avoids
        # "target wind-up" when the PD loop lags behind the IK command.
        commanded = joint_pos + torch.clamp(
            commanded - joint_pos, -2.0 * max_step, 2.0 * max_step
        )
        commanded = commanded.clamp(self._robust_joint_lower, self._robust_joint_upper)

        if bool(self._tracking_blocked.any()):
            held = joint_pos + torch.clamp(
                self._q_last_feasible - joint_pos, -max_step, max_step
            )
            commanded = torch.where(self._tracking_blocked.unsqueeze(-1), held, commanded)
        self._q_cmd[:] = commanded
        self._update_tracking_state(
            joint_pos,
            commanded,
            target_pos,
            target_quat,
            ee_pos_curr,
            ee_quat_curr,
            position_only,
        )
        return commanded

    def _update_tracking_state(
        self,
        joint_pos: torch.Tensor,
        joint_pos_des: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        ee_pos_curr: torch.Tensor,
        ee_quat_curr: torch.Tensor,
        position_only: torch.Tensor,
    ) -> None:
        """Update success, progress and failure-latch state from the current tracking error."""
        r_cfg: RobustIKCfg = self.cfg.robust_ik
        position_error = torch.linalg.vector_norm(target_pos - ee_pos_curr, dim=-1)
        ori_error = torch.linalg.vector_norm(
            orientation_error(ee_quat_curr, target_quat), dim=-1
        )
        active = ~self._tracking_blocked
        weighted_error = math.sqrt(self._projected_options.position_weight) * position_error
        weighted_error += torch.where(
            position_only,
            torch.zeros_like(ori_error),
            math.sqrt(self._projected_options.orientation_weight) * ori_error,
        )
        improved = (
            self._last_weighted_error - weighted_error
        ) > float(r_cfg.improvement_tolerance)
        stall = torch.where(
            improved & active,
            torch.zeros_like(self._stall_count),
            self._stall_count + 1,
        )
        success = (position_error <= float(r_cfg.position_tolerance)) & (
            position_only | (ori_error <= float(r_cfg.orientation_tolerance))
        )
        stall = torch.where(success, torch.zeros_like(stall), stall)
        newly_blocked = active & (~success) & (
            stall >= int(r_cfg.stall_patience)
        )
        self._stall_count[:] = stall
        self._last_weighted_error[:] = torch.where(
            active, weighted_error, self._last_weighted_error
        )
        if bool(success.any()):
            self._q_last_feasible[success] = joint_pos_des[success]
        if bool(newly_blocked.any()):
            self._tracking_blocked |= newly_blocked
            self._q_goal[newly_blocked] = self._q_last_feasible[newly_blocked]
            self._robust_stats["stall_latched"] += int(newly_blocked.sum())
            if bool(r_cfg.debug):
                print(
                    f"[IK robust] latency latch: {int(newly_blocked.sum())} env(s), "
                    f"total={self._robust_stats['stall_latched']}"
                )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def robust_ik_stats(self) -> dict[str, int]:
        """Cumulative robust-IK outcome counters for diagnostics."""

        return dict(self._robust_stats) if getattr(self, "_robust_stats", None) else {}

    @property
    def tracking_blocked(self) -> torch.Tensor:
        """Per-environment failure latch. Blocked environments hold the last feasible pose."""

        return self._tracking_blocked

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        if self.cfg.body_offset is not None:
            # 当前 EE（含 offset）在 root 系下的姿态
            _, ee_quat_b = self._compute_frame_pose()
            # 关键修复：把 body 系 offset 旋转到 root 系
            r_offset_b = math_utils.quat_apply(ee_quat_b, self._offset_pos)
            jacobian[:, 0:3, :] += torch.bmm(
                -math_utils.skew_symmetric_matrix(r_offset_b), jacobian[:, 3:, :]
            )
            # 旋转部分保持官方约定（offset rot 为 identity 时不变）
            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :]
            )
        return jacobian

    def _get_ee_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ee_pos_w  : (num_envs, 3)  EE 位置，世界坐标系
            ee_quat_w : (num_envs, 4)  EE 姿态，世界坐标系，wxyz 格式
        """
        # 1. gripper_base 在世界坐标系下的位姿
        #    self._body_idx 由父类 __init__ 解析（body_name="gripper_base"）
        body_pos_w  = self._asset.data.body_pos_w[:, self._body_idx, :]   # (N,3)
        body_quat_w = self._asset.data.body_quat_w[:, self._body_idx, :]  # (N,4) wxyz

        # 2. body_offset（父类已解析为 tensor，存在 self._offset_pos / self._offset_rot）
        #    shape: (1,3) / (1,4)，需 expand 到 (N,3)/(N,4)
        N = body_pos_w.shape[0]
        offset_pos  = self._offset_pos.expand(N, -1)   # (N,3)
        offset_quat = self._offset_rot.expand(N, -1)   # (N,4) wxyz

        # 3. T_world_ee = T_world_link6 ⊗ T_link6_ee
        ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
            body_pos_w, body_quat_w,
            offset_pos, offset_quat,
        )
        return ee_pos_w, ee_quat_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        """框架回调：开关 VisualizationMarkers。"""
        if debug_vis:
            if self._ee_vis_markers is None:
                marker_cfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/EE_Frame/ee_axis"
                )
                # 每个坐标轴 marker 由 3 个子 marker 组成（x/y/z），
                # 传入 num_envs 个位姿即可批量显示
                self._ee_vis_markers = VisualizationMarkers(marker_cfg)
            self._ee_vis_markers.set_visibility(True)
        else:
            if self._ee_vis_markers is not None:
                self._ee_vis_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """框架每帧回调：刷新 Marker 位姿。"""
        if self._ee_vis_markers is None:
            return

        ee_pos_w, ee_quat_w = self._get_ee_pose_world()

        scales = torch.tensor([[0.2, 0.2, 0.2]], device=ee_pos_w.device).expand(self.num_envs, -1)
    
        self._ee_vis_markers.visualize(
            translations=ee_pos_w,    # (N,3)
            orientations=ee_quat_w,   # (N,4) wxyz
            scales=scales,           # (3,) xyz 轴长度
        )


@configclass
class CommandDrivenIKActionCfg(ActionTermCfg):
    """
    CommandDrivenIKAction 的配置类。

    注意：不继承 DifferentialInverseKinematicsActionCfg 是为了避免
    action_space 被计算进 policy 的输出维度。
    如果你的框架要求所有 action term 都有 action_dim，
    可以改为继承 DifferentialInverseKinematicsActionCfg 并保持 class_type 指向本类。
    """

    class_type: type = CommandDrivenIKAction

    # --- 必须字段（与父类 DifferentialInverseKinematicsActionCfg 对齐）---

    joint_names: list[str] = MISSING
    """机械臂关节名称或正则，例如 ["arm_joint.*"]"""

    body_name: str = MISSING
    """末端执行器 body 名称，例如 "end_effector" """

    controller: DifferentialIKControllerCfg = MISSING
    """IK controller 配置；command_type='pose'，use_relative_mode 决定 7D 绝对或 6D 相对输入。"""

    # --- 本类新增字段 ---

    command_name: str = "ee_pose"
    """CommandManager 中目标位姿命令的 key，对应 CommandsCfg 里的属性名"""

    body_offset: mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg | None = None
    """EE frame 相对于 body frame 的偏移（可选）"""

    scale: float | tuple[float, ...] = 1.0
    """保留字段，本 action 中不使用（IK 直接接收绝对位姿）"""

    # --- B2-②/⑤：IK 输出的关节保护（DEF-049）---
    protect_joint_pos: bool = True
    """是否把 IK 解出的关节目标夹到关节位置限位内（B2-②；消融时可关掉）"""

    joint_limit_margin: float = 0.0
    """关节位置 clamp 的内缩余量（rad）：目标被夹到 `[lo+margin, hi-margin]`。0 = 夹到硬限位"""

    max_joint_vel: float = -1.0
    """单步关节目标的限速（rad/s）。**<0（默认 -1.0）= 关闭**；0 = 自动（从 actuator 实例的
    `velocity_limit` 读）；>0 = 用这个值。读不到限幅时也自动关闭（不猜数值）。

    ⚠️ **默认关闭是实测结论**（B2-⑤，DEF-054）：把 IK 目标的单步变化量限到
    `velocity_limit × step_dt` 会让"本来就到不了的目标"变成**长期存在的跟踪误差** ⇒
    PD 一直出力，`|tau|` 与 `|qd|` 反而**变大**（joint4 `|tau|` 均值 69.9→88.8 N·m、
    超速时间占比 84%→99.5%；joint5 76.5→90.4 / 35.7%→99.4%）。真要限速得从
    力矩/轨迹层做（或先做 B2-① 把不可达目标滤掉），不能只夹目标。

    实际限制的是"每个控制步的目标变化量"（`max_joint_vel × step_dt`），因为
    `velocity_limit` 在 `DelayedPDActuator` 里只参与力矩裁剪。
    注：这里用 `0.0` 而不是 `None` 当哨兵 —— hydra 覆盖 `None` 字段时会按 `NoneType`
    校验，连 `-1.0` 都传不进来（实测 `Expected: <class 'NoneType'>`）。"""

    vel_limit_scale: float = 1.0
    """限速系数的额外缩放（>1 更宽松，用于消融；1.0 = 直接用 actuator 的 velocity_limit）"""

    robust_ik: RobustIKCfg = RobustIKCfg()
    """Projected DLS + optional multi-start full IK configuration."""

    class_type: type = CommandDrivenIKAction
    command_name: str = "ee_pose"  # 对应CommandsCfg里的key
