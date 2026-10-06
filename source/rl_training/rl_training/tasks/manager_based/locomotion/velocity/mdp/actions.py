# mdp/actions.py
from __future__ import annotations
import os
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



class CommandDrivenIKAction(DifferentialInverseKinematicsAction):
    """
    从 CommandManager 直接读取世界坐标系下的目标 EE 位姿，
    完全绕过 policy 的 action 输出来驱动机械臂 IK。

    适用于装载机械臂的轮腿式机器人（floating base），
    底盘运动由其他 action term 控制，机械臂由本 term 接管。

    command 格式: (num_envs, 7) -> [x, y, z, qw, qx, qy, qz]，世界坐标系。
    """

    cfg: CommandDrivenIKActionCfg

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        
        self._ee_vis_markers: VisualizationMarkers | None = None
        super().__init__(cfg, env)
        self._init_joint_protection()

    def process_actions(self, actions: torch.Tensor):
        command = self._env.command_manager.get_command(self.cfg.command_name)
        # 当前末端真实位姿（root系）
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)

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
        """与基类同流程，但在写 PD 目标前先过 `_protect_joint_target()`。"""
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        joint_pos_des = self._protect_joint_target(joint_pos, joint_pos_des)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    @property
    def action_dim(self) -> int:
        return 0

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
    """IK controller 配置，必须设置 command_type='pose', use_relative_mode=False"""

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
    class_type: type = CommandDrivenIKAction
    command_name: str = "ee_pose"  # 对应CommandsCfg里的key
