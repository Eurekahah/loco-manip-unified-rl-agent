# mdp/actions.py
from __future__ import annotations
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
        step_dt = self._env.step_dt  # = physics_dt * decimation  # 0.02
        max_ee_lin_vel = 1.5   # m/s，按你的机械臂/任务合理设定
        max_ee_ang_vel = 6.0   # rad/s

        self.max_pos_step = max_ee_lin_vel * step_dt
        self.max_rot_step = max_ee_ang_vel * step_dt
        
        

    def process_actions(self, actions: torch.Tensor):
        command = self._env.command_manager.get_command(self.cfg.command_name)
        # print(f"CommandDrivenIKAction: command={command}")
        target_pos_b = command[:, 0:3]
        target_quat_b = command[:, 3:7]

        # 当前末端真实位姿（root系）
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # # ---- 位置增量限幅 ----
        # pos_error = target_pos_b - ee_pos_curr
        # pos_error_norm = torch.norm(pos_error, dim=-1, keepdim=True)
        # scale = torch.clamp(self.max_pos_step / (pos_error_norm + 1e-6), max=1.0)
        # clipped_pos_b = ee_pos_curr + pos_error * scale
        # # print(f"{pos_error_norm=}, {scale=}")
        # # print(f"target_pos_b: {target_pos_b}, ee_pos_curr: {ee_pos_curr}, pos_error: {pos_error}")

        # # ---- 姿态增量限幅 ----
        # # quat_error 满足: target = quat_error * current
        # quat_error = math_utils.quat_mul(target_quat_b, math_utils.quat_conjugate(ee_quat_curr))
        # # 保证走最短路径（四元数双重覆盖问题）
        # quat_error = torch.where(quat_error[:, 0:1] < 0, -quat_error, quat_error)
        # # print(f"{quat_error=}")

        # rotvec = math_utils.axis_angle_from_quat(quat_error)  # 方向=转轴, 模长=角度
        # angle = torch.norm(rotvec, dim=-1, keepdim=True)
        # axis = rotvec / (angle + 1e-6)
        # clipped_angle = torch.clamp(angle, max=self.max_rot_step)
        # clipped_quat_error = math_utils.quat_from_angle_axis(clipped_angle.squeeze(-1), axis)
        # clipped_quat_b = math_utils.quat_mul(clipped_quat_error, ee_quat_curr)
        # # print(f"{clipped_quat_error=}")
        # # ik_command = torch.cat([clipped_pos_b, clipped_quat_b], dim=-1)
        # ik_command = torch.cat([clipped_pos_b, target_quat_b], dim=-1)
        ik_command =  command # target_pos_b #
        self._ik_controller.set_command(ik_command, ee_pos_curr, ee_quat_curr)

    @property
    def action_dim(self) -> int:
        return 7  # 这里需要删掉的，临时debug设置

    def apply_actions(self):
        """调用父类的 apply_actions 来执行 IK 控制。"""
        super().apply_actions()

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

    # ------------------------------------------------------------------ #
    #  方式一：IsaacLab Debug Vis 框架（坐标轴 Marker）                    #
    # ------------------------------------------------------------------ #

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
    class_type: type = CommandDrivenIKAction
    command_name: str = "ee_pose"  # 对应CommandsCfg里的key