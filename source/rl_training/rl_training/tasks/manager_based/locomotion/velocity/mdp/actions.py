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
        

    def process_actions(self, actions: torch.Tensor):
        """忽略 policy 输出，直接从 CommandManager 读取目标位姿并转换到 root 系。"""
        pass

        # 将世界系目标位姿转换到机器人 root 系
        # 这对于 floating base（底盘在移动）是必须的！
        # root_pos_w = self._asset.data.root_pos_w       # (num_envs, 3)
        # root_quat_w = self._asset.data.root_quat_w     # (num_envs, 4) (w, x, y, z)

        # target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        #     root_pos_w, root_quat_w,
        #     target_pos_w, target_quat_w
        # )
        # # 打印第一个环境的信息
        # ee_pos_b, ee_quat_b = self._compute_frame_pose()
        # error = torch.norm(target_pos_b[0] - ee_pos_b[0]).item()
        # print(f"IK controlling joints: {self._joint_names}")
        # print(list(enumerate(self._asset.data.joint_names)))
        # print(f"joint_ids: {self._joint_ids}")
        # print(f"target_b: {target_pos_b[0].cpu().numpy().round(3)}, "
        #     f"ee_b:     {ee_pos_b[0].cpu().numpy().round(3)}, "
        #     f"error:    {error:.4f}")
        # print(f"root_pos_w:     {self._asset.data.root_pos_w[0].cpu().numpy().round(3)}")
        # print(f"arm_link6_pos_w:{self._asset.data.body_pos_w[0, self._body_idx].cpu().numpy().round(3)}")
        # print(f"body_idx: {self._body_idx}")
        # print(f"body_name at idx: {self._asset.data.body_names[self._body_idx]}")
        # 拼接为 (num_envs, 7) 送入 IK controller
        # use_relative_mode=False 时 set_command 只需要目标位姿，不需要当前 EE 位姿
        # ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        # # self._ik_controller.set_command(ik_command)
        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # self._ik_controller.set_command(ik_command, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        """调用父类的 apply_actions 来执行 IK 控制。"""
        # 从 CommandManager 读取世界系目标位姿
        # command shape: (num_envs, 7) -> [x, y, z, qw, qx, qy, qz]
        command = self._env.command_manager.get_command(self.cfg.command_name)

        self._target_pos_w = command[:, 0:3]
        self._target_quat_w = command[:, 3:7]  # (w, x, y, z)
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            self._target_pos_w, self._target_quat_w
        )
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        
        # 用最新位姿 set_command，然后立即 compute，不存在时序差
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(ik_command, ee_pos_curr, ee_quat_curr)
        super().apply_actions()

    def _get_ee_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ee_pos_w  : (num_envs, 3)  EE 位置，世界坐标系
            ee_quat_w : (num_envs, 4)  EE 姿态，世界坐标系，wxyz 格式
        """
        # 1. arm_link6 在世界坐标系下的位姿
        #    self._body_idx 由父类 __init__ 解析（body_name="arm_link6"）
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