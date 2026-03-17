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

    # def apply_actions(self):
    #     """
    #     重写 apply_actions 以正确处理 floating base 的 Jacobian 索引。

    #     PhysX 对 floating base 机器人返回的 Jacobian 最后一维：
    #       - 前 6 列对应 root 的 6-DoF（平移+旋转），不是关节
    #       - 之后才是各关节
    #     所以 joint_ids 需要偏移 +6 才能正确索引。
    #     """
    #     # 获取当前 EE 在 root 系下的位姿（父类私有方法，已验证存在）
    #     ee_pos_b, ee_quat_b = self._compute_frame_pose()

    #     # 获取当前关节角
    #     joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

    #     # 获取 Jacobian
    #     # 对于 floating base：PhysX Jacobian 最后一维前6列是root DoF，需要偏移
    #     if self._asset.is_fixed_base:
    #         print("[CommandDrivenIKAction] Fixed base detected, using joint_ids without offset.")
    #         jacobian = self._asset.root_physx_view.get_jacobians()[
    #             :, self._jacobi_body_idx, :, self._joint_ids
    #         ]
    #     else:
    #         print("[CommandDrivenIKAction] Floating base detected, applying +6 offset to joint_ids.")
    #         # floating base: joint_ids 在 PhysX Jacobian 中需要 +6 偏移
    #         joint_ids_tensor = torch.tensor(
    #             self._joint_ids, device=self._asset.device, dtype=torch.long
    #         )
    #         jacobian = self._asset.root_physx_view.get_jacobians()[
    #             :, self._jacobi_body_idx, :, joint_ids_tensor + 6
    #         ]

    #         # 将 Jacobian 从世界系旋转到 root 系
    #         # （父类已在最新版本中修复，但显式处理更安全）
    #         root_rot_matrix = math_utils.matrix_from_quat(
    #             math_utils.quat_inv(self._asset.data.root_quat_w)
    #         )  # (num_envs, 3, 3)
    #         jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
    #         jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])

    #     # 调用 IK controller 计算目标关节角
    #     joint_pos_des = self._ik_controller.compute(
    #         ee_pos_b, ee_quat_b, jacobian, joint_pos
    #     )

    #     # 设置关节位置目标
    #     self._asset.set_joint_position_target(
    #         joint_pos_des, joint_ids=self._joint_ids
    #     )


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