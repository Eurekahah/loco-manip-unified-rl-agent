# Copyright (c) 2024, Your Lab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PreTrainedPickAction 的 OpenVLA 替换版本。

改动说明：
  - 删除高层 policy.pt 推理，改为 OpenVLA 直接从腕部相机输出 EE 目标位姿
  - leg / wheel 速度命令固定为 0（机器狗静止，只操纵手臂）
  - gripper 从 VLA 输出的第 7 维（0/1）路由给外部 BinaryJointPositionAction
  - 其余低层控制逻辑（IK、关节控制）与原版完全一致
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import MISSING
from PIL import Image
from scipy.spatial.transform import Rotation as R

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationManager
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as mdp

class VLAPickAction(ActionTerm):
    """
    将原 PreTrainedPickAction 中的高层 EE 目标来源替换为 OpenVLA。

    改动点（相对原版）：
      1. 不再需要外部传入 ee_pose actions（process_actions 只接收 vel 命令）
      2. _run_vla_inference() 每 N 步调用，覆写 raw_actions[3:10]
      3. gripper 写入 raw_actions[10]，由外部 BinaryJointPositionAction 读取
      4. policy.pt 推理、低层三路 ActionTerm 逻辑与原版完全一致，不做任何修改
    """

    cfg: VLAPickActionCfg

    leg_joint_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]
    arm_joint_names = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    joint_names = leg_joint_names + wheel_joint_names + arm_joint_names

    def __init__(self, cfg: VLAPickActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ── 加载低层 policy.pt（与原版完全相同）─────────────────────────
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # raw_actions: [vel(3) | ee_pose(7) | gripper(1)] = 11 维
        # vel(3) 由外部高层传入；ee_pose(7) 由 VLA 覆写；gripper(1) 由 VLA 覆写
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # ── 三路低层 ActionTerm（与原版完全相同）─────────────────────────
        self._joint_pos_action_term: ActionTerm = cfg.low_level_leg_actions.class_type(
            cfg.low_level_leg_actions, env
        )
        self._wheel_vel_action_term: ActionTerm = cfg.low_level_wheel_actions.class_type(
            cfg.low_level_wheel_actions, env
        )
        self._ee_ik_action_term: ActionTerm = cfg.low_level_ee_actions.class_type(
            cfg.low_level_ee_actions, env
        )

        self._joint_pos_dim = self._joint_pos_action_term.action_dim
        self._wheel_vel_dim = self._wheel_vel_action_term.action_dim
        self._ee_ik_dim     = self._ee_ik_action_term.action_dim

        self.low_level_leg_actions   = torch.zeros(self.num_envs, self._joint_pos_dim, device=self.device)
        self.low_level_wheel_actions = torch.zeros(self.num_envs, self._wheel_vel_dim,  device=self.device)
        self.low_level_ee_actions    = torch.zeros(self.num_envs, self._ee_ik_dim,      device=self.device)

        # scale / clip（与原版完全相同）
        self._joint_pos_action_term.scale = {
            ".*_hipx_joint": 0.125,
            r'^(?!.*_hipx_joint)(?!.*arm_joint).*': 0.25
        }
        self._wheel_vel_action_term.scale = 20.0
        self._joint_pos_action_term.clip  = {".*": (-100.0, 100.0)}
        self._wheel_vel_action_term.clip  = {".*": (-100.0, 100.0)}
        self._joint_pos_action_term.joint_names = self.leg_joint_names
        self._wheel_vel_action_term.joint_names = self.wheel_joint_names

        # ── low-level obs 回调（与原版完全相同）─────────────────────────
        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_mask = env.episode_length_buf == 0
                self.low_level_leg_actions[reset_mask]   = 0
                self.low_level_wheel_actions[reset_mask] = 0
                self.low_level_ee_actions[reset_mask]    = 0
                self._raw_actions[reset_mask]            = 0
            return torch.cat([
                self.low_level_leg_actions,
                self.low_level_wheel_actions,
                self.low_level_ee_actions
            ], dim=-1)

        cfg.low_level_observations.actions.func   = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = {}

        # velocity_commands 从 raw_actions[:, :3] 读（外部高层传入，与原版相同）
        cfg.low_level_observations.velocity_commands.func   = lambda dummy_env: self._raw_actions[:, :3]
        cfg.low_level_observations.velocity_commands.params = {}

        # ee_goal 从 raw_actions[3:10] 读 ← VLA 每 N 步覆写这里
        cfg.low_level_observations.ee_goal.func   = lambda dummy_env: self._raw_actions[:, 3:10]
        cfg.low_level_observations.ee_goal.params = {}

        # 其余 obs 配置（与原版完全相同）
        cfg.low_level_observations.joint_pos.func = mdp.joint_pos_rel_without_wheel
        cfg.low_level_observations.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        cfg.low_level_observations.base_ang_vel.scale = 0.25
        cfg.low_level_observations.joint_pos.scale    = 1.0
        cfg.low_level_observations.joint_vel.scale    = 0.05
        cfg.low_level_observations.base_lin_vel       = None
        cfg.low_level_observations.height_scan        = None
        cfg.low_level_observations.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        cfg.low_level_observations.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self._ee_command_term = env.command_manager.get_term(cfg.ee_command_name)
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        # ── 缓存 EE body index，避免每步查找 ────────────────────────────
        self._ee_body_idx = self.robot.find_bodies(cfg.ee_body_name)[0][0]

        # ── 加载 OpenVLA ──────────────────────────────────────────────────
        self._load_openvla(cfg)
        self._vla_step = 0

        # 用当前实际 EE 位姿初始化，防止第一帧跳变
        self._init_ee_pose()

        # ── 直接在内部实例化 gripper ActionTerm ──────────────────────────
        from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
        
        gripper_cfg = BinaryJointPositionActionCfg(
            asset_name=cfg.asset_name,
            joint_names=["arm_joint7", "arm_joint8"],
            open_command_expr={"arm_joint7": 0.04, "arm_joint8": 0.04},
            close_command_expr={"arm_joint7": 0.0,  "arm_joint8": 0.0},
        )
        self._gripper_action_term: ActionTerm = gripper_cfg.class_type(gripper_cfg, env)

        # gripper 状态缓存：1.0=open, 0.0=close
        self._gripper_cmd = torch.zeros(self.num_envs, 1, device=self.device)

        self._counter = 0

    # ──────────────────────────────────────────────────────────────────────
    #   OpenVLA 加载
    # ──────────────────────────────────────────────────────────────────────

    def _load_openvla(self, cfg: VLAPickActionCfg):
        print(f"[VLAPickAction] Loading OpenVLA from: {cfg.vla_model_path}")
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            cfg.vla_model_path, trust_remote_code=True
        )
        self._vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(cfg.vla_device)
        self._vla.eval()
        for p in self._vla.parameters():
            p.requires_grad_(False)

        self._vla_device  = cfg.vla_device
        self._unnorm_key  = cfg.vla_unnorm_key
        self._task_prompt = f"In: What action should the robot take to {cfg.task_instruction}?\nOut:"
        self._pos_scale   = cfg.vla_pos_scale
        self._rot_scale   = cfg.vla_rot_scale
        print("[VLAPickAction] OpenVLA loaded and frozen ✓")

    def _init_ee_pose(self):
        """用当前实际 EE 位姿初始化 raw_actions[3:10]。"""
        try:
            pos  = self.robot.data.body_pos_w[:, self._ee_body_idx, :]
            quat = self.robot.data.body_quat_w[:, self._ee_body_idx, :]
            self._raw_actions[:, 3:6]  = pos
            self._raw_actions[:, 6:10] = quat
        except Exception as e:
            print(f"[VLAPickAction] Warning: could not init EE pose: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #   ActionTerm 接口
    # ──────────────────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        # [vel(3) | ee_pose(7) | gripper(1)] = 11
        # ee_pose 由 VLA 覆写，外部高层只需传入 vel(3)
        # 但为保持接口兼容，仍声明 11 维，高层传入的 actions[:, 3:] 会被 VLA 忽略
        return 11

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        """
        与原版逻辑相同，负责处理 vel 命令并做限幅。
        ee_pose 部分（actions[:, 3:10]）由 VLA 在 apply_actions 里覆写，
        所以这里即使写入了也会被覆盖，无需关心。
        """
        self._raw_actions[:] = actions

        r = self.cfg.low_level_command_ranges

        # vel 限幅（与原版完全相同）
        self._raw_actions[:, 0].clamp_(r.lin_vel_x[0], r.lin_vel_x[1])
        self._raw_actions[:, 1].clamp_(r.lin_vel_y[0], r.lin_vel_y[1])
        self._raw_actions[:, 2].clamp_(r.ang_vel_z[0], r.ang_vel_z[1])

        # ee_pose 的 clamp 和坐标转换移到 _run_vla_inference() 里做
        # 此处不再处理 actions[:, 3:]，等 VLA 覆写

    def apply_actions(self):
        # ── Step 1: VLA 推理，覆写 raw_actions[3:10] 和 raw_actions[10] ──
        if self._vla_step % self.cfg.vla_infer_every_n == 0:
            self._run_vla_inference()
        self._vla_step += 1

        # ── Step 2: gripper 控制（直接在内部驱动）───────────────────────
        self._gripper_action_term.process_actions(self._gripper_cmd)
        self._gripper_action_term.apply_actions()

        # ── Step 3: 低层 policy.pt 推理（与原版完全相同）────────────────
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # policy.pt 输入：low_level_obs（包含 ee_goal = raw_actions[3:10]）
            # 输出切分给三路 ActionTerm，与原版完全相同
            policy_output = self.policy(low_level_obs)
            self.low_level_leg_actions[:]   = policy_output[:, :self._joint_pos_dim]
            self.low_level_wheel_actions[:] = policy_output[:, self._joint_pos_dim:self._joint_pos_dim + self._wheel_vel_dim]
            self.low_level_ee_actions[:]    = policy_output[:, self._joint_pos_dim + self._wheel_vel_dim:self._joint_pos_dim + self._wheel_vel_dim + self._ee_ik_dim]

            # 更新 CommandManager（与原版完全相同）
            self._ee_command_term.pose_command_w[:] = self._raw_actions[:, 3:10]

            self._joint_pos_action_term.process_actions(self.low_level_leg_actions)
            self._wheel_vel_action_term.process_actions(self.low_level_wheel_actions)
            self._ee_ik_action_term.process_actions(self.low_level_ee_actions)
            self._counter = 0

        self._joint_pos_action_term.apply_actions()
        self._wheel_vel_action_term.apply_actions()
        self._ee_ik_action_term.apply_actions()
        self._counter += 1

    # ──────────────────────────────────────────────────────────────────────
    #   VLA 推理：覆写 raw_actions[3:10] 和 raw_actions[10]
    # ──────────────────────────────────────────────────────────────────────

    def _run_vla_inference(self):
        sensor   = self._env.scene.sensors[self.cfg.camera_sensor_name]
        rgb_tensor = sensor.data.output["rgb"]  # [N, H, W, 3] uint8

        new_pos_list    = []
        new_quat_list   = []
        new_gripper_list = []

        for env_idx in range(self.num_envs):
            # 1. 当前 EE 位姿（world frame）
            cur_pos_w  = self.robot.data.body_pos_w[env_idx, self._ee_body_idx]   # [3]
            cur_quat_w = self.robot.data.body_quat_w[env_idx, self._ee_body_idx]  # [4] wxyz

            # 2. OpenVLA 推理
            raw_np  = rgb_tensor[env_idx].cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(raw_np)

            with torch.inference_mode():
                inputs  = self._processor(
                    self._task_prompt, pil_img
                ).to(self._vla_device, dtype=torch.bfloat16)
                vla_out = self._vla.predict_action(
                    **inputs,
                    unnorm_key=self._unnorm_key,
                    do_sample=False,
                )  # numpy [7]: [dx, dy, dz, dRx, dRy, dRz, gripper]

            # 3. 解析
            delta_pos_t   = torch.tensor(
                (vla_out[0:3] * self._pos_scale).astype(np.float32), device=self.device
            )
            delta_euler_np = vla_out[3:6] * self._rot_scale

            # 4. Euler delta → quat delta (wxyz)
            from scipy.spatial.transform import Rotation as R_scipy
            xyzw = R_scipy.from_euler("xyz", delta_euler_np, degrees=False).as_quat()
            delta_quat_t = torch.tensor(
                [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float32, device=self.device
            )

            # 5. 叠加到当前 EE 位姿
            new_pos  = cur_pos_w + delta_pos_t
            new_quat = torch.nn.functional.normalize(
                math_utils.quat_mul(delta_quat_t.unsqueeze(0), cur_quat_w.unsqueeze(0)).squeeze(0),
                p=2, dim=-1
            )

            # 6. 位置限幅
            r = self.cfg.low_level_command_ranges
            new_pos[0].clamp_(r.ee_pos_x[0], r.ee_pos_x[1])
            new_pos[1].clamp_(r.ee_pos_y[0], r.ee_pos_y[1])
            new_pos[2].clamp_(r.ee_pos_z[0], r.ee_pos_z[1])

            new_pos_list.append(new_pos)
            new_quat_list.append(new_quat)
            new_gripper_list.append(float(vla_out[6]))

        # 7. 覆写 raw_actions[3:10]（policy.pt 的 ee_goal 输入）
        self._raw_actions[:, 3:6]  = torch.stack(new_pos_list,  dim=0)
        self._raw_actions[:, 6:10] = torch.stack(new_quat_list, dim=0)

        # 8. 最后写入 gripper
        self._gripper_cmd[:, 0] = torch.tensor(
            [1.0 if g > 0.5 else 0.0 for g in new_gripper_list], device=self.device
        )

# ══════════════════════════════════════════════════════════════════════════
#   Cfg
# ══════════════════════════════════════════════════════════════════════════

@configclass
class VLAPickActionCfg(ActionTermCfg):
    """VLAPickAction 配置。"""

    class_type: type[ActionTerm] = VLAPickAction

    # ── 机器人场景 ───────────────────────────────────────────────────────
    asset_name: str = MISSING
    ee_body_name: str = "arm_link6"         # ← 改成你的实际 EE body 名
    camera_sensor_name: str = "arm_camera"
    ee_command_name: str = "ee_pose"
    debug_vis: bool = True

    # ── 低层配置（与原版相同）────────────────────────────────────────────
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    low_level_leg_actions: ActionTermCfg = MISSING
    low_level_wheel_actions: ActionTermCfg = MISSING
    low_level_ee_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING   # ObservationGroupCfg

    # ── OpenVLA ──────────────────────────────────────────────────────────
    vla_model_path: str = "openvla/openvla-7b"
    vla_device: str = "cuda:0"
    vla_unnorm_key: str = "bridge_orig"
    task_instruction: str = "pick up the object"
    vla_infer_every_n: int = 4              # 推理频率 = sim_hz / 4
    vla_pos_scale: float = 1.0
    vla_rot_scale: float = 1.0

    # ── 动作限幅（与原版相同）────────────────────────────────────────────
    @configclass
    class LowLevelCommandRanges:
        lin_vel_x: tuple[float, float] = (-0.5, 0.5)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        ee_pos_x:  tuple[float, float] = (0.0, 1.0)
        ee_pos_y:  tuple[float, float] = (-0.3, 0.3)
        ee_pos_z:  tuple[float, float] = (0.3, 0.8)

    low_level_command_ranges: LowLevelCommandRanges = LowLevelCommandRanges()