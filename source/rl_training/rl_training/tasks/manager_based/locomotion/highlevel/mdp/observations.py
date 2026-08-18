# mdp/observations.py
from __future__ import annotations
import torch
import torchvision.transforms.functional as TF



import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply_inverse,
    yaw_quat,
    wrap_to_pi,
    quat_inv,
    quat_mul,
    quat_apply,
)

from .utils import robot_root_pos_w, robot_root_quat_w, object_root_pos_w, object_root_quat_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# def camera_feature_embedding(
#     env: ManagerBasedEnv,
#     sensor_name: str = "front_camera",
#     data_type: str = "rgb",
#     encoder_name: str = "dinov2_small",
#     image_size: tuple = (224, 224),
# ) -> torch.Tensor:
#     """
#     从相机获取图像，经过视觉编码器后返回 embedding。

#     - frozen 编码器：自动 torch.no_grad()
#     - trainable 编码器：保留计算图，供 RL 梯度回传

#     Returns:
#         shape: (num_envs, embed_dim)
#     """
#     # ── 1. 取原始图像 ──────────────────────────────────────────────────
#     sensor = env.scene.sensors[sensor_name]
#     raw = sensor.data.output[data_type]

#     # ── 2. 预处理 ──────────────────────────────────────────────────────
#     if data_type == "rgb":
#         imgs = raw[..., :3].float() / 255.0      # (N, H, W, 3)
#         imgs = imgs.permute(0, 3, 1, 2)           # (N, 3, H, W)
#     elif data_type == "depth":
#         imgs = raw.float()
#         imgs = imgs.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
#     else:
#         raise ValueError(f"Unsupported data_type: {data_type}")

#     imgs = TF.resize(imgs, list(image_size))

#     mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
#     imgs = (imgs - mean) / std

#     # ── 3. 编码 ────────────────────────────────────────────────────────
#     encoder = VisionEncoderRegistry.get_encoder(encoder_name, device=str(imgs.device))

#     if VisionEncoderRegistry.is_trainable(encoder_name):
#         # 可训练：保留梯度，BatchNorm 用 train() 模式
#         encoder.train()
#         embedding = encoder(imgs)
#     else:
#         # 冻结：关闭梯度，节省显存
#         encoder.eval()
#         with torch.no_grad():
#             embedding = encoder(imgs)

#     return embedding   # (N, embed_dim)

# =============================================================================
# Observations
# =============================================================================

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    use_heading_frame: bool = True,
) -> torch.Tensor:
    """
    Relative position of an object (table / target) expressed in the robot's
    heading frame (yaw-only rotation removed, roll/pitch kept at zero).

    Returns shape (N, 3): [x_rel, y_rel, z_rel] in robot heading frame.

    Args:
        robot_cfg:        SceneEntityCfg for the robot articulation.
        object_cfg:       SceneEntityCfg for the target rigid object.
        use_heading_frame: If True, express in yaw-only frame (recommended for
                           navigation — invariant to robot tilt).
                           If False, express in full body frame.
    """
    robot_pos_w   = robot_root_pos_w(env, robot_cfg)    # (N, 3)
    robot_quat_w  = robot_root_quat_w(env, robot_cfg)   # (N, 4)
    object_pos_w  = object_root_pos_w(env, object_cfg)  # (N, 3)

    # Vector from robot to object in world frame
    rel_pos_w = object_pos_w - robot_pos_w  # (N, 3)

    if use_heading_frame:
        # Strip roll/pitch — keep only yaw rotation
        heading_quat = yaw_quat(robot_quat_w)  # (N, 4)
        rel_pos_b = quat_apply_inverse(heading_quat, rel_pos_w)
    else:
        rel_pos_b = quat_apply_inverse(robot_quat_w, rel_pos_w)

    # print(f"Relative position(root): {rel_pos_b}")

    return rel_pos_b  # (N, 3)

def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    use_heading_frame: bool = True,
) -> torch.Tensor:
    """
    Relative pose of an object expressed in the robot's heading frame.
    Returns shape (N, 7): [x, y, z, qw, qx, qy, qz] in robot heading frame.

    Args:
        robot_cfg:         SceneEntityCfg for the robot articulation.
        object_cfg:        SceneEntityCfg for the target rigid object.
        use_heading_frame: If True, express in yaw-only frame (recommended for
                           navigation). If False, express in full body frame.
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)   # (N, 3)
    robot_quat_w = robot_root_quat_w(env, robot_cfg)  # (N, 4) wxyz
    object_pos_w = object_root_pos_w(env, object_cfg) # (N, 3)
    object_quat_w = object_root_quat_w(env, object_cfg) # (N, 4) wxyz

    # --- Relative position ---
    rel_pos_w = object_pos_w - robot_pos_w  # (N, 3)

    if use_heading_frame:
        ref_quat = yaw_quat(robot_quat_w)  # (N, 4) 只保留yaw
    else:
        ref_quat = robot_quat_w            # (N, 4) 完整姿态

    # 相对位置：旋转到参考帧下
    rel_pos_b = quat_apply_inverse(ref_quat, rel_pos_w)  # (N, 3)

    # --- Relative orientation ---
    # q_rel = q_ref^{-1} * q_object
    # isaaclab中 quat_mul 约定: wxyz
    ref_quat_inv = quat_inv(ref_quat)                          # (N, 4)
    rel_quat_b = quat_mul(ref_quat_inv, object_quat_w)        # (N, 4)

    # --- Concatenate (N, 3+4=7) ---
    return torch.cat([rel_pos_b, rel_quat_b], dim=-1)  # (N, 7)

def object_pose_in_ee_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    ee_link_name: str = "gripper_base",
    ee_offset_z: float = 0.135,
) -> torch.Tensor:
    """
    物体位姿相对于 EE frame 的表达。
    EE frame = gripper_base 沿其自身 z 轴方向偏移 ee_offset_z 处的虚拟帧。

    Returns shape (N, 7): [x, y, z, qw, qx, qy, qz] in EE frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # ── 1. 获取 gripper_base 在世界系下的位姿 ──────────────────────────────
    link_idx = robot.find_bodies(ee_link_name)[0][0]          # int 或 List[int]
    # body_pos_w / body_quat_w shape: (N, num_bodies, 3/4)
    link_pos_w  = robot.data.body_pos_w[:, link_idx, :]   # (N, 3)
    link_quat_w = robot.data.body_quat_w[:, link_idx, :]  # (N, 4) wxyz

    # ── 2. 计算虚拟 EE 位置：沿 link6 局部 z 轴偏移 0.135 m ────────────
    # 局部偏移向量 [0, 0, offset_z]，广播到 (N, 3)
    offset_local = link_pos_w.new_zeros(link_pos_w.shape)
    offset_local[:, 2] = ee_offset_z                      # z 分量

    # 将局部偏移旋转到世界系
    offset_world = quat_apply(link_quat_w, offset_local)  # (N, 3)
    ee_pos_w = link_pos_w + offset_world                  # (N, 3)

    # EE frame 姿态与 link6 相同（只做平移 offset，不额外旋转）
    ee_quat_w = link_quat_w                               # (N, 4)

    # ── 3. 获取物体在世界系下的位姿 ────────────────────────────────────
    obj_pos_w  = object_root_pos_w(env, object_cfg)       # (N, 3)
    obj_quat_w = object_root_quat_w(env, object_cfg)      # (N, 4)

    # ── 4. 计算物体相对于 EE frame 的位姿 ───────────────────────────────
    # 相对位置
    rel_pos_w = obj_pos_w - ee_pos_w                      # (N, 3)
    rel_pos_ee = quat_apply_inverse(ee_quat_w, rel_pos_w) # (N, 3)

    # 相对姿态: q_rel = q_ee^{-1} * q_obj
    ee_quat_inv = quat_inv(ee_quat_w)                     # (N, 4)
    rel_quat_ee = quat_mul(ee_quat_inv, obj_quat_w)       # (N, 4)

    return torch.cat([rel_pos_ee, rel_quat_ee], dim=-1)   # (N, 7)


def object_heading_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Yaw angle from the robot to the target object, expressed as
    [sin(angle), cos(angle)] to avoid angle-wrap discontinuity.

    Returns shape (N, 2).

    This is useful as a compact directional signal without distance magnitude.
    """
    robot_pos_w  = robot_root_pos_w(env, robot_cfg)   # (N, 3)
    robot_quat_w = robot_root_quat_w(env, robot_cfg)  # (N, 4)
    object_pos_w = object_root_pos_w(env, object_cfg) # (N, 3)

    rel_pos_w = object_pos_w - robot_pos_w  # (N, 3)

    # Yaw angle of relative vector in world frame
    target_angle_w = torch.atan2(rel_pos_w[:, 1], rel_pos_w[:, 0])  # (N,)

    # Robot yaw in world frame (from yaw_quat → extract z-rotation)
    heading_quat   = yaw_quat(robot_quat_w)  # (N, 4)
    # yaw_quat gives quat = [cos(y/2), 0, 0, sin(y/2)] (w,x,y,z)
    robot_yaw      = 2.0 * torch.atan2(heading_quat[:, 3], heading_quat[:, 0])  # (N,)

    # Relative yaw angle (wrapped)
    rel_yaw = wrap_to_pi(target_angle_w - robot_yaw)  # (N,)

    return torch.stack([torch.sin(rel_yaw), torch.cos(rel_yaw)], dim=-1)  # (N, 2)

