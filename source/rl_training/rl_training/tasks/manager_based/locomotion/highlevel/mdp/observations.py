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
)

from .utils import robot_root_pos_w, robot_root_quat_w, object_root_pos_w

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

    return rel_pos_b  # (N, 3)


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

