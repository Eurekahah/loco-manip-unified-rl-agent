# mdp/observations.py (在你已有的文件中添加)

import torch
import torchvision.transforms.functional as TF
from isaaclab.envs import ManagerBasedEnv
from .encoder import VisionEncoderRegistry, LidarEncoderRegistry


def camera_feature_embedding(
    env: ManagerBasedEnv,
    sensor_name: str = "front_camera",
    data_type: str = "rgb",
    encoder_name: str = "dinov3_small",
    image_size: tuple = (224, 224),
) -> torch.Tensor:
    """
    从相机获取图像，经过视觉编码器后返回 embedding。
    
    Returns:
        shape: (num_envs, embed_dim)
    """
    # 1. 取原始图像数据
    sensor = env.scene.sensors[sensor_name]
    # rgb: (num_envs, H, W, 4) RGBA uint8  /  depth: (num_envs, H, W, 1) float
    raw = sensor.data.output[data_type]

    # 2. 预处理
    if data_type == "rgb":
        # 去掉 alpha 通道，转为 float，归一化到 [0,1]
        imgs = raw[..., :3].float() / 255.0          # (N, H, W, 3)
        imgs = imgs.permute(0, 3, 1, 2)              # (N, 3, H, W)
    elif data_type == "depth":
        imgs = raw.float()                            # (N, H, W, 1)
        imgs = imgs.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)  # 深度复制为3通道

    # resize
    imgs = TF.resize(imgs, list(image_size))

    # ImageNet 归一化（DINOv2/CLIP 都需要）
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    # 3. 编码
    encoder = VisionEncoderRegistry.get_encoder(encoder_name, device=str(imgs.device))
    with torch.no_grad():
        embedding = encoder(imgs)   # (N, embed_dim)

    return embedding


def lidar_feature_embedding(
    env: ManagerBasedEnv,
    sensor_name: str = "lidar",
    encoder_name: str = "mini_pointnet",
    max_points: int = 1024,
    max_range: float = 20.0,
    min_range: float = 0.1,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]

    # 1. 取击中点（世界坐标）
    hits_w = sensor.data.ray_hits_w          # (N, num_rays, 3)
    num_envs, num_rays, _ = hits_w.shape

    # 2. 转换到局部坐标（减去传感器原点）
    sensor_pos = sensor.data.pos_w.unsqueeze(1)   # (N, 1, 3)
    xyz_local = hits_w - sensor_pos               # (N, num_rays, 3)

    # ✅ 距离由 xyz_local 计算，而不是从 sensor.data.distances 读取
    distances = torch.norm(xyz_local, dim=-1)     # (N, num_rays)

    # 3. 过滤无效点（超出量程 / 未击中）
    # RayCaster 未击中时 ray_hits_w 会填充极大值（如 1e6），用距离过滤即可
    valid_mask = (distances >= min_range) & (distances <= max_range)  # (N, num_rays)

    # 4. 采样 / padding 至固定点数 max_points
    xyz_fixed = _sample_fixed_points(
        xyz_local, valid_mask, max_points, hits_w.device
    )   # (N, max_points, 3)

    # 5. 归一化：按 max_range 缩放到 [-1, 1]
    xyz_fixed = xyz_fixed / max_range

    # 6. 编码
    encoder = LidarEncoderRegistry.get_encoder(
        encoder_name, device=str(xyz_fixed.device)
    )
    with torch.no_grad():
        embedding = encoder(xyz_fixed)  # (N, embed_dim)

    return embedding


# ── 工具函数 ──────────────────────────────────────────────────────

def _sample_fixed_points(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    max_points: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将每个环境的有效点数统一采样/padding 至 max_points。
    - 有效点 > max_points：随机下采样
    - 有效点 < max_points：用零点 padding
    """
    num_envs = xyz.shape[0]
    out = torch.zeros(num_envs, max_points, 3, device=device)

    for i in range(num_envs):
        pts = xyz[i][valid_mask[i]]      # (valid_n, 3)
        n   = pts.shape[0]
        if n == 0:
            continue
        elif n >= max_points:
            idx = torch.randperm(n, device=device)[:max_points]
            out[i] = pts[idx]
        else:
            out[i, :n] = pts
            # padding：用最近点重复填充（比零点更自然）
            repeat_idx = torch.randint(0, n, (max_points - n,), device=device)
            out[i, n:] = pts[repeat_idx]

    return out