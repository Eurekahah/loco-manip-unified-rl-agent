# encoder.py
import torch
import torch.nn as nn
from typing import Optional

class VisionEncoderRegistry:
    """全局单例，管理视觉编码器生命周期"""
    _encoders: dict = {}

    @classmethod
    def get_encoder(cls, name: str, device: str = "cuda") -> nn.Module:
        if name not in cls._encoders:
            cls._encoders[name] = cls._build_encoder(name, device)
        return cls._encoders[name]

    @classmethod
    def _build_encoder(cls, name: str, device: str) -> nn.Module:
        if name == "dinov2_small":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        elif name == "dinov2_base":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif name == "clip_vit":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            model = model.visual  # 只取视觉部分
        elif name == "my_custom_encoder":
            from my_project.models import MyResNetEncoder
            model = MyResNetEncoder()
            # 加载 checkpoint
            ckpt = torch.load("/path/to/checkpoint.pth", map_location=device)
            model.load_state_dict(ckpt["encoder"])
        else:
            raise ValueError(f"Unknown encoder: {name}")

        model = model.to(device).eval()
        # 冻结参数，不参与 RL 梯度
        for p in model.parameters():
            p.requires_grad_(False)
        return model

class _MiniPointNet(nn.Module):
    """轻量 PointNet，适合实时 RL 推理。输出固定维 embedding。"""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (N, num_points, 3)
        feat = self.mlp(xyz)          # (N, num_points, out_dim)
        return feat.max(dim=1).values  # 全局最大池化 → (N, out_dim)


class LidarEncoderRegistry:
    """全局单例，管理点云编码器生命周期。"""
    _encoders: dict = {}

    @classmethod
    def get_encoder(cls, name: str, device: str = "cuda") -> nn.Module:
        if name not in cls._encoders:
            cls._encoders[name] = cls._build(name, device)
        return cls._encoders[name]

    @classmethod
    def _build(cls, name: str, device: str) -> nn.Module:
        if name == "mini_pointnet":
            model = _MiniPointNet(out_dim=256)

        elif name == "pointnet_plus":
            # 需要 pip install torch-points3d 或 pointnet2_ops
            from pointnet2_ops.pointnet2_modules import PointnetSAModule
            # 这里可替换为你自己的 PointNet++ 结构
            raise NotImplementedError("请替换为你的 PointNet++ 实现")

        elif name == "my_custom_lidar_encoder":
            from my_project.models import MyLidarEncoder
            model = MyLidarEncoder()
            ckpt = torch.load("/path/to/lidar_encoder.pth", map_location=device)
            model.load_state_dict(ckpt["encoder"])

        else:
            raise ValueError(f"Unknown lidar encoder: {name}")

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model