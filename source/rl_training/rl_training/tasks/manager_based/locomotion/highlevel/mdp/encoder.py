# encoder.py

import torch
import torch.nn as nn
from typing import Optional


class VisionEncoderRegistry:
    """
    全局单例，管理视觉编码器生命周期。

    frozen  编码器：缓存复用，梯度关闭（DINOv2 / CLIP 等预训练模型）
    trainable 编码器：不缓存，由调用方持有并注册，梯度开启
    """

    _frozen_encoders: dict = {}
    _trainable_encoders: dict = {}   # key → nn.Module，由外部注册

    # ------------------------------------------------------------------ #
    #  对外接口                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_encoder(cls, name: str, device: str = "cuda") -> nn.Module:
        """获取编码器（frozen 自动构建并缓存；trainable 必须先 register）"""
        if name in cls._trainable_encoders:
            return cls._trainable_encoders[name]
        # frozen 路径
        if name not in cls._frozen_encoders:
            cls._frozen_encoders[name] = cls._build_frozen_encoder(name, device)
        return cls._frozen_encoders[name]

    @classmethod
    def register_trainable(cls, name: str, model: nn.Module):
        """
        将可训练编码器注册到 registry。
        应在 env / policy 初始化时调用一次，之后 mdp 函数可通过 name 取到同一实例。

        Example::
            encoder = TrainableCNNEncoder(embed_dim=256).to(device)
            VisionEncoderRegistry.register_trainable("trainable_cnn", encoder)
        """
        if name in cls._frozen_encoders:
            raise ValueError(f"'{name}' 已作为 frozen encoder 存在，请换一个名字。")
        cls._trainable_encoders[name] = model

    @classmethod
    def is_trainable(cls, name: str) -> bool:
        return name in cls._trainable_encoders

    # ------------------------------------------------------------------ #
    #  内部构建（frozen 专用）                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_frozen_encoder(cls, name: str, device: str) -> nn.Module:
        if name == "dinov2_small":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        elif name == "dinov2_base":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif name == "clip_vit":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model = model.visual
        elif name == "cnn":
            # 这里保留你原来的静态 CNN checkpoint 加载逻辑
            from my_project.models import MyResNetEncoder
            model = MyResNetEncoder()
            ckpt = torch.load("/path/to/checkpoint.pth", map_location=device)
            model.load_state_dict(ckpt["encoder"])
        else:
            raise ValueError(
                f"Unknown encoder: '{name}'。"
                f"如需可训练编码器，请先调用 VisionEncoderRegistry.register_trainable()。"
            )

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

# your_env/vision_encoders.py
"""
可训练的视觉编码器定义。
用 model_zoo_cfg 注入到官方 image_features，不需要改 IsaacLab 源码。
"""



# ---------------------------------------------------------------------------
# 1. 定义编码器网络
#    (a) 轻量 CNN —— 端到端可训练，适合小分辨率
#    (b) 解冻的 ResNet —— 在官方 resnet18 基础上开放梯度
# ---------------------------------------------------------------------------

class LightweightCNN(nn.Module):
    """三层卷积 + 全连接，输出 256 维 embedding，端到端可训练。"""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C) uint8 from TiledCamera
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)          # NHWC -> NCHW
        return self.net(x)


class UnfrozenResNet18(nn.Module):
    """解冻梯度的 ResNet18，输出 512 维，可随策略一起训练。"""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 去掉最后的分类头，保留到 avgpool
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (N,512,1,1)
        self.flatten = nn.Flatten()
        # 注意：不调用 .eval() / .requires_grad_(False)，保持可训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, H, W, C) uint8
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.flatten(self.encoder(x))


# ---------------------------------------------------------------------------
# 2. 按照官方 model_zoo_cfg 格式打包
#
#    model_zoo_cfg 的结构由 image_features.__init__ 内部解析，
#    官方源码中每个 entry 包含：
#      - "model":      nn.Module 实例（已 .to(device) 之前）
#      - "reset_fn":   可选，用于 episode reset 时的回调
#
#    实际上官方内部只要求传入一个可调用的工厂函数或已实例化的模块，
#    最稳妥的方式是传入工厂函数（lambda），让 image_features 在初始化时
#    调用它并 .to(device)。
# ---------------------------------------------------------------------------

def make_cnn_model_zoo_cfg() -> dict:
    """
    返回符合 image_features 期望的 model_zoo_cfg 字典。
    key   = model_name 字符串
    value = 无参工厂函数，返回 nn.Module
    """
    return {
        "nav_cnn": lambda: LightweightCNN(in_channels=3, feature_dim=256),
        "nav_resnet18_unfrozen": lambda: UnfrozenResNet18(),
    }