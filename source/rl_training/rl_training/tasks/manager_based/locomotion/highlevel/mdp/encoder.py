# encoder.py

import os

import torch
import torch.nn as nn
from typing import Optional

# --------------------------------------------------------------------------- #
# 本地权重优先 / 默认**不联网**（2026-09-30，见 docs/review/DEFECT_LOG_zh.md DEF-034）
#
# 背景：这几个 frozen encoder 以前一律走“联网下载”（`torch.hub.load(...)` /
# `open_clip(pretrained="openai")` / `torchvision(weights=DEFAULT)`），在离线机器
# （比如实验室集群、MuJoCo 部署机）上会直接失败或者卡住几十秒。
#
# 现在的策略：
#   1. 先在 ``RL_TRAINING_ENCODER_DIR``（默认 ``~/.cache/rl_training/encoders``）里找
#      ``<name>.pth``；找到就**只用本地权重**；
#   2. 找不到就**报错**，并在错误信息里写清楚该把权重放到哪、以及怎么显式允许下载；
#   3. 只有显式设 ``RL_TRAINING_ALLOW_ENCODER_DOWNLOAD=1`` 才会回退到联网下载。
# --------------------------------------------------------------------------- #
ENCODER_DIR = os.environ.get(
    "RL_TRAINING_ENCODER_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "rl_training", "encoders"),
)


def allow_encoder_download() -> bool:
    """是否允许联网下载预训练权重（默认否）。"""
    return os.environ.get("RL_TRAINING_ALLOW_ENCODER_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}


def local_encoder_weights(name: str) -> Optional[str]:
    """返回 ``<ENCODER_DIR>/<name>.pth``（存在时），否则 ``None``。"""
    path = os.path.join(ENCODER_DIR, f"{name}.pth")
    return path if os.path.isfile(path) else None


def _weights_missing(name: str, how_to: str) -> RuntimeError:
    return RuntimeError(
        f"[VisionEncoderRegistry] 找不到 '{name}' 的本地权重（默认不联网）。\n"
        f"  · 请把权重放到：{os.path.join(ENCODER_DIR, name + '.pth')}"
        f"（或设环境变量 RL_TRAINING_ENCODER_DIR 指向别的目录）\n"
        f"  · {how_to}\n"
        f"  · 如果确实想让它自己下载：设 RL_TRAINING_ALLOW_ENCODER_DOWNLOAD=1"
    )


class VisionEncoderRegistry:
    """
    全局单例，管理视觉编码器生命周期。

    frozen  编码器：缓存复用，梯度关闭（DINOv2 / CLIP 等预训练模型）
    trainable 编码器：不缓存，由调用方持有并注册，梯度开启
    """

    # key = (name, device)：以前只用 name 做 key，同一进程中先在某块卡上建过一次，
    # 之后即使换 device 也会拿到**建在旧设备上的模型**（训练/评估跨设备时静默出错）。
    # known_issues 工程债，2026-09-30 修（docs/review/DEFECT_LOG_zh.md DEF-034）。
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
        # frozen 路径：按 (name, device) 缓存（见 _frozen_encoders 的注释）
        key = (name, str(device))
        if key not in cls._frozen_encoders:
            cls._frozen_encoders[key] = cls._build_frozen_encoder(name, device)
        return cls._frozen_encoders[key]

    @classmethod
    def _frozen_names(cls) -> set:
        """已缓存的 frozen encoder 名字（忽略 device）。"""
        return {k[0] if isinstance(k, tuple) else k for k in cls._frozen_encoders}

    @classmethod
    def register_trainable(cls, name: str, model: nn.Module):
        """
        将可训练编码器注册到 registry。
        应在 env / policy 初始化时调用一次，之后 mdp 函数可通过 name 取到同一实例。

        Example::
            encoder = TrainableCNNEncoder(embed_dim=256).to(device)
            VisionEncoderRegistry.register_trainable("trainable_cnn", encoder)
        """
        if name in cls._frozen_names():
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
        """构建 frozen encoder：**本地权重优先，默认不联网**（见文件头注释）。"""
        ckpt = local_encoder_weights(name)
        if name == "dinov2_small":
            model = cls._build_dinov2("dinov2_vits14", ckpt, name, device)
        elif name == "dinov2_base":
            model = cls._build_dinov2("dinov2_vitb14", ckpt, name, device)
        elif name == "clip_vit":
            import open_clip
            if ckpt is not None:
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=ckpt)
            elif allow_encoder_download():
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            else:
                raise _weights_missing(
                    name,
                    "open_clip 的 ViT-B-32(openai) 需自备："
                    "`open_clip.create_model_and_transforms('ViT-B-32')` + "
                    "`torch.save(model.visual.state_dict(), '<ENCODER_DIR>/clip_vit.pth')`。",
                )
            model = model.visual
        elif name == "cnn":
            # 用本仓库自带的 LightweightCNN（原来这里 import 一个不存在的 `my_project.models`
            # ⇒ 死代码；现在改成"本地 checkpoint + 自带网络结构"）。
            if ckpt is None:
                raise _weights_missing(name, "先用 LightweightCNN 训练/或直接拷一个 checkpoint 过来。")
            model = LightweightCNN(in_channels=3, feature_dim=256)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state.get("encoder", state))
        else:
            raise ValueError(
                f"Unknown encoder: '{name}'。"
                f"如需可训练编码器，请先调用 VisionEncoderRegistry.register_trainable()。"
            )

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @classmethod
    def _build_dinov2(cls, arch: str, ckpt: Optional[str], name: str, device: str) -> nn.Module:
        """DINOv2：本地 checkpoint 优先；没有就报错（除非显式允许下载）。"""
        repo = "facebookresearch/dinov2"
        hub_cached = os.path.isdir(os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main"))
        if ckpt is not None:
            if not hub_cached and not allow_encoder_download():
                raise RuntimeError(
                    f"[VisionEncoderRegistry] 有本地权重 {ckpt}，但 torch.hub 里没有 DINOv2 的"
                    f"**网络结构代码**（{os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')}）。\n"
                    "  · 离线做法：把 dinov2 仓库也拷到 hub 目录（`git clone` 后改名），"
                    "或者用 `RL_TRAINING_ALLOW_ENCODER_DOWNLOAD=1` 先跑一次把代码缓存下来。"
                )
            model = torch.hub.load(repo, arch, pretrained=False, source="local")
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state.get("model", state))
            return model
        if allow_encoder_download():
            return torch.hub.load(repo, arch)
        raise _weights_missing(
            name,
            f"DINOv2 权重自备方式：`torch.hub.load('{repo}', '{arch}')` 后 "
            f"`torch.save(model.state_dict(), '<ENCODER_DIR>/{name}.pth')`（在有网的机器上做一次即可）。",
        )

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
        # 默认**不联网**：只建结构（weights=None）。想用 ImageNet 预训练权重的话：
        #   ① （推荐）把 resnet18 权重放到本地：<ENCODER_DIR>/resnet18.pth，然后走下面分支；
        #   ② 或者显式设 RL_TRAINING_ALLOW_ENCODER_DOWNLOAD=1 允许联网下载。
        ckpt = local_encoder_weights("resnet18")
        if ckpt is not None:
            backbone = models.resnet18(weights=None)
            state = torch.load(ckpt, map_location="cpu")
            backbone.load_state_dict(state)
        elif allow_encoder_download():
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)
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
