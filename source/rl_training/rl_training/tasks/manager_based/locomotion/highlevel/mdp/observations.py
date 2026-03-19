# mdp/observations.py

import torch
import torchvision.transforms.functional as TF


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