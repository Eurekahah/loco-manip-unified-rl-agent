# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Custom module added on top of rsl_rl_lib==3.1.2 for "Regularized Online Adaptation" (ROA)
# style training: the privileged encoder (the "teacher") is only ever available in simulation
# (it consumes ground-truth environment parameters such as friction, payload mass, terrain
# parameters, contact forces, etc. that a real robot cannot directly observe). Its latent output
# is used purely as a regression *target* for the HistoryEncoder -- gradients from this module
# should not depend on the HistoryEncoder, and vice versa.

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP


class PrivilegedEncoder(nn.Module):
    """Encodes privileged (simulation-only) environment information into a latent vector.

    This is the "teacher" / extrinsics encoder from RMA-style training. It is a simple MLP;
    nothing fancy is needed since the input is already a clean, low-dimensional vector of
    privileged scalars (as opposed to the noisy time-series the ``HistoryEncoder`` has to deal
    with).
    """

    def __init__(
        self,
        num_privileged_obs: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
    ) -> None:
        """Initialize the privileged encoder.

        Args:
            num_privileged_obs: Dimension of the privileged observation vector (sum of all obs
                terms assigned to the "privileged" obs_group in your IsaacLab env config).
            latent_dim: Dimension of the output latent (should match ``HistoryEncoder.latent_dim``).
            hidden_dims: Hidden layer sizes of the encoder MLP.
            activation: Activation function name.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(num_privileged_obs, latent_dim, list(hidden_dims), activation)

    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        """Compute the latent from the privileged observation.

        Args:
            privileged_obs: Tensor of shape ``(B, num_privileged_obs)``.

        Returns:
            Latent tensor of shape ``(B, latent_dim)``.
        """
        return self.encoder(privileged_obs)


class PrivilegedDecoder(nn.Module):
    """Teacher 的**重建头**（2026-10-16 新增，用于给 z_priv 加"锚"）。

    为什么需要它
    ------------
    RMA/ROA 式单阶段训练里，教师（PrivilegedEncoder）唯一的"强"梯度来自
    ``λ‖z_priv − sg(z_hist)‖``，即**被拉向学生**；而学生又被拉向教师。两个自由编码器互相吸引、
    没有锚点时，**"两者都输出同一个常数"是全局最优且稳定的解** —— 这正是我们实测到的塌陷
    （旧版教师输入敏感度掉到初值的 3%，见 docs/history_encoder_loss_cliff_zh.md）。

    这里加一条**不可能被常数满足**的监督：``decoder(z_priv) ≈ x_priv``（x_priv = 归一化后的特权
    向量）。若 z_priv 塌成常数，decoder 只能输出常数，而 x_priv 跨环境是变化的 ⇒ 重建误差不可能
    降到 0。于是教师被迫保留信息，塌陷不再是优解。这条损失只依赖 sim-only 的特权向量，**部署时
    完全不需要**（与 privileged encoder 一样只在训练期存在）。

    注意：参考实现（MarkFzp/DWL、Ericonaldo/vwb）**没有**这一条 —— 这是我们在复现塌陷后，
    超出参考代码的补强（见 docs/review/TODO_zh.md P1-6）。
    """

    def __init__(
        self,
        latent_dim: int,
        num_privileged_obs: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.decoder = MLP(latent_dim, num_privileged_obs, list(hidden_dims), activation)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
