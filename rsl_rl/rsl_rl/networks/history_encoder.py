# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Custom module added on top of rsl_rl_lib==3.1.2 for "Regularized Online Adaptation" (ROA)
# style training: a history encoder (the "adaptation module" / student in RMA terminology)
# regresses the latent produced by a PrivilegedEncoder (the "teacher") purely from a window
# of past proprioceptive observations, so that it can be used at deployment time when the
# privileged information is not available.

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class HistoryEncoder(nn.Module):
    """Encodes a window of past proprioceptive observations into a latent vector.

    The input is expected to be a *flattened* history tensor of shape ``(B, history_length * obs_dim)``,
    which is how IsaacLab's ``ObservationManager`` will hand you a "history" observation term
    (it concatenates ``history_length`` past steps along the last dimension). Internally this
    module reshapes it back to ``(B, obs_dim, history_length)`` and runs a small temporal
    convolutional network (TCN) over it, à la Kumar et al., "RMA: Rapid Motor Adaptation for
    Legged Robots" (2021).

    If you would rather use a recurrent encoder (GRU/LSTM) instead of a TCN, swap the body of
    ``__init__``/``forward`` -- the surrounding ``ActorCriticROA`` module only relies on
    ``output_dim`` and the ``forward(obs) -> latent`` contract, so it doesn't matter how you
    implement the inside.
    """

    def __init__(
        self,
        num_single_step_obs: int,
        history_length: int,
        latent_dim: int = 32,
        hidden_channels: tuple[int, ...] | list[int] = (32, 32, 32),
        kernel_sizes: tuple[int, ...] | list[int] = (4, 3, 2),
        strides: tuple[int, ...] | list[int] = (2, 1, 1),
        activation: str = "elu",
        projection_dim: int = 0,
    ) -> None:
        """Initialize the history encoder.

        Args:
            num_single_step_obs: Dimension of a single time-step's proprioceptive observation
                (i.e. the "policy" obs dim, NOT including the history window).
            history_length: Number of past steps stacked in the history observation term. Must
                match the ``history_length``/``flatten_history_dim`` you configured for the
                corresponding ``ObservationTermCfg`` in IsaacLab.
            latent_dim: Dimension of the output latent (should match ``PrivilegedEncoder.latent_dim``).
            hidden_channels: Number of channels for each Conv1d layer of the TCN.
            kernel_sizes: Kernel size for each Conv1d layer.
            strides: Stride for each Conv1d layer.
            activation: Activation function name (resolved via ``rsl_rl.utils.resolve_nn_activation``).
            projection_dim: 若 > 0，则先对**每个时间步单独**做一次共享的 ``Linear(num_single_step_obs -> projection_dim)``
                投影，再在投影后的特征上做时间卷积。参考实现（vwb ``StateHistoryEncoder``：
                ``nn.Linear(input_size, 3 * channel_size)`` + ``nn.Sequential(Conv1d...)``）就是这个结构；
                ``0``（默认）= 旧行为（直接对原始维度做卷积），保证已有 run 完全不受影响。
        """
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes) == len(strides), (
            "hidden_channels, kernel_sizes and strides must have the same length"
        )

        self.num_single_step_obs = num_single_step_obs
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.projection_dim = int(projection_dim or 0)

        activation_mod = resolve_nn_activation(activation)

        # 逐时间步的共享投影（参考实现的做法）：把异构量纲的 70/83 维先线性映射到统一宽度，
        # 之后的卷积只在这个"每步特征"上做时间聚合。
        if self.projection_dim > 0:
            self.proj: nn.Module = nn.Sequential(nn.Linear(num_single_step_obs, self.projection_dim), activation_mod)
            conv_in_channels = self.projection_dim
        else:
            self.proj = nn.Identity()
            conv_in_channels = num_single_step_obs

        # Temporal convolutional stack. We treat the per-step observation dimension as the
        # "channel" axis and the history length as the temporal/"length" axis -- this is the
        # standard layout used for proprioceptive-history encoders in RMA-style legged-robot RL.
        conv_layers: list[nn.Module] = []
        in_channels = conv_in_channels
        for out_channels, kernel_size, stride in zip(hidden_channels, kernel_sizes, strides):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            conv_layers.append(activation_mod)
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)

        # Figure out the flattened size after the conv stack with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, conv_in_channels, history_length)
            conv_out_dim = self.conv(dummy).flatten(1).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, 2 * latent_dim),
            activation_mod,
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        """Compute the latent from a flattened history observation.

        Args:
            history_obs: Tensor of shape ``(B, history_length * num_single_step_obs)``.

        Returns:
            Latent tensor of shape ``(B, latent_dim)``.
        """
        batch_size = history_obs.shape[0]
        # (B, T * D) -> (B, T, D) -> （可选）逐帧共享投影 -> (B, C, T) for Conv1d (channels-first).
        x = history_obs.view(batch_size, self.history_length, self.num_single_step_obs)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)
