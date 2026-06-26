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