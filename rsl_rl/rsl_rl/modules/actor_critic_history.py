# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Custom policy module added on top of rsl_rl_lib==3.1.2.
#
# This implements a "Regularized Online Adaptation" (ROA) style actor-critic:
#   - A `PrivilegedEncoder` ("teacher") compresses ground-truth, simulation-only environment
#     parameters into a latent z_priv. This is ONLY available in simulation.
#   - A `HistoryEncoder` ("adaptation module" / student) compresses a window of past
#     proprioceptive observations into a latent z_hist. This IS available on the real robot.
#   - The actor ALWAYS consumes z_hist (never z_priv) so that the network used at train time is
#     identical to the network used at deployment time -- no policy distillation/fine-tuning
#     step is required after RL training.
#   - z_priv is only used as a regression TARGET: during the PPO update, an auxiliary loss
#     pulls z_hist towards z_priv.detach() (see `rsl_rl/algorithms/ppo_history.py`). This is what
#     gives the adaptation module a clean supervised signal on top of the (slow, high-variance)
#     RL gradient that also flows into it through the actor.
#   - The critic is an asymmetric critic: it can see the privileged observations directly
#     (put your privileged obs terms into the "critic" obs_group too, in addition to
#     "privileged") since the critic is never deployed on the real robot.
#
# Required env obs_groups (configure these in your IsaacLab agent cfg's `obs_groups`):
#   "policy"      -> proprioception + commands etc. (single time-step, what the deployed
#                    network sees besides the latent)
#   "critic"      -> whatever you want the value function to see (can freely include
#                    privileged terms, since the critic is sim-only)
#   "history"     -> proprioceptive history, flattened to (history_length * single_step_dim,)
#   "privileged"  -> ground-truth simulation-only parameters (friction, mass, terrain, ...)

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.networks.history_encoder import HistoryEncoder
from rsl_rl.networks.privileged_encoder import PrivilegedEncoder


class ActorCriticHistory(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Latent / encoder configuration
        latent_dim: int = 32,
        history_length: int = 10,
        history_encoder_hidden_channels: tuple[int, ...] | list[int] = (32, 32, 32),
        history_encoder_kernel_sizes: tuple[int, ...] | list[int] = (4, 3, 2),
        history_encoder_strides: tuple[int, ...] | list[int] = (2, 1, 1),
        privileged_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        # Actor / critic configuration (same semantics as rsl_rl.modules.ActorCritic)
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        history_obs_normalization: bool = False,
        privileged_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticHistory.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        for required_set in ("history", "privileged"):
            if required_set not in obs_groups:
                raise ValueError(
                    f"ActorCriticHistory requires an '{required_set}' entry in `obs_groups` (got: "
                    f"{list(obs_groups.keys())}). Add it to your agent cfg, e.g. "
                    f'obs_groups["{required_set}"] = ["{required_set}"].'
                )

        # ---- Figure out observation dimensions from the example `obs` TensorDict ----
        def _group_dim(set_name: str) -> int:
            total = 0
            for obs_group in obs_groups[set_name]:
                assert len(obs[obs_group].shape) == 2, "ActorCriticHistory only supports 1D observations."
                total += obs[obs_group].shape[-1]
            return total

        num_actor_obs = _group_dim("policy")
        num_critic_obs = _group_dim("critic")
        num_history_obs = _group_dim("history")
        num_privileged_obs = _group_dim("privileged")

        if num_history_obs % history_length != 0:
            raise ValueError(
                f"The flattened 'history' obs dim ({num_history_obs}) is not divisible by "
                f"history_length ({history_length}). Make sure the 'history' obs_group's total "
                "dimension equals history_length * single_step_obs_dim."
            )
        num_single_step_obs = num_history_obs // history_length

        self.latent_dim = latent_dim

        # ---- Encoders ----
        self.history_encoder = HistoryEncoder(
            num_single_step_obs=num_single_step_obs,
            history_length=history_length,
            latent_dim=latent_dim,
            hidden_channels=history_encoder_hidden_channels,
            kernel_sizes=history_encoder_kernel_sizes,
            strides=history_encoder_strides,
            activation=activation,
        )
        self.privileged_encoder = PrivilegedEncoder(
            num_privileged_obs=num_privileged_obs,
            latent_dim=latent_dim,
            hidden_dims=privileged_encoder_hidden_dims,
            activation=activation,
        )

        # ---- Actor (always conditioned on the HISTORY latent -> train/deploy consistent) ----
        self.actor = MLP(num_actor_obs + latent_dim, num_actions, list(actor_hidden_dims), activation)
        print(f"Actor MLP: {self.actor}")
        print(f"History encoder: {self.history_encoder}")
        print(f"Privileged encoder (teacher, sim-only): {self.privileged_encoder}")

        # ---- Critic (asymmetric: sees whatever raw obs you put in the 'critic' obs_group) ----
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        print(f"Critic MLP: {self.critic}")

        # ---- Observation normalization ----
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
        self.history_obs_normalization = history_obs_normalization
        self.history_obs_normalizer = (
            EmpiricalNormalization(num_history_obs) if history_obs_normalization else nn.Identity()
        )
        self.privileged_obs_normalization = privileged_obs_normalization
        self.privileged_obs_normalizer = (
            EmpiricalNormalization(num_privileged_obs) if privileged_obs_normalization else nn.Identity()
        )

        # ---- Action noise (identical mechanism to rsl_rl.modules.ActorCritic) ----
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        # Cached latents from the most recent forward pass, exposed so that PPORoA can compute
        # the encoder-regularization loss without recomputing the encoders from scratch.
        self._last_history_latent: torch.Tensor | None = None
        self._last_privileged_latent: torch.Tensor | None = None

    # ------------------------------------------------------------------------------------- #
    # Boilerplate to match the rsl_rl.modules.ActorCritic interface expected by rsl_rl.PPO  #
    # ------------------------------------------------------------------------------------- #

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def history_latent(self) -> torch.Tensor:
        if self._last_history_latent is None:
            raise RuntimeError("No cached history latent found. Call `act(obs)` first.")
        return self._last_history_latent

    @property
    def privileged_latent(self) -> torch.Tensor:
        if self._last_privileged_latent is None:
            raise RuntimeError("No cached privileged latent found. Call `act(obs)` first.")
        return self._last_privileged_latent

    # --------------------------------------------------------------------------------- #
    # Observation plumbing                                                              #
    # --------------------------------------------------------------------------------- #

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["history"]]
        return torch.cat(obs_list, dim=-1)

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["privileged"]]
        return torch.cat(obs_list, dim=-1)

    def _compute_latents(self, obs: TensorDict) -> torch.Tensor:
        """Run both encoders and return the latent that the actor should condition on.

        Both latents are cached on `self` (history latent WITH grad, privileged latent WITH
        grad too -- `PPORoA` is responsible for `.detach()`-ing the target side of the
        regularization loss) so the algorithm can read them after calling `act`/`evaluate`
        without paying for a second forward pass through the encoders.
        """
        history_obs = self.get_history_obs(obs)
        history_obs = self.history_obs_normalizer(history_obs)
        history_latent = self.history_encoder(history_obs)

        privileged_obs = self.get_privileged_obs(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        privileged_latent = self.privileged_encoder(privileged_obs)

        self._last_history_latent = history_latent
        self._last_privileged_latent = privileged_latent
        return history_latent

    # --------------------------------------------------------------------------------- #
    # Standard ActorCritic-style API consumed by rsl_rl.algorithms.PPO / PPORoA         #
    # --------------------------------------------------------------------------------- #

    def _update_distribution(self, actor_in: torch.Tensor) -> None:
        mean = self.actor(actor_in)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent = self._compute_latents(obs)
        self._update_distribution(torch.cat([actor_obs, latent], dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deployment-time forward pass. Note this NEVER touches the privileged encoder/obs,
        only the history-based adaptation module -- exactly what you'd run on the real robot.
        """
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        history_obs = self.get_history_obs(obs)
        history_obs = self.history_obs_normalizer(history_obs)
        latent = self.history_encoder(history_obs)
        return self.actor(torch.cat([actor_obs, latent], dim=-1))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent_regularization_loss(self) -> torch.Tensor:
        """MSE between the history (student) latent and the privileged (teacher) latent.

        The teacher side is detached: gradients only flow into the HistoryEncoder, never into
        the PrivilegedEncoder, which is meant to stay a clean, low-noise summary of privileged
        information. Call `act(obs)` (or `_compute_latents(obs)`) first so the cached latents
        are up to date for the current batch.
        """
        if self._last_history_latent is None or self._last_privileged_latent is None:
            raise RuntimeError("Call `act(obs)` before `get_latent_regularization_loss()`.")
        return nn.functional.mse_loss(self._last_history_latent, self._last_privileged_latent.detach())

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
        if self.history_obs_normalization:
            self.history_obs_normalizer.update(self.get_history_obs(obs))
        if self.privileged_obs_normalization:
            self.privileged_obs_normalizer.update(self.get_privileged_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def infer_history_latent(self, obs: TensorDict) -> torch.Tensor:
        """Run ONLY the HistoryEncoder and return its latent (with grad).

        This is the student-side forward pass used by `PPORoA.update_dagger()` for the
        imitation loss. Unlike `act(obs)` / `_compute_latents(obs)`, this does NOT touch the
        PrivilegedEncoder or the actor, and does NOT overwrite the cached
        `_last_history_latent` / `_last_privileged_latent` used by `update()`'s constraint
        loss -- the two updates stay fully decoupled, exactly like the old
        `infer_hist_latent` / `infer_priv_latent` pair.
        """
        history_obs = self.get_history_obs(obs)
        history_obs = self.history_obs_normalizer(history_obs)
        return self.history_encoder(history_obs)
    
    def infer_privileged_latent(self, obs: TensorDict) -> torch.Tensor:
        """Run ONLY the PrivilegedEncoder and return its latent (with grad).

        Teacher-side forward pass. In `PPORoA.update_dagger()` this should be called inside
        `torch.inference_mode()` and then `.detach()`-ed, since it's only used as a frozen
        regression target there -- mirrors the old `infer_priv_latent`.
        """
        privileged_obs = self.get_privileged_obs(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        return self.privileged_encoder(privileged_obs)