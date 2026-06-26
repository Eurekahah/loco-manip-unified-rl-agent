# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Custom algorithm added on top of rsl_rl_lib==3.1.2.
#
# This is rsl_rl.algorithms.PPO with one addition: after each gradient step, it also computes
# an auxiliary "latent regularization" loss that pulls the HistoryEncoder's output (the online
# adaptation module, available at deployment) towards the PrivilegedEncoder's output (the
# teacher, sim-only) -- see `rsl_rl/modules/actor_critic_history.py` for the policy side of this.
#
# This is the RMA (Kumar et al. 2021) / ROA (Regularized Online Adaptation) idea collapsed into
# a SINGLE training phase: instead of (1) training an asymmetric teacher policy with PPO, then
# (2) freezing it and distilling an adaptation module via supervised regression in a second
# phase, both encoders are trained CONCURRENTLY with the policy. The only structural change vs.
# vanilla PPO is the extra loss term added below (search for "ROA loss" / "RMA loss").
#
# Everything else (rollout storage, GAE, clipped surrogate loss, RND, symmetry, multi-GPU
# gradient reduction, adaptive KL learning-rate schedule) is copied verbatim from
# `rsl_rl.algorithms.PPO` so that this class remains a drop-in replacement.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_history import ActorCriticHistory
from rsl_rl.utils import string_to_callable


class PPORoA(PPO):
    """PPO with a concurrent RMA/ROA-style history-encoder <-> privileged-encoder regularization loss."""

    policy: ActorCriticHistory

    def __init__(
        self,
        policy: ActorCriticHistory,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # ROA / RMA parameters
        # constraint_coef_schedule = (start_coef, end_coef, delay_updates, ramp_updates)
        #   - delay_updates: number of update() calls before the constraint loss starts ramping
        #   - ramp_updates:  number of update() calls over which it linearly anneals start->end
        # This exactly mirrors the old `priv_reg_coef_schedual` semantics.
        constraint_coef_schedule: tuple[float, float, int, int] = (0.0, 1.0, 0, 1),
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize PPO with the additional ROA/RMA latent-regularization loss.

        Args:
            constraint_coef_schedule: (start, end, delay, ramp) — see old `priv_reg_coef_schedual`.
            For `self.counter` (= number of completed update() calls):
                stage = clip((counter - delay) / ramp, 0, 1)
                coef  = start + stage * (end - start)
            (See ``rsl_rl.algorithms.PPO`` for all other arguments.)
        """
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.constraint_coef_schedule = constraint_coef_schedule
        self.constraint_coef = constraint_coef_schedule[0]
        # `counter` tracks number of completed update() calls (PPO + constraint term),
        # exactly like the old `self.counter` — incremented in BOTH update() and
        # update_dagger(), matching the old `update_counter()` call sites.
        self.counter = 0

        # Separate optimizer that ONLY updates the history encoder, used by update_dagger().
        # NOTE: adjust the attribute name below to whatever ActorCriticHistory actually
        # exposes (e.g. `self.policy.history_encoder`).
        self.hist_encoder_optimizer = optim.Adam(
            self.policy.history_encoder.parameters(), lr=learning_rate
        )

    def _get_constraint_coef(self) -> float:
        """Old-style delayed linear ramp: stage = clip((counter - delay) / ramp, 0, 1)."""
        start_coef, end_coef, delay, ramp = self.constraint_coef_schedule
        stage = min(max((self.counter - delay) / max(ramp, 1), 0.0), 1.0)
        return start_coef + stage * (end_coef - start_coef)

    def update_counter(self) -> None:
        self.counter += 1

    def update(self) -> dict[str, float]:
        self.constraint_coef = self._get_constraint_coef()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_constraint_loss = 0  # <- Term 2: privileged encoder constrained by history encoder (should prevent latent drift, ideally stays low and stable)
        mean_latent_distance = 0  # <- Just the raw distance between the latents (no gradients) for monitoring purposes -- ideally should decrease over training as the two encoders align
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions.
            # NOTE: `self.policy.act(...)` is `ActorCriticHistory.act`, which internally runs
            # BOTH the history encoder and the privileged encoder and caches their outputs
            # (`self.policy._last_history_latent` / `_last_privileged_latent`) -- that's what
            # lets us compute the ROA/RMA loss below without a redundant forward pass.
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ---------------------------------------------------------------------------- #
            # ROA / RMA loss: Bidirectional regularization as per the paper
            # Term 2: λ||z^μ - sg[z^φ]||₂  (constrain privileged encoder)
            # Term 3: ||sg[z^μ] - z^φ||₂    (adaptation module mimics privileged)
            # ---------------------------------------------------------------------------- #
            
            # Get the cached latents from the policy
            history_latent = self.policy.history_latent      # z^φ
            privileged_latent = self.policy.privileged_latent # z^μ
            
            # Term 2: Constrain privileged encoder (Lagrangian multiplier)
            # Gradient → privileged_latent → PrivilegedEncoder (μ)
            # This prevents μ from drifting too far during RL training
            constraint_loss = nn.functional.mse_loss(
                privileged_latent,
                history_latent.detach()
            )
            
            # Add to total loss
            loss = loss + self.constraint_coef * constraint_loss

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO (+ ROA/RMA loss, fused into `loss` above)
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_constraint_loss += constraint_loss.item()
            mean_latent_distance += torch.mean(torch.norm(history_latent - privileged_latent.detach(), dim=-1)).item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_constraint_loss /= num_updates 
        mean_latent_distance /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()
        self.update_counter()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "constraint": mean_constraint_loss,
            "latent_distance": mean_latent_distance,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
    
    # ------------------------------------------------------------------ #
    # Stage B: dagger / imitation update (student mimics teacher)
    #   -> SEPARATE backward, SEPARATE optimizer (only touches the history
    #      encoder), SEPARATE gradient clipping — exactly like the old
    #      update_dagger() / hist_encoder_optimizer.
    #   -> No coefficient: imitation weight is implicitly 1.0, since it
    #      never gets mixed additively with any other loss term.
    #   -> Call this instead of update() on the steps you want pure
    #      imitation (e.g. every `dagger_update_freq` rollouts), the same
    #      way the old code alternated between update() and update_dagger().
    # ------------------------------------------------------------------ #
    def update_dagger(self) -> dict[str, float]:
        mean_imitation_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            # Run the full policy under no-grad just to populate caches/hidden state
            # consistently (mirrors old `self.actor_critic.act(..., hist_encoding=True)`
            # inside `inference_mode()`), then recompute the privileged latent (teacher,
            # frozen target) and history latent (student, with grad) explicitly.
            with torch.inference_mode():
                self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
                priv_latent_batch = self.policy.privileged_latent.clone()

            hist_latent_batch = self.policy.infer_history_latent(obs_batch)

            imitation_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=-1).mean()

            self.hist_encoder_optimizer.zero_grad()
            imitation_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_imitation_loss += imitation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_imitation_loss /= num_updates

        self.storage.clear()
        self.update_counter()  # counter still advances, same as old update_dagger()

        return {"imitation": mean_imitation_loss}