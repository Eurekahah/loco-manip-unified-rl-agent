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
        constraint_coef_schedule: tuple[float, float, int, int] = (0.0, 1.0, 5000, 5000),
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Teacher 锚点（P1-6）：>0 时把 `MSE(decoder(z_priv), x_priv)` 加进 PPO 总损失。
        # 0 = 关闭（旧臂行为逐位不变）。注意：这是参考实现里没有的补强，见 DEF-034。
        teacher_recon_coef: float = 0.0,
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
        self.teacher_recon_coef = float(teacher_recon_coef)
        if self.teacher_recon_coef > 0.0 and getattr(self.policy, "privileged_decoder", None) is None:
            raise ValueError(
                "teacher_recon_coef>0 但 policy 没有 privileged_decoder —— 请在 policy cfg 里设置 "
                "`teacher_recon_hidden_dims=(128,128)`。"
            )
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

        # Latent 健康度指标（每次 update() 刷新；由 runner 打到 TensorBoard 的 `Latent/*`）。
        # 为什么需要：`Loss/constraint` / `Loss/latent_distance` 这两条"越降越好"的曲线**不能**
        # 说明 adaptation 通路是活的 —— 两个编码器一起塌成常数同样能把它们压到 ~0（见
        # docs/history_encoder_loss_cliff_zh.md）。判据是"教师对输入是否敏感 + latent 跨环境是否分散"。
        self.latent_metrics: dict[str, float] = {}
        #: ‖∂z_priv/∂x‖_F 的采样间隔（每次都要跑 latent_dim 遍 backward，没必要每步都算）
        self.latent_jac_every: int = 25
        self._jac_gen = torch.Generator(device="cpu")
        self._jac_gen.manual_seed(0)

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
        # Latent 健康度：留一个 mini-batch 的特权观测做"输入分散度"统计（很小的一坨）
        metric_priv_obs: torch.Tensor | None = None
        metric_z_priv: torch.Tensor | None = None
        mean_recon_loss = 0.0 if self.teacher_recon_coef > 0.0 else None

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
            # 参考实现（vwb `ppo.py:166`）在 PPO 的 forward 里用 `hist_encoding=False`：actor 走**教师模式**，
            # 即 surrogate/value 损失作用在"特权 latent 驱动的策略"上；历史 latent 只出现在下面的 priv_reg 项里。
            self.policy.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_state=hidden_states_batch[0])
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
            if metric_priv_obs is None:
                metric_priv_obs = self.policy.get_privileged_obs(obs_batch).detach().clone()
                metric_z_priv = privileged_latent.detach().clone()
            
            # Term 2: Constrain privileged encoder (Lagrangian multiplier)
            # Gradient → privileged_latent → PrivilegedEncoder (μ)
            # This prevents μ from drifting too far during RL training
            # 与参考实现一致：用 L2 范数（vwb `ppo.py:177`），MSE 会把量级压掉约 200 倍
            constraint_loss = (privileged_latent - history_latent.detach()).norm(p=2, dim=-1).mean()
            
            # Add to total loss
            loss = loss + self.constraint_coef * constraint_loss

            # Term 4（P1-6，可选）：教师锚点 —— decoder(z_priv) 重建归一化特权向量。
            # 这一项不可能被"常数 latent"满足（x 跨环境变化），因此把塌陷从"全局最优"变成"次优"。
            if self.teacher_recon_coef > 0.0:
                recon_loss = self.policy.teacher_reconstruction_loss(obs_batch)
                loss = loss + self.teacher_recon_coef * recon_loss

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
            # 探索噪声上界（P1-1）：entropy bonus 会把 log_std 一路顶高，这里投影回可行域。
            # 没有配置上界（max_noise_std=None）时是空操作。
            if hasattr(self.policy, "clamp_noise_std_"):
                self.policy.clamp_noise_std_()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_constraint_loss += constraint_loss.item()
            if mean_recon_loss is not None:
                mean_recon_loss += recon_loss.item()
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
        if mean_recon_loss is not None:
            mean_recon_loss /= num_updates
        mean_latent_distance /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()
        # 刷新 Latent 健康度指标（必须在 storage.clear() 之外、且用最后一批的缓存 latent）
        self._refresh_latent_metrics(metric_priv_obs, metric_z_priv)
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
        if mean_recon_loss is not None:
            loss_dict["recon"] = mean_recon_loss

        return loss_dict

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _refresh_latent_metrics(
        self, priv_obs: torch.Tensor | None, z_priv_batch: torch.Tensor | None = None
    ) -> None:
        """记录教师/学生 latent 的健康度（`Latent/*` 系列，长周期监控用）。

        * `z_priv_spread` / `z_hist_spread`：latent 在 batch 内的逐维 std 均值 —— 塌成常数时 →0；
        * `z_priv_over_hist`：两者之比，塌陷时教师会先掉下去；
        * `priv_input_spread`：特权输入本身的逐维 std 均值（归一化后应≈0.3~1.0；若≈0 说明
          缓存填成常数了 / 随机化没生效 —— 这是"缓存时序"那一类事故的哨兵）；
        * `teacher_jac`：‖∂z_priv/∂x‖_F（在 [-1,1]^D 上随机采样求均值，与离线探针同定义）——
          **这是"教师是不是常数函数"的直接判据**，每 `latent_jac_every` 次 update 算一次。
        """
        z_priv = self.policy._last_privileged_latent
        z_hist = self.policy._last_history_latent
        if z_priv is None or z_hist is None:
            return
        zp = z_priv.std(dim=0, unbiased=False).mean().item()
        zh = z_hist.std(dim=0, unbiased=False).mean().item()
        self.latent_metrics["z_priv_spread"] = zp
        self.latent_metrics["z_hist_spread"] = zh
        self.latent_metrics["z_priv_over_hist"] = zp / (zh + 1e-8)
        if priv_obs is not None and priv_obs.numel() > 0:
            self.latent_metrics["priv_input_spread"] = priv_obs.std(dim=0, unbiased=False).mean().item()
        if priv_obs is not None and z_priv_batch is not None:
            r2 = self.policy.teacher_recon_r2(z_priv_batch, priv_obs)
            if r2 is not None:
                self.latent_metrics["teacher_recon_r2"] = r2
        if self.counter % max(self.latent_jac_every, 1) == 0:
            jac = self._teacher_jacobian()
            if jac is not None:
                self.latent_metrics["teacher_jac"] = jac

    def _teacher_jacobian(self, n: int = 32) -> float | None:
        """‖∂z_priv/∂x‖_F 的样本均值（默认在 [-1,1]^D 上采样；V4 特权向量就活在这个盒子里）。"""
        try:
            enc = self.policy.privileged_encoder
            first = next(p for p in enc.parameters() if p.requires_grad)
            dim = int(enc.encoder[0].in_features) if hasattr(enc.encoder, "__getitem__") else None
            if dim is None:  # 兜底：从第一层 Linear 取
                dim = int(next(m for m in enc.modules() if isinstance(m, nn.Linear)).in_features)
            x = torch.rand(n, dim, generator=self._jac_gen).to(first.device)
            x = 2.0 * x - 1.0
            x.requires_grad_(True)
            with torch.enable_grad():
                z = enc(x)
                grads = []
                for i in range(z.shape[1]):
                    g = torch.autograd.grad(z[:, i].sum(), x, retain_graph=(i < z.shape[1] - 1))[0]
                    grads.append(g.reshape(x.shape[0], -1))
                jac = torch.cat(grads, dim=1).norm(dim=1).mean().item()
            return float(jac)
        except Exception as exc:  # noqa: BLE001 - 监控指标不能把训练搞崩
            print(f"[latent-metrics] WARN: teacher_jac 计算失败（{type(exc).__name__}: {exc}）")
            return None
    
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
