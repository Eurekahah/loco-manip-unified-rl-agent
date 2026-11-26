# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层 RL 的 RSL-RL runner 配置（只有一套）。

**基线 = M20 History-Anchor (h060)，唯一保留的一套**（交接单 §3 / D-04 / D-05）：

* 历史编码器：TCN，``history_length=10``，单步 70 维 ⇒ 700 维；
* 非对称 critic：``critic`` + ``privileged``（86 + 54 = 140）；
* 教师自编码锚：``teacher_recon_hidden_dims=(128, 128)`` + ``teacher_recon_coef=1.0``；
* 探索口径：``entropy_coef=0.0`` + ``max_noise_std=1.2``；
* 算法：``rsl_rl.algorithms.ppo_roa.PPORoA``（约束系数调度 0 → 0.1 @ 3000~7000）。

按 D-09 / §4，**不 import 任何 ``deeprobotics_m20`` 的东西**：本文件自带
``RslRlPpoActorCriticHistoryCfg`` / ``RslRlPpoAlgorithmHistoryCfg``（字段与 M20
的 History-Anchor 逐项一致）。
"""

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoActorCriticHistoryCfg(RslRlPpoActorCriticCfg):
    """``ActorCriticHistory`` 专用配置：在 PPO actor-critic 基础上增加历史编码器字段。"""

    class_name: str = "ActorCriticHistory"

    latent_dim: int = MISSING
    """历史编码器输出的潜变量维度。"""

    history_length: int = MISSING
    """历史观测的时间步数（必须与 ``observations.history`` 的 ``history_length`` 一致）。"""

    history_encoder_hidden_channels: tuple[int, ...] = MISSING
    """历史编码器（TCN）各层通道数。"""

    history_encoder_kernel_sizes: tuple[int, ...] = MISSING
    """历史编码器各层卷积核大小。"""

    history_encoder_strides: tuple[int, ...] = MISSING
    """历史编码器各层步长。"""

    history_encoder_projection_dim: int = 0
    """逐帧共享投影宽度；0 = 旧行为（不投影，直接对原始维度做时间卷积）。"""

    history_encoder_type: str = "tcn"
    """历史编码器结构：``tcn`` / ``mlp`` / ``gru``。"""

    history_encoder_mlp_hidden_dims: tuple[int, ...] = (256, 128)
    """``history_encoder_type="mlp"`` 时的隐藏层宽度。"""

    history_encoder_gru_hidden_dim: int = 64
    """``history_encoder_type="gru"`` 时的隐状态宽度。"""

    history_encoder_gru_num_layers: int = 1
    """``history_encoder_type="gru"`` 时的层数。"""

    privileged_encoder_hidden_dims: tuple[int, ...] = MISSING
    """特权信息编码器的隐藏层维度。"""

    teacher_recon_hidden_dims: tuple[int, ...] = ()
    """教师锚点重建头隐藏层；空 tuple = 关闭。非空时配合 ``teacher_recon_coef``。"""

    max_noise_std: float = 1.2
    """探索噪声（动作标准差）上界；0 = 不限制。默认 1.2（见 M20 DEF-039）。"""


@configclass
class RslRlPpoAlgorithmHistoryCfg(RslRlPpoAlgorithmCfg):
    """``PPORoA`` 专用配置：在 PPO 基础上增加约束调度与教师重建权重。"""

    class_name: str = "rsl_rl.algorithms.ppo_roa.PPORoA"

    constraint_coef_schedule: tuple[float, float, int, int] = MISSING
    """约束系数调度 ``(start_value, end_value, start_iteration, end_iteration)``。"""

    teacher_recon_coef: float = 0.0
    """教师锚点重建损失权重；0 = 关闭。需要 ``policy.teacher_recon_hidden_dims`` 非空。"""


@configclass
class AS2WPiperXHistoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AS2W-PiperX 低层 History-Anchor runner（字段 = 交接单 §9.9）。"""

    class_name = "OnPolicyRunnerHis"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "history_anchor_as2w_piperx"
    empirical_normalization = False
    clip_actions = 100
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic", "privileged"],
        "history": ["history"],
        "privileged": ["privileged"],
    }
    policy = RslRlPpoActorCriticHistoryCfg(
        class_name="ActorCriticHistory",
        init_noise_std=1.0,
        noise_std_type="log",
        # 不显式传 state_dependent_std：IsaacLab v2.3.0 的 RslRlPpoActorCriticCfg 没有该字段，
        # 而 ActorCriticHistory 本就通过 **kwargs 忽略它；本地新版默认值即 False，行为等价。
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        history_length=10,
        history_encoder_hidden_channels=(32, 32, 32),
        history_encoder_kernel_sizes=(4, 3, 2),
        history_encoder_strides=(2, 1, 1),
        history_encoder_projection_dim=0,
        history_encoder_type="tcn",
        history_encoder_mlp_hidden_dims=(256, 128),
        history_encoder_gru_hidden_dim=64,
        history_encoder_gru_num_layers=1,
        privileged_encoder_hidden_dims=(128, 64),
        teacher_recon_hidden_dims=(128, 128),
        max_noise_std=1.2,
    )
    algorithm = RslRlPpoAlgorithmHistoryCfg(
        class_name="rsl_rl.algorithms.ppo_roa.PPORoA",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=None,
        constraint_coef_schedule=(0.0, 0.1, 3000, 7000),
        teacher_recon_coef=1.0,
    )
