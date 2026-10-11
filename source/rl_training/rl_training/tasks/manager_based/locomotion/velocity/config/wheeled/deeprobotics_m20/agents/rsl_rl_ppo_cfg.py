# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20FlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat"

@configclass
class DeeproboticsM20NavFlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_nav_flat"

@configclass
class DeeproboticsM20WBCFlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "deeprobotics_m20_wbc_flat"

@configclass
class DeeproboticsM20ArmFlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """策略直接控制机械臂（关节空间）的 runner 配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "deeprobotics_m20_arm_flat"

from dataclasses import MISSING

@configclass
class RslRlPpoActorCriticHistoryCfg(RslRlPpoActorCriticCfg):
    """ActorCriticHistory 专用的配置,在原配置基础上新增字段。"""

    class_name: str = "rsl_rl.networks.actor_critic_history.ActorCriticHistory"

    latent_dim: int = MISSING
    """历史编码器输出的潜变量维度。"""

    history_length: int = MISSING
    """历史观测的长度(时间步数)。"""

    history_encoder_hidden_channels: tuple[int, ...] = MISSING
    """历史编码器(CNN)各层的通道数。"""

    history_encoder_kernel_sizes: tuple[int, ...] = MISSING
    """历史编码器各层卷积核大小。"""

    history_encoder_strides: tuple[int, ...] = MISSING
    """历史编码器各层步长。"""

    history_encoder_projection_dim: int = 0
    """历史编码器的"逐帧共享投影"宽度；**0 = 旧行为**（不投影，直接对原始维度做时间卷积）。

    参考实现（vwb `StateHistoryEncoder`）是"先对每个时间步做一次共享 `Linear(num_prop -> 3*channel_size)`，
    再在投影后的特征上做时间卷积"——即先把异构量纲（rad / rad·s⁻¹ / 归一化动作 / 指令）各自线性映射到
    统一宽度，再做时间聚合。设成正数即启用该结构（例如 30 = 参考实现的 `3 * channel_size(10)`）。
    """

    privileged_encoder_hidden_dims: tuple[int, ...] = MISSING
    """特权信息编码器的隐藏层维度。"""

    max_noise_std: float = 0.0
    """探索噪声（动作标准差）的上界；**0（默认）= 不限制**，即旧行为。

    P1-1（见 docs/review/DEFECT_LOG_zh.md DEF-023）：`entropy_coef` 的熵奖励 + 无上界的
    `log_std` 会把 `Policy/mean_noise_std` 顶到 ~1.5 并停在那，同时 adaptive 调度把学习率
    压到 1e-5（高熵 + 低学习率，精度上界被压住）。设成 1.2 之类可给探索噪声封顶：
    采样处 clamp + 每次 `optimizer.step()` 后把参数投影回可行域（`ActorCriticHistory.clamp_noise_std_`）。

    注意：这里不用 `float | None` —— IsaacLab 的 `update_class_from_dict` 是按**当前值的类型**
    校验 hydra 覆盖的（`value is None or isinstance(value, type(obj_mem))`），默认 None 会让
    `agent.policy.max_noise_std=1.2` 报 "Incorrect type ... Expected NoneType"。
    用法：`python .../train.py --task ... agent.policy.max_noise_std=1.2`
    """

@configclass
class RslRlPpoAlgorithmHistoryCfg(RslRlPpoAlgorithmCfg):
    """PPORoA 专用的配置,在原配置基础上新增字段。"""

    class_name: str = "rsl_rl.algorithms.ppo_roa.PPORoA"

    constraint_coef_schedule: tuple[float, float, int, int] = MISSING
    """约束系数的调度参数,格式为 (start_value, end_value, start_iteration, end_iteration)。"""

@configclass
class HistoryAdaptationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunnerHis"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "history_adaptation"
    empirical_normalization = False
    clip_actions = 100
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "history": ["history"],
        "privileged": ["privileged"],
    }
    policy = RslRlPpoActorCriticHistoryCfg(
        class_name="ActorCriticHistory",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim= 32,
        history_length= 10,
        history_encoder_hidden_channels= (32, 32, 32),
        history_encoder_kernel_sizes= (4, 3, 2),
        history_encoder_strides= (2, 1, 1),
        privileged_encoder_hidden_dims= (128, 64),
    )
    algorithm = RslRlPpoAlgorithmHistoryCfg(
        class_name="rsl_rl.algorithms.ppo_roa.PPORoA",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        constraint_coef_schedule= (0.0, 0.1, 3000, 7000),
    )


@configclass
class HistoryAdaptationV2PPORunnerCfg(HistoryAdaptationPPORunnerCfg):
    """V2 的 agent 配置：与参考实现一致地把**特权观测也喂给 critic**（非对称 critic）。

    只改 ``obs_groups["critic"]``（``critic`` + ``privileged``，86 + 89 = 175 维），其余超参
    与 ``HistoryAdaptationPPORunnerCfg`` 逐字段一致；配 ``FlatEnvWBCConfigV2`` 使用（V2 的历史
    窗口 = actor 自己的观测，含速度指令）。

    为什么：参考实现（vwb `actor_critic.py` 的 ``Critic.forward``：``obs[:, :num_prop+num_priv]``；
    DWL 同理）里 critic 直接吃 ``[prop, priv]``，值函数能"看见"摩擦/质量/负载/PD 增益等特权量
    ⇒ 优势估计更准。本仓库原来 critic 只有 86 维（policy obs + base_lin_vel），特权观测
    只喂了编码器、没喂值函数。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_adaptation_v2"
        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["critic", "privileged"],
            "history": ["history"],
            "privileged": ["privileged"],
        }


@configclass
class HistoryAdaptationV3PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """V3 = V2 + 参考实现的历史编码器结构（逐帧共享投影 → 时间卷积）。

    vwb `StateHistoryEncoder` 在 T=10 时的配置是：逐帧 `Linear(num_prop → 30)`，
    然后 `Conv1d(30→20, k=4, s=2)`、`Conv1d(20→10, k=2, s=1)`，最后线性输出 latent。
    这里用同一组超参（channel_size = 10 ⇒ 投影 30 / 卷积通道 20,10）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_adaptation_v3"
        self.policy.history_encoder_projection_dim = 30
        self.policy.history_encoder_hidden_channels = (20, 10)
        self.policy.history_encoder_kernel_sizes = (4, 2)
        self.policy.history_encoder_strides = (2, 1)
