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

    history_encoder_type: str = "tcn"
    """历史编码器结构：``"tcn"``（默认，时间卷积，可选 `history_encoder_projection_dim`）、
    ``"mlp"``（窗口展平后过 MLP）、``"gru"``（GRU 取最后一帧隐状态）。

    trade-off 直觉（见 `docs/` 的消融记录）：
      * TCN：参数量与窗口长度 T **无关**（只看通道/卷积核），但感受野受层数/卷积核限制；
      * MLP：输入维度 = T×D，第一层参数随 T **线性增长**（T=50 时首层就是 4150×256）；
      * GRU：参数量与 T 无关（只看 hidden size），理论感受野无限，但训练更慢、对顺序敏感。
    """

    history_encoder_mlp_hidden_dims: tuple[int, ...] = (256, 128)
    """`history_encoder_type="mlp"` 时的隐藏层宽度。"""

    history_encoder_gru_hidden_dim: int = 64
    """`history_encoder_type="gru"` 时的隐状态宽度。"""

    history_encoder_gru_num_layers: int = 1
    """`history_encoder_type="gru"` 时的层数（不做消融，默认 1 层）。"""

    privileged_encoder_hidden_dims: tuple[int, ...] = MISSING
    """特权信息编码器的隐藏层维度。"""

    teacher_recon_hidden_dims: tuple[int, ...] = ()
    """教师锚点（P1-6）的重建头隐藏层；**空 tuple = 关闭**（旧臂/旧 checkpoint 行为不变）。

    非空时 `ActorCriticHistory` 会建 `PrivilegedDecoder(z_priv → 归一化特权向量)`，配合
    `algorithm.teacher_recon_coef>0` 把重建损失加进 PPO 总损失 ⇒ 教师塌成常数不再是优解。
    理由与实测见 `docs/review/DEFECT_LOG_zh.md` DEF-032/034。
    """

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

    teacher_recon_coef: float = 0.0
    """教师锚点（P1-6）重建损失权重；0 = 关闭。需要 policy.teacher_recon_hidden_dims 非空。"""

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


@configclass
class HistoryAdaptationV4PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """V4 = V2（历史窗口/奖励/课程/非对称 critic 完全一致）+ V4 特权向量 + 对齐参考实现的探索口径。

    与 V2 只有两处超参差异，都是"复刻参考实现口径"：
      * ``entropy_coef`` 0.01 → **0.0**。参考实现是 0.0（DWL
        ``widowGo1_config.py:348``、vwb 低层 ``b1z1_config.py:379``），std 是一个**不动的标量**
        （初值 1.0、无熵奖励 ⇒ 没有梯度）。我们此前熵奖励把探索 std 顶到 0.9~2.4、并让自适应 KL
        把学习率反复压到 1e-5 地板（DEF-023）⇒ 教师那条 RL 梯度被高熵冲淡，而"拉向学生"的
        正则项不受影响（共享 Adam 会抹平 λ 的量级），这是塌陷的放大器之一。
      * ``max_noise_std`` 0.0（= 无上限）→ **1.2**。``entropy_coef=0`` 时 log_std 没有梯度、
        std 会一直停在 init 的 1.0，这个上限是"万一以后重开熵奖励"的保险。

    **刻意保留** ``learning_rate=1e-3 + schedule="adaptive"``（参考实现是 2e-4 fixed）：我们的预算
    （4096 env × 24 step × 20k iter ≈ 2e9 样本）只有 DWL（5000×40×40k ≈ 8e9）的 1/4，
    直接砍 5 倍学习率会变成"训练不足"，把"教师是否还活着"这个主问题污染掉。等主结论出来
    再单独做 LR 对照（见 TODO）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_adaptation_v4"
        self.algorithm.entropy_coef = 0.0
        self.policy.max_noise_std = 1.2


# ══════════════════════════════════════════════════════════════════════════════════
# 归因臂（2026-10-16）：把 V4 的两个改动**拆开**，做成 2×2，直接回答"哪一项救活了 adaptation"。
#
#            特权向量 89 维（旧）      特权向量 54 维（V4，全归一）
#   探索 0.01/无上限   fixV2（已有 20k）      `History-Abl-PrivV4-*`（本臂）
#   探索 0 / cap 1.2   `History-Abl-Expl0-*`（本臂）   V4（正在跑）
#
# 两条臂都:**只改一个变量**，env 的奖励/课程/历史窗口/critic 结构全部与对应父配置逐字段一致。
# ══════════════════════════════════════════════════════════════════════════════════
@configclass
class HistoryAblExpl0PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """归因臂 "Expl0"：**特权向量仍是旧的 89 维**，只把探索口径对齐参考实现。

    用途：若这一臂的 `Latent/teacher_jac` 也活了 ⇒ 说明主因是"熵奖励把探索噪声顶大 + 自适应 KL
    把学习率压到地板"稀释了教师那条 RL 梯度（DEF-023），而不是特权向量的量级问题。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_expl0"
        self.algorithm.entropy_coef = 0.0
        self.policy.max_noise_std = 1.2


@configclass
class HistoryAblPrivV4PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """归因臂 "PrivV4"：**探索口径仍是 V2 的**（entropy 0.01 / 无上限），只换 54 维特权向量。

    用途：若这一臂的教师仍塌 ⇒ 说明"光把特权向量收拾干净"不够，必须在损失侧加锚（第 3 步）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_privv4"


@configclass
class HistoryAnchorPPORunnerCfg(HistoryAdaptationV4PPORunnerCfg):
    """P1-6 锚点臂：V4（54 维归一化特权 + 参考探索口径）**再加教师自编码锚**。

    动机（实测，见 DEF-032/034）：V4 把特权向量的量级/死维修好，但如果 λ 开闸后教师仍被
    "拉向学生"拉塌，就说明输入口径不够，必须让**塌陷不再是优解**。做法：
        loss += 1.0 · MSE( decoder(z_priv), normalize(x_priv) )
    常数 z_priv 只能重建出常数，而 x_priv 跨环境变化 ⇒ 重建误差无法归零 ⇒ 教师必须保留信息。
    部署侧完全不增加东西（decoder / 特权向量都只在仿真里用）。

    与 V4 的差别**只有**这一条损失（env、探索口径、历史窗口、critic 全同）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_anchor"
        self.policy.teacher_recon_hidden_dims = (128, 128)
        self.algorithm.teacher_recon_coef = 1.0


@configclass
class HistoryAblNoHistPPORunnerCfg(HistoryAdaptationV4PPORunnerCfg):
    """P1-7 无历史臂：V4 的 env/agent 完全不变，只把 history 观测**整窗置零**（同宽）。

    用途：等 2×2 告诉我们"哪套配置的 adaptation 活下来"之后，用它回答"历史到底有没有用"。
    注意：这个臂的 history 编码器仍在、参数仍在、优化器仍在 —— 唯一删掉的是**信息**，
    所以性能差多少就是 adaptation 的真实贡献。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_nohist"


@configclass
class HistoryAnchorTerrainPPORunnerCfg(HistoryAnchorPPORunnerCfg):
    """多地形 Anchor 臂：与平地 Anchor 逐字段相同，只换 env（Rough-Slopes）与实验名。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_anchor_terrain"


@configclass
class HistoryAblNoHistTerrainPPORunnerCfg(HistoryAblNoHistPPORunnerCfg):
    """多地形 NoHist 对照：与 terrain Anchor 同 env/超参，只把历史换成同宽全零。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_nohist_terrain"


@configclass
class HistoryAdaptationV4TerrainPPORunnerCfg(HistoryAdaptationV4PPORunnerCfg):
    """多地形 V4 臂：与 terrain Anchor **只差"没有教师自编码锚"**（教师会塌）。

    用途：地形三臂对照 Anchor / V4 / NoHist —— 直接回答"锚点让教师活着，是否真的换来了
    地形任务上的性能"。env 与 Anchor 完全相同（V4 观测组 + Rough-Slopes）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_adaptation_v4_terrain"


# ══════════════════════════════════════════════════════════════════════════════════
# 消融臂（2026-10-11）：历史长度 {5,10,20,50} × 编码器结构 {TCN, TCN+投影, MLP, GRU}
#   * 长度档的卷积核用的是参考实现 vwb `StateHistoryEncoder` 按 T 分档的那套
#     （T=10: k4s2+k2s1 / T=20: k6s2+k4s2 / T=50: k8s4+k5s1+k5s1；T=5 是这里补的 k3s1+k2s1），
#     投影宽度统一 30（= 参考实现的 `3 * channel_size(10)`），这样"长度"是唯一变量。
#   * 结构档固定在 T=10：TCN（= 正在跑的 fixV2）、TCN+投影（= V3）、MLP、GRU。
# ══════════════════════════════════════════════════════════════════════════════════
def _apply_reference_stack(cfg, t: int) -> None:
    """把参考实现按 T 分档的卷积配置写到 policy cfg 上。"""
    cfg.policy.history_length = int(t)
    cfg.policy.history_encoder_projection_dim = 30
    if t == 5:
        cfg.policy.history_encoder_hidden_channels = (20, 10)
        cfg.policy.history_encoder_kernel_sizes = (3, 2)
        cfg.policy.history_encoder_strides = (1, 1)
    elif t == 10:
        cfg.policy.history_encoder_hidden_channels = (20, 10)
        cfg.policy.history_encoder_kernel_sizes = (4, 2)
        cfg.policy.history_encoder_strides = (2, 1)
    elif t == 20:
        cfg.policy.history_encoder_hidden_channels = (20, 10)
        cfg.policy.history_encoder_kernel_sizes = (6, 4)
        cfg.policy.history_encoder_strides = (2, 2)
    elif t == 50:
        cfg.policy.history_encoder_hidden_channels = (20, 10, 10)
        cfg.policy.history_encoder_kernel_sizes = (8, 5, 5)
        cfg.policy.history_encoder_strides = (4, 1, 1)
    else:
        raise ValueError(f"未定义的历史长度：{t}")


@configclass
class HistoryAnchorTerrainLen5PPORunnerCfg(HistoryAnchorTerrainPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_anchor_terrain_len5"
        _apply_reference_stack(self, 5)


@configclass
class HistoryAnchorTerrainLen20PPORunnerCfg(HistoryAnchorTerrainPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_anchor_terrain_len20"
        _apply_reference_stack(self, 20)


@configclass
class HistoryAnchorTerrainLen50PPORunnerCfg(HistoryAnchorTerrainPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_anchor_terrain_len50"
        _apply_reference_stack(self, 50)


@configclass
class HistoryAblLen5PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_len5"
        _apply_reference_stack(self, 5)


@configclass
class HistoryAblLen20PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_len20"
        _apply_reference_stack(self, 20)


@configclass
class HistoryAblLen50PPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_len50"
        _apply_reference_stack(self, 50)


@configclass
class HistoryAblMLPPPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """结构档：MLP（窗口展平后过 MLP，参数量随 T 线性增长）。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_mlp"
        self.policy.history_encoder_type = "mlp"
        self.policy.history_encoder_mlp_hidden_dims = (256, 128)


@configclass
class HistoryAblGRUPPORunnerCfg(HistoryAdaptationV2PPORunnerCfg):
    """结构档：GRU（取最后一帧隐状态），1 层 64 维隐状态。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "history_abl_gru"
        self.policy.history_encoder_type = "gru"
        self.policy.history_encoder_gru_hidden_dim = 64
        self.policy.history_encoder_gru_num_layers = 1
