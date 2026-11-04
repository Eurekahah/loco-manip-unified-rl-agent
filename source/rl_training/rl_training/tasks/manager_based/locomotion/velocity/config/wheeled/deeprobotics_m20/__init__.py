# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_nav_cfg:DeeproboticsM20FlatNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20NavFlatPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-WBC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20WBCFlatPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-WBC-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20WBCFlatPPORunnerCfg",
    },
)

# ==========================================
# 策略直接控制机械臂（关节空间）的隔离环境
#   - 去掉 ee_ik 动作项，新增 arm_joint_pos(6 维)
#   - 臂奖励不再受 arm_weight 门控，改由 ramp_reward_weight 课程逐步加入
# ==========================================
gym.register(
    id="Flat-Deeprobotics-M20-Piper-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_arm_cfg:DeeproboticsM20ArmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-Arm-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_arm_cfg:DeeproboticsM20ArmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="Rough-History-Adaptation-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="Rough-History-Adaptation-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="Rough-WO-Stairs-History-Adaptation-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughWOStairsEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

# ── 消融实验专用（2026-09-30）────────────────────────────────────────────────
# 只用于拆"加强扰动 / 静止惩罚 / 镜像符号"三项改动的贡献（见 flat_env_wbc_cfg.py 的
# _apply_ablation 注释与 docs/review/DONE_zh.md 第七节 §5）。**不要当日常配置用。**
gym.register(
    id="History-Ablation-PushOnly-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:AblPushOnlyEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="History-Ablation-RewardOnly-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:AblRewardOnlyEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

# 消融 C（2026-10-02）：三项改动**全退**（= main 行为 + ⑫ 等工程修复）。
# 与云端旧代码 run（2026-09-20_00-50-31，同 seed / 同 10k）对照 ⇒ 单独量 ⑫ 的影响。
gym.register(
    id="History-Ablation-LegacyAll-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:AblLegacyAllEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

# 多地形（随机粗糙 0.01~0.05 + 正/反斜坡 + 平地）—— 需求 3（2026-09-29）
gym.register(
    id="Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Slopes-History-Adaptation-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

# 多地形 + v_x 命令课程推迟一倍（2026-10-02，DEF-040 §2 的候选修法）。
# 其余（地形组成、奖励、其它课程）与上面那条完全一致 ⇒ 与 cloud_roughslopes20k 可直接对比；
# 注意它还同时吃到了新的全局默认 `max_noise_std=1.2`（DEF-039）⇒ 差异有两条，看结论时要一起说。
gym.register(
    id="Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesSlowVxEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)

# 同上的 PLAY 变体（2026-10-05 补）：`play.py --task=…-play-v0` 用完整难度、关课程。
gym.register(
    id="Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesSlowVxEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationPPORunnerCfg",
    },
)
# ── 多地形 + Anchor / NoHist（2026-10-19）：提高任务对环境的敏感度 ──────────────────
#   Anchor = Rough-Slopes 地形 + V4 观测组（54 维归一化特权 + 700 维历史）+ 教师自编码锚 + entropy 0
#   NoHist = 同一 env/超参，历史换 700 维全零 ⇒ 唯一变量是"历史里有没有信息"
gym.register(
    id="Rough-Slopes-History-Anchor-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvAnchor",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAnchorTerrainPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Slopes-History-Anchor-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvAnchor_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAnchorTerrainPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Slopes-History-NoHist-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvNoHist",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistTerrainPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Slopes-History-NoHist-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvNoHist_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistTerrainPPORunnerCfg",
    },
)

# 第三条臂：同 env（V4 观测组 + Rough-Slopes），但**没有教师锚**（V4 配置原样）
#   Anchor / V4 / NoHist 三臂对照 ⇒ 锚点是否带来地形性能收益
gym.register(
    id="Rough-Slopes-History-V4-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvAnchor",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV4TerrainPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Slopes-History-V4-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:RoughSlopesEnvAnchor_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV4TerrainPPORunnerCfg",
    },
)

# 多地形 Anchor 的历史长度消融：T=5/20/50，T=10 使用现有 TerrainAnchor。
_TERRAIN_HISTORY_LENGTH_ARMS = {
    5: ("RoughSlopesEnvAnchorLen5", "RoughSlopesEnvAnchorLen5_PLAY", "HistoryAnchorTerrainLen5PPORunnerCfg"),
    20: ("RoughSlopesEnvAnchorLen20", "RoughSlopesEnvAnchorLen20_PLAY", "HistoryAnchorTerrainLen20PPORunnerCfg"),
    50: ("RoughSlopesEnvAnchorLen50", "RoughSlopesEnvAnchorLen50_PLAY", "HistoryAnchorTerrainLen50PPORunnerCfg"),
}
for _history_length, (_env_cls, _env_play_cls, _runner_cls) in _TERRAIN_HISTORY_LENGTH_ARMS.items():
    gym.register(
        id=f"Rough-Slopes-History-Anchor-Len{_history_length}-Deeprobotics-M20-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:{_env_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_cls}",
        },
    )
    gym.register(
        id=f"Rough-Slopes-History-Anchor-Len{_history_length}-Deeprobotics-M20-play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:{_env_play_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_cls}",
        },
    )

# ── V2：对齐参考实现（DWL / visual_wholebody）的历史窗口内容 + 非对称 critic（2026-10-11）──
#   与 `History-Adaptation-*` 的差别只有两处：
#     ① 历史窗口每一帧 = actor 自己的观测（含速度指令 / EE 目标 / 体姿指令）—— 参考实现里
#        `hist = obs[:, -num_hist*num_prop:]`，而 num_prop 含 commands；
#     ② critic 的 obs_group 额外吃 `privileged`（86 + 89 = 175 维）—— 参考实现的非对称 critic。
#   算法侧还配套了 `rsl_rl` 的三处修复（actor 的 hist_encoding 双模式、PPO 走教师模式、
#   priv_reg 用 L2 范数），见 docs/history_encoder_loss_cliff_zh.md。
gym.register(
    id="History-Adaptation-V2-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV2PPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-V2-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV2PPORunnerCfg",
    },
)

# V3 = V2 + 参考实现的历史编码器结构（逐帧共享投影 → 时间卷积）；env 与 V2 相同。
gym.register(
    id="History-Adaptation-V3-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV3PPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-V3-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV3PPORunnerCfg",
    },
)

# ── V4：特权向量对齐参考实现（2026-10-16）────────────────────────────────────────
#   与 V2 的差别只有两处，都是为了回答"adaptation 通路是不是被特权向量 + 探索口径坑死的"：
#     ① 特权组换成单一 term `privileged_env_params_v4`（54 维、每通道归一到 [-1,1]、
#        丢掉 27 维惯量 Δ / 4 个恒 0 的轮 stiffness 死维、加回 base COM）；
#     ② 探索口径对齐参考实现（entropy_coef 0.01 → 0.0、max_noise_std 1.2）。
#   历史窗口 / 奖励 / 课程 / 非对称 critic 与 V2 逐字段一致 ⇒ 可直接与 fixV2 对比。
gym.register(
    id="History-Adaptation-V4-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV4PPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-V4-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAdaptationV4PPORunnerCfg",
    },
)

# ── 归因臂 2×2（2026-10-16）：把 V4 的"特权向量"与"探索口径"两个改动拆开 ──────────────
#   PrivV4 = **54 维 V4 特权组** + 原探索口径（entropy 0.01 / 无上限）
#   Expl0  = **旧 89 维特权组** + 参考探索口径（entropy 0.0 / cap 1.2）
#   与 fixV2（89+0.01）、V4（54+0.0）合起来正好是 2×2 ⇒ 哪一项救活了 teacher 一目了然。
gym.register(
    id="History-Abl-PrivV4-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblPrivV4PPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-PrivV4-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblPrivV4PPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-Expl0-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblExpl0PPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-Expl0-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV2_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblExpl0PPORunnerCfg",
    },
)

# ── 消融臂（2026-10-11）：历史长度 {5,20,50} + 编码器结构 {MLP,GRU}（T=10）──────────
#   （2026-10-17 追加在上面：P1-6 锚点臂 / P1-7 无历史臂）
# ── P1-6 / P1-7（2026-10-17）────────────────────────────────────────────────────
#   Anchor = V4 + 教师自编码锚（decoder(z_priv) 重建归一化特权向量，不可能被常数满足）
#   NoHist = V4 + history 整窗置零（同宽，只去掉信息）⇒ 回答"历史到底有没有用"
gym.register(
    id="History-Anchor-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAnchorPPORunnerCfg",
    },
)

gym.register(
    id="History-Anchor-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAnchorPPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-NoHist-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4NoHist",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistPPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-NoHist-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4NoHist_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistPPORunnerCfg",
    },
)

# 平地 + 700 维全零历史：用于把地形 NoHist 检查点迁移到平地时保持输入宽度一致。
gym.register(
    id="History-Abl-NoHistLegacy-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4NoHistLegacy",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistPPORunnerCfg",
    },
)

gym.register(
    id="History-Abl-NoHistLegacy-Deeprobotics-M20-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfigV4NoHistLegacy_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HistoryAblNoHistPPORunnerCfg",
    },
)

#   T=10 的两个结构锚点已经存在：TCN（`History-Adaptation-V2-*` = fixV2，正在跑）
#   与 TCN+逐帧投影（`History-Adaptation-V3-*`）。下面是补齐的 5 条臂，env 都基于 V2 的
#   （历史窗口 = actor 自身观测、含 commands；非对称 critic），只改"窗口长度"或"编码器结构"。
_ABLATION_ARMS = {
    "Len5": ("FlatEnvWBCConfigV2Len5", "FlatEnvWBCConfigV2Len5_PLAY", "HistoryAblLen5PPORunnerCfg"),
    "Len20": ("FlatEnvWBCConfigV2Len20", "FlatEnvWBCConfigV2Len20_PLAY", "HistoryAblLen20PPORunnerCfg"),
    "Len50": ("FlatEnvWBCConfigV2Len50", "FlatEnvWBCConfigV2Len50_PLAY", "HistoryAblLen50PPORunnerCfg"),
    "MLP": ("FlatEnvWBCConfigV2", "FlatEnvWBCConfigV2_PLAY", "HistoryAblMLPPPORunnerCfg"),
    "GRU": ("FlatEnvWBCConfigV2", "FlatEnvWBCConfigV2_PLAY", "HistoryAblGRUPPORunnerCfg"),
}
for _arm_name, (_env_cls, _env_play_cls, _runner_cls) in _ABLATION_ARMS.items():
    gym.register(
        id=f"History-Abl-{_arm_name}-Deeprobotics-M20-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:{_env_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_cls}",
        },
    )
    gym.register(
        id=f"History-Abl-{_arm_name}-Deeprobotics-M20-play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:{_env_play_cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_cls}",
        },
    )
