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
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatTrainerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatTrainerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_nav_cfg:DeeproboticsM20FlatNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20NavFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatNavTrainerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-WBC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20WBCFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatWBCTrainerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-Piper-WBC-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_wbc_cfg:FlatEnvWBCConfig_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20WBCFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatWBCTrainerCfg",
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

# ── 消融臂（2026-10-11）：历史长度 {5,20,50} + 编码器结构 {MLP,GRU}（T=10）──────────
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
