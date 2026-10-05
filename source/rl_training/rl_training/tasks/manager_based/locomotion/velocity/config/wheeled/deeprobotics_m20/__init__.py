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
