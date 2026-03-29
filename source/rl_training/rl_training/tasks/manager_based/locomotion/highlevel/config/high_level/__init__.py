# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Deeprobotics-High-Level-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.high_level_rough_env_cfg:HighLevelRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighLevelRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Deeprobotics-High-Level-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.high_level_flat_env_cfg:HighLevelFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighLevelFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Deeprobotics-High-Level-Nav-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_flat_nav_env_cfg:HLFlatNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighLevelNavFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# ==========================================
# 1. 教师任务：无相机，纯特权信息，训练速度极快
# ==========================================
# gym.register(
#     id="Isaac-Deeprobotics-High-Level-Nav-Flat-Teacher-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.hl_flat_nav_env_cfg:HLFlatNavTeacherEnvCfg", # 教师专用环境
#         "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HighLevelNavFlatTeacherPPORunnerCfg",
#     },
# )

# # ==========================================
# # 2. 学生任务（蒸馏）：有相机，视觉导航
# # ==========================================
# gym.register(
#     id="Isaac-Deeprobotics-High-Level-Nav-Flat-Student-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.hl_flat_nav_env_cfg:HLFlatNavEnvCfg", # 学生专用环境（带视觉）
#         "rsl_rl_distillation_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HighLevelNavFlatDistillationRunnerCfg",
#     },
# )

gym.register(
    id="Isaac-Deeprobotics-High-Level-Nav-Flat-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_flat_nav_env_cfg:HLFlatNavEnvCfg", # 学生专用环境（带视觉）
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HighLevelNavFlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{__name__}.agents.rsl_rl_distillation_cfg:HighLevelNavFlatDistillationRunnerCfg",

    },
)

# ==========================================
# 1. 教师任务：无相机，纯特权信息，训练速度极快
# ==========================================
gym.register(
    id="Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_flat_pick_env_cfg:HLFlatPickTeacherEnvCfg", # 教师专用环境
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HighLevelPickFlatTeacherPPORunnerCfg",
    },
)

# ==========================================
# 2. 学生任务（蒸馏）：有相机，视觉导航
# ==========================================
gym.register(
    id="Isaac-Deeprobotics-High-Level-Pick-Flat-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_flat_pick_env_cfg:HLFlatPickEnvCfg", # 学生专用环境（带视觉）
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HighLevelPickFlatDistillationRunnerCfg",
    },
)