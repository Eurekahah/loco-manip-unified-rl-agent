# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-HighLowServe-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HighLowServeTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HighLowServePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-HighLowServe-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HighLowServeTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HighLowServePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-HighLowServe-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HighLowServeCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HighLowServePPORunnerCfg",
    },
)
