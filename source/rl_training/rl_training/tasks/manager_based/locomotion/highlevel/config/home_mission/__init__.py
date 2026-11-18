# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-HomeMission-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HomeMissionTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HomeMissionPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-HomeMission-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HomeMissionTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HomeMissionPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-HomeMission-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HomeMissionCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:HomeMissionPPORunnerCfg",
    },
)
