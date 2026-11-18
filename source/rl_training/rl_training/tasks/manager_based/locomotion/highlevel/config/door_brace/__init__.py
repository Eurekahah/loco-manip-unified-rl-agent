# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-DoorBrace-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DoorBraceTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:DoorBracePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-DoorBrace-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DoorBraceTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:DoorBracePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-DoorBrace-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DoorBraceCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:DoorBracePPORunnerCfg",
    },
)
