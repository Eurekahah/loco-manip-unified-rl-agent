# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-LowShelf-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LowShelfPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:LowShelfPlacePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-LowShelf-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LowShelfPlaceTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:LowShelfPlacePPORunnerCfg",
    },
)

