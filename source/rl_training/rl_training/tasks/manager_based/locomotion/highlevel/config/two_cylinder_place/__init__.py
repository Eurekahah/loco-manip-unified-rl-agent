# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-TwoCylinder-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TwoCylinderPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:TwoCylinderPlacePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-TwoCylinder-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TwoCylinderPlaceTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:TwoCylinderPlacePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-TwoCylinder-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TwoCylinderPlaceCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:TwoCylinderPlacePPORunnerCfg",
    },
)
