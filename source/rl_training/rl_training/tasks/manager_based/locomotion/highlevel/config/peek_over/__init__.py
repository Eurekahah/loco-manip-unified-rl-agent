# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-PeekOver-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PeekOverTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:PeekOverPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-PeekOver-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PeekOverTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:PeekOverPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-PeekOver-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PeekOverCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:PeekOverPPORunnerCfg",
    },
)
