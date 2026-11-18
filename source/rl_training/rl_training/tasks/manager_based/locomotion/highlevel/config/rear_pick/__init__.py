# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-RearPick-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RearPickTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:RearPickPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-RearPick-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RearPickTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:RearPickPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-RearPick-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RearPickCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:RearPickPPORunnerCfg",
    },
)
