# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-M20-Piper-BlindIntercept-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BlindInterceptTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:BlindInterceptPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-BlindIntercept-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BlindInterceptTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:BlindInterceptPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-M20-Piper-BlindIntercept-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BlindInterceptCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:BlindInterceptPPORunnerCfg",
    },
)
