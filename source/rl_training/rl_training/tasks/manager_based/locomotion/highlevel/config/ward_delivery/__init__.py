# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


for suffix, cfg_name in (
    ("Teacher", "WardDeliveryTeacherEnvCfg"),
    ("Teleop", "WardDeliveryTeleopEnvCfg"),
    ("Collect", "WardDeliveryCollectEnvCfg"),
):
    gym.register(
        id=f"Isaac-M20-Piper-WardDelivery-{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}:WardDeliveryPPORunnerCfg",
        },
    )
