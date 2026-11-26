# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""AS2-W X + Piper X 可视化任务注册。"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Visualize-AS2W-PiperX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:AS2WPiperXStandEnvCfg",
    },
)
