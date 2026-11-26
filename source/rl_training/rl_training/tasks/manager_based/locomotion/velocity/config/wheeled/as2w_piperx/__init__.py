# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""AS2-W X + Piper X 任务注册。

三个 task（交接单 D-03）：

* ``History-Adaptation-AS2W-PiperX-v0``       —— 平地低层训练（Anchor 口径）；
* ``History-Adaptation-AS2W-PiperX-play-v0``  —— 对应 PLAY；
* ``Visualize-AS2W-PiperX-v0``                —— 静止可视化（zero_agent）。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="History-Adaptation-AS2W-PiperX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.low_level.env_cfg:AS2WPiperXFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2WPiperXHistoryPPORunnerCfg",
    },
)

gym.register(
    id="History-Adaptation-AS2W-PiperX-play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.low_level.env_cfg:AS2WPiperXFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2WPiperXHistoryPPORunnerCfg",
    },
)

gym.register(
    id="Visualize-AS2W-PiperX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:AS2WPiperXStandEnvCfg",
    },
)