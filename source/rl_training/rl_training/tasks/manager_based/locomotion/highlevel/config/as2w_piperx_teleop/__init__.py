# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 遥操作任务注册。

与 ``Isaac-M20-Piper-Teleop-History-v0`` 同结构：高层 13 维遥操动作
（12 维 chassis/EE/body + 1 维夹爪）→ ``TeleopLLAction`` → 历史自适应低层策略。
"""

import gymnasium as gym


gym.register(
    id="Isaac-AS2W-PiperX-Teleop-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:AS2WPiperXTeleopEnvCfg",
        # 遥操不需要训练，但注册一个 runner 让 probe_task_cfg / play 等通用脚本可用。
        "rsl_rl_cfg_entry_point": (
            "rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.agents."
            "rsl_rl_ppo_cfg:HighLevelPickFlatTeacherPPORunnerCfg"
        ),
    },
)

# 便于沿用 AS2W 任务的无 ``Isaac-`` 前缀命名习惯。
gym.register(
    id="AS2W-PiperX-Teleop-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:AS2WPiperXTeleopEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.agents."
            "rsl_rl_ppo_cfg:HighLevelPickFlatTeacherPPORunnerCfg"
        ),
    },
)
