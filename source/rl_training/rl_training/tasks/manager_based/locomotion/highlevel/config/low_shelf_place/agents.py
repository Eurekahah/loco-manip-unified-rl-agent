# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.agents.rsl_rl_ppo_cfg import (
    HighLevelPickFlatTeacherPPORunnerCfg,
)


@configclass
class LowShelfPlacePPORunnerCfg(HighLevelPickFlatTeacherPPORunnerCfg):
    """State-based teacher configuration for the first pick-and-place task."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "low_shelf_place"

