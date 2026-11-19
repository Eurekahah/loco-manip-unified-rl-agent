# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.agents import (
    LowShelfPlacePPORunnerCfg,
)


@configclass
class FrontDeskPickupPPORunnerCfg(LowShelfPlacePPORunnerCfg):
    """Shared runner metadata for the office-themed collection task."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "front_desk_pickup"
