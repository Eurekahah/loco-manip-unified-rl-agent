# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for H2 SupplyRestock."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from rl_training.recorders import HDF5RecorderManagerCfg
from rl_training.tasks.manager_based.locomotion.highlevel.config.common_task import ThemeTaskEventCfg
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.hl_flat_pick_env_cfg import (
    HLFlatPickWBCActionsCfg,
    TeleopActionsCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.high_level_flat_env_cfg import (
    HighLevelFlatEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.rear_pick.env_cfg import (
    RearPickCommandsCfg,
    RearPickObservationsCfg,
    RearPickRewardsCfg,
    RearPickTerminationsCfg,
)

from .scene_cfg import (
    SupplyRestockSceneCfg,
    make_supply_restock_front_camera,
    make_supply_restock_top_camera,
    make_supply_restock_workcell_camera,
    make_supply_restock_wrist_camera,
)


@configclass
class SupplyRestockEventCfg(ThemeTaskEventCfg):
    """Start between the high cabinet and the low bedside destination."""

    reset_base = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.20, 0.30), "y": (1.10, 1.20), "yaw": (-0.72, -0.62)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class SupplyRestockTeacherEnvCfg(HighLevelFlatEnvCfg):
    """State teacher for high-to-low hospital supply restocking."""

    scene: SupplyRestockSceneCfg = SupplyRestockSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: SupplyRestockEventCfg = SupplyRestockEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: RearPickTerminationsCfg = RearPickTerminationsCfg()
    language_instruction: str = "从高层耗材柜取出药品，绕过推车和病床，把药品补到床边地面托盘。"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 150.0
        self.scene.warehouse = None
        self.scene.arm_camera = None
        self.scene.nav_camera = None
        self.scene.front_camera = None
        self.scene.workcell_camera = None
        self.scene.wrist_camera = None
        self.scene.top_camera = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.actions.pre_trained_pick_action.low_level_ee_actions.debug_vis = False
        self.sim.render_interval = self.actions.pre_trained_pick_action.low_level_decimation
        self.disable_zero_weight_rewards()


@configclass
class SupplyRestockTeleopEnvCfg(SupplyRestockTeacherEnvCfg):
    """Interactive teleoperation variant without collection recording."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 180.0
        self.disable_zero_weight_rewards()


@configclass
class SupplyRestockCollectEnvCfg(SupplyRestockTeleopEnvCfg):
    """Cameraized task for future hospital restocking demonstrations."""

    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_supply_restock_top_camera(with_depth=False)
        self.scene.workcell_camera = make_supply_restock_workcell_camera(with_depth=False)
        self.scene.front_camera = make_supply_restock_front_camera(with_depth=False)
        self.scene.wrist_camera = make_supply_restock_wrist_camera(with_depth=False)

        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-SupplyRestock-Collect-v0"
        recorder.task_profile = "supply_restock"
        recorder.action_term_name = "pre_trained_pick_action"
        recorder.camera_names = ("top_camera", "workcell_camera", "front_camera", "wrist_camera")
        recorder.tracked_entity_names = ("object", "target_pad", "distractor_bottle")
        recorder.primary_object_name = "object"
        recorder.primary_target_name = "target_pad"
        recorder.success_term_name = "place_success"
        recorder.ee_body_name = "gripper_base"
        recorder.base_link_name = "base_link"
        recorder.gripper_joint_names = ("gripper_joint1", "gripper_joint2")
