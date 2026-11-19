# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for H1 WardDelivery."""

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
    WardDeliverySceneCfg,
    make_ward_delivery_front_camera,
    make_ward_delivery_top_camera,
    make_ward_delivery_workcell_camera,
    make_ward_delivery_wrist_camera,
)


@configclass
class WardDeliveryEventCfg(ThemeTaskEventCfg):
    """Start in the ward entrance with the medicine shelf outside the front view."""

    reset_base = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.55, -0.55),
                "y": (-0.55, -0.55),
                "yaw": (-0.45, -0.35),
            },
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
class WardDeliveryTeacherEnvCfg(HighLevelFlatEnvCfg):
    """State teacher for hospital ward delivery through a constrained aisle."""

    scene: WardDeliverySceneCfg = WardDeliverySceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: WardDeliveryEventCfg = WardDeliveryEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: RearPickTerminationsCfg = RearPickTerminationsCfg()
    language_instruction: str = "在病房里找到医疗柜上的药品，穿过轮椅旁窄道，送到床头柜托盘。"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 120.0
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
class WardDeliveryTeleopEnvCfg(WardDeliveryTeacherEnvCfg):
    """Interactive teleoperation variant without collection recording."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 150.0
        self.disable_zero_weight_rewards()


@configclass
class WardDeliveryCollectEnvCfg(WardDeliveryTeleopEnvCfg):
    """Cameraized task for future hospital demonstration collection."""

    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_ward_delivery_top_camera(with_depth=False)
        self.scene.workcell_camera = make_ward_delivery_workcell_camera(with_depth=False)
        self.scene.front_camera = make_ward_delivery_front_camera(with_depth=False)
        self.scene.wrist_camera = make_ward_delivery_wrist_camera(with_depth=False)

        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-WardDelivery-Collect-v0"
        recorder.task_profile = "ward_delivery"
        recorder.action_term_name = "pre_trained_pick_action"
        recorder.camera_names = ("top_camera", "workcell_camera", "front_camera", "wrist_camera")
        recorder.tracked_entity_names = ("object", "target_pad", "distractor_bottle")
        recorder.primary_object_name = "object"
        recorder.primary_target_name = "target_pad"
        recorder.success_term_name = "place_success"
        recorder.ee_body_name = "gripper_base"
        recorder.base_link_name = "base_link"
        recorder.gripper_joint_names = ("gripper_joint1", "gripper_joint2")
