# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for P3 PeekOver."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as hl_mdp
import rl_training.tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from rl_training.recorders import HDF5RecorderManagerCfg
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
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.common_task import (
    SimpleTaskEventCfg,
)

from . import mdp
from .scene_cfg import (
    PeekOverSceneCfg,
    make_peek_over_front_camera,
    make_peek_over_top_camera,
    make_peek_over_workcell_camera,
    make_peek_over_wrist_camera,
)


@configclass
class PeekOverEventCfg(SimpleTaskEventCfg):
    reset_object_position_discontinuous = None

    reset_base = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.05, 0.05)},
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

    reset_peek_over_layout = EventTerm(
        func=mdp.reset_peek_over_layout,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
        },
    )

    reset_joints_default = EventTerm(
        func=vel_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class PeekOverTerminationsCfg:
    time_out = DoneTerm(func=vel_mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(
        func=vel_mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.20},
    )
    bad_orientation = DoneTerm(
        func=vel_mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.8},
    )
    object_dropped = DoneTerm(
        func=hl_mdp.object_dropped,
        params={"object_cfg": SceneEntityCfg("object"), "height_threshold": 0.16},
    )
    place_success = DoneTerm(
        func=mdp.peek_over_place_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class PeekOverTeacherEnvCfg(HighLevelFlatEnvCfg):
    """State teacher for PeekOver."""

    scene: PeekOverSceneCfg = PeekOverSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: PeekOverEventCfg = PeekOverEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: PeekOverTerminationsCfg = PeekOverTerminationsCfg()

    language_instruction: str = "越过隔断找到后方桌面上的糖盒，并把它放进托盘。"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 90.0
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
class PeekOverTeleopEnvCfg(PeekOverTeacherEnvCfg):
    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.disable_zero_weight_rewards()


@configclass
class PeekOverCollectEnvCfg(PeekOverTeleopEnvCfg):
    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_peek_over_top_camera(with_depth=False)
        self.scene.workcell_camera = make_peek_over_workcell_camera(with_depth=False)
        self.scene.front_camera = make_peek_over_front_camera(with_depth=False)
        self.scene.wrist_camera = make_peek_over_wrist_camera(with_depth=False)

        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-PeekOver-Collect-v0"
        recorder.task_profile = "peek_over"
        recorder.action_term_name = "pre_trained_pick_action"
        recorder.camera_names = (
            "top_camera",
            "workcell_camera",
            "front_camera",
            "wrist_camera",
        )
        recorder.tracked_entity_names = ("object", "target_pad")
        recorder.primary_object_name = "object"
        recorder.primary_target_name = "target_pad"
        recorder.success_term_name = "place_success"
        recorder.ee_body_name = "gripper_base"
        recorder.base_link_name = "base_link"
        recorder.gripper_joint_names = (
            "gripper_joint1",
            "gripper_joint2",
        )
