# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for the P1 RearPick task."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.env_cfg import (
    LowShelfPlaceCommandsCfg,
    LowShelfPlaceCriticCfg,
    LowShelfPlaceTeacherCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place import (
    mdp as low_shelf_mdp,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.common_task import (
    SimpleTaskEventCfg,
)

from . import mdp
from .scene_cfg import (
    RearPickSceneCfg,
    make_rear_pick_front_camera,
    make_rear_pick_top_camera,
    make_rear_pick_workcell_camera,
    make_rear_pick_wrist_camera,
)


@configclass
class RearPickTeacherCfg(LowShelfPlaceTeacherCfg):
    """State teacher with target, tray, body and gripper observations."""


@configclass
class RearPickCriticCfg(LowShelfPlaceCriticCfg):
    """Privileged critic for the rear-pick task."""


@configclass
class RearPickObservationsCfg:
    policy: RearPickTeacherCfg = RearPickTeacherCfg()
    teacher: RearPickTeacherCfg = RearPickTeacherCfg()
    critic: RearPickCriticCfg = RearPickCriticCfg()


@configclass
class RearPickCommandsCfg(LowShelfPlaceCommandsCfg):
    """Reuse the high-level EE command term without debug markers."""


@configclass
class RearPickEventCfg(SimpleTaskEventCfg):
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

    reset_rear_pick_layout = EventTerm(
        func=mdp.reset_rear_pick_layout,
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
class RearPickRewardsCfg:
    approach_object = RewTerm(
        func=hl_mdp.distance_to_target_reward_shift,
        weight=0.40,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "target_dist": 0.58,
            "sharpness": 4.0,
        },
    )
    heading_to_object = RewTerm(
        func=hl_mdp.heading_to_target_reward,
        weight=0.30,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
        },
    )
    object_in_front = RewTerm(
        func=mdp.object_in_front_reward,
        weight=0.50,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "distance_threshold": 1.20,
            "cosine_threshold": 0.60,
        },
    )
    gripper_alignment = RewTerm(
        func=hl_mdp.object_ee_symmetric_alignment_progress,
        weight=1.0,
        params={
            "min_finger_dist": 0.04,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg_finger1": SceneEntityCfg("robot", body_names="gripper_link1"),
            "ee_frame_cfg_finger2": SceneEntityCfg("robot", body_names="gripper_link2"),
            "sensitivity": 10.0,
        },
    )
    lift_object = RewTerm(
        func=hl_mdp.object_is_lifted_progress,
        weight=1.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "cmd_proximity_gate": 0.25,
            "action_term_name": "pre_trained_pick_action",
            "sensitivity": 10.0,
        },
    )
    object_to_target = RewTerm(
        func=low_shelf_mdp.object_to_target_reward,
        weight=1.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "sensitivity": 6.0,
        },
    )
    place_success = RewTerm(
        func=hl_mdp.is_terminated_term,
        weight=4.0,
        params={"term_keys": "place_success"},
    )
    undesired_arm_contacts = RewTerm(
        func=hl_mdp.undesired_contacts,
        weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg("arm_contact_forces", body_names=["arm_link[1-6]"]),
            "threshold": 5.0,
        },
    )
    action_rate = RewTerm(func=vel_mdp.action_rate_l2, weight=-0.001)


@configclass
class RearPickTerminationsCfg:
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
        func=mdp.rear_pick_place_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RearPickTeacherEnvCfg(HighLevelFlatEnvCfg):
    """State-based teacher environment for P1 RearPick."""

    scene: RearPickSceneCfg = RearPickSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: RearPickEventCfg = RearPickEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: RearPickTerminationsCfg = RearPickTerminationsCfg()

    language_instruction: str = "去后方桌面找到糖盒，并把它放进托盘。"

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
class RearPickTeleopEnvCfg(RearPickTeacherEnvCfg):
    """Interactive teleoperation variant without data recording."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.disable_zero_weight_rewards()


@configclass
class RearPickCollectEnvCfg(RearPickTeleopEnvCfg):
    """Cameraized teleoperation task for future demonstration collection."""

    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_rear_pick_top_camera(with_depth=False)
        self.scene.workcell_camera = make_rear_pick_workcell_camera(with_depth=False)
        self.scene.front_camera = make_rear_pick_front_camera(with_depth=False)
        self.scene.wrist_camera = make_rear_pick_wrist_camera(with_depth=False)

        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-RearPick-Collect-v0"
        recorder.task_profile = "rear_pick"
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
