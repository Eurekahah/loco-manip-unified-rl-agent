# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as hl_mdp
import rl_training.tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.hl_flat_pick_env_cfg import (
    CriticCfg as PickCriticCfg,
    HLFlatPickCommandCfg,
    HLFlatPickWBCActionsCfg,
    TeacherCfg as PickTeacherCfg,
    TeleopActionsCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.high_level_flat_env_cfg import (
    HighLevelFlatEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import (
    EventCfg as HighLevelEventCfg,
)

from . import mdp
from .recording import LowShelfHDF5RecorderManagerCfg
from .scene_cfg import LowShelfPlaceSceneCfg, make_top_camera, make_workcell_camera, make_wrist_camera


@configclass
class LowShelfPlaceTeacherCfg(PickTeacherCfg):
    """Teacher observations: object and target poses plus body/gripper state."""

    target_pad_rel_pose = ObsTerm(
        func=hl_mdp.object_pose_in_robot_root_frame,
        params={
            "object_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    target_pad_rel_pose_ee = ObsTerm(
        func=hl_mdp.object_pose_in_ee_frame,
        params={
            "object_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_link_name": "gripper_base",
            "ee_offset_z": 0.135,
        },
    )
    body_pose_state = ObsTerm(
        func=mdp.body_pose_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
        },
    )
    gripper_pos = ObsTerm(
        func=mdp.gripper_joint_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LowShelfPlaceCriticCfg(PickCriticCfg):
    """Privileged critic observations mirror the teacher task geometry."""

    target_pad_rel_pose = ObsTerm(
        func=hl_mdp.object_pose_in_robot_root_frame,
        params={
            "object_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    target_pad_rel_pose_ee = ObsTerm(
        func=hl_mdp.object_pose_in_ee_frame,
        params={
            "object_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_link_name": "gripper_base",
            "ee_offset_z": 0.135,
        },
    )
    body_pose_state = ObsTerm(
        func=mdp.body_pose_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
        },
    )
    gripper_pos = ObsTerm(
        func=mdp.gripper_joint_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LowShelfPlaceObservationsCfg:
    # ``HighLevelFlatEnvCfg`` expects a ``policy`` group during post-init. The RL runner
    # still selects ``teacher`` as its actor input; this alias keeps the shared base class
    # compatible without changing the task's teacher/critic interface.
    policy: LowShelfPlaceTeacherCfg = LowShelfPlaceTeacherCfg()
    teacher: LowShelfPlaceTeacherCfg = LowShelfPlaceTeacherCfg()
    critic: LowShelfPlaceCriticCfg = LowShelfPlaceCriticCfg()


@configclass
class LowShelfPlaceCommandsCfg(HLFlatPickCommandCfg):
    def __post_init__(self):
        self.ee_pose.debug_vis = False


@configclass
class LowShelfPlaceEventCfg(HighLevelEventCfg):
    reset_base = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)},
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

    reset_object_position_discontinuous = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.04, 0.02),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.01),
                "yaw": (-0.17, 0.17),
            },
            "velocity_range": {},
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
class LowShelfPlaceRewardsCfg:
    approach_object = RewTerm(
        func=hl_mdp.distance_to_target_reward_shift,
        weight=0.20,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "target_dist": 0.55,
            "sharpness": 4.0,
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
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "cmd_proximity_gate": 0.25,
            "action_term_name": "pre_trained_pick_action",
            "sensitivity": 10.0,
        },
    )
    object_to_target = RewTerm(
        func=mdp.object_to_target_reward,
        weight=1.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "sensitivity": 6.0,
        },
    )
    body_pitch_reach = RewTerm(
        func=mdp.body_pitch_reach_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "desired_pitch": -0.15,
            "tolerance": 0.12,
            "object_height_gate": 0.45,
        },
    )
    place_success = RewTerm(
        func=hl_mdp.is_terminated_term,
        weight=4.0,
        params={"term_keys": "place_success"},
    )
    action_rate = RewTerm(func=vel_mdp.action_rate_l2, weight=-0.001)


@configclass
class LowShelfPlaceTerminationsCfg:
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
        func=mdp.object_placed_on_target,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
            "xy_threshold": 0.08,
            "height_threshold": 0.035,
            "target_height_offset": 0.025,
            "lift_height": 0.04,
            "hold_duration": 0.5,
        },
    )


@configclass
class LowShelfPlaceEnvCfg(HighLevelFlatEnvCfg):
    """Single-task low-shelf pick-and-place environment for the M20 + Piper."""

    scene: LowShelfPlaceSceneCfg = LowShelfPlaceSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: LowShelfPlaceObservationsCfg = LowShelfPlaceObservationsCfg()
    commands: LowShelfPlaceCommandsCfg = LowShelfPlaceCommandsCfg()
    events: LowShelfPlaceEventCfg = LowShelfPlaceEventCfg()
    rewards: LowShelfPlaceRewardsCfg = LowShelfPlaceRewardsCfg()
    terminations: LowShelfPlaceTerminationsCfg = LowShelfPlaceTerminationsCfg()

    language_instruction: str = "把下层托盘里的糖盒放到上层桌面。"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # The low-shelf MDP only uses rigid-body poses and velocities. Remove
        # inherited full-body sensors after the parent post-init has configured
        # them, otherwise they still run on every physics step.
        self.scene.height_scanner_base = None
        self.scene.contact_forces = None
        self.scene.gripper_link1_contact_forces = None
        self.scene.gripper_link2_contact_forces = None
        self.scene.arm_contact_forces = None
        # Avoid a shutdown-time marker callback after the scene is destroyed.
        self.actions.pre_trained_pick_action.low_level_ee_actions.debug_vis = False
        # The high-level controller runs at 5 Hz. Rendering only once per
        # high-level step makes the GUI look frozen (~0.65 FPS on the current
        # WBC pipeline), so refresh the viewer at the low-level control rate.
        self.sim.render_interval = self.actions.pre_trained_pick_action.low_level_decimation
        self.scene.warehouse = None
        self.scene.arm_camera = None
        self.scene.nav_camera = None
        self.scene.workcell_camera = None
        self.scene.wrist_camera = None
        self.scene.top_camera = None
        self.disable_zero_weight_rewards()


@configclass
class LowShelfPlaceTeleopEnvCfg(LowShelfPlaceEnvCfg):
    """Interactive teleoperation variant without cameras or data recording."""

    actions: TeleopActionsCfg = TeleopActionsCfg()
    language_instruction: str = "把下层托盘里的糖盒放到上层桌面。"

    def __post_init__(self):
        super().__post_init__()
        # Teleop itself must remain lightweight. Cameraized data collection is a
        # separate task (LowShelfPlaceCollectEnvCfg).
        self.scene.workcell_camera = None
        self.scene.wrist_camera = None
        self.scene.top_camera = None
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0
        self.disable_zero_weight_rewards()


@configclass
class LowShelfPlaceCollectEnvCfg(LowShelfPlaceTeleopEnvCfg):
    """Explicit RGB collection task for keyboard tests and VR demonstrations."""

    recorders: LowShelfHDF5RecorderManagerCfg = LowShelfHDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        # SmolVLA is trained on RGB only. Depth rendering and readback add
        # latency and HDF5 size without changing the policy input.
        self.scene.workcell_camera = make_workcell_camera(with_depth=False)
        self.scene.wrist_camera = make_wrist_camera(with_depth=False)
        self.scene.top_camera = make_top_camera(with_depth=False)
        self.recorders.record_hdf5.language_instruction = self.language_instruction
        self.recorders.record_hdf5.task = "Isaac-M20-Piper-LowShelf-Collect-v0"

