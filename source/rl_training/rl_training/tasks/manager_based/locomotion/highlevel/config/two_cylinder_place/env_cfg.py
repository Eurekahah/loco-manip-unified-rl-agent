# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a compact two-cylinder pick-and-place task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.highlevel.mdp as hl_mdp
import rl_training.tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_level.hl_flat_pick_env_cfg import (
    TeleopActionsCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.env_cfg import (
    LowShelfPlaceEnvCfg,
    LowShelfPlaceTerminationsCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.recording import (
    LowShelfHDF5RecorderManagerCfg,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place import (
    mdp as low_shelf_mdp,
)

from . import mdp
from .scene_cfg import (
    TwoCylinderPlaceSceneCfg,
    make_two_cylinder_top_camera,
    make_two_cylinder_workcell_camera,
    make_two_cylinder_wrist_camera,
)


@configclass
class TwoCylinderPlaceRewardsCfg:
    approach_object = RewTerm(
        func=hl_mdp.distance_to_target_reward_shift_progress,
        weight=0.05,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "sensitivity": 5.0,
            "penalty_scale": 0.0,
            "stop_reward_dist": 0.50,
            "overshoot_penalty_scale": 0.2,
        },
    )
    heading_to_object = RewTerm(
        func=hl_mdp.heading_to_target_reward_progress,
        weight=0.02,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("object"),
            "sensitivity": 10.0,
            "penalty_scale": 0.0,
            "near_dist_gate": 0.30,
        },
    )
    gripper_alignment = RewTerm(
        func=hl_mdp.object_ee_symmetric_alignment_progress,
        weight=0.25,
        params={
            "min_finger_dist": 0.04,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg_finger1": SceneEntityCfg("robot", body_names="gripper_link1"),
            "ee_frame_cfg_finger2": SceneEntityCfg("robot", body_names="gripper_link2"),
            "sensitivity": 10.0,
        },
    )
    ee_object_progress = RewTerm(
        func=mdp.ee_object_progress_reward,
        weight=0.25,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "sensitivity": 20.0,
        },
    )
    ee_orientation = RewTerm(
        func=hl_mdp.ee_orientation_to_object,
        weight=0.02,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("robot", body_names="gripper_base"),
            "dist_gate": 0.01,
        },
    )
    standpick = RewTerm(
        func=mdp.standpick_reward,
        weight=0.10,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "distance_threshold": 0.60,
            "forward_speed_threshold": 0.08,
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
    pick_up = RewTerm(
        func=mdp.pick_up_bonus,
        weight=3.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "lift_height": 0.04,
            "ee_distance_threshold": 0.12,
            "gripper_closed_threshold": 0.03,
        },
    )
    object_to_target = RewTerm(
        func=mdp.object_to_target_reward_gated,
        weight=1.5,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "lift_height": 0.04,
            "sensitivity": 6.0,
        },
    )
    body_pitch_reach = RewTerm(
        func=low_shelf_mdp.body_pitch_reach_reward,
        weight=0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "desired_pitch": -0.14,
            "tolerance": 0.12,
            "object_height_gate": 0.42,
        },
    )
    body_height_for_place = RewTerm(
        func=mdp.body_height_for_place_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "low_height": 0.40,
            "high_height": 0.52,
            "lift_transition_margin": 0.10,
            "tolerance": 0.07,
        },
    )
    place_success = RewTerm(
        func=hl_mdp.is_terminated_term,
        weight=4.0,
        params={"term_keys": "place_success"},
    )
    action_rate = RewTerm(func=vel_mdp.action_rate_l2, weight=-0.001)


@configclass
class TwoCylinderPlaceTerminationsCfg(LowShelfPlaceTerminationsCfg):
    place_success = DoneTerm(
        func=low_shelf_mdp.object_placed_on_target,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_pad"),
            "robot_cfg": SceneEntityCfg("robot"),
            "xy_threshold": 0.07,
            "height_threshold": 0.035,
            # Rotated sugar-box half-height plus the target-pad top.
            "target_height_offset": 0.088,
            "lift_height": 0.04,
            "hold_duration": 0.5,
        },
    )


@configclass
class TwoCylinderPlaceEnvCfg(LowShelfPlaceEnvCfg):
    """State-based RL teacher for the two-cylinder task."""

    scene: TwoCylinderPlaceSceneCfg = TwoCylinderPlaceSceneCfg(num_envs=16, env_spacing=3.0)
    rewards: TwoCylinderPlaceRewardsCfg = TwoCylinderPlaceRewardsCfg()
    terminations: TwoCylinderPlaceTerminationsCfg = TwoCylinderPlaceTerminationsCfg()

    language_instruction: str = "把糖盒从矮圆柱台放到高圆柱台顶面。"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 24.0
        self.scene.warehouse = None
        self.scene.arm_camera = None
        self.scene.nav_camera = None
        self.scene.workcell_camera = None
        self.scene.wrist_camera = None
        self.scene.top_camera = None
        self.disable_zero_weight_rewards()


@configclass
class TwoCylinderPlaceTeleopEnvCfg(TwoCylinderPlaceEnvCfg):
    """Interactive teleoperation variant without cameras or recording."""

    actions: TeleopActionsCfg = TeleopActionsCfg()
    language_instruction: str = "把糖盒从矮圆柱台放到高圆柱台顶面。"

    def __post_init__(self):
        super().__post_init__()
        self.scene.workcell_camera = None
        self.scene.wrist_camera = None
        self.scene.top_camera = None
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0
        self.disable_zero_weight_rewards()


@configclass
class TwoCylinderPlaceCollectEnvCfg(TwoCylinderPlaceTeleopEnvCfg):
    """Explicit RGB collection variant for keyboard tests and VR demos."""

    recorders: LowShelfHDF5RecorderManagerCfg = LowShelfHDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.workcell_camera = make_two_cylinder_workcell_camera(with_depth=False)
        self.scene.wrist_camera = make_two_cylinder_wrist_camera(with_depth=False)
        self.scene.top_camera = make_two_cylinder_top_camera(with_depth=False)
        self.recorders.record_hdf5.language_instruction = self.language_instruction
        self.recorders.record_hdf5.task = "Isaac-M20-Piper-TwoCylinder-Collect-v0"
