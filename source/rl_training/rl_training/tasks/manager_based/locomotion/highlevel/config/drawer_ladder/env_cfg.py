# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for W1 DrawerLadder."""

from __future__ import annotations

from isaaclab.utils import configclass

from rl_training.recorders import HDF5RecorderManagerCfg
from rl_training.tasks.manager_based.locomotion.highlevel.config.common_task import SimpleTaskEventCfg
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
    DrawerLadderSceneCfg,
    make_drawer_ladder_front_camera,
    make_drawer_ladder_top_camera,
    make_drawer_ladder_workcell_camera,
    make_drawer_ladder_wrist_camera,
)


@configclass
class DrawerLadderTeacherEnvCfg(HighLevelFlatEnvCfg):
    scene: DrawerLadderSceneCfg = DrawerLadderSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: SimpleTaskEventCfg = SimpleTaskEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: RearPickTerminationsCfg = RearPickTerminationsCfg()
    language_instruction: str = "打开柜体抽屉，取出糖盒并放进托盘。"

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
class DrawerLadderTeleopEnvCfg(DrawerLadderTeacherEnvCfg):
    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 150.0
        self.disable_zero_weight_rewards()


@configclass
class DrawerLadderCollectEnvCfg(DrawerLadderTeleopEnvCfg):
    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_drawer_ladder_top_camera(with_depth=False)
        self.scene.workcell_camera = make_drawer_ladder_workcell_camera(with_depth=False)
        self.scene.front_camera = make_drawer_ladder_front_camera(with_depth=False)
        self.scene.wrist_camera = make_drawer_ladder_wrist_camera(with_depth=False)
        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-DrawerLadder-Collect-v0"
        recorder.task_profile = "drawer_ladder"
        recorder.action_term_name = "pre_trained_pick_action"
        recorder.camera_names = ("top_camera", "workcell_camera", "front_camera", "wrist_camera")
        recorder.tracked_entity_names = ("object", "target_pad")
        recorder.primary_object_name = "object"
        recorder.primary_target_name = "target_pad"
        recorder.success_term_name = "place_success"
        recorder.ee_body_name = "gripper_base"
        recorder.base_link_name = "base_link"
        recorder.gripper_joint_names = ("gripper_joint1", "gripper_joint2")
