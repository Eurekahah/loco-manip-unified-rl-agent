# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for P2 BlindIntercept."""

from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
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
    CONVEYOR_PRIM_PATH,
    CONVEYOR_SURFACE_PRIM_PATH,
    BlindInterceptSceneCfg,
    make_blind_intercept_front_camera,
    make_blind_intercept_top_camera,
    make_blind_intercept_workcell_camera,
    make_blind_intercept_wrist_camera,
)

CONVEYOR_BELT_SPEED = 0.35
TARGET_START_X = 1.35
TARGET_RESET_X = 1.05
TARGET_END_X = 2.65
TARGET_SIDE_Y = -0.24
TARGET_START_Z = 0.82


def drive_official_conveyor_belt(env, env_ids=None) -> None:
    """Drive the official conveyor mesh with PhysX surface velocity."""
    del env_ids
    import omni.usd
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim_path_expr in (CONVEYOR_PRIM_PATH, CONVEYOR_SURFACE_PRIM_PATH):
        is_visual_asset = prim_path_expr == CONVEYOR_PRIM_PATH
        for env_prim_path in env.scene.env_prim_paths:
            root_prim = stage.GetPrimAtPath(
                prim_path_expr.replace("{ENV_REGEX_NS}", env_prim_path)
            )
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if is_visual_asset:
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    continue
                if not (
                    prim.IsA(UsdGeom.Mesh)
                    or prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ):
                    continue
                api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
                api.GetSurfaceVelocityEnabledAttr().Set(True)
                api.GetSurfaceVelocityLocalSpaceAttr().Set(False)
                api.GetSurfaceVelocityAttr().Set(Gf.Vec3f(CONVEYOR_BELT_SPEED, 0.0, 0.0))


def reset_conveyor_target(
    env,
    env_ids,
    object_cfg: SceneEntityCfg,
    *,
    target_x: float,
) -> None:
    """Place the conveyor target at a repeatable point on the belt."""
    obj = env.scene[object_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = obj.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    state[:, 0] = origins[:, 0] + target_x
    state[:, 1] = origins[:, 1] + TARGET_SIDE_Y
    state[:, 2] = origins[:, 2] + TARGET_START_Z
    state[:, 7:] = 0.0
    obj.write_root_state_to_sim(state, env_ids=env_ids)


def recycle_conveyor_target(
    env,
    env_ids,
    object_cfg: SceneEntityCfg,
) -> None:
    """Return the target to the upstream end before it falls off the belt."""
    obj = env.scene[object_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    positions = obj.data.root_pos_w[env_ids]
    origins = env.scene.env_origins[env_ids]
    recycle_mask = positions[:, 0] - origins[:, 0] > TARGET_END_X
    if not recycle_mask.any():
        return
    recycle_ids = env_ids[recycle_mask]
    reset_conveyor_target(env, recycle_ids, object_cfg, target_x=TARGET_RESET_X)


@configclass
class BlindInterceptEventCfg(SimpleTaskEventCfg):
    """Startup event that drives the official conveyor asset."""

    drive_conveyor_belt = EventTerm(
        func=drive_official_conveyor_belt,
        mode="prestartup",
    )

    reset_conveyor_target = EventTerm(
        func=reset_conveyor_target,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_x": TARGET_START_X,
        },
    )

    recycle_conveyor_target = EventTerm(
        func=recycle_conveyor_target,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class BlindInterceptTeacherEnvCfg(HighLevelFlatEnvCfg):
    scene: BlindInterceptSceneCfg = BlindInterceptSceneCfg(num_envs=16, env_spacing=3.0)
    actions: HLFlatPickWBCActionsCfg = HLFlatPickWBCActionsCfg()
    observations: RearPickObservationsCfg = RearPickObservationsCfg()
    commands: RearPickCommandsCfg = RearPickCommandsCfg()
    events: BlindInterceptEventCfg = BlindInterceptEventCfg()
    rewards: RearPickRewardsCfg = RearPickRewardsCfg()
    terminations: RearPickTerminationsCfg = RearPickTerminationsCfg()
    language_instruction: str = "追踪并拦截传送带上的目标，把它放进托盘。"

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
class BlindInterceptTeleopEnvCfg(BlindInterceptTeacherEnvCfg):
    actions: TeleopActionsCfg = TeleopActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.disable_zero_weight_rewards()


@configclass
class BlindInterceptCollectEnvCfg(BlindInterceptTeleopEnvCfg):
    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_camera = make_blind_intercept_top_camera(with_depth=False)
        self.scene.workcell_camera = make_blind_intercept_workcell_camera(with_depth=False)
        self.scene.front_camera = make_blind_intercept_front_camera(with_depth=False)
        self.scene.wrist_camera = make_blind_intercept_wrist_camera(with_depth=False)
        recorder = self.recorders.record_hdf5
        recorder.language_instruction = self.language_instruction
        recorder.task = "Isaac-M20-Piper-BlindIntercept-Collect-v0"
        recorder.task_profile = "blind_intercept"
        recorder.action_term_name = "pre_trained_pick_action"
        recorder.camera_names = ("top_camera", "workcell_camera", "front_camera", "wrist_camera")
        recorder.tracked_entity_names = ("object", "target_pad")
        recorder.primary_object_name = "object"
        recorder.primary_target_name = "target_pad"
        recorder.success_term_name = "place_success"
        recorder.ee_body_name = "gripper_base"
        recorder.base_link_name = "base_link"
        recorder.gripper_joint_names = ("gripper_joint1", "gripper_joint2")
