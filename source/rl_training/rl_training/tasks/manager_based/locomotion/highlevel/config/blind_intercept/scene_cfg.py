# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for P2 BlindIntercept."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from rl_training.tasks.manager_based.locomotion.highlevel.config.lab_room import (
    make_lab_room_fill_light,
    make_lab_room_light,
    make_official_room,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
    make_top_camera,
    make_workcell_camera,
    make_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.rear_pick.scene_cfg import (
    make_rear_pick_front_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
OBJECT_ROT = (0.7071068, 0.7071068, 0.0, 0.0)
CONVEYOR_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A05.usd"
CONVEYOR_PRIM_PATH = "{ENV_REGEX_NS}/BlindInterceptConveyor"
CONVEYOR_SURFACE_PRIM_PATH = "{ENV_REGEX_NS}/BlindInterceptConveyorSurface"


def _conveyor_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


@configclass
class BlindInterceptSceneCfg(HighLevelSceneCfg):
    """Low conveyor lane with a moving-target placeholder and drop tray."""

    # PhysX surface velocity must be authored before the physics view starts.
    replicate_physics = False

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("BlindIntercept", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("BlindIntercept", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    conveyor: AssetBaseCfg = AssetBaseCfg(
        prim_path=CONVEYOR_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=CONVEYOR_USD_PATH,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Leave a clear gap from the robot's front wheel envelope.
            pos=(0.80, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # The official conveyor asset is a static visual mesh. This invisible
    # kinematic plate shares its belt height and carries PhysX surface velocity.
    conveyor_surface: RigidObjectCfg = RigidObjectCfg(
        prim_path=CONVEYOR_SURFACE_PRIM_PATH,
        spawn=sim_utils.CuboidCfg(
            size=(1.98, 0.90, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_conveyor_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.7975, 0.0, 0.758),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlindInterceptTray",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.24, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.48, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, -0.85, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.35, -0.24, 0.82),
            rot=OBJECT_ROT,
        ),
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_blind_intercept_top_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    return make_top_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        focal_length=14.0,
        pos=(0.90, 0.0, 2.80),
        target=(0.90, 0.0, 0.30),
        clipping_range=(0.1, 6.0),
    )


def make_blind_intercept_workcell_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    return make_workcell_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        pos=(0.20, -2.35, 1.25),
        target=(1.10, 0.0, 0.38),
    )


def make_blind_intercept_front_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    return make_rear_pick_front_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
    )


def make_blind_intercept_wrist_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    return make_wrist_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
    )
