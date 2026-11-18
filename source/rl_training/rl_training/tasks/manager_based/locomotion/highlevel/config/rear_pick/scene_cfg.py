# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the P1 RearPick task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
    CAMERA_CLIP_RANGE,
    make_top_camera,
    make_workcell_camera,
    make_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.lab_room import (
    make_lab_room_fill_light,
    make_lab_room_light,
    make_metal_material,
    make_official_room,
    make_painted_material,
    make_wood_material,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


REAR_TABLE_CENTER = (-1.35, 0.0)
REAR_TABLE_TOP_Z = 0.55
REAR_TABLE_SIZE = (0.85, 1.00, 0.05)

TARGET_SIDE_OFFSET_Y = 0.24
TRAY_SIDE_OFFSET_Y = 0.22
TARGET_INITIAL_Z = 0.645
TRAY_CENTER_Z = 0.56
TRAY_SIZE = (0.30, 0.22, 0.015)

OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
OBJECT_ROT = (0.7071068, 0.7071068, 0.0, 0.0)


def _static_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


@configclass
class RearPickSceneCfg(HighLevelSceneCfg):
    """Rear worktable with a target and a tray behind the robot."""

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("RearPick", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("RearPick", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    background_cabinet: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearPickBackgroundCabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 1.10, 1.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-2.95, 1.75, 0.60),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    rear_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTableTop",
        spawn=sim_utils.CuboidCfg(
            size=REAR_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((2.5, 2.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(REAR_TABLE_CENTER[0], REAR_TABLE_CENTER[1], REAR_TABLE_TOP_Z - 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    rear_table_leg_fl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTableLegFL",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1.71, -0.44, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    rear_table_leg_fr: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTableLegFR",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1.71, 0.44, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    rear_table_leg_bl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTableLegBL",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.99, -0.44, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    rear_table_leg_br: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTableLegBR",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.99, 0.44, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RearTray",
        spawn=sim_utils.CuboidCfg(
            size=TRAY_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.46, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(REAR_TABLE_CENTER[0], -TRAY_SIDE_OFFSET_Y, TRAY_CENTER_Z),
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
            pos=(
                REAR_TABLE_CENTER[0],
                TARGET_SIDE_OFFSET_Y,
                TARGET_INITIAL_Z,
            ),
            rot=OBJECT_ROT,
        ),
    )

    # Cameras are opt-in. Teacher and plain teleop keep the scene headless.
    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_rear_pick_front_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Create the forward RGB-D camera used for local reacquisition."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/FrontCamera",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=CAMERA_CLIP_RANGE,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # The M20 base_link mesh reaches x=0.434 m. Keep the camera ahead
            # of that silhouette so the lower image is not occluded by the
            # chassis or front wheel.
            pos=(0.58, 0.0, 0.12),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


def make_rear_pick_top_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Create an overhead camera that frames both robot and rear worktable."""
    return make_top_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        focal_length=14.0,
        pos=(-0.65, 0.0, 2.80),
        target=(-0.65, 0.0, 0.30),
        clipping_range=(0.1, 6.0),
    )


def make_rear_pick_workcell_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Create a side view of the robot and rear worktable."""
    return make_workcell_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        pos=(-0.55, -1.80, 1.15),
        target=(-0.85, 0.0, 0.35),
    )


def make_rear_pick_wrist_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Reuse the calibrated Piper wrist-camera mount."""
    return make_wrist_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
    )
