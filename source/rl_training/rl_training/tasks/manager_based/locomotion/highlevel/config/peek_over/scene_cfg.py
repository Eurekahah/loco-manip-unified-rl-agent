# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the P3 PeekOver task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from rl_training.tasks.manager_based.locomotion.highlevel.config.lab_room import (
    make_lab_room_fill_light,
    make_lab_room_light,
    make_metal_material,
    make_official_room,
    make_painted_material,
    make_wood_material,
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


FAR_TABLE_CENTER = (2.15, 0.0)
FAR_TABLE_TOP_Z = 0.45
FAR_TABLE_SIZE = (0.90, 1.15, 0.05)

PARTITION_X = 1.15
PARTITION_HEIGHT = 1.00
PARTITION_WIDTH = 1.80

TARGET_SIDE_OFFSET_Y = 0.34
TRAY_SIDE_OFFSET_Y = 0.30
TARGET_INITIAL_Z = 0.54
TRAY_CENTER_Z = 0.46
TRAY_SIZE = (0.28, 0.20, 0.015)

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
class PeekOverSceneCfg(HighLevelSceneCfg):
    """Room with a tall partition and a target table behind it."""

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("PeekOver", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("PeekOver", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    partition: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverPartition",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, PARTITION_WIDTH, PARTITION_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((1.5, 2.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PARTITION_X, 0.0, PARTITION_HEIGHT * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    partition_cap: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverPartitionCap",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, PARTITION_WIDTH + 0.08, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PARTITION_X, 0.0, PARTITION_HEIGHT + 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    background_cabinet: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverBackgroundCabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 1.20, 1.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.35, 1.65, 0.65),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    far_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverFarTableTop",
        spawn=sim_utils.CuboidCfg(
            size=FAR_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((2.0, 2.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(FAR_TABLE_CENTER[0], FAR_TABLE_CENTER[1], FAR_TABLE_TOP_Z - 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    far_table_leg_fl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverTableLegFL",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.78, -0.50, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    far_table_leg_fr: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverTableLegFR",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.78, 0.50, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    far_table_leg_bl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverTableLegBL",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.52, -0.50, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    far_table_leg_br: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverTableLegBR",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.52, 0.50, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PeekOverTray",
        spawn=sim_utils.CuboidCfg(
            size=TRAY_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.48, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(FAR_TABLE_CENTER[0], -TRAY_SIDE_OFFSET_Y, TRAY_CENTER_Z),
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
            pos=(FAR_TABLE_CENTER[0], TARGET_SIDE_OFFSET_Y, TARGET_INITIAL_Z),
            rot=OBJECT_ROT,
        ),
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_peek_over_top_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Frame the robot, partition and far table from above."""
    return make_top_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        focal_length=14.0,
        pos=(1.10, 0.0, 2.80),
        target=(1.10, 0.0, 0.30),
        clipping_range=(0.1, 6.0),
    )


def make_peek_over_workcell_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
) -> TiledCameraCfg:
    """Side view showing the robot, partition and target table."""
    return make_workcell_camera(
        width=width,
        height=height,
        update_period=update_period,
        with_depth=with_depth,
        pos=(0.45, -2.30, 1.30),
        target=(1.55, 0.0, 0.48),
    )


def make_peek_over_front_camera(
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


def make_peek_over_wrist_camera(
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
