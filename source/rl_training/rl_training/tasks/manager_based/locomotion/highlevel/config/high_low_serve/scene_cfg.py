# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for W5 HighLowServe."""

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


LOW_OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
HIGH_OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
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
class HighLowServeSceneCfg(HighLevelSceneCfg):
    """Room with a low cabinet and a high shelf, plus two target objects."""

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("HighLowServe", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("HighLowServe", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    low_shelf: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighLowLowShelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.80, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((1.5, 1.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.20, -0.62, 0.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    high_shelf: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighLowHighShelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 0.90, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((1.8, 1.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.65, 0.60, 1.30),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    high_shelf_support_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighLowShelfSupportLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 1.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.30, 0.24, 0.65),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    high_shelf_support_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighLowShelfSupportRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 1.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_metal_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.30, 0.96, 0.65),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighLowTray",
        spawn=sim_utils.CuboidCfg(
            size=(0.34, 0.26, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.48, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.85, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=LOW_OBJECT_USD_PATH,
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
            pos=(1.20, -0.62, 0.42),
            rot=OBJECT_ROT,
        ),
    )

    object_high: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectHigh",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HIGH_OBJECT_USD_PATH,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.65, 0.60, 1.42),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_high_low_top_camera(
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
        pos=(0.95, 0.0, 2.85),
        target=(0.95, 0.0, 0.60),
        clipping_range=(0.1, 6.0),
    )


def make_high_low_workcell_camera(
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
        pos=(0.10, -2.45, 1.55),
        target=(1.05, 0.0, 0.75),
    )


def make_high_low_front_camera(
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


def make_high_low_wrist_camera(
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
