# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Shared assets and rigid-body helpers for environment-themed tasks."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


OFFICE_ROOM_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
HOSPITAL_ROOM_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"

OFFICE_PROPS_DIR = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props"
HOSPITAL_PROPS_DIR = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Props"
YCB_PROPS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"
SEKTION_CABINET_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
HOSPITAL_PILL_SET_USD_PATH = str(
    Path(__file__).resolve().parents[6] / "assets" / "isaac_hospital_pill_set_01b_rigid.usda"
)
HOSPITAL_BOTTLE_A_USD_PATH = str(
    Path(__file__).resolve().parents[6] / "assets" / "isaac_hospital_bottle_a_rigid.usda"
)

OFFLINE_TASK_OBJECTS = {
    "cracker_box": f"{YCB_PROPS_DIR}/003_cracker_box.usd",
    "sugar_box": f"{YCB_PROPS_DIR}/004_sugar_box.usd",
    "tomato_soup_can": f"{YCB_PROPS_DIR}/005_tomato_soup_can.usd",
    "mustard_bottle": f"{YCB_PROPS_DIR}/006_mustard_bottle.usd",
}

UPRIGHT_BOX_ROT = (0.7071068, 0.7071068, 0.0, 0.0)


def make_static_material() -> sim_utils.RigidBodyMaterialCfg:
    """High-friction material for authored task furniture."""
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


def make_ycb_object(
    *,
    prim_path: str,
    object_key: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = UPRIGHT_BOX_ROT,
    mass: float = 0.25,
) -> RigidObjectCfg:
    """Create one physically simulated task object from the installed YCB set."""
    try:
        usd_path = OFFLINE_TASK_OBJECTS[object_key]
    except KeyError as exc:
        available = ", ".join(sorted(OFFLINE_TASK_OBJECTS))
        raise ValueError(f"unknown YCB object '{object_key}'; available: {available}") from exc

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def make_usd_rigid_object(
    *,
    prim_path: str,
    usd_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mass: float = 0.25,
) -> RigidObjectCfg:
    """Wrap an Isaac prop whose root already carries RigidBodyAPI as a task object."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def make_static_asset(
    *,
    prim_path: str,
    usd_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AssetBaseCfg:
    """Create static furniture from an Isaac USD asset without inventing a rigid body."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=scale,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def make_kinematic_box(
    *,
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    visual_material: sim_utils.MdlFileCfg | sim_utils.PreviewSurfaceCfg | None = None,
) -> RigidObjectCfg:
    """Create one collision-enabled task structure with an explicit material."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=make_static_material(),
            visual_material=(
                visual_material
                if visual_material is not None
                else sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.73, 0.70))
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def make_tray(
    *,
    prim_path: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float] = (0.32, 0.24, 0.02),
    color: tuple[float, float, float] = (0.05, 0.36, 0.78),
) -> RigidObjectCfg:
    """Create the kinematic target region recorded by the shared HDF5 schema."""
    return make_kinematic_box(
        prim_path=prim_path,
        size=size,
        pos=pos,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
