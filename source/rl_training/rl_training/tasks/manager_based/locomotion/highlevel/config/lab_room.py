# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable enclosed lab-room background for high-level manipulation tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR


LAB_ROOM_CENTER = (0.20, 0.0)
LAB_ROOM_SIZE = (7.20, 6.00, 3.20)
LAB_ROOM_WALL_THICKNESS = 0.08

_FLOOR_COLOR = (0.48, 0.47, 0.45)
_WALL_COLOR = (0.68, 0.70, 0.72)
_ACCENT_COLOR = (0.42, 0.50, 0.58)
_CEILING_COLOR = (0.80, 0.80, 0.78)

# Simple_Room is authored with its interior floor at z=-0.78 m. The robot and
# task assets use the simulator ground plane at z=0, so the room needs to be
# lifted before it can replace the default dark grid floor.
SIMPLE_ROOM_Z_OFFSET = 0.78
INTERIOR_HDRI = f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr"

_WOOD_MDL = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl"
_PAINT_MDL = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Miscellaneous/Paint_Matte.mdl"
_METAL_MDL = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless.mdl"
_PLASTIC_MDL = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Plastics/Plastic_ABS.mdl"


def _room_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=0.95,
        dynamic_friction=0.85,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


def make_wood_material(texture_scale: tuple[float, float] = (2.0, 2.0)) -> sim_utils.MdlFileCfg:
    """Textured oak surface for tables, shelves and door panels."""
    return sim_utils.MdlFileCfg(
        mdl_path=_WOOD_MDL,
        project_uvw=True,
        albedo_brightness=1.15,
        texture_scale=texture_scale,
    )


def make_painted_material(texture_scale: tuple[float, float] = (1.5, 1.5)) -> sim_utils.MdlFileCfg:
    """Textured matte paint for frames, cabinets and room fixtures."""
    return sim_utils.MdlFileCfg(
        mdl_path=_PAINT_MDL,
        project_uvw=True,
        albedo_brightness=1.25,
        texture_scale=texture_scale,
    )


def make_metal_material(texture_scale: tuple[float, float] = (1.0, 1.0)) -> sim_utils.MdlFileCfg:
    """Brushed stainless-steel surface for rails and supports."""
    return sim_utils.MdlFileCfg(
        mdl_path=_METAL_MDL,
        project_uvw=True,
        albedo_brightness=1.15,
        texture_scale=texture_scale,
    )


def make_plastic_material(texture_scale: tuple[float, float] = (1.0, 1.0)) -> sim_utils.MdlFileCfg:
    """Textured ABS surface for conveyors and trays."""
    return sim_utils.MdlFileCfg(
        mdl_path=_PLASTIC_MDL,
        project_uvw=True,
        albedo_brightness=1.20,
        texture_scale=texture_scale,
    )


def make_lab_room_floor(prefix: str = "LabRoom") -> RigidObjectCfg:
    """Visual floor slab that hides the infinite dark grid in camera data."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}Floor",
        spawn=sim_utils.CuboidCfg(
            size=(LAB_ROOM_SIZE[0], LAB_ROOM_SIZE[1], 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_FLOOR_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(LAB_ROOM_CENTER[0], LAB_ROOM_CENTER[1], 0.004),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_lab_room_wall(prefix: str, side: str) -> RigidObjectCfg:
    """Create one collision-enabled room wall."""
    width, depth, height = LAB_ROOM_SIZE
    half_width = width * 0.5
    half_depth = depth * 0.5
    thickness = LAB_ROOM_WALL_THICKNESS
    color = _ACCENT_COLOR if side in {"back", "left"} else _WALL_COLOR
    if side == "back":
        size = (thickness, depth, height)
        pos = (LAB_ROOM_CENTER[0] - half_width, LAB_ROOM_CENTER[1], height * 0.5)
    elif side == "front":
        size = (thickness, depth, height)
        pos = (LAB_ROOM_CENTER[0] + half_width, LAB_ROOM_CENTER[1], height * 0.5)
    elif side == "left":
        size = (width, thickness, height)
        pos = (LAB_ROOM_CENTER[0], LAB_ROOM_CENTER[1] - half_depth, height * 0.5)
    elif side == "right":
        size = (width, thickness, height)
        pos = (LAB_ROOM_CENTER[0], LAB_ROOM_CENTER[1] + half_depth, height * 0.5)
    else:
        raise ValueError(f"unsupported room wall side: {side}")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}Wall{side.title()}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_room_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_lab_room_ceiling(prefix: str = "LabRoom") -> RigidObjectCfg:
    """Visual ceiling so upward-looking and panoramic views do not show sky."""
    width, depth, height = LAB_ROOM_SIZE
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}Ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(width, depth, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CEILING_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(LAB_ROOM_CENTER[0], LAB_ROOM_CENTER[1], height - 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_lab_room_light(prefix: str, index: int, y: float) -> AssetBaseCfg:
    """Create an interior ceiling light."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}Light{index}",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.95, 0.86),
            intensity=12000.0,
            radius=0.28,
            treat_as_point=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(LAB_ROOM_CENTER[0] + 0.45, y, LAB_ROOM_SIZE[2] - 0.18),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_lab_room_fill_light() -> AssetBaseCfg:
    """Image-based indoor fill using Isaac's calibrated HDRI assets."""
    return AssetBaseCfg(
        prim_path="/World/WBCInteriorFillLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 0.98, 0.95),
            intensity=6000.0,
            texture_file=INTERIOR_HDRI,
            texture_format="latlong",
            visible_in_primary_ray=False,
        ),
    )


def make_official_room(room: str = "Simple_Room") -> AssetBaseCfg:
    """Use an Isaac-provided textured room instead of primitive gray-box walls."""
    room_specs = {
        "Simple_Room": (
            f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
            SIMPLE_ROOM_Z_OFFSET,
        ),
        "Office": (f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd", 0.0),
        "Hospital": (f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd", 0.0),
        "Simple_Warehouse": (
            f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            0.0,
        ),
    }
    if room not in room_specs:
        raise ValueError(f"unsupported official room: {room}")
    usd_path, z_offset = room_specs[room]
    return AssetBaseCfg(
        prim_path="/World/WBCRoom",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, z_offset),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
