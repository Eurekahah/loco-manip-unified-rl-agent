# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital supply-room scene for H2 SupplyRestock."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

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
from rl_training.tasks.manager_based.locomotion.highlevel.config.theme_task import (
    HOSPITAL_BOTTLE_A_USD_PATH,
    HOSPITAL_PILL_SET_USD_PATH,
    HOSPITAL_PROPS_DIR,
    make_kinematic_box,
    make_static_asset,
    make_tray,
    make_usd_rigid_object,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


SUPPLY_SHELF_TOP_Z = 1.40
GROUND_TRAY_Z = 0.011


@configclass
class SupplyRestockSceneCfg(HighLevelSceneCfg):
    """High supply cabinet, hospital bed, wheelchair, cart and low floor tray."""

    room_scene: AssetBaseCfg = make_official_room("HospitalWard")
    room_light_a: AssetBaseCfg = make_lab_room_light("SupplyRestock", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("SupplyRestock", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    supply_cabinet = make_static_asset(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCabinet",
        usd_path=f"{HOSPITAL_PROPS_DIR}/SM_SupplyCabinet_01c.usd",
        pos=(-2.40, 2.30, 0.0),
    )

    hospital_bed = make_static_asset(
        prim_path="{ENV_REGEX_NS}/SupplyRestockHospitalBed",
        usd_path=f"{HOSPITAL_PROPS_DIR}/SM_HospitalBed_01b.usd",
        pos=(1.20, -1.00, 0.0),
    )

    wheelchair = make_static_asset(
        prim_path="{ENV_REGEX_NS}/SupplyRestockWheelchair",
        usd_path=f"{HOSPITAL_PROPS_DIR}/SM_WheelChair_01a.usd",
        pos=(-0.40, -2.35, 0.0),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
    )

    supply_shelf = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockTaskShelf",
        size=(0.42, 0.58, 0.04),
        pos=(-2.03, 2.30, SUPPLY_SHELF_TOP_Z - 0.02),
        visual_material=make_wood_material((1.4, 1.4)),
    )

    supply_shelf_support_left = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockShelfSupportLeft",
        size=(0.035, 0.035, 0.34),
        pos=(-2.21, 2.06, SUPPLY_SHELF_TOP_Z - 0.19),
        visual_material=make_metal_material(),
    )

    supply_shelf_support_right = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockShelfSupportRight",
        size=(0.035, 0.035, 0.34),
        pos=(-2.21, 2.54, SUPPLY_SHELF_TOP_Z - 0.19),
        visual_material=make_metal_material(),
    )

    supply_cart_top = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartTop",
        size=(0.92, 0.62, 0.05),
        pos=(-0.15, 0.32, 0.74),
        visual_material=make_metal_material(),
    )

    supply_cart_lower_shelf = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartLowerShelf",
        size=(0.92, 0.62, 0.04),
        pos=(-0.15, 0.32, 0.34),
        visual_material=make_metal_material(),
    )

    supply_cart_leg_fl = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartLegFL",
        size=(0.035, 0.035, 0.74),
        pos=(-0.56, 0.05, 0.37),
        visual_material=make_metal_material(),
    )

    supply_cart_leg_fr = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartLegFR",
        size=(0.035, 0.035, 0.74),
        pos=(0.26, 0.05, 0.37),
        visual_material=make_metal_material(),
    )

    supply_cart_leg_bl = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartLegBL",
        size=(0.035, 0.035, 0.74),
        pos=(-0.56, 0.59, 0.37),
        visual_material=make_metal_material(),
    )

    supply_cart_leg_br = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/SupplyRestockCartLegBR",
        size=(0.035, 0.035, 0.74),
        pos=(0.26, 0.59, 0.37),
        visual_material=make_metal_material(),
    )

    target_pad = make_tray(
        prim_path="{ENV_REGEX_NS}/SupplyRestockFloorTray",
        pos=(0.80, -2.46, GROUND_TRAY_Z),
        size=(0.32, 0.24, 0.02),
        color=(0.04, 0.48, 0.28),
    )

    object = make_usd_rigid_object(
        prim_path="{ENV_REGEX_NS}/Object",
        usd_path=HOSPITAL_PILL_SET_USD_PATH,
        pos=(-2.05, 2.30, SUPPLY_SHELF_TOP_Z + 0.052),
        mass=0.22,
    )

    distractor_bottle = make_usd_rigid_object(
        prim_path="{ENV_REGEX_NS}/SupplyRestockBottleDistractor",
        usd_path=HOSPITAL_BOTTLE_A_USD_PATH,
        pos=(-2.05, 2.52, SUPPLY_SHELF_TOP_Z + 0.125),
        mass=0.18,
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_supply_restock_top_camera(
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
        focal_length=13.0,
        pos=(-0.15, 0.05, 2.90),
        target=(-0.15, 0.05, 0.62),
        clipping_range=(0.1, 6.0),
    )


def make_supply_restock_workcell_camera(
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
        pos=(0.15, -2.55, 1.65),
        target=(-0.35, 0.10, 0.80),
    )


def make_supply_restock_front_camera(
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


def make_supply_restock_wrist_camera(
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
