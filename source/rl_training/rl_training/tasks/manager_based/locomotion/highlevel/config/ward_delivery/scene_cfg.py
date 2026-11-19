# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital ward scene for H1 WardDelivery."""

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


MEDICINE_SHELF_TOP_Z = 1.32
BEDSIDE_TABLE_TOP_Z = 0.901


@configclass
class WardDeliverySceneCfg(HighLevelSceneCfg):
    """Hospital ward with a bed, medical cabinet, wheelchair and bedside target."""

    room_scene: AssetBaseCfg = make_official_room("HospitalWard")
    room_light_a: AssetBaseCfg = make_lab_room_light("WardDelivery", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("WardDelivery", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    hospital_bed = make_static_asset(
        prim_path="{ENV_REGEX_NS}/WardHospitalBed",
        usd_path=f"{HOSPITAL_PROPS_DIR}/SM_HospitalBed_01b.usd",
        pos=(1.20, -1.00, 0.0),
    )

    wheelchair = make_static_asset(
        prim_path="{ENV_REGEX_NS}/WardWheelchair",
        usd_path=f"{HOSPITAL_PROPS_DIR}/SM_WheelChair_01a.usd",
        pos=(-0.10, -2.30, 0.0),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
    )

    medicine_shelf = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/WardMedicineShelf",
        size=(0.45, 0.55, 0.04),
        pos=(-2.22, 2.35, MEDICINE_SHELF_TOP_Z - 0.02),
        visual_material=make_wood_material((1.5, 1.5)),
    )

    medicine_shelf_support_left = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/WardMedicineShelfSupportLeft",
        size=(0.035, 0.035, 0.30),
        pos=(-2.40, 2.12, MEDICINE_SHELF_TOP_Z - 0.17),
        visual_material=make_metal_material(),
    )

    medicine_shelf_support_right = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/WardMedicineShelfSupportRight",
        size=(0.035, 0.035, 0.30),
        pos=(-2.40, 2.58, MEDICINE_SHELF_TOP_Z - 0.17),
        visual_material=make_metal_material(),
    )

    target_pad = make_tray(
        prim_path="{ENV_REGEX_NS}/WardBedsideTray",
        pos=(1.24, 1.15, BEDSIDE_TABLE_TOP_Z + 0.011),
        size=(0.28, 0.22, 0.02),
    )

    object = make_usd_rigid_object(
        prim_path="{ENV_REGEX_NS}/Object",
        usd_path=HOSPITAL_PILL_SET_USD_PATH,
        pos=(-2.22, 2.35, MEDICINE_SHELF_TOP_Z + 0.052),
        mass=0.22,
    )

    distractor_bottle = make_usd_rigid_object(
        prim_path="{ENV_REGEX_NS}/WardBottleDistractor",
        usd_path=HOSPITAL_BOTTLE_A_USD_PATH,
        pos=(-2.22, 2.52, MEDICINE_SHELF_TOP_Z + 0.125),
        mass=0.18,
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_ward_delivery_top_camera(
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
        pos=(-0.45, 0.25, 2.85),
        target=(-0.45, 0.35, 0.75),
        clipping_range=(0.1, 6.0),
    )


def make_ward_delivery_workcell_camera(
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
        pos=(-0.20, -2.45, 1.55),
        target=(-0.60, 0.85, 0.85),
    )


def make_ward_delivery_front_camera(
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


def make_ward_delivery_wrist_camera(
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
