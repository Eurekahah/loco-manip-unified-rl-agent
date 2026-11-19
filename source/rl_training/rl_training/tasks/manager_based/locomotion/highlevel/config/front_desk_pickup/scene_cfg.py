# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Office reception scene for R1 FrontDeskPickup."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.highlevel.config.lab_room import (
    make_lab_room_fill_light,
    make_lab_room_light,
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
    OFFICE_PROPS_DIR,
    UPRIGHT_BOX_ROT,
    make_kinematic_box,
    make_static_asset,
    make_tray,
    make_ycb_object,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


RECEPTION_DESK_TOP_Z = 0.98
LOW_TABLE_TOP_Z = 0.46


@configclass
class FrontDeskPickupSceneCfg(HighLevelSceneCfg):
    """Isaac Office scene with a rear-facing visitor pickup challenge."""

    room_scene: AssetBaseCfg = make_official_room("Office")
    room_light_a: AssetBaseCfg = make_lab_room_light("FrontDeskPickup", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("FrontDeskPickup", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    visitor_table = make_static_asset(
        prim_path="{ENV_REGEX_NS}/FrontDeskVisitorTable",
        usd_path=f"{OFFICE_PROPS_DIR}/SM_TableCoffee.usd",
        pos=(-0.10, -1.45, 0.0),
    )

    waiting_chair = make_static_asset(
        prim_path="{ENV_REGEX_NS}/FrontDeskWaitingChair",
        usd_path=f"{OFFICE_PROPS_DIR}/SM_Chair.usd",
        pos=(0.95, -2.05, 0.0),
        rot=(0.9238795, 0.0, 0.0, 0.3826834),
    )

    waiting_sofa = make_static_asset(
        prim_path="{ENV_REGEX_NS}/FrontDeskWaitingSofa",
        usd_path=f"{OFFICE_PROPS_DIR}/SM_Sofa.usd",
        pos=(1.45, 1.10, 0.0),
    )

    reception_pedestal = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/FrontDeskReceptionPedestal",
        size=(0.62, 1.35, 0.92),
        pos=(-2.15, 0.0, 0.46),
        visual_material=make_wood_material((2.5, 2.5)),
    )

    reception_counter = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/FrontDeskReceptionCounter",
        size=(0.86, 1.55, 0.06),
        pos=(-2.15, 0.0, RECEPTION_DESK_TOP_Z - 0.03),
        visual_material=make_wood_material((2.5, 2.5)),
    )

    reception_privacy_panel = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/FrontDeskPrivacyPanel",
        size=(0.06, 1.45, 0.32),
        pos=(-2.54, 0.0, RECEPTION_DESK_TOP_Z + 0.16),
        visual_material=make_wood_material((1.0, 3.0)),
    )

    target_pad = make_tray(
        prim_path="{ENV_REGEX_NS}/FrontDeskTray",
        pos=(-0.10, -1.45, LOW_TABLE_TOP_Z + 0.011),
    )

    object = make_ycb_object(
        prim_path="{ENV_REGEX_NS}/Object",
        object_key="sugar_box",
        pos=(-2.18, -0.48, RECEPTION_DESK_TOP_Z + 0.092),
        rot=UPRIGHT_BOX_ROT,
        mass=0.30,
    )

    distractor_cracker = make_ycb_object(
        prim_path="{ENV_REGEX_NS}/FrontDeskCrackerDistractor",
        object_key="cracker_box",
        pos=(-2.18, 0.42, RECEPTION_DESK_TOP_Z + 0.109),
        rot=UPRIGHT_BOX_ROT,
        mass=0.35,
    )

    distractor_can = make_ycb_object(
        prim_path="{ENV_REGEX_NS}/FrontDeskCanDistractor",
        object_key="tomato_soup_can",
        pos=(-2.18, -0.66, RECEPTION_DESK_TOP_Z + 0.035),
        rot=(1.0, 0.0, 0.0, 0.0),
        mass=0.20,
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_front_desk_pickup_top_camera(
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
        pos=(-0.75, -0.55, 2.80),
        target=(-0.75, -0.55, 0.58),
        clipping_range=(0.1, 6.0),
    )


def make_front_desk_pickup_workcell_camera(
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
        target=(-1.25, -0.55, 0.70),
    )


def make_front_desk_pickup_front_camera(
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


def make_front_desk_pickup_wrist_camera(
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
