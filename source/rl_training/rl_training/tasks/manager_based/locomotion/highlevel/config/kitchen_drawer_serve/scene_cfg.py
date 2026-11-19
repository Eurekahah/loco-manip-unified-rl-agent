# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Independent KitchenRoom scene for K1 KitchenDrawerServe."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.highlevel.config.lab_room import (
    make_lab_room_fill_light,
    make_lab_room_light,
    make_metal_material,
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
from rl_training.tasks.manager_based.locomotion.highlevel.config.theme_task import (
    SEKTION_CABINET_USD_PATH,
    UPRIGHT_BOX_ROT,
    make_kinematic_box,
    make_tray,
    make_ycb_object,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


KITCHEN_ROOM_SIZE = (5.60, 4.40, 3.00)
KITCHEN_ROOM_WALL_THICKNESS = 0.08
COUNTER_TOP_Z = 0.85


def _cabinet_actuators() -> dict[str, ImplicitActuatorCfg]:
    return {
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
            effort_limit_sim=87.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "doors": ImplicitActuatorCfg(
            joint_names_expr=["door_left_joint", "door_right_joint"],
            effort_limit_sim=87.0,
            stiffness=18.0,
            damping=4.0,
        ),
    }


@configclass
class KitchenDrawerServeSceneCfg(HighLevelSceneCfg):
    """A dedicated textured kitchen room built around the Isaac Sektion cabinet."""

    kitchen_floor: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomFloor",
        size=(KITCHEN_ROOM_SIZE[0], KITCHEN_ROOM_SIZE[1], 0.02),
        pos=(0.0, 0.0, 0.01),
        visual_material=make_wood_material((5.0, 4.0)),
    )

    kitchen_back_wall: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomBackWall",
        size=(KITCHEN_ROOM_WALL_THICKNESS, KITCHEN_ROOM_SIZE[1], KITCHEN_ROOM_SIZE[2]),
        pos=(KITCHEN_ROOM_SIZE[0] * 0.5, 0.0, KITCHEN_ROOM_SIZE[2] * 0.5),
        visual_material=make_painted_material((4.0, 3.0)),
    )

    kitchen_front_wall: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomFrontWall",
        size=(KITCHEN_ROOM_WALL_THICKNESS, KITCHEN_ROOM_SIZE[1], KITCHEN_ROOM_SIZE[2]),
        pos=(-KITCHEN_ROOM_SIZE[0] * 0.5, 0.0, KITCHEN_ROOM_SIZE[2] * 0.5),
        visual_material=make_painted_material((4.0, 3.0)),
    )

    kitchen_left_wall: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomLeftWall",
        size=(KITCHEN_ROOM_SIZE[0], KITCHEN_ROOM_WALL_THICKNESS, KITCHEN_ROOM_SIZE[2]),
        pos=(0.0, -KITCHEN_ROOM_SIZE[1] * 0.5, KITCHEN_ROOM_SIZE[2] * 0.5),
        visual_material=make_painted_material((4.0, 3.0)),
    )

    kitchen_right_wall: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomRightWall",
        size=(KITCHEN_ROOM_SIZE[0], KITCHEN_ROOM_WALL_THICKNESS, KITCHEN_ROOM_SIZE[2]),
        pos=(0.0, KITCHEN_ROOM_SIZE[1] * 0.5, KITCHEN_ROOM_SIZE[2] * 0.5),
        visual_material=make_painted_material((4.0, 3.0)),
    )

    kitchen_ceiling: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenRoomCeiling",
        size=(KITCHEN_ROOM_SIZE[0], KITCHEN_ROOM_SIZE[1], 0.06),
        pos=(0.0, 0.0, KITCHEN_ROOM_SIZE[2] - 0.03),
        visual_material=make_painted_material((4.0, 4.0)),
    )

    room_light_a: AssetBaseCfg = make_lab_room_light("KitchenDrawerServe", 0, -1.25)
    room_light_b: AssetBaseCfg = make_lab_room_light("KitchenDrawerServe", 1, 1.25)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/KitchenBaseCabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SEKTION_CABINET_USD_PATH,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.55, 0.0, 0.40),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.18,
            },
        ),
        actuators=_cabinet_actuators(),
    )

    overhead_cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/KitchenOverheadCabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SEKTION_CABINET_USD_PATH,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.82, 0.05, 1.25),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": -0.72,
                "door_right_joint": 0.72,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators=_cabinet_actuators(),
    )

    countertop: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenCountertop",
        size=(0.72, 1.24, 0.05),
        pos=(1.55, -0.18, COUNTER_TOP_Z - 0.025),
        visual_material=make_wood_material((2.5, 2.5)),
    )

    backsplash: RigidObjectCfg = make_kinematic_box(
        prim_path="{ENV_REGEX_NS}/KitchenBacksplash",
        size=(0.03, 1.65, 0.55),
        pos=(2.10, 0.0, 1.12),
        visual_material=make_metal_material(),
    )

    target_pad = make_tray(
        prim_path="{ENV_REGEX_NS}/KitchenCounterTray",
        pos=(1.37, -0.58, COUNTER_TOP_Z + 0.011),
        size=(0.30, 0.22, 0.02),
    )

    object = make_ycb_object(
        prim_path="{ENV_REGEX_NS}/Object",
        object_key="tomato_soup_can",
        pos=(1.45, 0.04, 0.697),
        rot=(1.0, 0.0, 0.0, 0.0),
        mass=0.20,
    )

    distractor_box = make_ycb_object(
        prim_path="{ENV_REGEX_NS}/KitchenBoxDistractor",
        object_key="sugar_box",
        pos=(1.55, 0.38, COUNTER_TOP_Z + 0.092),
        rot=UPRIGHT_BOX_ROT,
        mass=0.30,
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_kitchen_drawer_serve_top_camera(
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
        pos=(0.70, 0.0, 2.85),
        target=(0.70, 0.0, 0.65),
        clipping_range=(0.1, 6.0),
    )


def make_kitchen_drawer_serve_workcell_camera(
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
        pos=(-0.15, -2.05, 1.45),
        target=(1.10, 0.0, 0.72),
    )


def make_kitchen_drawer_serve_front_camera(
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


def make_kitchen_drawer_serve_wrist_camera(
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
