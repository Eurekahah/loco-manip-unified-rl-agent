# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for W2 DoorBrace."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
DOOR_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"


@configclass
class DoorBraceSceneCfg(HighLevelSceneCfg):
    """Room with an official hinged-door articulation and a target behind it."""

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("DoorBrace", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("DoorBrace", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    hinged_door: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/HingedDoor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOOR_USD_PATH,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.35, 0.0, 0.384),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=1.0,
            ),
            "drawers_locked": ImplicitActuatorCfg(
                joint_names_expr=["drawer_bottom_joint", "drawer_top_joint"],
                effort_limit_sim=87.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DoorBraceTray",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.24, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.48, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.82, 0.02),
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
            pos=(1.90, 0.22, 0.54),
            rot=OBJECT_ROT,
        ),
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_door_brace_top_camera(
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
        pos=(0.90, 0.0, 2.85),
        target=(0.90, 0.0, 0.40),
        clipping_range=(0.1, 6.0),
    )


def make_door_brace_workcell_camera(
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
        pos=(0.10, -2.45, 1.45),
        target=(1.10, 0.0, 0.70),
    )


def make_door_brace_front_camera(
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


def make_door_brace_wrist_camera(
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
