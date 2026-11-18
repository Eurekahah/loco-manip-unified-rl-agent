# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Composite scene for L1 HomeMission."""

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


def _static_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


@configclass
class HomeMissionSceneCfg(HighLevelSceneCfg):
    """Room with cabinet, doorway, blue target and destination basket."""

    room_scene: AssetBaseCfg = make_official_room("Simple_Room")
    room_light_a: AssetBaseCfg = make_lab_room_light("HomeMission", 0, -1.35)
    room_light_b: AssetBaseCfg = make_lab_room_light("HomeMission", 1, 1.35)
    room_fill_light: AssetBaseCfg = make_lab_room_fill_light()

    cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/HomeCabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.55, -1.45, 0.40),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    door_frame_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HomeDoorFrameLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.12, 2.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.55, 0.68, 1.10),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    door_frame_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HomeDoorFrameRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.12, 2.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.55, 1.84, 1.10),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    door_lintel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HomeDoorLintel",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 1.28, 0.14),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_painted_material(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.55, 1.26, 2.13),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    door_panel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HomeDoorPanel",
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 1.02, 1.95),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=make_wood_material((3.0, 6.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.62, 1.22, 1.02),
            rot=(0.9238795, 0.0, 0.0, 0.3826834),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HomeBasket",
        spawn=sim_utils.CuboidCfg(
            size=(0.36, 0.28, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.48, 0.82)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.45, 1.30, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.14),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.30, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.15, -1.25, 0.52),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    front_camera: TiledCameraCfg | None = None
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None


def make_home_mission_top_camera(
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
        pos=(0.85, 0.0, 2.90),
        target=(0.85, 0.0, 0.35),
        clipping_range=(0.1, 6.0),
    )


def make_home_mission_workcell_camera(
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
        pos=(0.05, -2.55, 1.55),
        target=(1.15, 0.0, 0.70),
    )


def make_home_mission_front_camera(
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


def make_home_mission_wrist_camera(
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
