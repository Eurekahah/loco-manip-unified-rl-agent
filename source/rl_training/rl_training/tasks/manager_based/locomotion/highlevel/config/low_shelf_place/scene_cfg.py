# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


SHELF_LOWER_TOP_Z = 0.30
SHELF_UPPER_TOP_Z = 0.55
SHELF_FRONT_X = 0.55
SHELF_BACK_X = 1.05

OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"


def _static_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


def make_workcell_camera() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/WorkcellCamera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "depth"],
        debug_vis=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.20, -1.10, 1.20),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


@configclass
class LowShelfPlaceSceneCfg(HighLevelSceneCfg):
    """Compact two-level workcell for a body-pitch-assisted pick-and-place task."""

    lower_shelf: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LowerShelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 0.72, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.44, 0.48)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.80, 0.0, SHELF_LOWER_TOP_Z - 0.015),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    upper_shelf: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UpperShelf",
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 0.72, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.52, 0.56)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.80, 0.0, SHELF_UPPER_TOP_Z - 0.015),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    shelf_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShelfBack",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.72, 0.55),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.36)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SHELF_BACK_X, 0.0, 0.275),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPad",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.48, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.78, 0.20, SHELF_UPPER_TOP_Z + 0.005),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            activate_contact_sensors=True,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.68, -0.15, 0.34),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    workcell_camera = make_workcell_camera()

