# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Scene for the compact two-cylinder pick-and-place task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
    make_workcell_camera,
    make_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelSceneCfg


LOW_PLATFORM_CENTER = (0.60, -0.22)
HIGH_PLATFORM_CENTER = (0.72, 0.38)
PLATFORM_RADIUS = 0.20
LOW_PLATFORM_TOP_Z = 0.30
HIGH_PLATFORM_TOP_Z = 0.52
TARGET_PAD_SIZE = (0.13, 0.13, 0.008)

# YCB sugar box. It is rotated so its thin axis fits the Piper gripper's
# measured ~0.063 m opening while resting on a stable long face.
OBJECT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"


def _static_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


@configclass
class TwoCylinderPlaceSceneCfg(HighLevelSceneCfg):
    """Two offset cylindrical tables with a single YCB target object."""

    low_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LowPlatform",
        spawn=sim_utils.CylinderCfg(
            radius=PLATFORM_RADIUS,
            height=LOW_PLATFORM_TOP_Z,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.43, 0.48)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(LOW_PLATFORM_CENTER[0], LOW_PLATFORM_CENTER[1], LOW_PLATFORM_TOP_Z * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    high_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HighPlatform",
        spawn=sim_utils.CylinderCfg(
            radius=PLATFORM_RADIUS,
            height=HIGH_PLATFORM_TOP_Z,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_static_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.34, 0.48)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(HIGH_PLATFORM_CENTER[0], HIGH_PLATFORM_CENTER[1], HIGH_PLATFORM_TOP_Z * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_pad: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPad",
        spawn=sim_utils.CuboidCfg(
            size=TARGET_PAD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.52, 0.78)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                HIGH_PLATFORM_CENTER[0],
                HIGH_PLATFORM_CENTER[1],
                HIGH_PLATFORM_TOP_Z + TARGET_PAD_SIZE[2] * 0.5,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH,
            scale=(1.0, 1.0, 1.0),
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
            pos=(
                LOW_PLATFORM_CENTER[0],
                LOW_PLATFORM_CENTER[1],
                LOW_PLATFORM_TOP_Z + 0.050,
            ),
            # Rotate the box about local X so its thin (about 0.042 m) axis
            # faces the gripper opening and it rests on a stable long face.
            rot=(0.7071068, 0.7071068, 0.0, 0.0),
        ),
    )

    # Cameraized collection is opt-in. Training and plain teleoperation keep
    # these fields empty so headless performance is unchanged.
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None


def make_two_cylinder_workcell_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
) -> TiledCameraCfg:
    """Frame both platforms from the front-left side of the workcell."""
    return make_workcell_camera(
        width=width,
        height=height,
        update_period=update_period,
        pos=(0.12, -1.08, 0.80),
        target=(0.72, 0.0, 0.42),
        clipping_range=(0.1, 4.0),
    )


def make_two_cylinder_wrist_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
) -> TiledCameraCfg:
    """Use the same calibrated Piper wrist-camera mount as the low-shelf task."""
    return make_wrist_camera(width=width, height=height, update_period=update_period)
