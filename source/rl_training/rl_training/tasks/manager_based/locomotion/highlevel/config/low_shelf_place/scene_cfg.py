# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

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


def _look_at_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Return a world-convention camera quaternion (+X forward, +Z up)."""
    eye_np = [float(value) for value in eye]
    target_np = [float(value) for value in target]
    forward = [target_np[i] - eye_np[i] for i in range(3)]
    forward_norm = math.sqrt(sum(value * value for value in forward))
    if forward_norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    forward = [value / forward_norm for value in forward]
    up_np = [float(value) for value in up]
    right = [
        up_np[1] * forward[2] - up_np[2] * forward[1],
        up_np[2] * forward[0] - up_np[0] * forward[2],
        up_np[0] * forward[1] - up_np[1] * forward[0],
    ]
    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm < 1e-9:
        up_np = (1.0, 0.0, 0.0)
        right = [
            up_np[1] * forward[2] - up_np[2] * forward[1],
            up_np[2] * forward[0] - up_np[0] * forward[2],
            up_np[0] * forward[1] - up_np[1] * forward[0],
        ]
        right_norm = math.sqrt(sum(value * value for value in right))
    right = [value / right_norm for value in right]
    camera_up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ]
    # World convention: camera +X is forward and +Z is up. The rotation
    # columns are [forward, up x forward, up].
    rotation = (
        (forward[0], right[0], camera_up[0]),
        (forward[1], right[1], camera_up[1]),
        (forward[2], right[2], camera_up[2]),
    )
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(float(value / norm) for value in quat)


def _static_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.9,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )


def make_workcell_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
    pos: tuple[float, float, float] = (0.10, -1.05, 0.75),
    target: tuple[float, float, float] = (0.78, 0.0, 0.45),
    rot: tuple[float, float, float, float] | None = None,
    clipping_range: tuple[float, float] = (0.1, 5.0),
) -> TiledCameraCfg:
    """Create the static scene camera used by data collection."""
    camera_rot = rot if rot is not None else _look_at_quat_world(pos, target)
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/WorkcellCamera",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=pos,
            rot=camera_rot,
            convention="world",
        ),
    )


def make_wrist_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
    pos: tuple[float, float, float] = (-0.10, 0.0, 0.08),
    # In the Piper gripper frame, +Z points forward and +X points down. Look
    # 30 degrees down while keeping image-up aligned with world +Z.
    rot: tuple[float, float, float, float] = (0.8660254, 0.0, -0.5, 0.0),
    clipping_range: tuple[float, float] = (0.03, 2.0),
) -> TiledCameraCfg:
    """Create an RGB-D camera rigidly attached to the Piper gripper base."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_base/WristCamera",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        # The wrist camera moves with the arm, so extrinsics must be refreshed.
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=pos,
            rot=rot,
            convention="world",
        ),
    )


def make_top_camera(
    *,
    width: int = 224,
    height: int = 224,
    update_period: float = 0.1,
    with_depth: bool = True,
    focal_length: float = 20.0,
    pos: tuple[float, float, float] = (0.80, 0.0, 1.45),
    target: tuple[float, float, float] = (0.80, 0.0, 0.35),
    rot: tuple[float, float, float, float] | None = None,
    clipping_range: tuple[float, float] = (0.1, 4.0),
) -> TiledCameraCfg:
    """Create an overhead camera for the upper shelf and placement area."""
    camera_rot = rot if rot is not None else _look_at_quat_world(pos, target)
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=pos,
            rot=camera_rot,
            convention="world",
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.68, -0.15, 0.34),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Cameras are intentionally opt-in. The training and plain teleop configs
    # therefore keep the same headless performance as the state-only task.
    workcell_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None
    top_camera: TiledCameraCfg | None = None

