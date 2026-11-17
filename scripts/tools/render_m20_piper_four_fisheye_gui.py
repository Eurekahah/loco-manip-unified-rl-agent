#!/usr/bin/env python3
"""Render a four-camera M20-Piper rig and stitch an equirectangular panorama.

Two modes are supported:

* ``real_calibration`` uses the four measured Kalibr omni-radtan models.
* ``ideal_horizontal_90`` ignores calibration and uses four colocated ideal
  cameras whose optical axes are exactly 90 degrees apart in the base frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="M20-Piper four-camera radtan GUI probe.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-Teleop-History-v0")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--equirect_width", type=int, default=1280)
parser.add_argument("--equirect_height", type=int, default=640)
parser.add_argument("--mount_pos", type=float, nargs=3, default=(-0.44, 0.0, 0.34))
parser.add_argument("--calibration_width", type=int, default=1920)
parser.add_argument("--calibration_height", type=int, default=1200)
parser.add_argument("--stitch_distance", type=float, default=4.0)
parser.add_argument("--blend_sigma_deg", type=float, default=24.0)
parser.add_argument(
    "--rig_mode",
    choices=("real_calibration", "ideal_horizontal_90"),
    default="ideal_horizontal_90",
    help="Use measured intrinsics or an ideal four-camera fisheye rig.",
)
parser.add_argument(
    "--ideal_fov_deg",
    type=float,
    default=200.0,
    help="Shared circular fisheye field of view for ideal modes.",
)
parser.add_argument(
    "--native_projection",
    choices=("ftheta",),
    default="ftheta",
    help="Native RTX projection. Real mode fits omni-radtan; ideal mode uses shared F-theta.",
)
parser.add_argument(
    "--ftheta_max_fov_deg",
    type=float,
    default=200.0,
    help="F-theta field of view used to fit each measured omni-radtan camera.",
)
parser.add_argument("--vx", type=float, default=0.20)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.10)
parser.add_argument("--measure_frames", type=int, default=20)
parser.add_argument("--warmup_frames", type=int, default=5)
parser.add_argument("--save_dir", type=str, default="outputs/four_fisheye_m20_piper")
parser.add_argument("--exit_after_measure", action="store_true")
parser.add_argument("--validate_follow", action="store_true")
parser.add_argument("--render_interval", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

if args_cli.measure_frames < 1:
    parser.error("--measure_frames must be >= 1")
if args_cli.warmup_frames < 0:
    parser.error("--warmup_frames must be >= 0")
if args_cli.stitch_distance <= 0.0:
    parser.error("--stitch_distance must be positive")
if not 0.0 < args_cli.ideal_fov_deg <= 220.0:
    parser.error("--ideal_fov_deg must be in (0, 220]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import omni.ui as ui  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import rl_training.tasks  # noqa: F401, E402
from fisheye_equirect import (  # noqa: E402
    build_four_fisheye_equirect_maps,
    build_four_ideal_fisheye_equirect_maps,
    camera_chain_poses,
    fit_ftheta_polynomial_from_omni,
    remap_four_fisheye_to_equirect,
)


enable_extension("isaacsim.sensors.camera")


CAMERA_TRANSFORMS = [
    np.array(
        [
            [0.04116687897, -0.0276385374, 0.9987699431, 0.05021831211],
            [0.1665883759, 0.985815193, 0.02041367655, 0.006245770687],
            [-0.9851667885, 0.1655430954, 0.04518719379, -0.04919725701],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [-0.03811326474, -0.1609567999, 0.9862252722, 0.05323725412],
            [-0.07300694028, 0.9847533198, 0.1578951732, -0.001858979024],
            [-0.9966029128, -0.06598338901, -0.04928312746, -0.04772791948],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            [-0.009392638788, 0.06358064125, 0.9979325029, 0.04468950362],
            [-0.06540657735, 0.9958003032, -0.06406040689, -0.0009904701698],
            [-0.9978144908, -0.0658730457, -0.005194600423, -0.05213309627],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
]

CAMERA_INTRINSICS = [
    (1468.523182, 1627.102873, 946.2186359, 664.5179211),
    (1415.365267, 1570.675126, 973.476261, 649.2967091),
    (1365.697685, 1512.373764, 960.2479936, 660.9491504),
    (1486.558502, 1633.116365, 938.1931787, 620.1732316),
]

CAMERA_DISTORTIONS = [
    np.array([1.399638243, -0.3102268723, 0.07304549906, 0.004086048883, 0.005162579018]),
    np.array([1.337249251, -0.3059052059, 0.05776048608, 0.002653574904, -0.002883056306]),
    np.array([1.181656886, -0.367289844, 0.09820716037, 0.00395071916, -0.001051910866]),
    np.array([1.391238099, -0.3286616208, 0.05390951061, 0.01292611884, 0.001809468783]),
]


@dataclass
class TimingSeries:
    values: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        self.values.append(float(value_ms))

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {}
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
            "samples": int(values.size),
        }


def _rotation_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _make_fisheye_camera(
    *,
    prim_path: str,
    position: np.ndarray,
    rotation_optical_to_base: np.ndarray,
    optical_center: tuple[float, float],
    ftheta_polynomial: np.ndarray,
    ftheta_max_fov: float,
) -> CameraCfg:
    # USD camera axes are OpenGL: +X right, +Y up, -Z forward.  The supplied
    # camera frame is OpenCV/ROS: +X right, +Y down, +Z forward.
    optical_to_opengl = np.diag([1.0, -1.0, -1.0])
    local_rotation = rotation_optical_to_base @ optical_to_opengl
    cx, cy = optical_center
    coefficients = np.zeros(6, dtype=np.float64)
    source_coefficients = np.asarray(ftheta_polynomial, dtype=np.float64)
    coefficients[: min(6, len(source_coefficients))] = source_coefficients[:6]
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        debug_vis=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            clipping_range=(0.03, 1000.0),
            focal_length=24.0,
            focus_distance=2.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955,
            fisheye_nominal_width=float(args_cli.width),
            fisheye_nominal_height=float(args_cli.height),
            fisheye_optical_centre_x=float(cx),
            fisheye_optical_centre_y=float(cy),
            fisheye_max_fov=float(ftheta_max_fov),
            fisheye_polynomial_a=float(coefficients[0]),
            fisheye_polynomial_b=float(coefficients[1]),
            fisheye_polynomial_c=float(coefficients[2]),
            fisheye_polynomial_d=float(coefficients[3]),
            fisheye_polynomial_e=float(coefficients[4]),
            fisheye_polynomial_f=float(coefficients[5]),
            lock_camera=True,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in position),
            rot=_rotation_quat_wxyz(local_rotation),
            convention="opengl",
        ),
    )


def _real_ftheta_render_model(
    intrinsic: tuple[float, float, float, float],
    fit: tuple[np.ndarray, float, float, float],
) -> tuple[tuple[float, float], np.ndarray, float]:
    _, _, cx, cy = intrinsic
    scale_x = args_cli.width / float(args_cli.calibration_width)
    scale_y = args_cli.height / float(args_cli.calibration_height)
    radial_scale = 0.5 * (scale_x + scale_y)
    coefficients = np.zeros(6, dtype=np.float64)
    for index, coefficient in enumerate(fit[0][:6]):
        coefficients[index] = coefficient / (radial_scale**index)
    return (cx * scale_x, cy * scale_y), coefficients, fit[1]


def _ideal_ftheta_render_model(
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[tuple[float, float], np.ndarray, float, float]:
    radius = min(width, height) * 0.5
    coefficients = np.zeros(6, dtype=np.float64)
    coefficients[1] = np.deg2rad(fov_deg * 0.5) / radius
    return (width * 0.5, height * 0.5), coefficients, fov_deg, radius


# Camera optical axes are OpenCV/ROS style: +X right, +Y down, +Z forward.
# This rotation maps the forward-looking camera into the base frame.
R_BASE_CAM0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _horizontal_ring_rotations() -> list[np.ndarray]:
    """Return four camera-to-base rotations at 0, 90, 180, and 270 degrees."""
    rotations = []
    for index in range(4):
        yaw = math.radians(90.0 * index)
        c, s = math.cos(yaw), math.sin(yaw)
        yaw_z = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotations.append(yaw_z @ R_BASE_CAM0)
    return rotations


def _sensor_rgb(sensor) -> np.ndarray:
    rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
    return _as_rgb(rgb)


def _as_rgb(data: object) -> np.ndarray:
    rgb = np.asarray(data)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"unexpected RGB shape: {rgb.shape}")
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(rgb.astype(np.uint8))


class EquirectWindow:
    def __init__(self, width: int, height: int, mode: str):
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            f"M20-Piper Four-Camera Panorama - {mode}",
            width=width + 24,
            height=height + 52,
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"{mode} -> {width}x{height} equirectangular")
                ui.ImageWithProvider(self._provider, width=width, height=height)

    def update(self, rgb: np.ndarray) -> float:
        height, width = rgb.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        t0 = time.perf_counter()
        self._provider.set_bytes_data(rgba.reshape(-1).tolist(), [width, height])
        return (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        self._window.visible = False


def _measure(
    env,
    sensors,
    actions: torch.Tensor,
    maps,
    weights,
    panorama_window: EquirectWindow | None,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> tuple[dict[str, dict[str, float]], list[np.ndarray], np.ndarray]:
    step_times = TimingSeries()
    readback_times = TimingSeries()
    stitch_times = TimingSeries()
    ui_times = TimingSeries()
    loop_times = TimingSeries()
    last_images = None
    last_equirect = None

    def one_frame():
        t0 = time.perf_counter()
        env.step(actions)
        t1 = time.perf_counter()
        images = [_sensor_rgb(sensor) for sensor in sensors]
        t2 = time.perf_counter()
        equirect = remap_four_fisheye_to_equirect(images, maps, weights)
        t3 = time.perf_counter()
        ui_ms = panorama_window.update(equirect) if panorama_window is not None else 0.0
        t4 = time.perf_counter()
        return images, equirect, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, ui_ms, (
            t4 - t0
        ) * 1000.0

    for _ in range(warmup_frames):
        one_frame()

    for frame_idx in range(measure_frames):
        images, equirect, step_ms, readback_ms, stitch_ms, ui_ms, loop_ms = one_frame()
        step_times.add(step_ms)
        readback_times.add(readback_ms)
        stitch_times.add(stitch_ms)
        if panorama_window is not None:
            ui_times.add(ui_ms)
        loop_times.add(loop_ms)
        last_images = images
        last_equirect = equirect
        if (frame_idx + 1) % 10 == 0:
            print(
                f"[four-fisheye] frame {frame_idx + 1}/{measure_frames}: "
                f"step={step_ms:.1f} ms stitch={stitch_ms:.1f} ms ui={ui_ms:.1f} ms",
                flush=True,
            )

    if last_images is None or last_equirect is None:
        raise RuntimeError("No four-camera frames were captured")
    return (
        {
            "motion_step_ms": step_times.summary(),
            "readback_four_ms": readback_times.summary(),
            "four_stitch_ms": stitch_times.summary(),
            "panorama_ui_ms": ui_times.summary(),
            "loop_ms": loop_times.summary(),
        },
        last_images,
        last_equirect,
    )


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\n[four-fisheye + panorama]")
    for name, stats in summary.items():
        if not stats:
            continue
        print(
            f"  {name:22s} mean={stats['mean_ms']:8.3f} ms  "
            f"p50={stats['p50_ms']:8.3f} ms  p95={stats['p95_ms']:8.3f} ms  n={stats['samples']}"
        )


def _validate_follow(env, robot, sensor, actions: torch.Tensor) -> None:
    def camera_world_position() -> np.ndarray:
        position = sensor._view.get_world_poses(usd=True)[0][0]
        if hasattr(position, "detach"):
            position = position.detach().cpu().numpy()
        return np.asarray(position, dtype=np.float64)

    root_pose = robot.data.root_pose_w.clone()
    root_before = robot.data.root_pos_w[0].detach().cpu().numpy()
    camera_before = camera_world_position()
    image_before = _sensor_rgb(sensor)
    moved = root_pose.clone()
    moved[0, 0] += 0.15
    moved[0, 1] += 0.05
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    robot.write_root_pose_to_sim(moved, env_ids=env_ids)
    env.step(actions)
    root_after = robot.data.root_pos_w[0].detach().cpu().numpy()
    camera_after = camera_world_position()
    image_after = _sensor_rgb(sensor)
    print(
        f"[four-fisheye] follow test: root_delta={np.round(root_after - root_before, 4)} "
        f"camera_delta={np.round(camera_after - camera_before, 4)} "
        f"image_mean_abs_diff={np.mean(np.abs(image_after.astype(np.int16) - image_before.astype(np.int16))):.3f}",
        flush=True,
    )
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.step(actions)


def main() -> None:
    save_dir = Path(args_cli.save_dir).resolve() / args_cli.rig_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.rig_mode == "ideal_horizontal_90":
        rotations_base = _horizontal_ring_rotations()
        rotations_cam0 = [R_BASE_CAM0.T @ rotation for rotation in rotations_base]
        translations_cam0 = [np.zeros(3, dtype=np.float64) for _ in range(4)]
    else:
        rotations_cam0, translations_cam0 = camera_chain_poses(CAMERA_TRANSFORMS)
    if args_cli.rig_mode == "real_calibration":
        ftheta_fits = [
            fit_ftheta_polynomial_from_omni(
                intrinsic=intrinsic,
                distortion=distortion,
                calibration_width=args_cli.calibration_width,
                calibration_height=args_cli.calibration_height,
                max_theta_deg=args_cli.ftheta_max_fov_deg * 0.5,
            )
            for intrinsic, distortion in zip(CAMERA_INTRINSICS, CAMERA_DISTORTIONS)
        ]
        camera_render_models = [
            _real_ftheta_render_model(intrinsic, fit)
            for intrinsic, fit in zip(CAMERA_INTRINSICS, ftheta_fits)
        ]
    else:
        ideal_center, ideal_coefficients, ideal_fov, ideal_radius = _ideal_ftheta_render_model(
            args_cli.width,
            args_cli.height,
            args_cli.ideal_fov_deg,
        )
        camera_render_models = [
            (ideal_center, ideal_coefficients, ideal_fov) for _ in range(4)
        ]
        ftheta_fits = [
            (ideal_coefficients, ideal_fov, ideal_radius, 0.0) for _ in range(4)
        ]
    mount_pos = np.asarray(args_cli.mount_pos, dtype=np.float64)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.render_interval = args_cli.render_interval

    camera_scene_keys = []
    for index, (rotation_cam0, translation_cam0) in enumerate(zip(rotations_cam0, translations_cam0)):
        scene_key = f"four_fisheye_cam{index}"
        camera_scene_keys.append(scene_key)
        optical_center, coefficients, max_fov = camera_render_models[index]
        setattr(
            env_cfg.scene,
            scene_key,
            _make_fisheye_camera(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link/M20FourCam{index}",
                position=mount_pos + R_BASE_CAM0 @ translation_cam0,
                rotation_optical_to_base=R_BASE_CAM0 @ rotation_cam0,
                optical_center=optical_center,
                ftheta_polynomial=coefficients,
                ftheta_max_fov=max_fov,
            ),
        )

    print("[four-fisheye] creating environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    print("[four-fisheye] environment ready", flush=True)
    env.sim.set_camera_view(eye=(2.8, -2.8, 1.8), target=(0.0, 0.0, 0.6))

    sensors = [env.scene[scene_key] for scene_key in camera_scene_keys]
    camera_paths = [str(sensor._view.prim_paths[0]) for sensor in sensors]

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, 0] = args_cli.vx
    actions[:, 1] = args_cli.vy
    actions[:, 2] = args_cli.wz

    if args_cli.validate_follow:
        _validate_follow(env, env.scene["robot"], sensors[0], actions)

    viewport_windows = []
    if not args_cli.headless:
        from isaacsim.core.utils.viewports import create_viewport_for_camera

        positions = [(40, 40), (380, 40), (40, 260), (380, 260)]
        for index, path in enumerate(camera_paths):
            viewport_windows.append(
                create_viewport_for_camera(
                    f"M20 Four Fisheye cam{index}",
                    path,
                    width=320,
                    height=200,
                    position_x=positions[index][0],
                    position_y=positions[index][1],
                )
            )

    if args_cli.rig_mode == "real_calibration":
        maps, weights, covered = build_four_fisheye_equirect_maps(
            rotations=rotations_cam0,
            translations=translations_cam0,
            intrinsics=CAMERA_INTRINSICS,
            distortions=CAMERA_DISTORTIONS,
            calibration_width=args_cli.calibration_width,
            calibration_height=args_cli.calibration_height,
            render_width=args_cli.width,
            render_height=args_cli.height,
            equirect_width=args_cli.equirect_width,
            equirect_height=args_cli.equirect_height,
            stitch_distance=args_cli.stitch_distance,
            blend_sigma_deg=args_cli.blend_sigma_deg,
        )
    else:
        maps, weights, covered = build_four_ideal_fisheye_equirect_maps(
            rotations=rotations_cam0,
            translations=translations_cam0,
            fov_deg=args_cli.ideal_fov_deg,
            render_width=args_cli.width,
            render_height=args_cli.height,
            equirect_width=args_cli.equirect_width,
            equirect_height=args_cli.equirect_height,
            stitch_distance=args_cli.stitch_distance,
            blend_sigma_deg=args_cli.blend_sigma_deg,
        )

    print(f"[four-fisheye] task          : {args_cli.task}", flush=True)
    print(f"[four-fisheye] rig mode      : {args_cli.rig_mode}", flush=True)
    print(f"[four-fisheye] motion command: vx={args_cli.vx:.3f} wz={args_cli.wz:.3f}", flush=True)
    print(f"[four-fisheye] mount_pos     : {tuple(args_cli.mount_pos)}", flush=True)
    print(f"[four-fisheye] render        : {args_cli.width}x{args_cli.height} each", flush=True)
    if args_cli.rig_mode == "real_calibration":
        print("[four-fisheye] camera model  : measured omni-radtan per camera", flush=True)
    else:
        print(
            "[four-fisheye] camera model  : shared ideal "
            f"{args_cli.ideal_fov_deg:.1f} deg fisheye, horizontal 90 deg ring, colocated",
            flush=True,
        )
    print(f"[four-fisheye] RTX projection: {args_cli.native_projection}", flush=True)
    for index, fit in enumerate(ftheta_fits):
        print(
            f"[four-fisheye] ftheta cam{index}   : max_fov={fit[1]:.2f} deg, "
            f"max_radius={fit[2]:.1f} px, rms={fit[3]:.3f} deg",
            flush=True,
        )
    print(
        f"[four-fisheye] equirect      : {args_cli.equirect_width}x{args_cli.equirect_height}, "
        f"coverage={100.0 * covered.mean():.1f}%",
        flush=True,
    )
    for index, path in enumerate(camera_paths):
        print(f"[four-fisheye] cam{index}          : {path}", flush=True)

    panorama_window = None
    if not args_cli.headless:
        panorama_window = EquirectWindow(
            args_cli.equirect_width,
            args_cli.equirect_height,
            args_cli.rig_mode,
        )

    summary, images, equirect = _measure(
        env,
        sensors,
        actions,
        maps,
        weights,
        panorama_window,
        warmup_frames=args_cli.warmup_frames,
        measure_frames=args_cli.measure_frames,
    )
    _print_summary(summary)

    for index, image in enumerate(images):
        cv2.imwrite(str(save_dir / f"cam{index}_rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "equirect_rgb.png"), cv2.cvtColor(equirect, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "equirect_valid_mask.png"), (covered.astype(np.uint8) * 255))
    thumbnails = [
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (args_cli.width // 2, args_cli.height // 2))
        for image in images
    ]
    view_grid = cv2.vconcat(
        [
            cv2.hconcat(thumbnails[:2]),
            cv2.hconcat(thumbnails[2:]),
        ]
    )
    cv2.imwrite(str(save_dir / "four_view_grid.png"), view_grid)

    report = {
        "metadata": {
            "task": args_cli.task,
            "rig_mode": args_cli.rig_mode,
            "motion_command": [args_cli.vx, args_cli.vy, args_cli.wz],
            "mount_pos": list(args_cli.mount_pos),
            "calibration_resolution": [args_cli.calibration_width, args_cli.calibration_height],
            "render_resolution": [args_cli.width, args_cli.height],
            "equirect_resolution": [args_cli.equirect_width, args_cli.equirect_height],
            "equirect_coverage_fraction": float(covered.mean()),
            "stitch_distance": args_cli.stitch_distance,
            "blend_sigma_deg": args_cli.blend_sigma_deg,
            "camera_model": (
                "measured-omni-radtan"
                if args_cli.rig_mode == "real_calibration"
                else "ideal-horizontal-90-colocated-equidistant-fisheye"
            ),
            "shared_intrinsics": args_cli.rig_mode != "real_calibration",
            "ideal_fov_deg": (
                args_cli.ideal_fov_deg
                if args_cli.rig_mode != "real_calibration"
                else None
            ),
            "camera_yaw_step_deg": 90.0 if args_cli.rig_mode == "ideal_horizontal_90" else None,
            "camera_baseline_m": 0.0 if args_cli.rig_mode == "ideal_horizontal_90" else None,
            "omni_xi": (
                [float(distortion[0]) for distortion in CAMERA_DISTORTIONS]
                if args_cli.rig_mode == "real_calibration"
                else None
            ),
            "radtan": (
                [
                    [float(value) for value in distortion[1:5]]
                    for distortion in CAMERA_DISTORTIONS
                ]
                if args_cli.rig_mode == "real_calibration"
                else None
            ),
            "native_projection": args_cli.native_projection,
            "ftheta_max_fov_deg": args_cli.ftheta_max_fov_deg,
            "ftheta_fits": [
                {
                    "coefficients": [float(value) for value in fit[0]],
                    "max_fov_deg": float(fit[1]),
                    "max_radius_px": float(fit[2]),
                    "rms_error_deg": float(fit[3]),
                }
                for fit in ftheta_fits
            ],
            "headless": args_cli.headless,
        },
        "timing": summary,
    }
    (save_dir / "timings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[four-fisheye] saved images/report to {save_dir}")

    if args_cli.headless or args_cli.exit_after_measure:
        print("[four-fisheye] exiting after measurement", flush=True)
        os._exit(0)

    print("[four-fisheye] real-time mode active; cameras follow base_link.")
    while simulation_app.is_running():
        env.step(actions)
        images = [_sensor_rgb(sensor) for sensor in sensors]
        equirect = remap_four_fisheye_to_equirect(images, maps, weights)
        if panorama_window is not None:
            panorama_window.update(equirect)


try:
    main()
finally:
    try:
        if "panorama_window" in locals() and panorama_window is not None:
            panorama_window.close()
    except Exception:
        pass
    try:
        if "viewport_windows" in locals():
            for viewport_window in viewport_windows:
                viewport_window.destroy()
    except Exception:
        pass
    try:
        if "env" in locals():
            env.close()
    except Exception:
        pass
    simulation_app.close()
