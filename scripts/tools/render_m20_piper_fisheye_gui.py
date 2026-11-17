#!/usr/bin/env python3
"""Render an Insta360-style dual-fisheye rig on the M20-Piper.

Both fisheye cameras are created as Isaac Lab camera sensors under
``Robot/base_link``.  Parenting a sensor to a rigid link is the supported way
to make it follow the robot without invalidating the PhysX articulation view at
runtime.

GUI mode opens:

* a native front fisheye viewport,
* a native back fisheye viewport,
* an omni.ui window with the stitched 360-degree equirectangular panorama.
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


parser = argparse.ArgumentParser(description="M20-Piper Insta360-style fisheye GUI probe.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-Teleop-History-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--width", type=int, default=960, help="Fisheye render width in pixels.")
parser.add_argument("--height", type=int, default=960, help="Fisheye render height in pixels.")
parser.add_argument("--fov_deg", type=float, default=200.0, help="Full field of view of each fisheye.")
parser.add_argument("--equirect_width", type=int, default=960)
parser.add_argument("--equirect_height", type=int, default=480)
parser.add_argument(
    "--mount_pos",
    type=float,
    nargs=3,
    default=(-0.44, 0.0, 0.34),
    metavar=("X", "Y", "Z"),
    help="Rig origin in the base_link frame (metres).",
)
parser.add_argument(
    "--pitch_deg",
    type=float,
    default=0.0,
    help="Downward pitch of both lenses. Keep 0 for full spherical coverage.",
)
parser.add_argument(
    "--lens_offset",
    type=float,
    default=0.02,
    help="Half distance between front/back lens optical centres (metres).",
)
parser.add_argument("--stitch_distance", type=float, default=4.0)
parser.add_argument("--blend_deg", type=float, default=4.0)
parser.add_argument("--vx", type=float, default=0.25, help="Forward velocity command for the low-level policy.")
parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command for the low-level policy.")
parser.add_argument("--wz", type=float, default=0.15, help="Yaw-rate command for the low-level policy.")
parser.add_argument("--measure_frames", type=int, default=60)
parser.add_argument("--warmup_frames", type=int, default=10)
parser.add_argument("--save_dir", type=str, default="outputs/fisheye_m20_piper")
parser.add_argument("--exit_after_measure", action="store_true")
parser.add_argument("--validate_follow", action="store_true")
parser.add_argument("--render_interval", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    parser.error("--num_envs must be 1 for this camera probe")
if args_cli.measure_frames < 1:
    parser.error("--measure_frames must be >= 1")
if args_cli.warmup_frames < 0:
    parser.error("--warmup_frames must be >= 0")
if args_cli.render_interval < 1:
    parser.error("--render_interval must be >= 1")
if args_cli.lens_offset < 0.0:
    parser.error("--lens_offset must be >= 0")
if args_cli.stitch_distance <= 0.0:
    parser.error("--stitch_distance must be > 0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require the simulator to be running.
import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import rl_training.tasks  # noqa: F401, E402
from fisheye_equirect import (  # noqa: E402
    build_dual_fisheye_equirect_maps,
    dual_camera_rotations,
    remap_dual_fisheye_to_equirect,
)


enable_extension("isaacsim.sensors.camera")


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


def _world_position(prim: Usd.Prim) -> np.ndarray:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(matrix, dtype=np.float64).T[:3, 3]


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
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    width: int,
    height: int,
    fov_deg: float,
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
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
            fisheye_nominal_width=float(width),
            fisheye_nominal_height=float(height),
            fisheye_optical_centre_x=width * 0.5,
            fisheye_optical_centre_y=height * 0.5,
            fisheye_max_fov=float(fov_deg),
            fisheye_polynomial_a=0.0,
            fisheye_polynomial_b=math.radians(fov_deg * 0.5) / (min(width, height) * 0.5),
            fisheye_polynomial_c=0.0,
            fisheye_polynomial_d=0.0,
            fisheye_polynomial_e=0.0,
            fisheye_polynomial_f=0.0,
            lock_camera=True,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )


def _sensor_rgb(sensor) -> np.ndarray:
    rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _blue_centroid(rgb: np.ndarray) -> np.ndarray:
    """Find the blue debug sphere on the robot as a visual tracking marker."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    mask = (b > 80) & (b > r + 35) & (b > g + 20)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


class EquirectWindow:
    def __init__(self, width: int, height: int):
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window("M20-Piper Insta360 Panorama", width=width + 24, height=height + 52)
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(
                    f"Front/back {args_cli.fov_deg:.0f} deg fisheye -> "
                    f"{width}x{height} equirectangular"
                )
                ui.ImageWithProvider(self._provider, width=width, height=height)

    def update(self, rgb: np.ndarray) -> float:
        height, width = rgb.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        t0 = time.perf_counter()
        # Kit 5.1 expects a Python list here. Passing bytes can block on Windows.
        self._provider.set_bytes_data(rgba.reshape(-1).tolist(), [width, height])
        return (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        self._window.visible = False


def _measure_fisheye_pair(
    env,
    front_sensor,
    back_sensor,
    actions: torch.Tensor,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    step_times = TimingSeries()
    readback_times = TimingSeries()
    last_front = None
    last_back = None

    for _ in range(warmup_frames):
        env.step(actions)
        _sensor_rgb(front_sensor)
        _sensor_rgb(back_sensor)

    for _ in range(measure_frames):
        t0 = time.perf_counter()
        env.step(actions)
        t1 = time.perf_counter()
        front = _sensor_rgb(front_sensor)
        back = _sensor_rgb(back_sensor)
        t2 = time.perf_counter()
        step_times.add((t1 - t0) * 1000.0)
        readback_times.add((t2 - t1) * 1000.0)
        last_front = front
        last_back = back

    if last_front is None or last_back is None:
        raise RuntimeError("No fisheye frames were captured")
    return {
        "motion_step_ms": step_times.summary(),
        "readback_both_ms": readback_times.summary(),
    }, last_front, last_back


def _measure_panorama(
    env,
    front_sensor,
    back_sensor,
    maps,
    panorama_window: EquirectWindow | None,
    actions: torch.Tensor,
    *,
    warmup_frames: int,
    measure_frames: int,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    step_times = TimingSeries()
    readback_times = TimingSeries()
    stitch_times = TimingSeries()
    ui_times = TimingSeries()
    loop_times = TimingSeries()
    last_front = None
    last_back = None
    last_equirect = None

    def one_frame():
        t0 = time.perf_counter()
        env.step(actions)
        t1 = time.perf_counter()
        front = _sensor_rgb(front_sensor)
        back = _sensor_rgb(back_sensor)
        t2 = time.perf_counter()
        equirect = remap_dual_fisheye_to_equirect(front, back, maps)
        t3 = time.perf_counter()
        ui_ms = panorama_window.update(equirect) if panorama_window is not None else 0.0
        t4 = time.perf_counter()
        return (
            front,
            back,
            equirect,
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t3 - t2) * 1000.0,
            ui_ms,
            (t4 - t0) * 1000.0,
        )

    for _ in range(warmup_frames):
        one_frame()

    for frame_idx in range(measure_frames):
        front, back, equirect, step_ms, readback_ms, stitch_ms, ui_ms, loop_ms = one_frame()
        step_times.add(step_ms)
        readback_times.add(readback_ms)
        stitch_times.add(stitch_ms)
        if panorama_window is not None:
            ui_times.add(ui_ms)
        loop_times.add(loop_ms)
        last_front = front
        last_back = back
        last_equirect = equirect
        if (frame_idx + 1) % 10 == 0:
            print(
                f"[fisheye-probe] panorama frame {frame_idx + 1}/{measure_frames}: "
                f"step={step_ms:.1f} ms stitch={stitch_ms:.1f} ms ui={ui_ms:.1f} ms",
                flush=True,
            )

    if last_front is None or last_back is None or last_equirect is None:
        raise RuntimeError("No panorama frames were captured")
    return (
        {
            "motion_step_ms": step_times.summary(),
            "readback_both_ms": readback_times.summary(),
            "dual_stitch_ms": stitch_times.summary(),
            "panorama_ui_ms": ui_times.summary(),
            "loop_ms": loop_times.summary(),
        },
        last_front,
        last_back,
        last_equirect,
    )


def _print_summary(label: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n[{label}]")
    for name, stats in summary.items():
        if not stats:
            continue
        print(
            f"  {name:22s} mean={stats['mean_ms']:8.3f} ms  "
            f"p50={stats['p50_ms']:8.3f} ms  p95={stats['p95_ms']:8.3f} ms  "
            f"min={stats['min_ms']:8.3f}  max={stats['max_ms']:8.3f}  n={stats['samples']}"
        )


def _validate_camera_follow(env, robot, front_sensor, zero_actions: torch.Tensor) -> None:
    """Move the root and verify that the camera sensor pose follows it."""
    root_pose = robot.data.root_pose_w.clone()
    root_before = robot.data.root_pos_w[0].detach().cpu().numpy()
    image_before = _sensor_rgb(front_sensor)
    blue_before = _blue_centroid(image_before)

    moved_pose = root_pose.clone()
    moved_pose[0, 0] += 0.15
    moved_pose[0, 1] += 0.05
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    robot.write_root_pose_to_sim(moved_pose, env_ids=env_ids)
    env.step(zero_actions)
    root_after = robot.data.root_pos_w[0].detach().cpu().numpy()
    image_after = _sensor_rgb(front_sensor)
    blue_after = _blue_centroid(image_after)
    print(
        "[fisheye-probe] follow test: "
        f"root_delta={np.round(root_after - root_before, 4)} "
        f"blue_centroid_delta={np.round(blue_after - blue_before, 2)}",
        flush=True,
    )
    save_dir = Path(args_cli.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / "follow_before.png"), cv2.cvtColor(image_before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "follow_after.png"), cv2.cvtColor(image_after, cv2.COLOR_RGB2BGR))

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.step(zero_actions)


def main() -> None:
    save_dir = Path(args_cli.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.render_interval = args_cli.render_interval

    front_rotation, back_rotation = dual_camera_rotations(args_cli.pitch_deg)
    front_rot = _rotation_quat_wxyz(front_rotation)
    back_rot = _rotation_quat_wxyz(back_rotation)
    front_pos = (
        args_cli.mount_pos[0] + args_cli.lens_offset,
        args_cli.mount_pos[1],
        args_cli.mount_pos[2],
    )
    back_pos = (
        args_cli.mount_pos[0] - args_cli.lens_offset,
        args_cli.mount_pos[1],
        args_cli.mount_pos[2],
    )
    env_cfg.scene.front_fisheye = _make_fisheye_camera(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/M20PanoFront",
        pos=front_pos,
        rot=front_rot,
        width=args_cli.width,
        height=args_cli.height,
        fov_deg=args_cli.fov_deg,
    )
    env_cfg.scene.back_fisheye = _make_fisheye_camera(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/M20PanoBack",
        pos=back_pos,
        rot=back_rot,
        width=args_cli.width,
        height=args_cli.height,
        fov_deg=args_cli.fov_deg,
    )

    print("[fisheye-probe] creating environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[fisheye-probe] resetting environment", flush=True)
    env.reset()
    print("[fisheye-probe] environment ready", flush=True)
    env.sim.set_camera_view(eye=(2.8, -2.8, 1.8), target=(0.0, 0.0, 0.6))

    front_sensor = env.scene["front_fisheye"]
    back_sensor = env.scene["back_fisheye"]
    front_path = str(front_sensor._view.prim_paths[0])
    back_path = str(back_sensor._view.prim_paths[0])

    if args_cli.validate_follow:
        zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        _validate_camera_follow(env, env.scene["robot"], front_sensor, zero_actions)

    viewport_windows = []
    if not args_cli.headless:
        from isaacsim.core.utils.viewports import create_viewport_for_camera

        viewport_size = min(480, args_cli.width)
        viewport_windows = [
            create_viewport_for_camera(
                "M20 Front Fisheye",
                front_path,
                width=viewport_size,
                height=viewport_size,
                position_x=80,
                position_y=80,
            ),
            create_viewport_for_camera(
                "M20 Back Fisheye",
                back_path,
                width=viewport_size,
                height=viewport_size,
                position_x=80 + viewport_size + 20,
                position_y=80,
            ),
        ]

    maps = build_dual_fisheye_equirect_maps(
        fisheye_width=args_cli.width,
        fisheye_height=args_cli.height,
        fov_deg=args_cli.fov_deg,
        equirect_width=args_cli.equirect_width,
        equirect_height=args_cli.equirect_height,
        pitch_deg=args_cli.pitch_deg,
        lens_offset=args_cli.lens_offset,
        stitch_distance=args_cli.stitch_distance,
        blend_deg=args_cli.blend_deg,
    )
    valid_mask = maps[4]

    print(f"[fisheye-probe] task          : {args_cli.task}", flush=True)
    print(
        f"[fisheye-probe] motion command: vx={args_cli.vx:.3f} vy={args_cli.vy:.3f} wz={args_cli.wz:.3f}",
        flush=True,
    )
    print(f"[fisheye-probe] front camera  : {front_path}", flush=True)
    print(f"[fisheye-probe] back camera   : {back_path}", flush=True)
    print(f"[fisheye-probe] mount_pos     : {tuple(args_cli.mount_pos)} m in base_link", flush=True)
    print(f"[fisheye-probe] lens_offset   : +/- {args_cli.lens_offset:.3f} m", flush=True)
    print(
        f"[fisheye-probe] fisheye       : {args_cli.width}x{args_cli.height}, "
        f"{args_cli.fov_deg:.1f} deg each",
        flush=True,
    )
    print(
        f"[fisheye-probe] equirect      : {args_cli.equirect_width}x{args_cli.equirect_height}, "
        f"coverage={100.0 * valid_mask.mean():.1f}%",
        flush=True,
    )

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, 0] = args_cli.vx
    actions[:, 1] = args_cli.vy
    actions[:, 2] = args_cli.wz
    robot = env.scene["robot"]
    root_pos_start = robot.data.root_pos_w[0].detach().cpu().numpy()
    print("[fisheye-probe] measuring front/back fisheye viewports", flush=True)
    fisheye_summary, front_rgb, back_rgb = _measure_fisheye_pair(
        env,
        front_sensor,
        back_sensor,
        actions,
        warmup_frames=args_cli.warmup_frames,
        measure_frames=args_cli.measure_frames,
    )
    _print_summary("dual fisheye native viewports", fisheye_summary)
    root_pos_mid = robot.data.root_pos_w[0].detach().cpu().numpy()
    print(
        f"[fisheye-probe] low-level motion delta after warmup+measure: "
        f"{np.round(root_pos_mid - root_pos_start, 4)} m",
        flush=True,
    )

    panorama_window = None
    if not args_cli.headless:
        panorama_window = EquirectWindow(args_cli.equirect_width, args_cli.equirect_height)

    print("[fisheye-probe] measuring dual-fisheye panorama", flush=True)
    panorama_summary, front_rgb, back_rgb, equirect_rgb = _measure_panorama(
        env,
        front_sensor,
        back_sensor,
        maps,
        panorama_window,
        actions,
        warmup_frames=args_cli.warmup_frames,
        measure_frames=args_cli.measure_frames,
    )
    _print_summary("dual fisheye + equirectangular panorama", panorama_summary)
    root_pos_end = robot.data.root_pos_w[0].detach().cpu().numpy()

    cv2.imwrite(str(save_dir / "front_fisheye_rgb.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "back_fisheye_rgb.png"), cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "equirect_rgb.png"), cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "equirect_valid_mask.png"), (valid_mask.astype(np.uint8) * 255))

    report = {
        "metadata": {
            "task": args_cli.task,
            "motion_command": [args_cli.vx, args_cli.vy, args_cli.wz],
            "front_camera_path": front_path,
            "back_camera_path": back_path,
            "mount_pos": list(args_cli.mount_pos),
            "pitch_deg": args_cli.pitch_deg,
            "lens_offset": args_cli.lens_offset,
            "stitch_distance": args_cli.stitch_distance,
            "blend_deg": args_cli.blend_deg,
            "fisheye_resolution": [args_cli.width, args_cli.height],
            "fov_deg": args_cli.fov_deg,
            "equirect_resolution": [args_cli.equirect_width, args_cli.equirect_height],
            "equirect_coverage_fraction": float(valid_mask.mean()),
            "headless": args_cli.headless,
            "follow_validated": args_cli.validate_follow,
            "low_level_motion_delta": (root_pos_end - root_pos_start).tolist(),
            "note": "Both camera sensors are children of Robot/base_link and follow it rigidly.",
        },
        "dual_fisheye_native_viewports": fisheye_summary,
        "dual_fisheye_plus_equirectangular_panorama": panorama_summary,
    }
    report_path = save_dir / "timings.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fisheye-probe] saved images/report to {save_dir}")

    if args_cli.headless or args_cli.exit_after_measure:
        print("[fisheye-probe] exiting after measurement", flush=True)
        os._exit(0)

    print("[fisheye-probe] real-time mode active; both cameras follow base_link.")
    while simulation_app.is_running():
        env.step(actions)
        front = _sensor_rgb(front_sensor)
        back = _sensor_rgb(back_sensor)
        equirect = remap_dual_fisheye_to_equirect(front, back, maps)
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
