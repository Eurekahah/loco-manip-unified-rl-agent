#!/usr/bin/env python3
"""Render and save one low-shelf scene-camera and wrist-camera sample."""

from __future__ import annotations

import argparse
import os

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test low-shelf RGB-D cameras.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-LowShelf-Collect-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--output_dir", type=str, default="recordings/camera_smoke")
parser.add_argument("--camera_width", type=int, default=224)
parser.add_argument("--camera_height", type=int, default=224)
parser.add_argument("--record_depth", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--mark_success", action="store_true", help="Test the manual-success recording path before exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (  # noqa: E402
    make_top_camera as make_low_shelf_top_camera,
    make_workcell_camera as make_low_shelf_workcell_camera,
    make_wrist_camera as make_low_shelf_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.two_cylinder_place.scene_cfg import (  # noqa: E402
    make_two_cylinder_top_camera,
    make_two_cylinder_workcell_camera,
    make_two_cylinder_wrist_camera,
)
import rl_training.tasks  # noqa: F401, E402


def _save_depth(depth: np.ndarray, path: str) -> None:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(finite, 2.0))
        hi = float(np.percentile(finite, 98.0))
        scale = max(hi - lo, 1e-6)
        preview = np.clip((depth - lo) / scale, 0.0, 1.0)
        preview = (preview * 255.0).astype(np.uint8)
    cv2.imwrite(path, preview)


def main() -> None:
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    is_two_cylinder = "TwoCylinder" in args_cli.task
    make_workcell_camera = (
        make_two_cylinder_workcell_camera if is_two_cylinder else make_low_shelf_workcell_camera
    )
    make_wrist_camera = (
        make_two_cylinder_wrist_camera if is_two_cylinder else make_low_shelf_wrist_camera
    )
    make_top_camera = make_two_cylinder_top_camera if is_two_cylinder else make_low_shelf_top_camera
    env_cfg.scene.workcell_camera = make_workcell_camera(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        with_depth=args_cli.record_depth,
    )
    env_cfg.scene.wrist_camera = make_wrist_camera(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        with_depth=args_cli.record_depth,
    )
    env_cfg.scene.top_camera = make_top_camera(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        with_depth=args_cli.record_depth,
    )
    camera_names = ("top_camera", "wrist_camera", "workcell_camera")
    env_cfg.recorders.record_hdf5.camera_names = camera_names
    env_cfg.recorders.record_hdf5.output_path = os.path.join(args_cli.output_dir, "camera_smoke.hdf5")
    env_cfg.recorders.record_hdf5.overwrite = True
    env_cfg.recorders.record_hdf5.source = "camera_smoke"

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args_cli.steps):
        env.step(actions)

    for name in camera_names:
        sensor = env.scene.sensors[name]
        data = sensor.data
        rgb = data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
        rgb_path = os.path.join(args_cli.output_dir, f"{name}_rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[camera-smoke] {name}: rgb={rgb.shape}")
        print(f"[camera-smoke] {name}: rgb_path={os.path.abspath(rgb_path)}")
        print(f"[camera-smoke] {name}: K=\n{data.intrinsic_matrices[0].detach().cpu().numpy()}")
        print(f"[camera-smoke] {name}: pos_w={data.pos_w[0].detach().cpu().numpy()}")
        print(f"[camera-smoke] {name}: quat_w_world={data.quat_w_world[0].detach().cpu().numpy()}")
        depth = data.output.get("depth")
        if depth is not None:
            depth_np = depth[0].detach().cpu().numpy().astype(np.float32)
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            depth_path = os.path.join(args_cli.output_dir, f"{name}_depth.png")
            _save_depth(depth_np, depth_path)
            print(f"[camera-smoke] {name}: depth={depth_np.shape}")
            print(f"[camera-smoke] {name}: depth_path={os.path.abspath(depth_path)}")
            finite = depth_np[np.isfinite(depth_np)]
            if finite.size:
                print(
                    f"[camera-smoke] {name}: depth_min={finite.min():.4f} depth_max={finite.max():.4f}"
                )

    recorder = getattr(env, "low_shelf_hdf5_recorder", None)
    if recorder is not None:
        if args_cli.mark_success:
            recorder.mark_outcome(True)
            env.reset()
        recorder.close(recorder.output_path)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
