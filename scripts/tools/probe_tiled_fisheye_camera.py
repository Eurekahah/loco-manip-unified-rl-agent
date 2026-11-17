"""Probe whether Isaac Lab 5.1 TiledCamera accepts a fisheye spawn config."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe TiledCamera with an Isaac fisheye spawn.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-Teleop-History-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=200)
parser.add_argument("--fov-deg", type=float, default=200.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import rl_training.tasks  # noqa: F401, E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    width, height = args_cli.width, args_cli.height
    env_cfg.scene.tiled_fisheye = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/TiledFisheye",
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
            fisheye_max_fov=float(args_cli.fov_deg),
            fisheye_polynomial_a=0.0,
            fisheye_polynomial_b=math.radians(args_cli.fov_deg * 0.5)
            / (min(width, height) * 0.5),
            fisheye_polynomial_c=0.0,
            fisheye_polynomial_d=0.0,
            fisheye_polynomial_e=0.0,
            fisheye_polynomial_f=0.0,
            lock_camera=True,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.15),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="opengl",
        ),
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        env.reset()
        actions = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )
        for _ in range(3):
            env.step(actions)
        sensor = env.scene["tiled_fisheye"]
        rgb = sensor.data.output["rgb"]
        print(f"[tiled-fisheye] sensor={sensor}")
        print(
            f"[tiled-fisheye] rgb_shape={tuple(rgb.shape)} dtype={rgb.dtype} "
            f"frame={sensor.frame.detach().cpu().tolist()} "
            f"render_products={sensor.render_product_paths}"
        )
        if tuple(rgb.shape) != (args_cli.num_envs, height, width, 3):
            raise RuntimeError(f"Unexpected tiled fisheye RGB shape: {tuple(rgb.shape)}")
        print("[tiled-fisheye] PASS")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
