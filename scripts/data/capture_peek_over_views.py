#!/usr/bin/env python3
"""Capture P3 PeekOver scene and camera screenshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture P3 PeekOver views.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-PeekOver-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="recordings/peek_over_views")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--fisheye_width", type=int, default=480)
parser.add_argument("--fisheye_height", type=int, default=300)
parser.add_argument("--panorama_width", type=int, default=960)
parser.add_argument("--panorama_height", type=int, default=480)
parser.add_argument("--settle_steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.rendering_mode = "quality"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fisheye_teleop import FisheyePanorama, add_fisheye_rig, read_fisheye_images  # noqa: E402

import rl_training.tasks  # noqa: F401, E402
from rl_training.tasks.manager_based.locomotion.highlevel.config.common_task import (  # noqa: E402
    configure_quality_rendering,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.peek_over.scene_cfg import (  # noqa: E402
    make_peek_over_front_camera,
    make_peek_over_top_camera,
    make_peek_over_workcell_camera,
    make_peek_over_wrist_camera,
)


def _save_rgb(sensor, path: str) -> np.ndarray:
    rgb = sensor.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb


def _label_tile(rgb: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    tile = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    cv2.rectangle(tile, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _montage(images: list[tuple[str, np.ndarray]], path: str, columns: int = 2) -> None:
    tile_w = max(image.shape[1] for _, image in images)
    tile_h = max(image.shape[0] for _, image in images)
    tiles = [_label_tile(image, label, (tile_w, tile_h)) for label, image in images]
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(path, np.concatenate(rows, axis=0))


def _capture_standard(env, output_dir: str, prefix: str) -> list[tuple[str, np.ndarray]]:
    captures = [
        ("Top view", "top_camera"),
        ("Workcell view", "workcell_camera"),
        ("Front camera", "front_camera"),
        ("Wrist camera", "wrist_camera"),
    ]
    images: list[tuple[str, np.ndarray]] = []
    for label, sensor_name in captures:
        path = os.path.join(output_dir, f"{prefix}_{sensor_name}.png")
        rgb = _save_rgb(env.scene.sensors[sensor_name], path)
        images.append((label, rgb))
        print(f"[peek-over-views] saved {path}", flush=True)
    _montage(images, os.path.join(output_dir, f"{prefix}_montage.png"))
    return images


def _capture_panorama(env, spec, panorama: FisheyePanorama, output_dir: str, prefix: str) -> None:
    raw_images = read_fisheye_images(env, spec)
    panorama_rgb = panorama.stitch(raw_images)
    panorama_path = os.path.join(output_dir, f"{prefix}_panorama.png")
    cv2.imwrite(panorama_path, cv2.cvtColor(panorama_rgb, cv2.COLOR_RGB2BGR))
    raw_montage = np.concatenate(
        [
            _label_tile(image, f"Fisheye {index}", (args_cli.fisheye_width, args_cli.fisheye_height))
            for index, image in enumerate(raw_images)
        ],
        axis=1,
    )
    raw_path = os.path.join(output_dir, f"{prefix}_fisheye_montage.png")
    cv2.imwrite(raw_path, raw_montage)
    print(f"[peek-over-views] saved {panorama_path}", flush=True)
    print(f"[peek-over-views] saved {raw_path}", flush=True)


def main() -> None:
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    configure_quality_rendering(env_cfg)
    env_cfg.seed = 11
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env_cfg.scene.top_camera = make_peek_over_top_camera(
        width=args_cli.width,
        height=args_cli.height,
        with_depth=False,
    )
    env_cfg.scene.workcell_camera = make_peek_over_workcell_camera(
        width=args_cli.width,
        height=args_cli.height,
        with_depth=False,
    )
    env_cfg.scene.front_camera = make_peek_over_front_camera(
        width=args_cli.width,
        height=args_cli.height,
        with_depth=False,
    )
    env_cfg.scene.wrist_camera = make_peek_over_wrist_camera(
        width=args_cli.width,
        height=args_cli.height,
        with_depth=False,
    )
    fisheye_spec = add_fisheye_rig(
        env_cfg,
        mode="four",
        width=args_cli.fisheye_width,
        height=args_cli.fisheye_height,
        update_period=0.1,
        fov_deg=200.0,
        mount_pos=(-0.44, 0.0, 0.58),
        lens_offset=0.02,
        with_depth=False,
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    _capture_standard(env, args_cli.output_dir, "peek_over_initial")
    panorama = FisheyePanorama(
        fisheye_spec,
        equirect_width=args_cli.panorama_width,
        equirect_height=args_cli.panorama_height,
        backend="cpu",
    )
    _capture_panorama(env, fisheye_spec, panorama, args_cli.output_dir, "peek_over_initial")

    root_state = torch.zeros(1, 13, device=env.device)
    root_state[0, :3] = torch.tensor((1.55, -1.10, 0.55), device=env.device)
    # Face +Y after moving around the left side of the partition.
    root_state[0, 3:7] = torch.tensor((0.7071068, 0.0, 0.0, 0.7071068), device=env.device)
    robot.write_root_state_to_sim(root_state, env_ids=torch.tensor([0], device=env.device))
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    action_term.recalibrate()
    for _ in range(24):
        env.step(actions)

    _capture_standard(env, args_cli.output_dir, "peek_over_side_approach")
    _capture_panorama(env, fisheye_spec, panorama, args_cli.output_dir, "peek_over_side_approach")

    with open(
        os.path.join(args_cli.output_dir, "peek_over_initial_state.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "robot_root": robot.data.root_state_w[0].detach().cpu().tolist(),
                "object_root": env.scene["object"].data.root_state_w[0].detach().cpu().tolist(),
                "target_root": env.scene["target_pad"].data.root_state_w[0].detach().cpu().tolist(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
