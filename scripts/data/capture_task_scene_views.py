#!/usr/bin/env python3
"""Capture balanced-rendering screenshots for generated task scenes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture generated task scene views.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=(
        "Isaac-M20-Piper-BlindIntercept-Teleop-v0",
        "Isaac-M20-Piper-DrawerLadder-Teleop-v0",
        "Isaac-M20-Piper-DoorBrace-Teleop-v0",
        "Isaac-M20-Piper-FrontDeskPickup-Teleop-v0",
        "Isaac-M20-Piper-HighLowServe-Teleop-v0",
        "Isaac-M20-Piper-HomeMission-Teleop-v0",
        "Isaac-M20-Piper-KitchenDrawerServe-Teleop-v0",
        "Isaac-M20-Piper-SupplyRestock-Teleop-v0",
        "Isaac-M20-Piper-WardDelivery-Teleop-v0",
    ),
)
parser.add_argument("--output_dir", type=str, required=True)
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
from rl_training.tasks.manager_based.locomotion.highlevel.config.blind_intercept.scene_cfg import (  # noqa: E402
    make_blind_intercept_front_camera,
    make_blind_intercept_top_camera,
    make_blind_intercept_workcell_camera,
    make_blind_intercept_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.blind_intercept.env_cfg import (  # noqa: E402
    CONVEYOR_BELT_SPEED,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.drawer_ladder.scene_cfg import (  # noqa: E402
    make_drawer_ladder_front_camera,
    make_drawer_ladder_top_camera,
    make_drawer_ladder_workcell_camera,
    make_drawer_ladder_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.door_brace.scene_cfg import (  # noqa: E402
    make_door_brace_front_camera,
    make_door_brace_top_camera,
    make_door_brace_workcell_camera,
    make_door_brace_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_low_serve.scene_cfg import (  # noqa: E402
    make_high_low_front_camera,
    make_high_low_top_camera,
    make_high_low_workcell_camera,
    make_high_low_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.home_mission.scene_cfg import (  # noqa: E402
    make_home_mission_front_camera,
    make_home_mission_top_camera,
    make_home_mission_workcell_camera,
    make_home_mission_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.front_desk_pickup.scene_cfg import (  # noqa: E402
    make_front_desk_pickup_front_camera,
    make_front_desk_pickup_top_camera,
    make_front_desk_pickup_workcell_camera,
    make_front_desk_pickup_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.ward_delivery.scene_cfg import (  # noqa: E402
    make_ward_delivery_front_camera,
    make_ward_delivery_top_camera,
    make_ward_delivery_workcell_camera,
    make_ward_delivery_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.kitchen_drawer_serve.scene_cfg import (  # noqa: E402
    make_kitchen_drawer_serve_front_camera,
    make_kitchen_drawer_serve_top_camera,
    make_kitchen_drawer_serve_workcell_camera,
    make_kitchen_drawer_serve_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.supply_restock.scene_cfg import (  # noqa: E402
    make_supply_restock_front_camera,
    make_supply_restock_top_camera,
    make_supply_restock_workcell_camera,
    make_supply_restock_wrist_camera,
)


REGISTRY = {
    "Isaac-M20-Piper-BlindIntercept-Teleop-v0": {
        "prefix": "blind_intercept",
        "top": make_blind_intercept_top_camera,
        "workcell": make_blind_intercept_workcell_camera,
        "front": make_blind_intercept_front_camera,
        "wrist": make_blind_intercept_wrist_camera,
    },
    "Isaac-M20-Piper-DrawerLadder-Teleop-v0": {
        "prefix": "drawer_ladder",
        "top": make_drawer_ladder_top_camera,
        "workcell": make_drawer_ladder_workcell_camera,
        "front": make_drawer_ladder_front_camera,
        "wrist": make_drawer_ladder_wrist_camera,
    },
    "Isaac-M20-Piper-DoorBrace-Teleop-v0": {
        "prefix": "door_brace",
        "top": make_door_brace_top_camera,
        "workcell": make_door_brace_workcell_camera,
        "front": make_door_brace_front_camera,
        "wrist": make_door_brace_wrist_camera,
    },
    "Isaac-M20-Piper-FrontDeskPickup-Teleop-v0": {
        "prefix": "front_desk_pickup",
        "top": make_front_desk_pickup_top_camera,
        "workcell": make_front_desk_pickup_workcell_camera,
        "front": make_front_desk_pickup_front_camera,
        "wrist": make_front_desk_pickup_wrist_camera,
        "fisheye_mount": (-0.44, 0.0, 0.58),
    },
    "Isaac-M20-Piper-HighLowServe-Teleop-v0": {
        "prefix": "high_low_serve",
        "top": make_high_low_top_camera,
        "workcell": make_high_low_workcell_camera,
        "front": make_high_low_front_camera,
        "wrist": make_high_low_wrist_camera,
    },
    "Isaac-M20-Piper-HomeMission-Teleop-v0": {
        "prefix": "home_mission",
        "top": make_home_mission_top_camera,
        "workcell": make_home_mission_workcell_camera,
        "front": make_home_mission_front_camera,
        "wrist": make_home_mission_wrist_camera,
    },
    "Isaac-M20-Piper-KitchenDrawerServe-Teleop-v0": {
        "prefix": "kitchen_drawer_serve",
        "top": make_kitchen_drawer_serve_top_camera,
        "workcell": make_kitchen_drawer_serve_workcell_camera,
        "front": make_kitchen_drawer_serve_front_camera,
        "wrist": make_kitchen_drawer_serve_wrist_camera,
        "fisheye_mount": (-0.44, 0.0, 0.58),
    },
    "Isaac-M20-Piper-SupplyRestock-Teleop-v0": {
        "prefix": "supply_restock",
        "top": make_supply_restock_top_camera,
        "workcell": make_supply_restock_workcell_camera,
        "front": make_supply_restock_front_camera,
        "wrist": make_supply_restock_wrist_camera,
        "fisheye_mount": (-0.44, 0.0, 0.58),
    },
    "Isaac-M20-Piper-WardDelivery-Teleop-v0": {
        "prefix": "ward_delivery",
        "top": make_ward_delivery_top_camera,
        "workcell": make_ward_delivery_workcell_camera,
        "front": make_ward_delivery_front_camera,
        "wrist": make_ward_delivery_wrist_camera,
        "fisheye_mount": (-0.44, 0.0, 0.58),
    },
}


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


def main() -> None:
    spec = REGISTRY[args_cli.task]
    prefix = spec["prefix"]
    os.makedirs(args_cli.output_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    configure_quality_rendering(env_cfg)
    env_cfg.seed = 17
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 180.0
    env_cfg.scene.top_camera = spec["top"](
        width=args_cli.width, height=args_cli.height, with_depth=False
    )
    env_cfg.scene.workcell_camera = spec["workcell"](
        width=args_cli.width, height=args_cli.height, with_depth=False
    )
    env_cfg.scene.front_camera = spec["front"](
        width=args_cli.width, height=args_cli.height, with_depth=False
    )
    env_cfg.scene.wrist_camera = spec["wrist"](
        width=args_cli.width, height=args_cli.height, with_depth=False
    )
    fisheye_spec = add_fisheye_rig(
        env_cfg,
        mode="four",
        width=args_cli.fisheye_width,
        height=args_cli.fisheye_height,
        update_period=0.1,
        fov_deg=200.0,
        mount_pos=spec.get("fisheye_mount", (-0.44, 0.0, 0.34)),
        lens_offset=0.02,
        with_depth=False,
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    captures = [
        ("Top view", "top_camera"),
        ("Workcell view", "workcell_camera"),
        ("Front camera", "front_camera"),
        ("Wrist camera", "wrist_camera"),
    ]
    images = []
    for label, sensor_name in captures:
        path = os.path.join(args_cli.output_dir, f"{prefix}_{sensor_name}.png")
        rgb = _save_rgb(env.scene.sensors[sensor_name], path)
        images.append((label, rgb))
        print(f"[task-views] saved {path}", flush=True)
    _montage(images, os.path.join(args_cli.output_dir, f"{prefix}_montage.png"))

    panorama = FisheyePanorama(
        fisheye_spec,
        equirect_width=args_cli.panorama_width,
        equirect_height=args_cli.panorama_height,
        backend="cpu",
    )
    raw_images = read_fisheye_images(env, fisheye_spec)
    panorama_rgb = panorama.stitch(raw_images)
    panorama_path = os.path.join(args_cli.output_dir, f"{prefix}_panorama.png")
    cv2.imwrite(panorama_path, cv2.cvtColor(panorama_rgb, cv2.COLOR_RGB2BGR))
    fisheye_montage = np.concatenate(
        [
            _label_tile(image, f"Fisheye {index}", (args_cli.fisheye_width, args_cli.fisheye_height))
            for index, image in enumerate(raw_images)
        ],
        axis=1,
    )
    fisheye_path = os.path.join(args_cli.output_dir, f"{prefix}_fisheye_montage.png")
    cv2.imwrite(fisheye_path, fisheye_montage)
    print(f"[task-views] saved {panorama_path}", flush=True)
    print(f"[task-views] saved {fisheye_path}", flush=True)

    robot = env.scene["robot"]
    state = {
        "task": args_cli.task,
        "robot_root": robot.data.root_state_w[0].detach().cpu().tolist(),
        "object_root": env.scene["object"].data.root_state_w[0].detach().cpu().tolist(),
        "target_root": env.scene["target_pad"].data.root_state_w[0].detach().cpu().tolist(),
    }
    if spec["prefix"] == "door_brace":
        door = env.scene["hinged_door"]
        state["hinged_door"] = {
            "prim_path": str(door.root_physx_view.prim_paths[0]),
            "joint_names": list(door.joint_names),
            "joint_pos": door.data.joint_pos[0].detach().cpu().tolist(),
        }
    if spec["prefix"] == "blind_intercept":
        state["conveyor"] = {
            "prim_path": str(env.scene["conveyor"].prims[0].GetPrimPath()),
            "belt_speed_x": CONVEYOR_BELT_SPEED,
        }
    with open(os.path.join(args_cli.output_dir, f"{prefix}_state.json"), "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
