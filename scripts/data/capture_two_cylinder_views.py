#!/usr/bin/env python3
"""Capture the three SmolVLA views while the target object is visible.

The default collection pose leaves the wrist camera looking at the horizon.
This script lowers the body and moves the end effector to the two-cylinder
pre-grasp pose before saving ``camera1`` (top), ``camera2`` (wrist), and
``camera3`` (front-left workcell) images.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture three target-visible camera views.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="recordings/two_cylinder_target_views")
parser.add_argument("--camera_width", type=int, default=224)
parser.add_argument("--camera_height", type=int, default=224)
parser.add_argument("--max_move_steps", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from rl_training.tasks.manager_based.locomotion.highlevel.config.two_cylinder_place.scene_cfg import (  # noqa: E402
    make_two_cylinder_top_camera,
    make_two_cylinder_workcell_camera,
    make_two_cylinder_wrist_camera,
)
import rl_training.tasks  # noqa: F401, E402


def _clamp_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def _save_rgb(sensor, path: str) -> np.ndarray:
    rgb = sensor.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb


def _montage(images: list[tuple[str, np.ndarray]], path: str) -> None:
    tiles = []
    for label, rgb in images:
        tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    cv2.imwrite(path, np.concatenate(tiles, axis=1))


def main() -> None:
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 11
    env_cfg.episode_length_s = 120.0
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env_cfg.scene.top_camera = make_two_cylinder_top_camera(
        width=args_cli.camera_width, height=args_cli.camera_height, with_depth=False
    )
    env_cfg.scene.wrist_camera = make_two_cylinder_wrist_camera(
        width=args_cli.camera_width, height=args_cli.camera_height, with_depth=False
    )
    env_cfg.scene.workcell_camera = make_two_cylinder_workcell_camera(
        width=args_cli.camera_width, height=args_cli.camera_height, with_depth=False
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0

    def step() -> None:
        env.step(actions)

    def move_ee(target_w: torch.Tensor, max_steps: int, label: str) -> None:
        for step_index in range(max_steps):
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0]
            target_b = math_utils.quat_apply(
                math_utils.quat_inv(root_quat), target_w - root_pos
            )
            actual_b, actual_quat_b = math_utils.subtract_frame_transforms(
                root_pos,
                root_quat,
                robot.data.body_pos_w[0, ee_body_id],
                robot.data.body_quat_w[0, ee_body_id],
            )
            if torch.linalg.vector_norm(target_b - actual_b) < 0.012:
                print(f"[views] {label} reached in {step_index} steps")
                break
            target_b = actual_b + _clamp_norm(target_b - actual_b, 0.018)
            actions[:, 3:6] = (target_b - action_term._default_ee_pos_b[0]).unsqueeze(0)
            action_term._default_ee_quat_b[0] = actual_quat_b
            actions[:, 6:9] = 0.0
            step()

    # Lower and pitch the body so the wrist camera points into the work area.
    actions[:, 9] = 0.40 - action_term._default_body_pose[0, 0].item()
    actions[:, 10] = -0.12 - action_term._default_body_pose[0, 1].item()
    for _ in range(32):
        step()

    object_pos = obj.data.root_pos_w[0].clone()
    pre_grasp = object_pos + torch.tensor([-0.15, 0.02, 0.07], device=env.device)
    grasp_pos = object_pos + torch.tensor([-0.09, 0.02, 0.0], device=env.device)
    move_ee(pre_grasp, args_cli.max_move_steps, "pre-grasp")

    # Hold the pose long enough for the 10 Hz cameras to refresh.
    for _ in range(12):
        step()

    captures = [
        ("camera1_top", "top_camera"),
        ("camera2_wrist", "wrist_camera"),
        ("camera3_workcell", "workcell_camera"),
    ]
    images = []
    for output_name, sensor_name in captures:
        path = os.path.join(args_cli.output_dir, f"{output_name}_rgb.png")
        rgb = _save_rgb(env.scene.sensors[sensor_name], path)
        images.append((output_name, rgb))
        print(f"[views] saved {path}")

    _montage(images, os.path.join(args_cli.output_dir, "three_view_target_visible.png"))
    move_ee(grasp_pos, args_cli.max_move_steps, "grasp")
    for _ in range(12):
        step()
    grasp_images = []
    for output_name, sensor_name in captures:
        path = os.path.join(args_cli.output_dir, f"grasp_{output_name}_rgb.png")
        grasp_images.append((output_name, _save_rgb(env.scene.sensors[sensor_name], path)))
        print(f"[views] saved {path}")
    _montage(
        grasp_images,
        os.path.join(args_cli.output_dir, "three_view_target_visible_grasp.png"),
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
