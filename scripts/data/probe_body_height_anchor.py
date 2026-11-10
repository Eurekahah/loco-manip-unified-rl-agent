#!/usr/bin/env python3
"""Probe how base anchoring affects the WBC body-height command."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe body-height tracking under base anchoring.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 7
    env_cfg.episode_length_s = 120.0
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    actions[:, 9] = 0.40 - action_term._default_body_pose[0, 0]
    actions[:, 10] = -0.12 - action_term._default_body_pose[0, 1]

    for mode in ("none", "xy", "xy_yaw", "xy_yaw_vel"):
        env.reset()
        anchor = robot.data.root_pos_w[0, :2].clone()
        _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
        anchor_yaw = yaw[0].clone()
        for _ in range(args_cli.steps):
            env.step(actions)
            if mode == "none":
                continue
            pos = robot.data.root_pos_w[0].clone()
            quat = robot.data.root_quat_w[0].clone()
            pos[0], pos[1] = anchor
            if mode != "xy":
                roll, pitch, _ = math_utils.euler_xyz_from_quat(quat.unsqueeze(0))
                quat = math_utils.quat_from_euler_xyz(roll, pitch, anchor_yaw.unsqueeze(0))[0]
            robot.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))
            if mode == "xy_yaw_vel":
                robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        root = robot.data.root_pos_w[0]
        rpy = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
        print(
            f"[anchor-probe] mode={mode:<11} root={root.detach().cpu().numpy().round(3).tolist()} "
            f"rpy=({rpy[0][0].item():.3f},{rpy[1][0].item():.3f},{rpy[2][0].item():.3f})",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
