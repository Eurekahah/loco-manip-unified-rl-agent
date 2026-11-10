#!/usr/bin/env python3
"""Probe the deployed base-velocity sign convention for the M20."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe M20 base velocity signs.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 3
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    wheel_ids, wheel_names = robot.find_joints([".*wheel.*"], preserve_order=True)
    print(f"[probe] wheels={list(zip(wheel_ids, wheel_names))}", flush=True)
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0

    def run(label: str, vx: float, vy: float, wz: float) -> None:
        start = robot.data.root_pos_w[0, :2].clone()
        actions[:, 0] = vx
        actions[:, 1] = vy
        actions[:, 2] = wz
        for index in range(args_cli.steps):
            env.step(actions)
            if index % 10 == 0:
                root = robot.data.root_pos_w[0]
                wheel_vel = robot.data.joint_vel[0, wheel_ids]
                print(
                    f"[probe] {label} step={index} root={root.detach().cpu().numpy().round(3).tolist()} "
                    f"wheel_vel={wheel_vel.detach().cpu().numpy().round(2).tolist()}",
                    flush=True,
                )
        end = robot.data.root_pos_w[0, :2].clone()
        delta = (end - start).detach().cpu().numpy()
        print(f"[probe] {label} delta_xy={delta.round(3).tolist()} cmd=({vx},{vy},{wz})", flush=True)

    run("+vx", 0.3, 0.0, 0.0)
    run("-vx", -0.3, 0.0, 0.0)
    run("+vy", 0.0, 0.2, 0.0)
    run("-vy", 0.0, -0.2, 0.0)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

