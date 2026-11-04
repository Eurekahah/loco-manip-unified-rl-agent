# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the low-shelf workcell geometry and object settling behavior."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe the low-shelf pick-and-place scene.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-LowShelf-Place-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--disable_workcell_camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.terminations.time_out = None
    if args_cli.disable_workcell_camera:
        env_cfg.scene.workcell_camera = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    print("=" * 78)
    reset_term = env.cfg.events.reset_object_position_discontinuous
    print(f"object reset func = {getattr(reset_term.func, '__name__', reset_term.func)}")
    print(f"object reset params = {reset_term.params}")
    for name in ("lower_shelf", "upper_shelf", "shelf_back", "target_pad", "object"):
        asset = env.scene[name]
        pos = asset.data.root_pos_w[0, :3].detach().cpu().tolist()
        print(f"{name:<16} root_pos_w = {[round(v, 5) for v in pos]}")
    print(f"object default_root_state = {env.scene['object'].data.default_root_state[0, :7].detach().cpu().tolist()}")
    print(f"object root_state_w      = {env.scene['object'].data.root_state_w[0, :7].detach().cpu().tolist()}")

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for step in range(args_cli.steps):
        env.step(actions)
        obj = env.scene["object"]
        root = env.scene["robot"]
        obj_pos = obj.data.root_pos_w[0, :3].detach().cpu().tolist()
        root_pos = root.data.root_pos_w[0, :3].detach().cpu().tolist()
        print(
            f"step={step:<2} object_z={obj_pos[2]:.5f} object_xyz="
            f"{[round(v, 5) for v in obj_pos]} robot_xyz={[round(v, 5) for v in root_pos]}"
        )
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    os._exit(0)

