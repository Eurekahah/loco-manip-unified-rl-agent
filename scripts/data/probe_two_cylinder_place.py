#!/usr/bin/env python3
"""Probe two-cylinder placement, collision support, and rest stability."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe the two-cylinder pick-and-place scene.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def _world_bounds(stage: Usd.Stage, prim_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return tuple(bounds.GetMin()), tuple(bounds.GetMax())


def _print_object_state(env, label: str) -> None:
    obj = env.scene["object"]
    pos = obj.data.root_pos_w[0].detach().cpu()
    vel = obj.data.root_lin_vel_w[0].detach().cpu()
    ang_vel = obj.data.root_ang_vel_w[0].detach().cpu()
    print(
        f"[probe] {label}: pos={pos.tolist()} "
        f"lin_vel_norm={vel.norm().item():.6f} ang_vel_norm={ang_vel.norm().item():.6f}",
        flush=True,
    )


def _settle(env, steps: int) -> None:
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(steps):
        env.step(actions)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 7
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    stage = env.sim.stage
    object_path = f"/World/envs/env_0/{env.scene['object'].cfg.prim_path.split('/')[-1]}"
    robot = env.scene["robot"]
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    ee_pos = robot.data.body_pos_w[0, ee_body_id].detach().cpu()
    ee_quat = robot.data.body_quat_w[0, ee_body_id].detach().cpu()
    print(f"[probe] ee_pos_w={ee_pos.tolist()}", flush=True)
    print(f"[probe] ee_quat_wxyz={ee_quat.tolist()}", flush=True)

    _print_object_state(env, "low platform initial")
    _settle(env, args_cli.settle_steps)
    _print_object_state(env, f"low platform after {args_cli.settle_steps} steps")
    bbox_min, bbox_max = _world_bounds(stage, object_path)
    print(f"[probe] object world bbox min={bbox_min} max={bbox_max}", flush=True)
    print(
        "[probe] object world size="
        f"{tuple(round(hi - lo, 5) for lo, hi in zip(bbox_min, bbox_max))}",
        flush=True,
    )

    target = env.scene["target_pad"]
    target_pos = target.data.root_pos_w[0].detach().cpu()
    root_state = torch.zeros(1, 13, device=env.device)
    root_state[0, 0:3] = target_pos.to(env.device)
    root_state[0, 2] += 0.12
    root_state[0, 3:7] = torch.tensor(
        env.scene["object"].cfg.init_state.rot, dtype=torch.float32, device=env.device
    )
    env.scene["object"].write_root_state_to_sim(root_state, env_ids=torch.tensor([0], device=env.device))
    env.scene["object"].write_root_pose_to_sim(root_state[:, :7], env_ids=torch.tensor([0], device=env.device))
    print(
        f"[probe] requested high-platform pose={root_state[0, :7].detach().cpu().tolist()}",
        flush=True,
    )
    env.sim.step(render=False)
    _print_object_state(env, "high platform immediately after write")
    _settle(env, args_cli.settle_steps)
    _print_object_state(env, "high platform target drop")
    bbox_min, bbox_max = _world_bounds(stage, object_path)
    print(f"[probe] object on target world bbox min={bbox_min} max={bbox_max}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
