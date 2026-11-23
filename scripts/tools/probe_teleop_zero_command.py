#!/usr/bin/env python3
"""Measure standing drift when the teleop action path receives an all-zero command.

This uses the same TeleopHistoryEnvCfg and 13-D teleop action layout as
scripts/teleoperation/teleop_m20_piper_keyboard.py, but feeds zeros directly so
the result does not depend on keyboard focus or GUI input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleop zero-command standing drift probe")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-Teleop-History-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Low-level policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500, help="Control steps to record after warmup")
parser.add_argument("--warmup", type=int, default=50, help="Control steps discarded before recording")
parser.add_argument("--no_push", action="store_true", help="Disable random push events")
parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

os.environ["RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY"] = str(
    Path(args_cli.checkpoint).resolve()
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def _split_step_result(result):
    if len(result) == 5:
        obs, rew, terminated, truncated, extras = result
        return obs, rew, terminated | truncated, extras
    obs, rew, done, extras = result
    return obs, rew, done, extras


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None
    if args_cli.no_push and hasattr(env_cfg.events, "randomize_push_robot"):
        env_cfg.events.randomize_push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    n_envs = env.num_envs
    device = env.device
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros(n_envs, action_dim, device=device)

    for _ in range(args_cli.warmup):
        _split_step_result(env.step(zero_action))

    start_xy = robot.data.root_pos_w[:, :2].clone()
    prev_xy = start_xy.clone()
    path_length = torch.zeros(n_envs, device=device)
    speeds = []
    yaw_rates = []
    heights = []
    pitches = []
    rolls = []
    resets = torch.zeros(n_envs, dtype=torch.long, device=device)
    falls = torch.zeros(n_envs, dtype=torch.long, device=device)
    body_cmd = env.command_manager.get_term("body_pose") if "body_pose" in env.command_manager.active_terms else None

    term_names = list(env.termination_manager.active_terms)
    for _ in range(args_cli.steps):
        xy_before = robot.data.root_pos_w[:, :2].clone()
        lin_vel = robot.data.root_lin_vel_b[:, :2].clone()
        yaw_rate = robot.data.root_ang_vel_b[:, 2].clone()
        quat = robot.data.root_quat_w.clone()
        roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
        height = robot.data.root_pos_w[:, 2].clone()

        _, _, done, _ = _split_step_result(env.step(zero_action))
        done = done.bool()
        xy_after = robot.data.root_pos_w[:, :2]
        step_dist = torch.norm(xy_after - xy_before, dim=1)
        path_length += torch.where(done, torch.zeros_like(step_dist), step_dist)

        speeds.append(lin_vel.cpu())
        yaw_rates.append(yaw_rate.cpu())
        heights.append(height.cpu())
        pitches.append(pitch.cpu())
        rolls.append(roll.cpu())

        if done.any():
            resets += done.long()
            for name in term_names:
                if name in ("bad_orientation_2", "root_height_below_minimum"):
                    falls += env.termination_manager.get_term(name).bool().long() * done.long()
        prev_xy = xy_after.clone()

    speed_t = torch.stack(speeds)
    yaw_t = torch.stack(yaw_rates)
    height_t = torch.stack(heights)
    pitch_t = torch.stack(pitches)
    roll_t = torch.stack(rolls)
    final_xy = robot.data.root_pos_w[:, :2].clone()
    no_reset = resets == 0
    net_disp = torch.norm(final_xy - start_xy, dim=1)

    payload = {
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "num_envs": n_envs,
        "action_dim": action_dim,
        "steps": args_cli.steps,
        "warmup": args_cli.warmup,
        "push_enabled": not args_cli.no_push,
        "net_displacement_m_mean": float(net_disp.mean()),
        "net_displacement_m_per_env": [float(x) for x in net_disp],
        "path_length_m_mean": float(path_length.mean()),
        "mean_speed_mps": float(torch.norm(speed_t, dim=-1).mean()),
        "rms_speed_mps": float(torch.sqrt(torch.mean(torch.sum(speed_t**2, dim=-1)))),
        "max_speed_mps": float(torch.norm(speed_t, dim=-1).max()),
        "mean_abs_yaw_rate_radps": float(yaw_t.abs().mean()),
        "max_abs_yaw_rate_radps": float(yaw_t.abs().max()),
        "mean_height_m": float(height_t.mean()),
        "height_std_m": float(height_t.std()),
        "mean_pitch_rad": float(pitch_t.mean()),
        "mean_roll_rad": float(roll_t.mean()),
        "resets_total": int(resets.sum()),
        "fall_terminations_total": int(falls.sum()),
        "termination_terms": term_names,
        "fully_stationary_envs": int(no_reset.sum()),
        "body_pose_command_mean": (
            [float(x) for x in body_cmd.command.mean(dim=0).cpu()] if body_cmd is not None else None
        ),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args_cli.out:
        out_path = Path(args_cli.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[probe] wrote {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
    os._exit(0)
