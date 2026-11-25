#!/usr/bin/env python3
"""Measure wheel/leg slip under a fixed zero velocity command in the PLAY task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="PLAY zero-command wheel/leg slip probe")
parser.add_argument("--task", type=str, default="History-Adaptation-Deeprobotics-M20-play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wheel_radius", type=float, default=0.09)
parser.add_argument(
    "--command",
    type=str,
    default="0,0,0",
    help='固定底盘命令 "vx,vy,wz"；默认全零，也可用 "0,0.5,0.5" 测"仅 vx=0"',
)
parser.add_argument(
    "--low-speed-filter",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="是否启用 M20 轮/腿动作项的低速滤波（默认启用，便于 A/B 对比）",
)
parser.add_argument("--out", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerHis  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    command = tuple(float(x) for x in args_cli.command.split(","))
    if len(command) != 3:
        raise ValueError(f'--command must be "vx,vy,wz", got {args_cli.command!r}')

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.ranges.lin_vel_x = (command[0], command[0])
    cmd_cfg.ranges.lin_vel_y = (command[1], command[1])
    cmd_cfg.ranges.ang_vel_z = (command[2], command[2])
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_zero_cmd_envs = 0.0
    cmd_cfg.rel_only_lin_x_envs = 0.0
    cmd_cfg.rel_only_lin_y_envs = 0.0
    cmd_cfg.rel_only_ang_z_envs = 0.0
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.debug_vis = False

    # Isolate zero-command standing: pin the body-pose command to the default pose.
    body_cfg = env_cfg.commands.body_pose
    body_cfg.height_range = (0.513, 0.513)
    body_cfg.pitch_range = (0.0, 0.0)
    body_cfg.roll_range = (0.0, 0.0)
    body_cfg.default_height = 0.513
    body_cfg.default_pitch = 0.0
    body_cfg.default_roll = 0.0
    body_cfg.rel_default_pose_envs = 1.0
    body_cfg.resampling_time_range = (1.0e9, 1.0e9)

    # Match play.py: remove pushes for a clean standing-slip measurement.
    if hasattr(env_cfg.events, "randomize_push_robot"):
        env_cfg.events.randomize_push_robot = None
    if env_cfg.curriculum is not None and hasattr(env_cfg.curriculum, "command_levels"):
        env_cfg.curriculum.command_levels = None

    # The low-speed filter lives on the action terms, so the probe exercises the
    # exact same code path as training.  Keep an A/B switch for diagnosing it.
    low_speed_filter_applied = False
    for name in ("joint_pos", "joint_vel"):
        action_cfg = getattr(env_cfg.actions, name, None)
        filt = getattr(action_cfg, "low_speed_filter", None) if action_cfg is not None else None
        if filt is not None:
            filt.enabled = bool(args_cli.low_speed_filter)
            low_speed_filter_applied = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    n_envs = unwrapped.num_envs
    device = unwrapped.device

    runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=device)

    term = unwrapped.command_manager.get_term("base_velocity")
    term._resample_command = lambda env_ids: None  # noqa: ARG005
    fixed_cmd = torch.tensor(command, device=device, dtype=torch.float32).repeat(n_envs, 1)
    wheel_action_term = unwrapped.action_manager.get_term("joint_vel")

    wheel_ids, wheel_names = robot.find_joints(".*_wheel_joint", preserve_order=True)
    wheel_body_ids, wheel_body_names = robot.find_bodies(".*_wheel", preserve_order=True)
    leg_ids, leg_names = robot.find_joints(
        [".*hipx_joint", ".*hipy_joint", ".*knee_joint"], preserve_order=True
    )

    obs = env.get_observations()
    for _ in range(args_cli.warmup):
        term.vel_command_b[:] = fixed_cmd
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))

    start_xy = robot.data.root_pos_w[:, :2].clone()
    path = torch.zeros(n_envs, device=device)
    root_speed = []
    wheel_qvel = []
    leg_qvel = []
    wheel_z = []
    heights = []
    pitches = []
    rolls = []
    resets = torch.zeros(n_envs, dtype=torch.long, device=device)
    falls = torch.zeros(n_envs, dtype=torch.long, device=device)
    term_names = list(unwrapped.termination_manager.active_terms)

    for _ in range(args_cli.steps):
        xy_before = robot.data.root_pos_w[:, :2].clone()
        root_speed.append(torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1).cpu())
        wheel_qvel.append(robot.data.joint_vel[:, wheel_ids].cpu())
        leg_qvel.append(robot.data.joint_vel[:, leg_ids].cpu())
        wheel_z.append(robot.data.body_pos_w[:, wheel_body_ids, 2].cpu())
        quat = robot.data.root_quat_w.clone()
        roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
        rolls.append(roll.cpu())
        pitches.append(pitch.cpu())
        heights.append(robot.data.root_pos_w[:, 2].cpu())

        term.vel_command_b[:] = fixed_cmd
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        done = dones.bool()
        xy_after = robot.data.root_pos_w[:, :2]
        step_dist = torch.norm(xy_after - xy_before, dim=1)
        path += torch.where(done, torch.zeros_like(step_dist), step_dist)
        if done.any():
            resets += done.long()
            for name in term_names:
                if name in ("bad_orientation_2", "root_height_below_minimum"):
                    falls += unwrapped.termination_manager.get_term(name).bool().long() * done.long()

    root_speed_t = torch.stack(root_speed)
    wheel_qvel_t = torch.stack(wheel_qvel).abs()
    leg_qvel_t = torch.stack(leg_qvel).abs()
    wheel_z_t = torch.stack(wheel_z)
    wheel_surface = wheel_qvel_t * args_cli.wheel_radius
    final_xy = robot.data.root_pos_w[:, :2]
    net_disp = torch.norm(final_xy - start_xy, dim=1)

    resets_cpu = resets.cpu()
    payload = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "num_envs": n_envs,
        "steps": args_cli.steps,
        "warmup": args_cli.warmup,
        "command": {"vx": command[0], "vy": command[1], "wz": command[2]},
        "low_speed_filter": bool(args_cli.low_speed_filter),
        "low_speed_filter_applied": low_speed_filter_applied,
        "low_speed_filter_gate_fraction": (
            float(wheel_action_term.standstill_gate.float().mean())
            if hasattr(wheel_action_term, "standstill_gate")
            else None
        ),
        "wheel_radius_m": args_cli.wheel_radius,
        "wheel_joint_names": list(wheel_names),
        "leg_joint_names": list(leg_names),
        "mean_root_speed_mps": float(root_speed_t.mean()),
        "rms_root_speed_mps": float(torch.sqrt(torch.mean(root_speed_t**2))),
        "net_displacement_m_mean": float(net_disp.mean()),
        "net_displacement_m_max": float(net_disp.max()),
        "path_length_m_mean": float(path.mean()),
        "mean_wheel_abs_qvel_radps": float(wheel_qvel_t.mean()),
        "max_wheel_abs_qvel_radps": float(wheel_qvel_t.max()),
        "mean_wheel_surface_speed_mps": float(wheel_surface.mean()),
        "mean_wheel_slip_speed_mps": float((wheel_surface - root_speed_t.unsqueeze(-1)).abs().mean()),
        "wheel_surface_to_root_speed_ratio": float(wheel_surface.mean() / torch.clamp(root_speed_t.mean(), min=1.0e-6)),
        "wheel_air_fraction_z_gt_0_12": float((wheel_z_t > 0.12).float().mean()),
        "valid_envs": int((resets_cpu == 0).sum()),
        "valid_mean_root_speed_mps": float(root_speed_t[:, resets_cpu == 0].mean()) if (resets_cpu == 0).any() else None,
        "valid_mean_wheel_surface_speed_mps": float(wheel_surface[:, resets_cpu == 0].mean()) if (resets_cpu == 0).any() else None,
        "valid_mean_wheel_slip_speed_mps": float((wheel_surface - root_speed_t.unsqueeze(-1))[:, resets_cpu == 0].abs().mean()) if (resets_cpu == 0).any() else None,
        "valid_slip_ratio": float(wheel_surface[:, resets_cpu == 0].mean() / torch.clamp(root_speed_t[:, resets_cpu == 0].mean(), min=1.0e-6)) if (resets_cpu == 0).any() else None,
        "per_env_net_displacement_m": [float(x) for x in net_disp],
        "per_env_resets": [int(x) for x in resets],
        "mean_leg_abs_qvel_radps": float(leg_qvel_t.mean()),
        "max_leg_abs_qvel_radps": float(leg_qvel_t.max()),
        "wheel_z_mean_m": float(wheel_z_t.mean()),
        "wheel_z_min_m": float(wheel_z_t.min()),
        "wheel_z_max_m": float(wheel_z_t.max()),
        "mean_height_m": float(torch.stack(heights).mean()),
        "mean_pitch_rad": float(torch.stack(pitches).mean()),
        "mean_roll_rad": float(torch.stack(rolls).mean()),
        "resets_total": int(resets.sum()),
        "fall_terminations_total": int(falls.sum()),
        "termination_terms": term_names,
        "per_wheel_mean_abs_qvel_radps": [float(x) for x in wheel_qvel_t.mean(dim=(0, 1))],
        "per_wheel_mean_surface_speed_mps": [float(x) for x in wheel_surface.mean(dim=(0, 1))],
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
