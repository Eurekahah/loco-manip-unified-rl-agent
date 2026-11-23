# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Probe the M20 teleoperation-style command distribution and roll-course shutdown.

Usage::

    python scripts/reinforcement_learning/rsl_rl/probe_command_sampling.py \
        --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 64
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="验证 M20 分层速度采样与身体姿态 1:1 采样")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="History-Adaptation-Deeprobotics-M20-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resamples", type=int, default=24)
parser.add_argument("--tolerance", type=float, default=0.03)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    vel_term = env.command_manager.get_term("base_velocity")
    pose_term = env.command_manager.get_term("body_pose")
    ids = torch.arange(env.num_envs, device=env.device)

    print("=" * 78)
    print(f"[probe] task={args_cli.task} num_envs={env.num_envs} resamples={args_cli.resamples}")
    print(
        f"[probe] velocity ratios: zero={vel_term.cfg.rel_zero_cmd_envs:.2f} "
        f"vx={vel_term.cfg.rel_only_lin_x_envs:.2f} vy={vel_term.cfg.rel_only_lin_y_envs:.2f} "
        f"yaw={vel_term.cfg.rel_only_ang_z_envs:.2f} "
        f"uniform={1.0 - sum([vel_term.cfg.rel_zero_cmd_envs, vel_term.cfg.rel_only_lin_x_envs, vel_term.cfg.rel_only_lin_y_envs, vel_term.cfg.rel_only_ang_z_envs]):.2f}"
    )
    print(
        f"[probe] pose: default_ratio={pose_term.cfg.rel_default_pose_envs:.2f} "
        f"height={pose_term.cfg.height_range} pitch={pose_term.cfg.pitch_range} "
        f"roll={pose_term.cfg.roll_range}"
    )

    if tuple(pose_term.cfg.roll_range) != (0.0, 0.0):
        raise RuntimeError(f"roll_range should be (0, 0), got {pose_term.cfg.roll_range}")
    if env_cfg.curriculum.body_pose_roll_range_s3 is not None:
        raise RuntimeError("body_pose_roll_range_s3 curriculum is still enabled")
    if env_cfg.curriculum.body_roll_rew_s3 is not None:
        raise RuntimeError("body_roll_rew_s3 curriculum is still enabled")

    # Open the height/pitch ranges in this probe instance so the random half of the
    # body-pose samples is actually distinguishable from the default pose.
    pose_term.cfg.height_range = (0.33, 0.55)
    pose_term.cfg.pitch_range = (-0.35, 0.35)
    print(
        "[probe] opened pose ranges for verification: "
        f"height={pose_term.cfg.height_range} pitch={pose_term.cfg.pitch_range}"
    )

    category_names = {
        vel_term.CATEGORY_ZERO: "zero",
        vel_term.CATEGORY_LIN_X: "vx",
        vel_term.CATEGORY_LIN_Y: "vy",
        vel_term.CATEGORY_ANG_Z: "yaw",
        vel_term.CATEGORY_UNIFORM: "uniform",
    }
    counts = {name: 0 for name in category_names.values()}
    default_pose_count = 0
    total = 0
    threshold = vel_term.cfg.min_command_lin_xy
    default_pose = torch.tensor(
        [pose_term.cfg.default_height, pose_term.cfg.default_pitch, pose_term.cfg.default_roll],
        device=env.device,
    )

    for _ in range(args_cli.resamples):
        vel_term._resample_command(ids)
        pose_term._resample_command(ids)
        cat = vel_term.command_category
        cmd = vel_term.command

        zero = cat == vel_term.CATEGORY_ZERO
        vx = cat == vel_term.CATEGORY_LIN_X
        vy = cat == vel_term.CATEGORY_LIN_Y
        yaw = cat == vel_term.CATEGORY_ANG_Z
        if zero.any() and not torch.all(cmd[zero].abs() <= 1e-6):
            raise RuntimeError("zero category has nonzero velocity")
        if vx.any() and not (cmd[vx, 1:].abs() <= 1e-6).all():
            raise RuntimeError("pure vx category has nonzero vy/yaw")
        if vy.any() and not (cmd[vy][:, [0, 2]].abs() <= 1e-6).all():
            raise RuntimeError("pure vy category has nonzero vx/yaw")
        if yaw.any() and not (cmd[yaw, :2].abs() <= 1e-6).all():
            raise RuntimeError("pure yaw category has nonzero vx/vy")
        if vx.any() and not (cmd[vx, 0].abs() > threshold).all():
            raise RuntimeError("pure vx sample fell under the dead zone")
        if vy.any() and not (cmd[vy, 1].abs() > threshold).all():
            raise RuntimeError("pure vy sample fell under the dead zone")
        if yaw.any() and not (cmd[yaw, 2].abs() > threshold).all():
            raise RuntimeError("pure yaw sample fell under the dead zone")

        for category, name in category_names.items():
            counts[name] += int((cat == category).sum())

        is_default = pose_term.default_pose_env
        default_pose_count += int(is_default.sum())
        if is_default.any() and not torch.all(pose_term.command[is_default] == default_pose):
            raise RuntimeError("default body-pose samples do not match the configured default pose")
        random_pose = ~is_default
        if random_pose.any():
            random_cmd = pose_term.command[random_pose]
            if not (
                (random_cmd[:, 0] >= pose_term.cfg.height_range[0])
                & (random_cmd[:, 0] <= pose_term.cfg.height_range[1])
                & (random_cmd[:, 1] >= pose_term.cfg.pitch_range[0])
                & (random_cmd[:, 1] <= pose_term.cfg.pitch_range[1])
            ).all():
                raise RuntimeError("random body-pose samples are outside the configured ranges")
        if not torch.all(pose_term.command[:, 2] == 0.0):
            raise RuntimeError("roll command is not identically zero")
        total += env.num_envs

    print("[probe] sampled category counts")
    for name, count in counts.items():
        print(f"    {name:<8} {count / total:.3f}")
    print(f"    default pose ratio {default_pose_count / total:.3f}")

    expected = {"zero": 0.2, "vx": 0.15, "vy": 0.15, "yaw": 0.15, "uniform": 0.35}
    for name, ratio in expected.items():
        observed = counts[name] / total
        if abs(observed - ratio) > args_cli.tolerance:
            raise RuntimeError(f"{name} ratio {observed:.3f} outside {ratio:.3f} +/- {args_cli.tolerance}")
    pose_ratio = default_pose_count / total
    if abs(pose_ratio - 0.5) > args_cli.tolerance:
        raise RuntimeError(f"default pose ratio {pose_ratio:.3f} outside 0.5 +/- {args_cli.tolerance}")

    print("[probe] sample rows: [vx, vy, yaw, height, pitch, roll, default_pose, category]")
    for i in range(min(12, env.num_envs)):
        print(
            f"    {cmd[i, 0].item():+.3f} {cmd[i, 1].item():+.3f} {cmd[i, 2].item():+.3f} "
            f"{pose_term.command[i, 0].item():.3f} {pose_term.command[i, 1].item():+.3f} "
            f"{pose_term.command[i, 2].item():+.3f} {bool(pose_term.default_pose_env[i].item())} "
            f"{category_names[int(cat[i].item())]}"
        )
    print("[probe] PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
    os._exit(0)
