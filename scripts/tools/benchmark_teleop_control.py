"""Benchmark task-generic teleoperation control loops with a fixed action.

This tool intentionally measures only the simulator-side control loop. Input
device parsing, networking, GUI rendering, and dataset writing are excluded so
that Isaac Lab's manager path and this repository's fast path can be compared
on the same machine and task.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


TOOLS_DIR = Path(__file__).resolve().parent
TELEOP_DIR = TOOLS_DIR.parent / "teleoperation"
for path in (TOOLS_DIR, TELEOP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

parser = argparse.ArgumentParser(description="Benchmark a teleoperation control loop.")
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--mode",
    choices=("manager", "fast", "fast_hot"),
    default="manager",
    help="manager is the official ManagerBasedRLEnv.step() path.",
)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fast-render-hz", type=float, default=15.0)
parser.add_argument("--fast-write-each-tick", action="store_true")
parser.add_argument("--fast-robot-hot-period", type=float, default=0.01)
parser.add_argument("--fast-robot-only-write", action="store_true")
parser.add_argument("--ik-update-period", type=float, default=0.0)
parser.add_argument("--profile-detailed", action="store_true")
parser.add_argument("--profile-json", default=None)
parser.add_argument("--output", default="outputs/fast_teleop/control_benchmark.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import rl_training.tasks  # noqa: F401, E402
from fast_teleop import DetailedStepProfiler, FastTeleopStepper  # noqa: E402
from fisheye_profiler import RuntimeProfiler  # noqa: E402


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
    return ordered[index]


def _configure_m20_teleop(env_cfg) -> None:
    action_cfg = getattr(env_cfg.actions, "pre_trained_pick_action", None)
    if action_cfg is None:
        return
    if hasattr(action_cfg, "debug_vis"):
        action_cfg.debug_vis = False
    if hasattr(action_cfg, "debug_vis_ee_target_only"):
        action_cfg.debug_vis_ee_target_only = False
    low_level_ee = getattr(action_cfg, "low_level_ee_actions", None)
    if low_level_ee is not None and hasattr(low_level_ee, "debug_vis"):
        low_level_ee.debug_vis = False
    if hasattr(action_cfg, "ik_update_period_s"):
        action_cfg.ik_update_period_s = args_cli.ik_update_period


def _disable_debug_vis(env) -> None:
    for manager_name in ("command_manager", "action_manager"):
        manager = getattr(env, manager_name, None)
        if manager is not None and hasattr(manager, "set_debug_vis"):
            manager.set_debug_vis(False)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(env_cfg, "terminations", None) is not None:
        env_cfg.terminations.time_out = None
    _configure_m20_teleop(env_cfg)

    profiler = RuntimeProfiler(args_cli.profile_json, interval_s=1.0e9)
    detailed_profiler = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    _disable_debug_vis(env)
    action_term = None
    if hasattr(env.action_manager, "active_terms"):
        try:
            action_term = env.action_manager.get_term("pre_trained_pick_action")
            action_term.cfg.absolute_commands = True
        except (KeyError, AttributeError):
            action_term = None

    if args_cli.profile_detailed:
        detailed_profiler = DetailedStepProfiler(
            env,
            profiler,
            action_term_name=(
                "pre_trained_pick_action"
                if action_term is not None
                else env.action_manager.active_terms[0]
            ),
        )
        detailed_profiler.install()

    stepper = None
    if args_cli.mode != "manager":
        stepper = FastTeleopStepper(
            env,
            render_hz=args_cli.fast_render_hz,
            write_targets_each_tick=args_cli.fast_write_each_tick,
            robot_hot_loop=args_cli.mode == "fast_hot",
            robot_hot_period_s=args_cli.fast_robot_hot_period,
            robot_only_write=args_cli.fast_robot_only_write,
            profiler=profiler,
        )

    env.reset()
    if action_term is not None and hasattr(action_term, "recalibrate"):
        action_term.recalibrate()

    action_shape = tuple(env.action_space.shape)
    actions = torch.zeros(action_shape, dtype=torch.float32, device=env.device)
    loop_ms: list[float] = []
    step_ms: list[float] = []

    for index in range(args_cli.warmup + args_cli.steps):
        loop_started = time.perf_counter()
        step_started = time.perf_counter()
        with torch.inference_mode():
            if stepper is None:
                env.step(actions)
            else:
                stepper.step(actions)
        step_elapsed_ms = (time.perf_counter() - step_started) * 1000.0
        loop_elapsed_ms = (time.perf_counter() - loop_started) * 1000.0
        if index < args_cli.warmup:
            continue
        step_ms.append(step_elapsed_ms)
        loop_ms.append(loop_elapsed_ms)
        profiler.record("benchmark_step_ms", step_elapsed_ms)
        profiler.record("benchmark_loop_ms", loop_elapsed_ms)

    result = {
        "task": args_cli.task,
        "mode": args_cli.mode,
        "device": str(env.device),
        "num_envs": args_cli.num_envs,
        "decimation": int(env.cfg.decimation),
        "physics_dt": float(env.physics_dt),
        "step_dt": float(env.step_dt),
        "warmup": args_cli.warmup,
        "steps": args_cli.steps,
        "ik_update_period_s": args_cli.ik_update_period,
        "robot_hot_period_s": (
            args_cli.fast_robot_hot_period if args_cli.mode == "fast_hot" else None
        ),
        "loop": {
            "fps": 1000.0 / statistics.fmean(loop_ms) if loop_ms else 0.0,
            "mean_ms": statistics.fmean(loop_ms) if loop_ms else 0.0,
            "p50_ms": _percentile(loop_ms, 0.50),
            "p95_ms": _percentile(loop_ms, 0.95),
            "max_ms": max(loop_ms, default=0.0),
        },
        "step": {
            "mean_ms": statistics.fmean(step_ms) if step_ms else 0.0,
            "p50_ms": _percentile(step_ms, 0.50),
            "p95_ms": _percentile(step_ms, 0.95),
            "max_ms": max(step_ms, default=0.0),
        },
        "profile": profiler.snapshot(),
    }

    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args_cli.profile_json:
        profiler.write(force=True)

    print(
        f"[control-benchmark] task={args_cli.task} mode={args_cli.mode} "
        f"device={env.device} fps={result['loop']['fps']:.2f} "
        f"loop_mean={result['loop']['mean_ms']:.2f}ms "
        f"loop_p95={result['loop']['p95_ms']:.2f}ms"
    )
    print(f"[control-benchmark] output={output_path}")

    if detailed_profiler is not None:
        detailed_profiler.uninstall()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
