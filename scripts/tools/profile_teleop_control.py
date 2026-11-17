"""Compare arm joint trajectories across normal and fast teleoperation stepping."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


TOOLS_DIR = Path(__file__).resolve().parent
TELEOP_DIR = TOOLS_DIR.parent / "teleoperation"
for path in (TOOLS_DIR, TELEOP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

parser = argparse.ArgumentParser(description="Profile M20 teleoperation arm trajectories.")
parser.add_argument("--task", default="Isaac-M20-Piper-Teleop-History-v0")
parser.add_argument("--mode", choices=("normal", "fast", "fast_hot"), default="normal")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ik-update-period", type=float, default=0.02)
parser.add_argument("--fast-render-hz", type=float, default=15.0)
parser.add_argument("--fast-write-each-tick", action="store_true")
parser.add_argument("--fast-robot-hot-period", type=float, default=0.005)
parser.add_argument("--fast-robot-only-write", action="store_true")
parser.add_argument("--output", default="outputs/fast_teleop/arm_trace.npz")
parser.add_argument(
    "--action",
    type=float,
    nargs=13,
    default=(0.0,) * 13,
    help="Fixed high-level action used during the trace.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import rl_training.tasks  # noqa: F401, E402
from fast_teleop import FastTeleopStepper  # noqa: E402


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jitter_metrics(name: str, values: np.ndarray) -> dict[str, float]:
    if len(values) < 3:
        return {}
    first_diff = np.diff(values, axis=0)
    second_diff = np.diff(values, n=2, axis=0)
    return {
        f"{name}_step_rms": float(np.sqrt(np.mean(first_diff**2))),
        f"{name}_second_diff_rms": float(np.sqrt(np.mean(second_diff**2))),
        f"{name}_second_diff_max": float(np.max(np.abs(second_diff))),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.actions.pre_trained_pick_action.debug_vis = False
    env_cfg.actions.pre_trained_pick_action.debug_vis_ee_target_only = False
    env_cfg.actions.pre_trained_pick_action.low_level_ee_actions.debug_vis = False
    env_cfg.actions.pre_trained_pick_action.ik_update_period_s = args_cli.ik_update_period

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.command_manager.set_debug_vis(False)
    env.action_manager.set_debug_vis(False)
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    action_term.cfg.absolute_commands = True

    robot = env.scene["robot"]
    arm_ids, arm_names = robot.find_joints("arm_joint.*")
    stepper = None
    if args_cli.mode != "normal":
        stepper = FastTeleopStepper(
            env,
            render_hz=args_cli.fast_render_hz,
            write_targets_each_tick=args_cli.fast_write_each_tick,
            robot_hot_loop=args_cli.mode == "fast_hot",
            robot_hot_period_s=args_cli.fast_robot_hot_period,
            robot_only_write=args_cli.fast_robot_only_write,
        )

    env.reset()
    action_term.recalibrate()
    actions = torch.tensor(args_cli.action, dtype=torch.float32, device=env.device).repeat(
        env.num_envs, 1
    )

    q_values = []
    dq_values = []
    target_values = []
    for step in range(args_cli.warmup + args_cli.steps):
        with torch.inference_mode():
            if stepper is None:
                env.step(actions)
            else:
                stepper.step(actions)
        if step < args_cli.warmup:
            continue
        q_values.append(_to_numpy(robot.data.joint_pos[0, arm_ids]))
        dq_values.append(_to_numpy(robot.data.joint_vel[0, arm_ids]))
        target_values.append(_to_numpy(robot.data.joint_pos_target[0, arm_ids]))

    q = np.stack(q_values)
    dq = np.stack(dq_values)
    target = np.stack(target_values)
    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        q=q,
        dq=dq,
        target=target,
        arm_joint_names=np.asarray(arm_names),
        mode=np.asarray(args_cli.mode),
    )

    print(f"[arm-trace] mode={args_cli.mode} steps={len(q)} output={output_path}")
    metrics = {
        **_jitter_metrics("q", q),
        **_jitter_metrics("dq", dq),
        **_jitter_metrics("target", target),
    }
    for name, value in metrics.items():
        print(f"[arm-trace] {name}={value:.8f}")
    print(
        f"[arm-trace] dq_abs_max={float(np.max(np.abs(dq))):.6f} "
        f"q_range={float(np.max(q.max(axis=0) - q.min(axis=0))):.6f}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
