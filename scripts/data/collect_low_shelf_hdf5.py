#!/usr/bin/env python3
"""Collect low-shelf RGB-D demonstrations with keyboard or VR teleoperation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect low-shelf pick-and-place demonstrations to HDF5.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-LowShelf-Collect-v0")
parser.add_argument("--input", choices=("keyboard", "vr"), default="keyboard")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N environment steps; 0 runs until app exit.")
parser.add_argument("--camera_width", type=int, default=224)
parser.add_argument("--camera_height", type=int, default=224)
parser.add_argument("--camera_update_period", type=float, default=0.1)
parser.add_argument("--scene_camera", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wrist_camera", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.scene_camera and not args_cli.wrist_camera:
    parser.error("At least one of --scene_camera or --wrist_camera must be enabled")
if args_cli.num_envs != 1:
    parser.error("Interactive collection currently requires --num_envs 1")

# Camera data is part of the collection contract. Do not rely on the user to
# remember --enable_cameras, especially for headless smoke tests.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_training.devices.keyboard_loco_manip_teleop import KeyboardLocoManipTeleop, KeyboardLocoManipTeleopCfg  # noqa: E402
from rl_training.devices.vr_loco_manip_teleop import VRLocoManipTeleop, VRLocoManipTeleopCfg  # noqa: E402
from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (  # noqa: E402
    make_workcell_camera,
    make_wrist_camera,
)
import rl_training.tasks  # noqa: F401, E402


def _default_output_path(source: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("recordings", "low_shelf_place", f"{stamp}_{source}.hdf5")


def main() -> None:
    output_path = args_cli.output or _default_output_path(args_cli.input)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1

    camera_names: list[str] = []
    env_cfg.scene.workcell_camera = None
    env_cfg.scene.wrist_camera = None
    if args_cli.scene_camera:
        env_cfg.scene.workcell_camera = make_workcell_camera(
            width=args_cli.camera_width,
            height=args_cli.camera_height,
            update_period=args_cli.camera_update_period,
        )
        camera_names.append("workcell_camera")
    if args_cli.wrist_camera:
        env_cfg.scene.wrist_camera = make_wrist_camera(
            width=args_cli.camera_width,
            height=args_cli.camera_height,
            update_period=args_cli.camera_update_period,
        )
        camera_names.append("wrist_camera")

    recorder_cfg = env_cfg.recorders.record_hdf5
    recorder_cfg.output_path = output_path
    recorder_cfg.source = args_cli.input
    recorder_cfg.camera_names = tuple(camera_names)
    recorder_cfg.overwrite = args_cli.overwrite
    recorder_cfg.language_instruction = env_cfg.language_instruction
    # Keep collection visuals focused on the commanded EE pose in GUI mode.
    env_cfg.actions.pre_trained_pick_action.debug_vis = not args_cli.headless
    env_cfg.actions.pre_trained_pick_action.debug_vis_ee_target_only = True

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    action_term.cfg.absolute_commands = args_cli.input == "vr"
    recorder = env.low_shelf_hdf5_recorder

    pending_outcome: bool | None = None

    def request_reset(success: bool) -> None:
        nonlocal pending_outcome
        pending_outcome = success

    if args_cli.input == "vr":
        teleop = VRLocoManipTeleop(VRLocoManipTeleopCfg())
        teleop.add_callback("R", lambda: request_reset(False))
        teleop.add_callback("N", lambda: request_reset(True))
        teleop.add_callback("S", lambda: action_term.recalibrate())
    else:
        teleop = KeyboardLocoManipTeleop(
            KeyboardLocoManipTeleopCfg(
                v_x_sensitivity=0.6,
                v_y_sensitivity=0.4,
                omega_z_sensitivity=0.8,
                height_sensitivity=0.05,
                pitch_sensitivity=0.05,
                roll_sensitivity=0.05,
                ee_pos_sensitivity=0.05,
                ee_orn_sensitivity=0.05,
            )
        )
        teleop.add_callback("L", lambda: request_reset(False))
        teleop.add_callback("K", lambda: request_reset(True))

    print(teleop)
    print("\n" + "=" * 64)
    print("  Low-shelf RGB-D collection")
    print("=" * 64)
    print(f"  task       : {args_cli.task}")
    print(f"  input      : {args_cli.input}")
    print(f"  cameras    : {', '.join(camera_names)}")
    print(f"  HDF5       : {os.path.abspath(output_path)}")
    if args_cli.input == "vr":
        print("  VR start   : right B (start + recalibrate)")
        print("  VR success : left Y")
        print("  VR failure : left X")
    else:
        print("  keyboard success : K")
        print("  keyboard failure : L")
    print("=" * 64 + "\n")

    env.reset()
    teleop.reset()
    if args_cli.input == "vr":
        action_term.recalibrate()

    step_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                cmd13 = teleop.advance()

                if pending_outcome is not None:
                    recorder.mark_outcome(pending_outcome)
                    env.reset()
                    teleop.reset()
                    if args_cli.input == "vr":
                        action_term.recalibrate()
                    print(f"[Collect] saved manual outcome success={pending_outcome}")
                    pending_outcome = None
                    continue

                actions = (
                    torch.tensor(cmd13, dtype=torch.float32, device=env.device)
                    .unsqueeze(0)
                    .expand(env.num_envs, -1)
                    .clone()
                )
                _, _, terminated, truncated, _ = env.step(actions)
                step_count += 1

                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    print(
                        f"[Collect] episode ended: terminated={bool(terminated[0].item())} "
                        f"truncated={bool(truncated[0].item())}"
                    )
                    teleop.reset()
                    if args_cli.input == "vr":
                        action_term.recalibrate()

                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    print(f"[Collect] reached --max_steps={args_cli.max_steps}")
                    break
    except KeyboardInterrupt:
        print("[Collect] interrupted; saving the open episode")
    finally:
        # Flush an open episode before env.close() tears down the recorder manager.
        recorder.close(recorder.output_path)
        env.close()
        print(f"[Collect] HDF5 written to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
