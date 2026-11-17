# scripts/teleoperation/teleop_m20_loco_keyboard.py

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from isaaclab.app import AppLauncher

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fisheye_cli import add_fisheye_args, validate_fisheye_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N environment steps; 0 runs until exit.")
add_fisheye_args(parser, include_panorama=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance")
args = parser.parse_args()
validate_fisheye_args(parser, args, include_panorama=True)
if args.max_steps < 0:
    parser.error("--max-steps must be >= 0")
if args.show_wrist_camera:
    parser.error("--show-wrist-camera is not available in this locomotion-only script")
args.enable_cameras = bool(args.fisheye_rig != "none" or args.show_panorama)
if args.fisheye_rig != "none" and not any(
    item.startswith("--rendering_mode") for item in sys.argv[1:]
):
    args.rendering_mode = "balanced"
    print("[Teleop] fisheye enabled; using rendering_mode=balanced unless explicitly overridden")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 以下在 sim 启动后 import ---
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.devices.keyboard_loco_teleop import (
    KeyboardLocoTeleop, KeyboardLocoTeleopCfg
)
import rl_training.tasks  # noqa: F401
from fisheye_profiler import RuntimeProfiler
from fisheye_teleop import TeleopFisheyeDisplay, add_fisheye_rig

def build_13d_action(
    cmd6: np.ndarray,   # [vx, vy, wz, delta_height, delta_pitch, delta_roll]
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """
    把键盘 6 维指令映射到 13 维 action tensor。

    ActionTerm 的 action_dim=13:
      [0]   vx
      [1]   vy
      [2]   wz
      [3:6] delta_ee_pos  (body系，填0 → EE不动)
      [6:9] delta_ee_orn  (body系，填0 → EE不动)
      [9]   delta_height
      [10]  delta_pitch
      [11]  delta_roll
      [12]  gripper       (填0 → 夹爪不动)
    """
    action_12 = np.zeros(13, dtype=np.float32)
    action_12[0]  = cmd6[0]   # vx
    action_12[1]  = cmd6[1]   # vy
    action_12[2]  = cmd6[2]   # wz
    # [3:9] 保持 0 → EE 增量为零 → target_ee_pos_b 不变
    action_12[9]  = cmd6[3]   # delta_height
    action_12[10] = cmd6[4]   # delta_pitch
    action_12[11] = cmd6[5]   # delta_roll
    # [12] 保持 0 → 夹爪不动
    action_t = torch.tensor(action_12, dtype=torch.float32, device=device)
    return action_t.unsqueeze(0).expand(num_envs, -1).clone()  # (num_envs, 13)


def main():
    profiler = RuntimeProfiler(args.profile_json, interval_s=args.profile_interval)
    # 1. 创建环境（你已有的 manager-based 环境）
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None
    fisheye_spec = add_fisheye_rig(
        env_cfg,
        mode=args.fisheye_rig,
        width=args.fisheye_width,
        height=args.fisheye_height,
        update_period=args.fisheye_update_period,
        fov_deg=args.fisheye_fov_deg,
        mount_pos=args.fisheye_mount_pos,
        lens_offset=args.fisheye_lens_offset,
    )

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    fisheye_display = TeleopFisheyeDisplay(
        fisheye_spec,
        show_panorama=args.show_panorama,
        panorama_width=args.panorama_width,
        panorama_height=args.panorama_height,
        wrist_display_hz=args.wrist_display_hz,
        panorama_display_hz=args.panorama_display_hz,
        preview_hz=args.preview_hz,
        panorama_backend=args.panorama_backend,
        panorama_window_pos=tuple(args.panorama_window_pos),
        profiler=profiler,
        ui_snapshot_dir=args.ui_snapshot_dir,
    )

    # 2. 创建设备
    device_cfg = KeyboardLocoTeleopCfg(
        v_x_sensitivity=0.6,
        v_y_sensitivity=0.4,
        omega_z_sensitivity=0.8,
        height_sensitivity=1.0,
        pitch_sensitivity=1.0,
        roll_sensitivity=1.0,
    )
    teleop_interface = KeyboardLocoTeleop(device_cfg)

    # 3. 注册 reset 回调（按 L 重置环境）
    teleop_interface.add_callback("RESET", lambda: env.reset())
    teleoperation_active = True
    should_reset_recording_instance = False

    # 用 nonlocal flag 的闭包写法，延迟到 simulation loop 中执行 reset
    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    print("\n=== 底盘遥操作控制说明 ===")
    print("W/S      → 前进/后退 (v_x)")
    print("A/D      → 左移/右移 (v_y)")
    print("Q/E      → 左转/右转 (w_z)")
    print("R/F      → 升高/降低机身 (height)")
    print("T/G      → 抬头/低头 (pitch)")
    print("Y/H      → 左倾/右倾 (roll)")
    print("P        → 重置姿态")
    print("L        → 重置环境")
    print("=" * 28)

    teleop_interface.add_callback("RESET", reset_recording_instance)
    teleop_interface.add_callback("START", start_teleoperation)
    teleop_interface.add_callback("STOP", stop_teleoperation)

    env.reset()
    teleop_interface.reset()
    profiler.reset_clock()

    # simulate environment
    step_count = 0
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                cmd6 = teleop_interface.advance() # np.ndarray (6,)

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = build_13d_action(cmd6, args.num_envs, env.device)   # (num_envs, 13)
                    print(f"Teleop action (6D): {cmd6}")
                    # print(f"Teleop action (13D): {actions[0]}")
                    # apply actions
                    with profiler.measure("env_step_ms"):
                        env.step(actions)
                else:
                    with profiler.measure("env_step_ms"):
                        env.sim.render()

                fisheye_display.update(env)
                step_count += 1

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
                if args.max_steps > 0 and step_count >= args.max_steps:
                    print(f"Reached --max-steps={args.max_steps}")
                    break
                profiler.tick()
        except Exception as e:
            print(f"Error during simulation step: {e}")
            break

    fisheye_display.save_ui_snapshots()
    fisheye_display.close()
    env.close()
    profiler.write(force=True)
    profiler.print_summary()

if __name__ == "__main__":
    main()
    simulation_app.close()
