# scripts/teleoperation/teleop_wheeled_legged.py

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 以下在 sim 启动后 import ---
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.devices.se2_keyboard_extended import (
    Se2KeyboardExtended, Se2KeyboardExtendedCfg
)
import rl_training.tasks  # noqa: F401

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
    # 1. 创建环境（你已有的 manager-based 环境）
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None


    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # 2. 创建设备
    device_cfg = Se2KeyboardExtendedCfg(
        v_x_sensitivity=0.6,
        v_y_sensitivity=0.4,
        omega_z_sensitivity=0.8,
        height_sensitivity=1.0,
        pitch_sensitivity=1.0,
        roll_sensitivity=1.0,
    )
    teleop_interface = Se2KeyboardExtended(device_cfg)

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

    # simulate environment
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
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            print(f"Error during simulation step: {e}")
            break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()