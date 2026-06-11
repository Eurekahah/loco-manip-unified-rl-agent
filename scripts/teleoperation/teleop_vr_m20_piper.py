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
from rl_training.devices.vr_extented import Se2VRExtended, Se2VRExtendedCfg
import rl_training.tasks  # noqa: F401


def main():
    # ── 1. 创建环境 ────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # ── 2. 创建 VR 设备 ────────────────────────────────────────────────
    device_cfg = Se2VRExtendedCfg(
        # v_x_sensitivity=0.6,
        # v_y_sensitivity=0.4,
        # omega_z_sensitivity=0.8,
        # height_sensitivity=1.0,
        # pitch_sensitivity=1.0,
        # roll_sensitivity=1.0,
        # ee_pos_sensitivity=1.0,
        # ee_orn_sensitivity=1.0,
    )
    teleop_interface = Se2VRExtended(device_cfg)
    print(teleop_interface)

    # ── 3. 控制标志 ────────────────────────────────────────────────────
    teleoperation_active = True
    should_reset = False

    def request_reset() -> None:
        nonlocal should_reset
        should_reset = True
        print("[Teleop] Reset requested — will execute on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("[Teleop] Teleoperation ACTIVATED")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[Teleop] Teleoperation DEACTIVATED")

    # L 键 → 重置环境（Se2VRExtended 内部 L 已 reset 命令向量，此处额外 reset env）
    teleop_interface.add_callback("L", request_reset)

    print("\n" + "=" * 56)
    print("  遥操作控制说明（底盘 + 机身姿态 + 机械臂 + 夹爪）")
    print("=" * 56)
    print("  ── 底盘速度 ──────────────────────────────────────")
    print("  ↑ / ↓          前进 / 后退  (v_x)")
    print("  ← / →          左移 / 右移  (v_y)")
    print("  Z  / X         左转 / 右转  (ω_z)")
    print("  ── 机身姿态（增量，松键归零）──────────────────────")
    print("  C  / V         升高 / 降低机身  (Δheight)")
    print("  B  / N         抬头 / 低头      (Δpitch)")
    print("  [  / ]         左倾 / 右倾      (Δroll)")
    print("  ── 机械臂末端位置（数字键盘，增量）────────────────")
    print("  Num8 / Num2    EE Δx +/-")
    print("  Num4 / Num6    EE Δy +/-")
    print("  Num7 / Num9    EE Δz +/-")
    print("  ── 机械臂末端姿态（数字键盘，增量）────────────────")
    print("  Num1 / Num3    EE Δroll  +/-")
    print("  Num0 / Num.    EE Δpitch +/-")
    print("  Num+ / Num-    EE Δyaw   +/-")
    print("  ── 夹爪 ──────────────────────────────────────────")
    print("  G              切换夹爪开/合（toggle）")
    print("  ── 全局 ──────────────────────────────────────────")
    print("  L              重置环境 + 命令清零")
    print("=" * 56 + "\n")

    env.reset()
    teleop_interface.reset()

    # ── 4. 仿真主循环 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # 读取 13 维指令（已包含 gripper）
                # layout: [vx, vy, wz, Δex, Δey, Δez, Δer, Δep, Δeyaw,
                #          Δbh, Δbp, Δbr, gripper]
                cmd13: np.ndarray = teleop_interface.advance()   # (13,)

                if teleoperation_active:
                    actions = (
                        torch.tensor(cmd13, dtype=torch.float32, device=env.device)
                        .unsqueeze(0)
                        .expand(args.num_envs, -1)
                        .clone()
                    )   # (num_envs, 13)
                    # print(
                    #     f"[Teleop] chassis=({cmd13[0]:.2f},{cmd13[1]:.2f},{cmd13[2]:.2f}) "
                    #     f"ee_pos=({cmd13[3]:.3f},{cmd13[4]:.3f},{cmd13[5]:.3f}) "
                    #     f"ee_orn=({cmd13[6]:.3f},{cmd13[7]:.3f},{cmd13[8]:.3f}) "
                    #     f"body=({cmd13[9]:.3f},{cmd13[10]:.3f},{cmd13[11]:.3f}) "
                    #     f"grip={int(cmd13[12])}"
                    # )
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("[Teleop] Environment reset complete")

        except Exception as e:
            print(f"[Teleop] Error during simulation step: {e}")
            raise

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()