# scripts/teleoperation/teleop_m20_piper_vr.py

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
# ── 手感门控：每项独立开关，便于真机逐项 A/B（默认只有 stick 整形开，其余全关 = 旧手感）──
g = parser.add_argument_group("VR 手感门控")
g.add_argument("--no-stick-shaping", action="store_true", help="关闭步骤1：摇杆死区/曲线/斜率限制（回到纯线性）")
g.add_argument("--arm-filter", action="store_true", help="步骤2：机械臂手柄信号连续死区 + 低通")
g.add_argument("--arm-clutch", action="store_true", help="步骤3：机械臂离合键（按住右 grip 才跟随，松开冻结）")
g.add_argument("--body-filter", action="store_true", help="步骤4a：机体姿态连续死区 + 低通")
g.add_argument("--body-clutch", action="store_true", help="步骤4b：机体姿态离合键（按住左 grip 才跟手）")
g.add_argument("--all-feel", action="store_true", help="一次打开步骤 2/3/4 全部门控")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 以下在 sim 启动后 import ---
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.devices.vr_loco_manip_teleop import VRLocoManipTeleop, VRLocoManipTeleopCfg
import rl_training.tasks  # noqa: F401


def main():
    # ── 1. 创建环境 ────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None
    env_cfg.actions.pre_trained_pick_action.debug_vis = not args.headless
    env_cfg.actions.pre_trained_pick_action.debug_vis_ee_target_only = True
    # VR 遥操：执行端采用"绝对目标 = 标定基准 + 手柄偏移"语义（不滚动累加），
    # 避免手柄微小偏差被积分放大到极限；键盘脚本不设置此项，保持增量语义
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    # 拿到执行端 action term，用于 B 键标定时重锚基准
    action_term = env.action_manager.get_term("pre_trained_pick_action")

    # ── 2. 创建 VR 设备 ────────────────────────────────────────────────
    device_cfg = VRLocoManipTeleopCfg(
        stick_shaping_enabled=not args.no_stick_shaping,
        arm_filter_enabled=args.arm_filter or args.all_feel,
        arm_clutch_enabled=args.arm_clutch or args.all_feel,
        body_filter_enabled=args.body_filter or args.all_feel,
        body_clutch_enabled=args.body_clutch or args.all_feel,
        # v_x_sensitivity=0.6,
        # v_y_sensitivity=0.4,
        # omega_z_sensitivity=0.8,
        # height_sensitivity=1.0,
        # pitch_sensitivity=1.0,
        # roll_sensitivity=1.0,
        # ee_pos_sensitivity=1.0,
        # ee_orn_sensitivity=1.0,
    )
    print(
        "[VR feel gates] "
        f"stick_shaping={device_cfg.stick_shaping_enabled} arm_filter={device_cfg.arm_filter_enabled} "
        f"arm_clutch={device_cfg.arm_clutch_enabled} body_filter={device_cfg.body_filter_enabled} "
        f"body_clutch={device_cfg.body_clutch_enabled}"
    )
    teleop_interface = VRLocoManipTeleop(device_cfg)
    # 摇杆斜率限制按仿真步长（而非墙钟）计算，不受仿真实时因子影响
    _step_dt = getattr(env, "step_dt", None)
    if _step_dt is not None:
        teleop_interface.set_step_dt(float(_step_dt))
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

    def request_recalibrate() -> None:
        action_term.recalibrate()
        print("[Teleop] Action-term reference recalibrated to current robot pose")

    # 左 X / 左 Y → 重置环境（VRLocoManipTeleop 内部已 reset 命令向量，此处额外 reset env + 重锚基准）
    teleop_interface.add_callback("R", request_reset)
    teleop_interface.add_callback("N", request_reset)
    # 右 B → 开始控制 + 重新标定（重锚执行端基准到机器人当前位姿）
    teleop_interface.add_callback("S", request_recalibrate)

    print("\n" + "=" * 56)
    print("  VR 遥操作控制说明（底盘 + 机身姿态 + 机械臂 + 夹爪）")
    print("=" * 56)
    print("  ── 底盘速度 ──────────────────────────────────────")
    print("  左摇杆 Y        前进 / 后退  (v_x)")
    print("  左摇杆 X        左移 / 右移  (v_y)")
    print("  右摇杆 X        左转 / 右转  (ω_z)")
    print("  ── 机身姿态（相对标定原点，静止即保持）────────────")
    print("  左手 Z 移动     升高 / 降低机身  (Δheight)")
    print("  左手旋转        机身 pitch / roll")
    print("  左手扳机        按住持续升高机身")
    print("  ── 机械臂末端（相对标定原点，静止即保持）──────────")
    print("  右手位置偏移    EE 位置（x/y/z）")
    print("  右手旋转偏移    EE 姿态（roll/pitch/yaw）")
    print("  ── 夹爪 ──────────────────────────────────────────")
    print("  右手扳机        按住开爪，松开合爪")
    print("  ── 全局 ──────────────────────────────────────────")
    print("  右 B           开始控制 + 重新标定")
    print("  左 X / 左 Y    重置环境 + 命令清零")
    print("=" * 56 + "\n")

    env.reset()
    teleop_interface.reset()
    action_term.recalibrate()

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
                    action_term.recalibrate()
                    should_reset = False
                    print("[Teleop] Environment reset complete")

        except Exception as e:
            print(f"[Teleop] Error during simulation step: {e}")
            raise

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
