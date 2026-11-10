# scripts/teleoperation/teleop_m20_piper_keyboard.py

import argparse
import copy

import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--render-interval",
    type=int,
    default=4,
    help="Isaac Sim 每多少个物理步渲染一次；默认 4（每个 env step 渲染一次）。",
)
parser.add_argument(
    "--camera-view",
    choices=("free", "chest", "wrist", "both"),
    default="free",
    help="GUI 视角：free=自由视角，chest=胸前，wrist=腕部，both=胸前主视口 + 独立腕部视口。",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(rendering_mode="performance")
args = parser.parse_args()
if args.render_interval < 1:
    parser.error("--render-interval 必须 >= 1")
if args.headless and args.camera_view != "free":
    parser.error("--camera-view 需要 GUI；headless 模式只能使用 free。")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 以下在 sim 启动后 import ---
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.devices.keyboard_loco_manip_teleop import KeyboardLocoManipTeleop, KeyboardLocoManipTeleopCfg
import rl_training.tasks  # noqa: F401


class _TeleopCameraRig:
    """Bind the main viewport and an optional second viewport to teleop cameras."""

    def __init__(self, env, mode: str):
        self.extra_viewport = None
        if mode == "free":
            return

        import omni.kit.commands
        from omni.kit.viewport.utility import get_active_viewport

        active_viewport = get_active_viewport()
        nav_camera_path = (
            str(env.scene["nav_camera"]._view.prim_paths[0])
            if mode in {"chest", "both"}
            else None
        )
        arm_camera_path = (
            str(env.scene["arm_camera"]._view.prim_paths[0])
            if mode in {"wrist", "both"}
            else None
        )
        primary_path = nav_camera_path if mode in {"chest", "both"} else arm_camera_path
        omni.kit.commands.execute(
            "SetViewportCamera",
            camera_path=primary_path,
            viewport_api=active_viewport,
        )

        if mode == "both":
            from isaacsim.core.utils.viewports import create_viewport_for_camera

            self.extra_viewport = create_viewport_for_camera(
                "Wrist Camera",
                arm_camera_path,
                width=640,
                height=480,
                position_x=80,
                position_y=80,
            )

    def close(self):
        if self.extra_viewport is not None:
            self.extra_viewport.destroy()
            self.extra_viewport = None


def main():
    # ── 1. 创建环境 ────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None
    env_cfg.sim.render_interval = int(args.render_interval)

    if args.camera_view != "free":
        if args.camera_view in {"chest", "both"}:
            from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import (
                HighLevelSceneCfg,
            )

            reference_scene_cfg = HighLevelSceneCfg(
                num_envs=args.num_envs, env_spacing=5.0
            )
            nav_cfg = copy.deepcopy(reference_scene_cfg.nav_camera)
            nav_cfg.prim_path = "{ENV_REGEX_NS}/Robot/base_link/nav_camera"
            nav_cfg.data_types = ["rgb"]
            env_cfg.scene.nav_camera = nav_cfg
        if args.camera_view in {"wrist", "both"}:
            from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
                make_wrist_camera,
            )

            arm_cfg = make_wrist_camera(width=224, height=224, update_period=0.1)
            arm_cfg.data_types = ["rgb"]
            env_cfg.scene.arm_camera = arm_cfg

    env_cfg.actions.pre_trained_pick_action.debug_vis = False
    env_cfg.actions.pre_trained_pick_action.debug_vis_ee_target_only = False
    if getattr(env_cfg, "commands", None) is not None and hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.actions.pre_trained_pick_action.low_level_ee_actions.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.cfg.sim.render_interval = int(args.render_interval)
    env.sim.cfg.render_interval = int(args.render_interval)
    env.command_manager.set_debug_vis(False)
    env.action_manager.set_debug_vis(False)

    import omni.usd
    from pxr import UsdGeom

    visuals_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/Visuals")
    if visuals_prim.IsValid():
        UsdGeom.Imageable(visuals_prim).MakeInvisible()

    camera_rig = _TeleopCameraRig(env, args.camera_view)

    # ── 2. 创建键盘设备 ────────────────────────────────────────────────
    device_cfg = KeyboardLocoManipTeleopCfg(
        v_x_sensitivity=0.6,
        v_y_sensitivity=0.4,
        omega_z_sensitivity=0.8,
        height_sensitivity=0.05,
        pitch_sensitivity=0.05,
        roll_sensitivity=0.05,
        ee_pos_sensitivity=0.05,
        ee_orn_sensitivity=0.05,
    )
    teleop_interface = KeyboardLocoManipTeleop(device_cfg)
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

    # L 键 → 重置环境（KeyboardLocoManipTeleop 内部 L 已 reset 命令向量，此处额外 reset env）
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

    camera_rig.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
