# scripts/teleoperation/teleop_m20_piper_vr.py

import argparse
import copy
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
parser.add_argument(
    "--relative-ik",
    action="store_true",
    help="IK 对比门控：把标定目标差分成每步相对位姿后送 IK；默认走绝对目标 IK",
)
parser.add_argument(
    "--render-interval",
    type=int,
    default=4,
    help=(
        "Isaac Sim 每多少个物理步渲染一次。默认 4（每个 env step 渲染一次，"
        "保证第一人称画面连续）；增大可降低渲染频率。"
    ),
)
parser.add_argument(
    "--camera-view",
    choices=("free", "chest", "wrist", "both"),
    default="free",
    help="GUI 视角：free=自由视角，chest=胸前，wrist=腕部，both=胸前主视口 + 独立腕部视口。",
)
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N environment steps; 0 runs until exit.")
parser.add_argument(
    "--fast-teleop",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use the lightweight teleoperation stepping path without training managers.",
)
parser.add_argument(
    "--fast-render-hz",
    type=float,
    default=15.0,
    help="Wall-clock render frequency for fast teleoperation.",
)
parser.add_argument(
    "--fast-write-each-tick",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Write action targets every 5 ms physics tick; default writes once per 20 ms control tick.",
)
parser.add_argument(
    "--ik-update-period",
    type=float,
    default=0.0,
    help="IK recompute period in simulation seconds; 0 keeps every physics tick.",
)
parser.add_argument(
    "--profile-detailed",
    action="store_true",
    help="Profile action manager, low-level policy, IK, scene, simulation and render separately.",
)
add_fisheye_args(parser, include_panorama=True)
parser.add_argument("--follow-chest-camera", action="store_true", help=argparse.SUPPRESS)
# 兼容旧启动命令：这三项现在已经是默认行为，保留为无操作开关，避免旧命令报错。
parser.add_argument("--arm-clutch", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--body-clutch", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--no-stick-shaping", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
# 遥操优先低延迟；仍可用 --rendering_mode balanced/quality 显式覆盖。
parser.set_defaults(rendering_mode="performance")
args = parser.parse_args()
camera_view = "chest" if args.follow_chest_camera else args.camera_view
if args.headless and camera_view != "free":
    parser.error("--camera-view 需要 GUI；headless 模式只能使用 free。")
if args.max_steps < 0:
    parser.error("--max-steps must be >= 0")
if args.fast_render_hz <= 0.0:
    parser.error("--fast-render-hz must be > 0")
if args.ik_update_period < 0.0:
    parser.error("--ik-update-period must be >= 0")
validate_fisheye_args(parser, args, include_panorama=True)
args.enable_cameras = bool(
    getattr(args, "enable_cameras", False)
    or args.fisheye_rig != "none"
    or camera_view != "free"
    or args.show_panorama
    or args.show_wrist_camera
)
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
from rl_training.devices.vr_loco_manip_teleop import VRLocoManipTeleop, VRLocoManipTeleopCfg
import rl_training.tasks  # noqa: F401
from fisheye_profiler import RuntimeProfiler
from fisheye_teleop import TeleopFisheyeDisplay, add_fisheye_rig
from fast_teleop import DetailedStepProfiler, FastTeleopStepper


class _TeleopCameraRig:
    """Drive the main GUI viewport and an optional second wrist viewport."""

    def __init__(self, env, mode: str):
        self.mode = mode
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
    profiler = RuntimeProfiler(args.profile_json, interval_s=args.profile_interval)
    # ── 1. 创建环境 ────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.terminations.time_out = None
    if args.render_interval < 1:
        parser.error("--render-interval 必须 >= 1")
    env_cfg.sim.render_interval = int(args.render_interval)

    if camera_view != "free" or args.show_wrist_camera:
        if camera_view in {"chest", "both"}:
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
        if camera_view in {"wrist", "both"} or args.show_wrist_camera:
            from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
                make_wrist_camera,
            )

            arm_cfg = make_wrist_camera(
                width=224,
                height=224,
                update_period=0.0,
            )
            arm_cfg.data_types = ["rgb"]
            env_cfg.scene.arm_camera = arm_cfg

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

    # 第一人称视角下 EE 目标箭头贴近镜头，会遮挡整个画面；遥操不需要该 marker。
    env_cfg.actions.pre_trained_pick_action.debug_vis = False
    env_cfg.actions.pre_trained_pick_action.debug_vis_ee_target_only = False
    if getattr(env_cfg, "commands", None) is not None and hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = False
    term_cfg = env_cfg.actions.pre_trained_pick_action
    # VR 遥操：执行端采用"绝对目标 = 标定基准 + 手柄偏移"语义（不滚动累加），
    # 避免手柄微小偏差被积分放大到极限；键盘脚本不设置此项，保持增量语义
    term_cfg.absolute_commands = True
    # IK 对比门控：两种模式都保留“目标偏移”作为高层命令/观测；
    # relative_ik 只改变送进 IK 的位姿形式（绝对 7D vs 每步相对 6D）。
    term_cfg.relative_ik_commands = bool(args.relative_ik)
    term_cfg.low_level_ee_actions.controller.command_type = "pose"
    term_cfg.low_level_ee_actions.controller.use_relative_mode = bool(args.relative_ik)
    term_cfg.low_level_ee_actions.debug_vis = False
    term_cfg.ik_update_period_s = args.ik_update_period
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    # 保险：关闭 Manager 级 debug 订阅，覆盖 command/action term 自己创建的 marker。
    env.command_manager.set_debug_vis(False)
    env.action_manager.set_debug_vis(False)
    # 再隐藏整个 /Visuals 调试层，避免已经 spawn 的 marker prim 仍然可见。
    import omni.usd
    from pxr import UsdGeom

    visuals_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/Visuals")
    if visuals_prim.IsValid():
        UsdGeom.Imageable(visuals_prim).MakeInvisible()
    # gym.make 可能持有自己的 cfg 副本；运行期再写一次，确保 render_interval 真正生效。
    env.cfg.sim.render_interval = int(args.render_interval)
    env.sim.cfg.render_interval = int(args.render_interval)
    # 拿到执行端 action term，用于 B 键标定时重锚基准
    action_term = env.action_manager.get_term("pre_trained_pick_action")

    camera_rig = _TeleopCameraRig(env, camera_view)
    fisheye_display = TeleopFisheyeDisplay(
        fisheye_spec,
        show_panorama=args.show_panorama,
        panorama_width=args.panorama_width,
        panorama_height=args.panorama_height,
        show_wrist_camera=args.show_wrist_camera,
        wrist_scene_key="arm_camera",
        wrist_display_hz=args.wrist_display_hz,
        panorama_display_hz=args.panorama_display_hz,
        preview_hz=args.preview_hz,
        panorama_backend=args.panorama_backend,
        wrist_window_width=args.wrist_window_width,
        wrist_window_height=args.wrist_window_height,
        wrist_window_pos=tuple(args.wrist_window_pos),
        panorama_window_pos=tuple(args.panorama_window_pos),
        profiler=profiler,
        ui_snapshot_dir=args.ui_snapshot_dir,
    )

    # ── 2. 创建 VR 设备 ────────────────────────────────────────────────
    # 默认值即当前最舒服的组合：
    #   --no-stick-shaping --arm-clutch --body-clutch
    device_cfg = VRLocoManipTeleopCfg()
    print(
        "[VR feel] "
        f"stick_shaping={device_cfg.stick_shaping_enabled} "
        f"arm_clutch={device_cfg.arm_clutch_enabled} body_clutch={device_cfg.body_clutch_enabled} "
        f"relative_ik={term_cfg.relative_ik_commands} render_interval={env.cfg.sim.render_interval} "
        f"camera_view={camera_view}"
    )
    teleop_interface = VRLocoManipTeleop(device_cfg)
    detailed_profiler = None
    fast_stepper = None
    if args.profile_detailed:
        detailed_profiler = DetailedStepProfiler(env, profiler)
        detailed_profiler.install()
    if args.fast_teleop:
        fast_stepper = FastTeleopStepper(
            env,
            render_hz=args.fast_render_hz,
            write_targets_each_tick=args.fast_write_each_tick,
            profiler=profiler,
        )
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
    profiler.reset_clock()

    # ── 4. 仿真主循环 ──────────────────────────────────────────────────
    step_count = 0
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
                    with profiler.measure("env_step_ms"):
                        if fast_stepper is not None:
                            fast_stepper.step(actions)
                        else:
                            env.step(actions)
                else:
                    with profiler.measure("env_step_ms"):
                        env.sim.render()

                fisheye_display.update(env)
                step_count += 1

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    action_term.recalibrate()
                    should_reset = False
                    print("[Teleop] Environment reset complete")
                if args.max_steps > 0 and step_count >= args.max_steps:
                    print(f"[Teleop] reached --max-steps={args.max_steps}")
                    break
                profiler.tick()

        except Exception as e:
            print(f"[Teleop] Error during simulation step: {e}")
            raise

    fisheye_display.save_ui_snapshots()
    fisheye_display.close()
    camera_rig.close()
    if detailed_profiler is not None:
        detailed_profiler.uninstall()
    env.close()
    profiler.write(force=True)
    profiler.print_summary()


if __name__ == "__main__":
    main()
    simulation_app.close()
