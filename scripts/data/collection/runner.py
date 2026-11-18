"""Generic interactive HDF5 collection loop."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
from rl_training.datasets.sampling import align_period_to_step
from rl_training.devices.keyboard_loco_manip_teleop import (
    KeyboardLocoManipTeleop,
    KeyboardLocoManipTeleopCfg,
)
from rl_training.devices.vr_loco_manip_teleop import (
    VRLocoManipTeleop,
    VRLocoManipTeleopCfg,
)
import rl_training.tasks  # noqa: F401

from fisheye_profiler import RuntimeProfiler
from fisheye_teleop import TeleopFisheyeDisplay, add_fisheye_rig

TELEOP_DIR = Path(__file__).resolve().parents[2] / "teleoperation"
if str(TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(TELEOP_DIR))

from fast_teleop import DetailedStepProfiler, FastTeleopStepper  # noqa: E402

from .profiles import CollectionTaskProfile, resolve_profile
from .termination import (
    compute_fast_termination_status,
    validate_success_termination_term,
)


class KeyboardAbsoluteAdapter:
    """Convert keyboard deltas to the absolute-offset collection action space."""

    def __init__(self, profile: CollectionTaskProfile) -> None:
        self._offset = np.zeros(profile.command_dim, dtype=np.float32)
        self._slices = profile.keyboard_absolute_slices
        if profile.keyboard_limits:
            self._limits = profile.keyboard_limits
        else:
            vr_cfg = VRLocoManipTeleopCfg()
            self._limits = (
                (-vr_cfg.arm_clutch_pos_limit, vr_cfg.arm_clutch_pos_limit),
                (-vr_cfg.arm_clutch_rot_limit, vr_cfg.arm_clutch_rot_limit),
                (-vr_cfg.body_clutch_limit, vr_cfg.body_clutch_limit),
            )
        if len(self._limits) != len(self._slices):
            raise ValueError("keyboard_limits must match keyboard_absolute_slices")

    def reset(self) -> None:
        self._offset[:] = 0.0

    def __call__(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).copy()
        if command.shape != self._offset.shape:
            raise ValueError(
                f"Teleop command shape {command.shape} does not match profile "
                f"command_dim {self._offset.shape[0]}"
            )
        for (start, stop), (lower, upper) in zip(self._slices, self._limits):
            self._offset[start:stop] = np.clip(
                self._offset[start:stop] + command[start:stop],
                lower,
                upper,
            )
            command[start:stop] = self._offset[start:stop]
        return command


def _default_output_path(profile: CollectionTaskProfile, source: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("recordings", profile.output_dir, f"{stamp}_{source}.hdf5")


def _configure_standard_cameras(
    env_cfg,
    profile: CollectionTaskProfile,
    args_cli: argparse.Namespace,
    *,
    update_period: float,
) -> list[str]:
    camera_names: list[str] = []
    enabled = {
        "top_camera": args_cli.top_camera,
        "wrist_camera": args_cli.wrist_camera,
        "workcell_camera": args_cli.scene_camera,
        "front_camera": args_cli.front_camera,
    }
    for camera_name, is_enabled in enabled.items():
        setattr(env_cfg.scene, camera_name, None)
        if not is_enabled:
            continue
        try:
            factory = profile.camera_factories[camera_name]
        except KeyError as exc:
            raise ValueError(
                f"Profile '{profile.key}' does not define camera '{camera_name}'"
            ) from exc
        setattr(
            env_cfg.scene,
            camera_name,
            factory(
                width=args_cli.camera_width,
                height=args_cli.camera_height,
                update_period=update_period,
                with_depth=args_cli.record_depth,
            ),
        )
        camera_names.append(camera_name)
    return camera_names


def run_collection(
    args_cli: argparse.Namespace,
    parser: argparse.ArgumentParser,
    simulation_app,
) -> None:
    profile = resolve_profile(
        args_cli.task,
        profile_key=args_cli.profile,
        profile_modules=tuple(args_cli.profile_module),
    )
    output_path = args_cli.output or _default_output_path(profile, args_cli.input)
    profiler = RuntimeProfiler(args_cli.profile_json, interval_s=args_cli.profile_interval)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    if profile.configure_env is not None:
        profile.configure_env(env_cfg)

    record_period = 1.0 / float(args_cli.record_hz)
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    camera_update_period = align_period_to_step(
        min(
            args_cli.camera_update_period
            if args_cli.camera_update_period > 0.0
            else record_period,
            record_period,
        ),
        control_dt,
    )
    fisheye_update_period = align_period_to_step(
        min(
            args_cli.fisheye_update_period
            if args_cli.fisheye_update_period > 0.0
            else record_period,
            record_period,
        ),
        control_dt,
    )

    camera_names = _configure_standard_cameras(
        env_cfg,
        profile,
        args_cli,
        update_period=camera_update_period,
    )
    default_fisheye_mount = (-0.44, 0.0, 0.34)
    fisheye_mount_pos = (
        profile.fisheye_mount_pos
        if profile.fisheye_mount_pos is not None
        and tuple(args_cli.fisheye_mount_pos) == default_fisheye_mount
        else tuple(args_cli.fisheye_mount_pos)
    )
    fisheye_spec = add_fisheye_rig(
        env_cfg,
        mode=args_cli.fisheye_rig,
        width=args_cli.fisheye_width,
        height=args_cli.fisheye_height,
        update_period=fisheye_update_period,
        fov_deg=args_cli.fisheye_fov_deg,
        mount_pos=fisheye_mount_pos,
        lens_offset=args_cli.fisheye_lens_offset,
        with_depth=args_cli.record_depth,
    )
    camera_names.extend(fisheye_spec.scene_keys)

    if not hasattr(env_cfg, "recorders") or not hasattr(env_cfg.recorders, "record_hdf5"):
        raise ValueError(
            f"Task '{args_cli.task}' does not expose recorders.record_hdf5. "
            "Use a *-Collect task or configure HDF5RecorderManagerCfg in the task env cfg."
        )
    recorder_cfg = env_cfg.recorders.record_hdf5
    recorder_cfg.output_path = output_path
    recorder_cfg.source = args_cli.input
    recorder_cfg.camera_names = tuple(camera_names)
    recorder_cfg.task = args_cli.task
    recorder_cfg.task_profile = profile.key
    recorder_cfg.action_term_name = profile.action_term_name
    recorder_cfg.tracked_entity_names = profile.tracked_entity_names
    recorder_cfg.primary_object_name = profile.primary_object_name
    recorder_cfg.primary_target_name = profile.primary_target_name
    recorder_cfg.success_term_name = profile.success_term_name
    recorder_cfg.ee_body_name = profile.ee_body_name
    recorder_cfg.base_link_name = profile.base_link_name
    recorder_cfg.gripper_joint_names = profile.gripper_joint_names
    recorder_cfg.env_attr_name = profile.recorder_attr_name
    recorder_cfg.language_instruction = getattr(env_cfg, profile.instruction_attr, "")
    recorder_cfg.overwrite = args_cli.overwrite
    recorder_cfg.record_hz = args_cli.record_hz
    recorder_cfg.preview_hz = args_cli.preview_hz
    recorder_cfg.compression = args_cli.compression
    recorder_cfg.async_write = args_cli.async_write
    recorder_cfg.queue_size = args_cli.writer_queue_size
    recorder_cfg.record_panorama = args_cli.record_panorama
    recorder_cfg.panorama_width = args_cli.panorama_width
    recorder_cfg.panorama_height = args_cli.panorama_height

    metadata = dict(profile.extra_metadata)
    metadata["stepping_mode"] = "fast" if args_cli.fast_stepping else "manager"
    if args_cli.fast_stepping:
        metadata.update(
            {
                "fast_render_hz": args_cli.fast_render_hz,
                "fast_robot_hot_period_s": args_cli.fast_robot_hot_period,
                "fast_robot_only_write": args_cli.fast_robot_only_write,
                "ik_update_period_s": args_cli.ik_update_period,
                "automatic_termination": True,
            }
        )
    if fisheye_spec.enabled:
        metadata.update(
            {
                "fisheye_rig": fisheye_spec.mode,
                "fisheye_projection": "equidistant",
                "fisheye_fov_deg": fisheye_spec.fov_deg,
                "fisheye_mount_pos_base_link": list(fisheye_mount_pos),
                "fisheye_lens_offset_m": fisheye_spec.lens_offset,
                "fisheye_camera_names": list(fisheye_spec.scene_keys),
                "fisheye_projection_model": "ideal_equidistant",
                "record_panorama": args_cli.record_panorama,
                "panorama_width": args_cli.panorama_width,
                "panorama_height": args_cli.panorama_height,
            }
        )
    recorder_cfg.extra_metadata = metadata

    action_cfg = getattr(env_cfg.actions, profile.action_term_name, None)
    if action_cfg is not None and hasattr(action_cfg, "debug_vis"):
        action_cfg.debug_vis = not args_cli.headless
    if action_cfg is not None and hasattr(action_cfg, "debug_vis_ee_target_only"):
        action_cfg.debug_vis_ee_target_only = True
    if action_cfg is not None and hasattr(action_cfg, "ik_update_period_s"):
        action_cfg.ik_update_period_s = args_cli.ik_update_period

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    action_term = env.action_manager.get_term(profile.action_term_name)
    validate_success_termination_term(
        env.termination_manager,
        profile.success_term_name,
    )
    if hasattr(action_term.cfg, "absolute_commands"):
        action_term.cfg.absolute_commands = True
    if profile.configure_action_term is not None:
        profile.configure_action_term(action_term)

    detailed_profiler = None
    fast_stepper = None
    if args_cli.profile_detailed:
        detailed_profiler = DetailedStepProfiler(
            env,
            profiler,
            action_term_name=profile.action_term_name,
        )
        detailed_profiler.install()
    if args_cli.fast_stepping:
        fast_stepper = FastTeleopStepper(
            env,
            render_hz=args_cli.fast_render_hz,
            robot_hot_loop=True,
            robot_hot_period_s=args_cli.fast_robot_hot_period,
            robot_only_write=args_cli.fast_robot_only_write or profile.fast_robot_only_write,
            profiler=profiler,
        )

    recorder = getattr(env, profile.recorder_attr_name)
    recorder.set_profiler(profiler)
    fisheye_display = TeleopFisheyeDisplay(
        fisheye_spec,
        show_panorama=args_cli.show_panorama,
        panorama_width=args_cli.panorama_width,
        panorama_height=args_cli.panorama_height,
        show_wrist_camera=args_cli.show_wrist_camera,
        wrist_scene_key="wrist_camera",
        wrist_display_hz=args_cli.wrist_display_hz,
        panorama_display_hz=args_cli.panorama_display_hz,
        preview_hz=args_cli.preview_hz,
        panorama_backend=args_cli.panorama_backend,
        wrist_window_width=args_cli.wrist_window_width,
        wrist_window_height=args_cli.wrist_window_height,
        wrist_window_pos=tuple(args_cli.wrist_window_pos),
        panorama_window_pos=tuple(args_cli.panorama_window_pos),
        record_panorama=args_cli.record_panorama,
        profiler=profiler,
        ui_snapshot_dir=args_cli.ui_snapshot_dir,
    )
    if args_cli.record_panorama:
        env._fisheye_panorama_provider = fisheye_display.capture_panorama

    pending_outcome: bool | None = None
    keyboard_adapter = (
        KeyboardAbsoluteAdapter(profile) if args_cli.input == "keyboard" else None
    )

    def request_reset(success: bool) -> None:
        nonlocal pending_outcome
        pending_outcome = success

    def recalibrate() -> None:
        if hasattr(action_term, "recalibrate"):
            action_term.recalibrate()

    if args_cli.input == "vr":
        teleop = VRLocoManipTeleop(VRLocoManipTeleopCfg())
        teleop.add_callback("R", lambda: request_reset(False))
        teleop.add_callback("N", lambda: request_reset(True))
        teleop.add_callback("S", recalibrate)
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
    print(f"  {profile.display_name} collection")
    print("=" * 64)
    print(f"  task       : {args_cli.task}")
    print(f"  profile    : {profile.key}")
    print(f"  input      : {args_cli.input}")
    print(f"  cameras    : {', '.join(camera_names)}")
    print(f"  fisheye    : {args_cli.fisheye_rig}")
    print(f"  depth      : {'rgb+depth' if args_cli.record_depth else 'rgb only'}")
    print(
        f"  sampling   : {args_cli.record_hz:.1f} Hz, compression={args_cli.compression}, "
        f"async={args_cli.async_write}, queue={args_cli.writer_queue_size}"
    )
    print(
        f"  preview    : wrist={args_cli.show_wrist_camera} panorama={args_cli.show_panorama} "
        f"backend={args_cli.panorama_backend}"
    )
    print(f"  stepping   : {'fast' if args_cli.fast_stepping else 'manager'}")
    if args_cli.fast_stepping:
        print(
            "  fast note  : automatic termination/success switching enabled; "
            "manual success/failure still overrides"
        )
    print(f"  HDF5       : {os.path.abspath(output_path)}")
    print("  action     : absolute offsets from the recalibrated anchor")
    if args_cli.input == "vr":
        print("  VR start   : right B (start + recalibrate)")
        print("  VR success : left Y")
        print("  VR failure : left X")
    else:
        print("  keyboard success : K")
        print("  keyboard failure : L")
    print("=" * 64 + "\n")

    def reset_episode_state() -> None:
        teleop.reset()
        if keyboard_adapter is not None:
            keyboard_adapter.reset()
        recalibrate()

    env.reset()
    reset_episode_state()
    profiler.reset_clock()

    step_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                command = teleop.advance()
                if keyboard_adapter is not None:
                    command = keyboard_adapter(command)

                if pending_outcome is not None:
                    recorder.mark_outcome(pending_outcome)
                    env.reset()
                    reset_episode_state()
                    print(f"[Collect] saved manual outcome success={pending_outcome}")
                    pending_outcome = None
                    continue

                actions = (
                    torch.tensor(command, dtype=torch.float32, device=env.device)
                    .unsqueeze(0)
                    .expand(env.num_envs, -1)
                    .clone()
                )
                if fast_stepper is not None:
                    with profiler.measure("env_step_ms"):
                        fast_stepper.process_action(actions)
                        recorder.record_pre_step()
                        fast_stepper.advance()
                        reset_buf, terminated, truncated = compute_fast_termination_status(
                            env.termination_manager
                        )
                        recorder.record_post_step()
                    episode_done = bool(reset_buf[0].item())
                else:
                    with profiler.measure("env_step_ms"):
                        _, _, terminated, truncated, _ = env.step(actions)
                    episode_done = bool(terminated[0].item()) or bool(truncated[0].item())
                fisheye_display.update(env)
                step_count += 1

                if episode_done:
                    print(
                        f"[Collect] episode ended: terminated={bool(terminated[0].item())} "
                        f"truncated={bool(truncated[0].item())}"
                    )
                    if fast_stepper is not None:
                        # Fast stepping bypasses ManagerBasedRLEnv.step(), which normally
                        # resets terminated envs after recording their final frame.
                        env.reset()
                    reset_episode_state()

                if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                    print(f"[Collect] reached --max_steps={args_cli.max_steps}")
                    break
                stats = recorder.writer_stats
                profiler.set_counter("writer_queue_length", stats["queue_length"])
                profiler.set_counter("writer_dropped_frames", stats["dropped_frames"])
                profiler.set_counter("writer_write_ms_mean", stats["hdf5_write_ms_mean"])
                profiler.tick()
    except KeyboardInterrupt:
        print("[Collect] interrupted; saving the open episode")
    finally:
        fisheye_display.save_ui_snapshots()
        fisheye_display.close()
        if detailed_profiler is not None:
            detailed_profiler.uninstall()
        recorder.close(recorder.output_path)
        env.close()
        profiler.write(force=True)
        profiler.print_summary()
        print(f"[Collect] HDF5 written to {os.path.abspath(output_path)}")
