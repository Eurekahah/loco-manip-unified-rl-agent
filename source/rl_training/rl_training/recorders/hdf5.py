# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Task-agnostic camera and state recorder for teleoperation episodes."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.recorder_manager import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from rl_training.datasets.hdf5 import (
    ACTION_LAYOUT,
    HDF5Writer,
    invert_transform,
    pose_to_matrix,
)
from rl_training.datasets.outcomes import resolve_episode_outcome

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class HDF5Recorder(RecorderTerm):
    """Record configured robot, scene-entity, camera and termination data."""

    def __init__(self, cfg: HDF5RecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg: HDF5RecorderCfg
        if not cfg.action_term_name:
            raise ValueError("HDF5RecorderCfg.action_term_name must be configured by the task")
        if not cfg.tracked_entity_names:
            raise ValueError("HDF5RecorderCfg.tracked_entity_names must not be empty")
        self._pending_frames: dict[int, dict | None] = {
            env_id: None for env_id in range(env.num_envs)
        }
        self._last_outcomes: dict[int, dict] = {
            env_id: {} for env_id in range(env.num_envs)
        }
        self._episode_names: dict[int, str | None] = {
            env_id: None for env_id in range(env.num_envs)
        }
        self._episode_step_counts: dict[int, int] = {
            env_id: 0 for env_id in range(env.num_envs)
        }
        self._last_sensor_frames: dict[int, dict[str, int]] = {
            env_id: {} for env_id in range(env.num_envs)
        }
        self._next_record_sim_time: dict[int, float] = {
            env_id: 0.0 for env_id in range(env.num_envs)
        }
        self._recording_active: dict[int, bool] = {
            env_id: True for env_id in range(env.num_envs)
        }
        self._manual_outcomes: dict[int, bool] = {}
        # RecorderManager is instantiated before TerminationManager, so resolve
        # termination terms lazily once env.step() has initialized all managers.
        self._termination_names: tuple[str, ...] = ()
        self._camera_names = tuple(cfg.camera_names)
        if cfg.record_panorama and "fisheye_panorama" not in self._camera_names:
            self._camera_names = (*self._camera_names, "fisheye_panorama")
        self._fisheye_names = set(cfg.extra_metadata.get("fisheye_camera_names", ()))
        robot = env.scene[cfg.robot_name]
        self._gripper_joint_ids, _ = (
            robot.find_joints(cfg.gripper_joint_names)
            if cfg.gripper_joint_names
            else ([], [])
        )
        ee_ids, _ = robot.find_bodies(cfg.ee_body_name) if cfg.ee_body_name else ([], [])
        base_ids, _ = robot.find_bodies(cfg.base_link_name) if cfg.base_link_name else ([], [])
        self._ee_body_id = ee_ids[0] if ee_ids else None
        self._base_link_id = base_ids[0] if base_ids else None
        self._profiler = None
        self._writer = HDF5Writer(
            cfg.output_path,
            task=cfg.task,
            instruction=cfg.language_instruction or getattr(env.cfg, "language_instruction", ""),
            source=cfg.source,
            camera_names=self._camera_names,
            action_layout=ACTION_LAYOUT,
            metadata={
                "sim_dt": env.cfg.sim.dt,
                "control_dt": env.step_dt,
                "decimation": env.cfg.decimation,
                "render_interval": env.cfg.sim.render_interval,
                "num_envs": env.num_envs,
                "record_hz": cfg.record_hz,
                "preview_hz": cfg.preview_hz,
                "record_panorama": cfg.record_panorama,
                "joint_names": tuple(robot.data.joint_names),
                "termination_names": self._termination_names,
                "task_profile": cfg.task_profile,
                "tracked_entity_names": cfg.tracked_entity_names,
                "primary_object_name": cfg.primary_object_name,
                "primary_target_name": cfg.primary_target_name,
                "success_term_name": cfg.success_term_name,
                **cfg.extra_metadata,
            },
            overwrite=cfg.overwrite,
            compression=cfg.compression,
            record_hz=cfg.record_hz,
            async_write=cfg.async_write,
            queue_size=cfg.queue_size,
        )
        # Expose the term for the interactive collection script without relying on
        # RecorderManager private state.
        setattr(env, cfg.env_attr_name, self)

    @property
    def output_path(self) -> str:
        return self._writer.path

    @property
    def writer_stats(self) -> dict[str, float | int]:
        return self._writer.stats

    def set_profiler(self, profiler) -> None:
        self._profiler = profiler

    def set_recording_active(self, active: bool, env_id: int = 0) -> None:
        """Gate frame sampling without discarding already streamed data."""
        self._recording_active[int(env_id)] = bool(active)
        self._next_record_sim_time[int(env_id)] = 0.0
        self._last_sensor_frames[int(env_id)].clear()
        self._last_outcomes[int(env_id)].clear()

    def has_open_episode(self, env_id: int = 0) -> bool:
        return self._episode_names.get(int(env_id)) is not None

    def mark_outcome(self, success: bool, env_id: int = 0) -> None:
        """Attach a manual success/failure label to the next reset."""
        self._manual_outcomes[int(env_id)] = bool(success)

    def _capture_camera(
        self,
        env_id: int,
        camera_name: str,
        sensor,
    ) -> dict[str, np.ndarray]:
        data = sensor.data
        rgb = data.output.get("rgb")
        depth = data.output.get("depth")
        if rgb is None:
            raise RuntimeError(
                f"Camera '{sensor.cfg.prim_path}' has no RGB output. "
                "Create the collection env with cameras enabled and wait for one rendered reset/step."
            )

        rgb_np = _to_numpy(rgb[env_id]).astype(np.uint8)
        pos_w = _to_numpy(data.pos_w[env_id]).astype(np.float32)
        quat_world = _to_numpy(data.quat_w_world[env_id]).astype(np.float32)
        quat_ros = _to_numpy(data.quat_w_ros[env_id]).astype(np.float32)
        transform_world = pose_to_matrix(pos_w, quat_world).astype(np.float32)
        transform_ros = pose_to_matrix(pos_w, quat_ros).astype(np.float32)
        output = {
            "rgb": rgb_np,
            "pos_w": pos_w,
            "quat_w_world": quat_world,
            "quat_w_ros": quat_ros,
            "T_world_camera_world": transform_world,
            "T_world_camera_ros": transform_ros,
            "T_camera_world_world": invert_transform(transform_world).astype(np.float32),
            "T_camera_ros_world": invert_transform(transform_ros).astype(np.float32),
            "sensor_frame": np.int64(_to_numpy(sensor.frame).item()),
        }
        if camera_name not in self._fisheye_names:
            output["intrinsics"] = _to_numpy(data.intrinsic_matrices[env_id]).astype(np.float32)
        if depth is not None:
            depth_np = _to_numpy(depth[env_id]).astype(np.float32)
            if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            output["depth"] = depth_np
        return output

    def _camera_metadata(self) -> dict[str, dict]:
        metadata: dict[str, dict] = {}
        for camera_name in self._camera_names:
            if camera_name == "fisheye_panorama":
                metadata[camera_name] = {
                    "projection_model": "equirectangular",
                    "intrinsics_applicable": False,
                    "width": self.cfg.panorama_width,
                    "height": self.cfg.panorama_height,
                }
                continue
            sensor = self._env.scene.sensors[camera_name]
            camera_metadata = {
                "width": int(sensor.cfg.width),
                "height": int(sensor.cfg.height),
                "projection_model": "fisheye_equidistant"
                if camera_name in self._fisheye_names
                else "pinhole",
                "intrinsics_applicable": camera_name not in self._fisheye_names,
            }
            if camera_name in self._fisheye_names:
                camera_metadata.update(
                    {
                        "fisheye_model": "ideal_equidistant",
                        "fisheye_fov_deg": float(
                            self.cfg.extra_metadata.get("fisheye_fov_deg", 0.0)
                        ),
                        "fisheye_mount_pos_base_link": self.cfg.extra_metadata.get(
                            "fisheye_mount_pos_base_link", ()
                        ),
                        "fisheye_lens_offset_m": float(
                            self.cfg.extra_metadata.get("fisheye_lens_offset_m", 0.0)
                        ),
                    }
                )
            metadata[camera_name] = camera_metadata
        return metadata

    def _capture_panorama(self, env_id: int) -> dict[str, np.ndarray]:
        provider = getattr(self._env, "_fisheye_panorama_provider", None)
        if provider is None:
            raise RuntimeError("--record-panorama requires a configured fisheye panorama provider")
        rgb = np.asarray(provider(self._env), dtype=np.uint8)
        robot = self._env.scene[self.cfg.robot_name]
        if self._base_link_id is None:
            pos_w = _to_numpy(robot.data.root_pos_w[env_id]).astype(np.float32)
            quat_world = _to_numpy(robot.data.root_quat_w[env_id]).astype(np.float32)
        else:
            pos_w = _to_numpy(robot.data.body_pos_w[env_id, self._base_link_id]).astype(np.float32)
            quat_world = _to_numpy(
                robot.data.body_quat_w[env_id, self._base_link_id]
            ).astype(np.float32)
        transform_world = pose_to_matrix(pos_w, quat_world).astype(np.float32)
        return {
            "rgb": rgb,
            "pos_w": pos_w,
            "quat_w_world": quat_world,
            "T_world_camera_world": transform_world,
            "T_camera_world_world": invert_transform(transform_world).astype(np.float32),
            "sensor_frame": np.int64(self._env.common_step_counter),
        }

    def _capture_rigid_entity(self, env_id: int, entity_name: str) -> dict[str, np.ndarray]:
        entity = self._env.scene[entity_name]
        return {
            "root_pos_w": _to_numpy(entity.data.root_pos_w[env_id]).astype(np.float32),
            "root_quat_w": _to_numpy(entity.data.root_quat_w[env_id]).astype(np.float32),
            "root_lin_vel_w": _to_numpy(entity.data.root_lin_vel_w[env_id]).astype(np.float32),
            "root_ang_vel_w": _to_numpy(entity.data.root_ang_vel_w[env_id]).astype(np.float32),
        }

    def _episode_metadata(self, env_id: int) -> dict:
        return {
            "env_id": env_id,
            "record_hz": self.cfg.record_hz,
            "compression": self.cfg.compression,
            "camera_metadata": self._camera_metadata(),
        }

    def record_pre_step(self):
        """Capture the observation/action pair before physics advances."""
        period = 1.0 / float(self.cfg.record_hz)
        sim_time = float(self._env.common_step_counter) * float(self._env.step_dt)
        if not any(self._recording_active.values()):
            return None, None
        due_env_ids = [
            env_id
            for env_id in range(self._env.num_envs)
            if self._recording_active[env_id]
            and sim_time + 1e-9 >= self._next_record_sim_time[env_id]
        ]
        if not due_env_ids:
            return None, None

        robot = self._env.scene[self.cfg.robot_name]
        action_term = self._env.action_manager.get_term(self.cfg.action_term_name)
        raw_actions = self._env.action_manager.action
        processed_actions = torch.cat(
            [
                self._env.action_manager.get_term(name).processed_actions
                for name in self._env.action_manager.active_terms
            ],
            dim=-1,
        )
        timestamp = time.time()
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)

        for env_id in due_env_ids:
            camera_data = {}
            sensor_frames: dict[str, int] = {}
            unchanged = False
            for camera_name in self._camera_names:
                if camera_name == "fisheye_panorama":
                    continue
                sensor = self._env.scene.sensors[camera_name]
                sensor_frame = int(_to_numpy(sensor.frame).item())
                sensor_frames[camera_name] = sensor_frame
                if self._last_sensor_frames[env_id].get(camera_name) == sensor_frame:
                    unchanged = True
                camera_data[camera_name] = self._capture_camera(env_id, camera_name, sensor)
            if unchanged:
                self._next_record_sim_time[env_id] = sim_time + period
                if self._profiler is not None:
                    self._profiler.increment("skipped_duplicate_camera_ticks")
                continue
            if self.cfg.record_panorama:
                camera_data["fisheye_panorama"] = self._capture_panorama(env_id)

            entities = {
                entity_name: self._capture_rigid_entity(env_id, entity_name)
                for entity_name in self.cfg.tracked_entity_names
            }
            frame = {
                "timestamp": np.float64(timestamp),
                "sim_time": np.float64(sim_time),
                "step_index": np.int64(self._env.episode_length_buf[env_id].item()),
                "action": _to_numpy(raw_actions[env_id]).astype(np.float32),
                "processed_action": _to_numpy(processed_actions[env_id]).astype(np.float32),
                "ll_command": _to_numpy(action_term.ll_command[env_id]).astype(np.float32),
                "robot": {
                    "root_pos_w": _to_numpy(robot.data.root_pos_w[env_id]).astype(np.float32),
                    "root_quat_w": _to_numpy(robot.data.root_quat_w[env_id]).astype(np.float32),
                    "root_lin_vel_w": _to_numpy(robot.data.root_lin_vel_w[env_id]).astype(np.float32),
                    "root_ang_vel_w": _to_numpy(robot.data.root_ang_vel_w[env_id]).astype(np.float32),
                    "joint_pos": _to_numpy(robot.data.joint_pos[env_id]).astype(np.float32),
                    "joint_vel": _to_numpy(robot.data.joint_vel[env_id]).astype(np.float32),
                    "ee_pos_w": _to_numpy(
                        robot.data.body_pos_w[env_id, self._ee_body_id]
                        if self._ee_body_id is not None
                        else robot.data.root_pos_w[env_id]
                    ).astype(np.float32),
                    "ee_quat_w": _to_numpy(
                        robot.data.body_quat_w[env_id, self._ee_body_id]
                        if self._ee_body_id is not None
                        else robot.data.root_quat_w[env_id]
                    ).astype(np.float32),
                    "gripper_pos": _to_numpy(
                        robot.data.joint_pos[env_id, self._gripper_joint_ids]
                    ).astype(np.float32),
                    "roll": np.float32(roll[env_id].item()),
                    "pitch": np.float32(pitch[env_id].item()),
                    "yaw": np.float32(yaw[env_id].item()),
                },
                "entities": entities,
                "camera": camera_data,
            }
            if self.cfg.primary_object_name in entities:
                frame["object"] = entities[self.cfg.primary_object_name]
            if self.cfg.primary_target_name in entities:
                frame["target_pad"] = entities[self.cfg.primary_target_name]
            self._pending_frames[env_id] = frame
            self._last_sensor_frames[env_id] = sensor_frames
            self._next_record_sim_time[env_id] = sim_time + period
        return None, None

    def record_post_step(self):
        """Attach flags, retain the latest outcome, and enqueue any pending frame."""
        if not self._termination_names:
            self._termination_names = tuple(self._env.termination_manager.active_terms)
            self._writer.set_metadata(termination_names=self._termination_names)
        terminated = self._env.termination_manager.terminated
        truncated = self._env.termination_manager.time_outs
        has_pending_frame = any(
            frame is not None for frame in self._pending_frames.values()
        )
        has_reset = bool(terminated.any().item() or truncated.any().item())
        if not has_pending_frame and not has_reset:
            return None, None
        for env_id in range(self._env.num_envs):
            frame = self._pending_frames[env_id]
            flags = [
                bool(self._env.termination_manager.get_term(name)[env_id].item())
                for name in self._termination_names
            ]
            success = (
                np.bool_(flags[self._termination_names.index(self.cfg.success_term_name)])
                if self.cfg.success_term_name in self._termination_names
                else np.bool_(False)
            )
            self._last_outcomes[env_id] = {
                "termination_flags": np.asarray(flags, dtype=bool).copy(),
                "terminated": np.bool_(terminated[env_id].item()),
                "truncated": np.bool_(truncated[env_id].item()),
                "success": success,
            }
            if frame is None:
                continue
            frame["termination_flags"] = self._last_outcomes[env_id][
                "termination_flags"
            ]
            frame["terminated"] = self._last_outcomes[env_id]["terminated"]
            frame["truncated"] = self._last_outcomes[env_id]["truncated"]
            frame["success"] = success
            if self._episode_names[env_id] is None:
                self._episode_names[env_id] = self._writer.start_episode()
            enqueue_started = time.perf_counter()
            appended = self._writer.append_frame(
                self._episode_names[env_id],
                frame,
                episode_metadata=self._episode_metadata(env_id),
            )
            enqueue_ms = (time.perf_counter() - enqueue_started) * 1000.0
            if self._profiler is not None:
                self._profiler.record("hdf5_enqueue_ms", enqueue_ms)
            if appended:
                self._episode_step_counts[env_id] += 1
            elif self._profiler is not None:
                self._profiler.increment("dropped_record_frames")
            self._pending_frames[env_id] = None
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        """Persist complete episodes before Isaac Lab clears the terminated env."""
        if not self._termination_names:
            self._termination_names = tuple(self._env.termination_manager.active_terms)
            self._writer.set_metadata(termination_names=self._termination_names)
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        for env_id in [int(item) for item in env_ids]:
            frame = self._pending_frames[env_id]
            episode_name = self._episode_names[env_id]
            if frame is not None and episode_name is not None:
                if self._termination_names:
                    frame.setdefault(
                        "termination_flags",
                        np.zeros(len(self._termination_names), dtype=bool),
                    )
                    frame.setdefault("terminated", np.bool_(False))
                    frame.setdefault("truncated", np.bool_(False))
                    frame.setdefault("success", np.bool_(False))
                appended = self._writer.append_frame(
                    episode_name,
                    frame,
                    episode_metadata=self._episode_metadata(env_id),
                )
                if appended:
                    self._episode_step_counts[env_id] += 1
                self._pending_frames[env_id] = None
            if episode_name is None:
                continue
            last = frame if frame is not None else self._last_outcomes[env_id]
            manual_outcome = self._manual_outcomes.pop(env_id, None)
            success, termination_label, auto_success, _ = resolve_episode_outcome(
                last,
                manual_outcome,
                self._termination_names,
            )
            if manual_outcome is True:
                last["success"] = np.bool_(True)
            self._writer.finalize_episode(
                episode_name,
                success=success,
                termination_label=termination_label,
                termination_names=self._termination_names,
                episode_metadata={"env_id": env_id, "auto_success": auto_success},
            )
            print(
                f"[HDF5Recorder:{self.cfg.task_profile}] finalized {episode_name}: "
                f"steps={self._episode_step_counts[env_id]} success={success} "
                f"label={termination_label}"
            )
            self._episode_names[env_id] = None
            self._episode_step_counts[env_id] = 0
            self._last_sensor_frames[env_id].clear()
            self._last_outcomes[env_id].clear()
            self._next_record_sim_time[env_id] = 0.0
        return None, None

    def close(self, file_path: str):
        """Persist any episode that is still open when the app exits."""
        for env_id in range(self._env.num_envs):
            frame = self._pending_frames[env_id]
            episode_name = self._episode_names[env_id]
            if frame is not None and episode_name is not None:
                appended = self._writer.append_frame(
                    episode_name,
                    frame,
                    episode_metadata=self._episode_metadata(env_id),
                )
                if appended:
                    self._episode_step_counts[env_id] += 1
                self._pending_frames[env_id] = None
            if episode_name is None:
                continue
            last = frame if frame is not None else self._last_outcomes[env_id]
            manual_outcome = self._manual_outcomes.pop(env_id, None)
            success, _, auto_success, fired = resolve_episode_outcome(
                last,
                manual_outcome,
                self._termination_names,
            )
            label = "shutdown_success" if success else ("shutdown_manual_stop" if not fired else "|".join(fired))
            self._writer.finalize_episode(
                episode_name,
                success=success,
                termination_label=label,
                termination_names=self._termination_names,
                episode_metadata={"env_id": env_id, "auto_success": auto_success},
            )
            print(
                f"[HDF5Recorder:{self.cfg.task_profile}] wrote open episode on close: "
                f"{self._writer.path}"
            )
            self._episode_names[env_id] = None
            self._episode_step_counts[env_id] = 0
            self._last_outcomes[env_id].clear()
        self._writer.close()


@configclass
class HDF5RecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = HDF5Recorder

    output_path: str = "recordings/episode.hdf5"
    task: str = "unknown"
    task_profile: str = "generic"
    language_instruction: str = ""
    source: str = "unknown"
    # Keep the HDF5 source order aligned with the SmolVLA camera suffixes:
    # top -> camera1, wrist -> camera2, workcell -> camera3.
    camera_names: tuple[str, ...] = ()
    robot_name: str = "robot"
    action_term_name: str = ""
    tracked_entity_names: tuple[str, ...] = ()
    primary_object_name: str = ""
    primary_target_name: str = ""
    success_term_name: str = ""
    ee_body_name: str = ""
    base_link_name: str = ""
    gripper_joint_names: tuple[str, ...] = ()
    env_attr_name: str = "hdf5_recorder"
    extra_metadata: dict[str, object] = field(default_factory=dict)
    overwrite: bool = False
    record_hz: float = 10.0
    preview_hz: float = 15.0
    compression: str = "lzf"
    async_write: bool = True
    queue_size: int = 8
    record_panorama: bool = False
    panorama_width: int = 960
    panorama_height: int = 480


@configclass
class HDF5RecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    record_hdf5: HDF5RecorderCfg = HDF5RecorderCfg()
