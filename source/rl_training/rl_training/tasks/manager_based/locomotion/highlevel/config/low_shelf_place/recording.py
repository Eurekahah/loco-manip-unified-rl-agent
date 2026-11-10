# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Camera and state recorder for low-shelf teleoperation episodes."""

from __future__ import annotations

import time
from collections.abc import Sequence
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

from rl_training.datasets.low_shelf_hdf5 import (
    ACTION_LAYOUT,
    LowShelfHDF5Writer,
    invert_transform,
    pose_to_matrix,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class LowShelfHDF5Recorder(RecorderTerm):
    """Record pre-step observations/actions and post-step termination labels."""

    def __init__(self, cfg: LowShelfHDF5RecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg: LowShelfHDF5RecorderCfg
        self._frames: dict[int, list[dict]] = {env_id: [] for env_id in range(env.num_envs)}
        self._manual_outcomes: dict[int, bool] = {}
        # RecorderManager is instantiated before TerminationManager, so resolve
        # termination terms lazily once env.step() has initialized all managers.
        self._termination_names: tuple[str, ...] = ()
        self._camera_names = tuple(cfg.camera_names)
        self._gripper_joint_ids, _ = env.scene["robot"].find_joints(
            ("gripper_joint1", "gripper_joint2")
        )
        self._ee_body_id = env.scene["robot"].find_bodies("gripper_base")[0][0]
        self._writer = LowShelfHDF5Writer(
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
                "joint_names": tuple(env.scene["robot"].data.joint_names),
                "termination_names": self._termination_names,
            },
            overwrite=cfg.overwrite,
        )
        # Expose the term for the interactive collection script without relying on
        # RecorderManager private state.
        env.low_shelf_hdf5_recorder = self

    @property
    def output_path(self) -> str:
        return self._writer.path

    def mark_outcome(self, success: bool, env_id: int = 0) -> None:
        """Attach a manual success/failure label to the next reset."""
        self._manual_outcomes[int(env_id)] = bool(success)

    def _capture_camera(self, env_id: int, sensor) -> dict[str, np.ndarray]:
        data = sensor.data
        rgb = data.output.get("rgb")
        depth = data.output.get("depth")
        if rgb is None or depth is None:
            raise RuntimeError(
                f"Camera '{sensor.cfg.prim_path}' has no RGB-D output. "
                "Create the collection env with cameras enabled and wait for one rendered reset/step."
            )

        rgb_np = _to_numpy(rgb[env_id]).astype(np.uint8)
        depth_np = _to_numpy(depth[env_id]).astype(np.float32)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        intrinsics = _to_numpy(data.intrinsic_matrices[env_id]).astype(np.float32)
        pos_w = _to_numpy(data.pos_w[env_id]).astype(np.float32)
        quat_world = _to_numpy(data.quat_w_world[env_id]).astype(np.float32)
        quat_ros = _to_numpy(data.quat_w_ros[env_id]).astype(np.float32)
        transform_world = pose_to_matrix(pos_w, quat_world).astype(np.float32)
        transform_ros = pose_to_matrix(pos_w, quat_ros).astype(np.float32)
        return {
            "rgb": rgb_np,
            "depth": depth_np,
            "intrinsics": intrinsics,
            "pos_w": pos_w,
            "quat_w_world": quat_world,
            "quat_w_ros": quat_ros,
            "T_world_camera_world": transform_world,
            "T_world_camera_ros": transform_ros,
            "T_camera_world_world": invert_transform(transform_world).astype(np.float32),
            "T_camera_ros_world": invert_transform(transform_ros).astype(np.float32),
            "sensor_frame": np.int64(_to_numpy(sensor.frame).item()),
        }

    def record_pre_step(self):
        """Capture the observation/action pair before physics advances."""
        robot = self._env.scene["robot"]
        obj = self._env.scene["object"]
        target = self._env.scene["target_pad"]
        action_term = self._env.action_manager.get_term("pre_trained_pick_action")
        raw_actions = self._env.action_manager.action
        processed_actions = torch.cat(
            [
                self._env.action_manager.get_term(name).processed_actions
                for name in self._env.action_manager.active_terms
            ],
            dim=-1,
        )
        timestamp = time.time()
        sim_time = float(self._env.common_step_counter) * float(self._env.step_dt)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)

        for env_id in range(self._env.num_envs):
            camera_data = {}
            for camera_name in self._camera_names:
                sensor = self._env.scene.sensors[camera_name]
                camera_data[camera_name] = self._capture_camera(env_id, sensor)

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
                    "ee_pos_w": _to_numpy(robot.data.body_pos_w[env_id, self._ee_body_id]).astype(np.float32),
                    "ee_quat_w": _to_numpy(robot.data.body_quat_w[env_id, self._ee_body_id]).astype(np.float32),
                    "gripper_pos": _to_numpy(robot.data.joint_pos[env_id, self._gripper_joint_ids]).astype(np.float32),
                    "roll": np.float32(roll[env_id].item()),
                    "pitch": np.float32(pitch[env_id].item()),
                    "yaw": np.float32(yaw[env_id].item()),
                },
                "object": {
                    "root_pos_w": _to_numpy(obj.data.root_pos_w[env_id]).astype(np.float32),
                    "root_quat_w": _to_numpy(obj.data.root_quat_w[env_id]).astype(np.float32),
                    "root_lin_vel_w": _to_numpy(obj.data.root_lin_vel_w[env_id]).astype(np.float32),
                },
                "target_pad": {
                    "root_pos_w": _to_numpy(target.data.root_pos_w[env_id]).astype(np.float32),
                    "root_quat_w": _to_numpy(target.data.root_quat_w[env_id]).astype(np.float32),
                },
                "camera": camera_data,
            }
            self._frames[env_id].append(frame)
        return None, None

    def record_post_step(self):
        """Attach all termination flags to the just-recorded action/state frame."""
        if not self._termination_names:
            self._termination_names = tuple(self._env.termination_manager.active_terms)
            self._writer.set_metadata(termination_names=self._termination_names)
        terminated = self._env.termination_manager.terminated
        truncated = self._env.termination_manager.time_outs
        for env_id in range(self._env.num_envs):
            if not self._frames[env_id]:
                continue
            flags = [
                bool(self._env.termination_manager.get_term(name)[env_id].item())
                for name in self._termination_names
            ]
            frame = self._frames[env_id][-1]
            frame["termination_flags"] = np.asarray(flags, dtype=bool)
            frame["terminated"] = np.bool_(terminated[env_id].item())
            frame["truncated"] = np.bool_(truncated[env_id].item())
            frame["success"] = (
                np.bool_(flags[self._termination_names.index("place_success")])
                if "place_success" in self._termination_names
                else np.bool_(False)
            )
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        """Persist complete episodes before Isaac Lab clears the terminated env."""
        if not self._termination_names:
            self._termination_names = tuple(self._env.termination_manager.active_terms)
            self._writer.set_metadata(termination_names=self._termination_names)
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        for env_id in [int(item) for item in env_ids]:
            frames = self._frames[env_id]
            if not frames:
                continue
            last = frames[-1]
            flags = np.asarray(last.get("termination_flags", np.zeros(len(self._termination_names), dtype=bool)))
            fired = [name for name, value in zip(self._termination_names, flags) if bool(value)]
            manual_outcome = self._manual_outcomes.pop(env_id, None)
            auto_success = bool(last.get("success", False))
            success = auto_success if manual_outcome is None else manual_outcome
            if manual_outcome is True:
                termination_label = "manual_success"
                last["success"] = np.bool_(True)
            elif manual_outcome is False:
                termination_label = "manual_failure"
            elif fired:
                termination_label = "|".join(fired)
            else:
                termination_label = "manual_reset"
            episode_name = self._writer.write_episode(
                frames,
                success=success,
                termination_label=termination_label,
                termination_names=self._termination_names,
                episode_metadata={"env_id": env_id, "auto_success": auto_success},
            )
            print(
                f"[LowShelfHDF5Recorder] wrote {episode_name}: "
                f"steps={len(frames)} success={success} label={termination_label}"
            )
            self._frames[env_id] = []
        return None, None

    def close(self, file_path: str):
        """Persist any episode that is still open when the app exits."""
        for env_id, frames in self._frames.items():
            if not frames:
                continue
            last = frames[-1]
            flags = np.asarray(last.get("termination_flags", np.zeros(len(self._termination_names), dtype=bool)))
            fired = [name for name, value in zip(self._termination_names, flags) if bool(value)]
            manual_outcome = self._manual_outcomes.pop(env_id, None)
            success = bool(manual_outcome) if manual_outcome is not None else bool(last.get("success", False))
            label = "shutdown_success" if success else ("shutdown_manual_stop" if not fired else "|".join(fired))
            self._writer.write_episode(
                frames,
                success=success,
                termination_label=label,
                termination_names=self._termination_names,
                episode_metadata={"env_id": env_id, "auto_success": bool(last.get("success", False))},
            )
            print(f"[LowShelfHDF5Recorder] wrote open episode on close: {self._writer.path}")
            self._frames[env_id] = []
        self._writer.close()


@configclass
class LowShelfHDF5RecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LowShelfHDF5Recorder

    output_path: str = "recordings/low_shelf_place/episode.hdf5"
    task: str = "Isaac-M20-Piper-LowShelf-Collect-v0"
    language_instruction: str = "把下层托盘里的糖盒放到上层桌面。"
    source: str = "unknown"
    camera_names: tuple[str, ...] = ("workcell_camera", "wrist_camera", "top_camera")
    overwrite: bool = False


@configclass
class LowShelfHDF5RecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    record_hdf5: LowShelfHDF5RecorderCfg = LowShelfHDF5RecorderCfg()
