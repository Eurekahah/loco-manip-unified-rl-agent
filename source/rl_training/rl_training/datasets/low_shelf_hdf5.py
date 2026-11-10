# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""HDF5 schema and writer for low-shelf teleoperation episodes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np


LOW_SHELF_HDF5_SCHEMA = "low_shelf_pick_place/v1"
ACTION_SEMANTICS = "absolute_offset_from_recalibrated_anchor"

ACTION_LAYOUT = (
    "vx",
    "vy",
    "wz",
    "delta_ee_x",
    "delta_ee_y",
    "delta_ee_z",
    "delta_ee_roll",
    "delta_ee_pitch",
    "delta_ee_yaw",
    "delta_body_height",
    "delta_body_pitch",
    "delta_body_roll",
    "gripper",
)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert one ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Return a 4x4 ``T_world_camera`` transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _stack_frames(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.stack([_to_numpy(frame[key]) for frame in frames], axis=0)


def _collect_nested(frames: list[dict[str, Any]], key: str) -> dict[str, np.ndarray]:
    first = frames[0][key]
    if not isinstance(first, dict):
        return {key: _stack_frames(frames, key)}

    output: dict[str, np.ndarray] = {}
    for sub_key in first:
        output[sub_key] = _stack_frames([frame[key] for frame in frames], sub_key)
    return output


class LowShelfHDF5Writer:
    """Write one HDF5 file with a stable episode/camera schema.

    The writer intentionally accepts plain NumPy values so the recording term and
    offline inspection tooling do not depend on Isaac Sim imports.
    """

    def __init__(
        self,
        path: str,
        *,
        task: str,
        instruction: str,
        source: str,
        camera_names: tuple[str, ...] | list[str],
        action_layout: tuple[str, ...] | list[str] = ACTION_LAYOUT,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        if not path.endswith(".hdf5"):
            path += ".hdf5"
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing HDF5 file: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.path = path
        self.task = task
        self.instruction = instruction
        self.source = source
        self.camera_names = tuple(camera_names)
        self.action_layout = tuple(action_layout)
        self._episode_count = 0
        self._file = h5py.File(path, "w")
        self._episodes = self._file.create_group("episodes")

        self._file.attrs["schema"] = LOW_SHELF_HDF5_SCHEMA
        # Keyboard and VR collection share one action space: each arm/body
        # channel is an absolute offset from the recalibrated anchor.
        self._file.attrs["action_semantics"] = ACTION_SEMANTICS
        self._file.attrs["task"] = task
        self._file.attrs["instruction"] = instruction
        self._file.attrs["source"] = source
        self._file.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["action_layout"] = json.dumps(list(self.action_layout))
        self._file.attrs["camera_names"] = json.dumps(list(self.camera_names))
        self._file.attrs["quaternion_convention"] = "wxyz"
        self._file.attrs["camera_convention"] = "world (+X forward, +Z up) and ROS (+Z forward, -Y up)"
        self._file.attrs["depth_convention"] = "distance_to_image_plane, metres"
        self._file.attrs["intrinsics_convention"] = "row-major 3x3 pinhole matrix"
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                self._file.attrs[key] = value

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    def set_metadata(self, **metadata: Any) -> None:
        """Update root-level metadata after the writer has been created."""
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            self._file.attrs[key] = value
        self._file.flush()

    def write_episode(
        self,
        frames: list[dict[str, Any]],
        *,
        success: bool,
        termination_label: str,
        termination_names: tuple[str, ...] | list[str],
        episode_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write a non-empty episode and return its HDF5 group name."""
        if not frames:
            raise ValueError("Cannot write an empty episode")

        episode_name = f"episode_{self._episode_count:06d}"
        episode = self._episodes.create_group(episode_name)
        timestamps = _stack_frames(frames, "timestamp").astype(np.float64)
        sim_times = _stack_frames(frames, "sim_time").astype(np.float64)

        episode.attrs["num_steps"] = len(frames)
        episode.attrs["action_semantics"] = ACTION_SEMANTICS
        episode.attrs["success"] = bool(success)
        episode.attrs["termination_label"] = termination_label
        episode.attrs["termination_names"] = json.dumps(list(termination_names))
        episode.attrs["instruction"] = self.instruction
        episode.attrs["source"] = self.source
        episode.attrs["episode_index"] = self._episode_count
        episode.attrs["start_timestamp"] = float(timestamps[0])
        episode.attrs["end_timestamp"] = float(timestamps[-1])
        episode.attrs["start_sim_time"] = float(sim_times[0])
        episode.attrs["end_sim_time"] = float(sim_times[-1])
        if episode_metadata:
            for key, value in episode_metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                episode.attrs[key] = value

        episode.create_dataset("language_instruction", data=self.instruction, dtype=h5py.string_dtype("utf-8"))
        episode.create_dataset("timestamp", data=timestamps)
        episode.create_dataset("sim_time", data=sim_times)
        episode.create_dataset("step_index", data=_stack_frames(frames, "step_index").astype(np.int64))

        for key in ("action", "processed_action", "ll_command"):
            episode.create_dataset(key, data=_stack_frames(frames, key).astype(np.float32))

        for group_name in ("robot", "object", "target_pad"):
            nested = _collect_nested(frames, group_name)
            group = episode.create_group(group_name)
            for key, value in nested.items():
                group.create_dataset(key, data=value.astype(np.float32))

        termination = episode.create_group("termination")
        termination.create_dataset("flags", data=_stack_frames(frames, "termination_flags").astype(bool))
        termination.create_dataset("terminated", data=_stack_frames(frames, "terminated").astype(bool))
        termination.create_dataset("truncated", data=_stack_frames(frames, "truncated").astype(bool))
        termination.create_dataset("success", data=_stack_frames(frames, "success").astype(bool))
        termination.attrs["names"] = json.dumps(list(termination_names))
        termination.attrs["label"] = termination_label

        cameras = episode.create_group("camera")
        for camera_name in self.camera_names:
            camera_group = cameras.create_group(camera_name)
            camera_frames = [frame["camera"][camera_name] for frame in frames]
            rgb_shape = None
            for key, value in camera_frames[0].items():
                array = np.stack([_to_numpy(frame[key]) for frame in camera_frames], axis=0)
                if key == "rgb":
                    rgb_shape = array.shape
                if key in {"rgb", "depth"}:
                    chunks = (1, *array.shape[1:])
                    camera_group.create_dataset(
                        key,
                        data=array,
                        chunks=chunks,
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    camera_group.create_dataset(key, data=array)
            if rgb_shape is not None:
                camera_group.attrs["height"] = int(rgb_shape[1])
                camera_group.attrs["width"] = int(rgb_shape[2])
                camera_group.attrs["num_frames"] = int(rgb_shape[0])
            camera_group.attrs["rgb_dtype"] = "uint8"
            camera_group.attrs["depth_unit"] = "m" if "depth" in camera_frames[0] else "none"
            camera_group.attrs["extrinsics"] = "T_world_camera_world and T_world_camera_ros"

        self._episode_count += 1
        self._file.attrs["total_episodes"] = self._episode_count
        self._file.attrs["total_steps"] = int(self._file.attrs.get("total_steps", 0)) + len(frames)
        self._file.flush()
        return episode_name

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> LowShelfHDF5Writer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
