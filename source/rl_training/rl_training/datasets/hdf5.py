# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Shared streaming HDF5 schema and writer for teleoperation episodes."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np


TELEOP_HDF5_SCHEMA = "teleop_manipulation/v1"
LOW_SHELF_HDF5_SCHEMA = TELEOP_HDF5_SCHEMA
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


class HDF5Writer:
    """Stream one HDF5 file with a stable episode/camera schema.

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
        schema: str = TELEOP_HDF5_SCHEMA,
        overwrite: bool = False,
        compression: str = "lzf",
        record_hz: float = 10.0,
        async_write: bool = True,
        queue_size: int = 8,
    ) -> None:
        if compression not in {"none", "lzf", "gzip1", "gzip4"}:
            raise ValueError(f"unsupported HDF5 compression: {compression}")
        if record_hz <= 0.0:
            raise ValueError("record_hz must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
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
        self.metadata = dict(metadata or {})
        self.schema = schema
        self.compression = compression
        self.record_hz = float(record_hz)
        self.async_write = bool(async_write)
        self.queue_size = int(queue_size)
        self._episode_count = 0
        self._file: h5py.File | None = None
        self._episodes: h5py.Group | None = None
        self._open_episode_metadata: dict[str, dict[str, Any]] = {}
        self._pending: deque[tuple[str, tuple[Any, ...]]] = deque()
        self._queued_frames = 0
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._error: BaseException | None = None
        self._dropped_frames = 0
        self._write_samples_ms: deque[float] = deque(maxlen=512)

        if self.async_write:
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="low-shelf-hdf5-writer",
                daemon=True,
            )
            self._thread.start()
            if not self._ready.wait(timeout=30.0):
                raise RuntimeError("Timed out while opening the HDF5 writer")
            self._raise_if_failed()
        else:
            self._open_file()

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    def set_metadata(self, **metadata: Any) -> None:
        """Update root-level metadata after the writer has been created."""
        self.metadata.update(metadata)
        if self.async_write:
            self._put_control("metadata", metadata)
        else:
            self._apply_metadata(metadata)
            assert self._file is not None
            self._file.flush()

    def next_episode_name(self) -> str:
        episode_name = f"episode_{self._episode_count:06d}"
        self._episode_count += 1
        return episode_name

    def start_episode(
        self,
        *,
        episode_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Allocate an episode name without waiting for the first frame."""
        return self.next_episode_name()

    def append_frame(
        self,
        episode_name: str,
        frame: dict[str, Any],
        *,
        episode_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Queue one frame and return ``False`` when the bounded frame queue is full."""
        frame_np = _frame_to_numpy(frame)
        if self.async_write:
            with self._condition:
                if self._closed:
                    raise RuntimeError("Cannot append to a closed HDF5 writer")
                if self._queued_frames >= self.queue_size:
                    self._dropped_frames += 1
                    return False
                self._pending.append(
                    ("frame", (episode_name, frame_np, dict(episode_metadata or {})))
                )
                self._queued_frames += 1
                self._condition.notify()
            return True
        self._append_frame(episode_name, frame_np, episode_metadata=episode_metadata)
        return True

    def finalize_episode(
        self,
        episode_name: str,
        *,
        success: bool,
        termination_label: str,
        termination_names: tuple[str, ...] | list[str],
        episode_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalize an episode. Control commands are never dropped by frame backpressure."""
        command = (
            episode_name,
            bool(success),
            termination_label,
            tuple(termination_names),
            dict(episode_metadata or {}),
        )
        if self.async_write:
            self._put_control("finalize", *command)
        else:
            self._finalize_episode(*command)

    def write_episode(
        self,
        frames: list[dict[str, Any]],
        *,
        success: bool,
        termination_label: str,
        termination_names: tuple[str, ...] | list[str],
        episode_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Legacy helper that streams a complete frame list and finalizes it."""
        if not frames:
            raise ValueError("Cannot write an empty episode")
        episode_name = self.start_episode()
        for frame in frames:
            self.append_frame(episode_name, frame, episode_metadata=episode_metadata)
        self.finalize_episode(
            episode_name,
            success=success,
            termination_label=termination_label,
            termination_names=termination_names,
            episode_metadata=episode_metadata,
        )
        return episode_name

    def flush(self) -> None:
        if self.async_write:
            self._put_control("flush")
        else:
            assert self._file is not None
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        if self.async_write:
            self._put_control("stop")
            assert self._thread is not None
            self._thread.join(timeout=60.0)
            if self._thread.is_alive():
                raise TimeoutError("HDF5 writer thread did not stop within 60 seconds")
            self._raise_if_failed()
        elif self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._closed = True

    @property
    def stats(self) -> dict[str, float | int]:
        samples = list(self._write_samples_ms)
        return {
            "queue_length": self._queued_frames,
            "dropped_frames": self._dropped_frames,
            "hdf5_write_ms_mean": sum(samples) / len(samples) if samples else 0.0,
            "hdf5_write_ms_max": max(samples, default=0.0),
        }

    def _put_control(self, command: str, *args: Any) -> None:
        with self._condition:
            if self._closed:
                return
            self._pending.append((command, args))
            self._condition.notify()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("HDF5 writer thread failed") from self._error

    def _writer_loop(self) -> None:
        try:
            self._open_file()
            self._ready.set()
            while True:
                with self._condition:
                    while not self._pending:
                        self._condition.wait()
                    command, args = self._pending.popleft()
                    if command == "frame":
                        self._queued_frames -= 1
                if command == "stop":
                    break
                if command == "frame":
                    self._append_frame(*args)
                elif command == "finalize":
                    self._finalize_episode(*args)
                elif command == "metadata":
                    self._apply_metadata(args[0])
                elif command == "flush":
                    assert self._file is not None
                    self._file.flush()
            assert self._file is not None
            self._file.flush()
            self._file.close()
            self._file = None
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._ready.set()

    def _open_file(self) -> None:
        self._file = h5py.File(self.path, "w", libver="latest")
        self._episodes = self._file.create_group("episodes")
        self._write_root_attrs()

    def _write_root_attrs(self) -> None:
        assert self._file is not None
        self._file.attrs["schema"] = self.schema
        self._file.attrs["action_semantics"] = ACTION_SEMANTICS
        self._file.attrs["task"] = self.task
        self._file.attrs["instruction"] = self.instruction
        self._file.attrs["source"] = self.source
        self._file.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["action_layout"] = json.dumps(list(self.action_layout))
        self._file.attrs["camera_names"] = json.dumps(list(self.camera_names))
        self._file.attrs["quaternion_convention"] = "wxyz"
        self._file.attrs["camera_convention"] = "world (+X forward, +Z up) and ROS (+Z forward, -Y up)"
        self._file.attrs["depth_convention"] = "distance_to_image_plane, metres"
        self._file.attrs["intrinsics_convention"] = "row-major 3x3 pinhole matrix; not valid for fisheye projection"
        self._file.attrs["hdf5_compression"] = self.compression
        self._file.attrs["record_hz"] = self.record_hz
        self._file.attrs["async_write"] = self.async_write
        self._file.attrs["writer_queue_size"] = self.queue_size
        self._apply_metadata(self.metadata)

    def _apply_metadata(self, metadata: dict[str, Any]) -> None:
        assert self._file is not None
        for key, value in metadata.items():
            self._file.attrs[key] = _jsonable_attr(value)

    def _ensure_episode(
        self,
        episode_name: str,
        episode_metadata: dict[str, Any] | None,
    ) -> h5py.Group:
        assert self._episodes is not None
        if episode_name in self._episodes:
            return self._episodes[episode_name]
        metadata = dict(episode_metadata or {})
        episode = self._episodes.create_group(episode_name)
        episode.attrs["num_steps"] = 0
        episode.attrs["action_semantics"] = ACTION_SEMANTICS
        episode.attrs["instruction"] = self.instruction
        episode.attrs["source"] = self.source
        episode.attrs["episode_index"] = int(episode_name.rsplit("_", 1)[-1])
        for key, value in metadata.items():
            episode.attrs[key] = _jsonable_attr(value)
        episode.create_dataset(
            "language_instruction",
            data=self.instruction,
            dtype=h5py.string_dtype("utf-8"),
        )
        self._open_episode_metadata[episode_name] = metadata
        return episode

    def _append_frame(
        self,
        episode_name: str,
        frame: dict[str, np.ndarray],
        episode_metadata: dict[str, Any] | None = None,
    ) -> None:
        started = time.perf_counter()
        episode = self._ensure_episode(episode_name, episode_metadata)
        camera_metadata = dict((episode_metadata or {}).get("camera_metadata", {}))
        for path, value in _flatten_frame(frame):
            if path == "termination_flags":
                path = "termination/flags"
            elif path in {"terminated", "truncated", "success"}:
                path = f"termination/{path}"
            dataset = episode.get(path)
            if dataset is None:
                dataset = self._create_stream_dataset(episode, path, value)
                if path.startswith("camera/"):
                    camera_name = path.split("/")[1]
                    self._set_camera_attrs(
                        episode[path.rsplit("/", 1)[0]],
                        value,
                        camera_metadata.get(camera_name, {}),
                    )
            index = dataset.shape[0]
            dataset.resize((index + 1, *dataset.shape[1:]))
            dataset[index] = value
        self._write_samples_ms.append((time.perf_counter() - started) * 1000.0)

    def _create_stream_dataset(
        self,
        episode: h5py.Group,
        path: str,
        value: np.ndarray,
    ) -> h5py.Dataset:
        parent_path, _, dataset_name = path.rpartition("/")
        parent = episode
        if parent_path:
            parent = episode.require_group(parent_path)
        is_image = path.startswith("camera/") and path.endswith(("/rgb", "/depth"))
        if value.ndim == 0:
            chunks = (1024,)
        elif is_image:
            chunks = (1, *value.shape)
        else:
            chunks = (min(256, max(1, value.size)), *value.shape)
        compression, compression_opts = self._compression_args(is_image)
        return parent.create_dataset(
            dataset_name,
            shape=(0, *value.shape),
            maxshape=(None, *value.shape),
            dtype=value.dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )

    def _compression_args(self, is_image: bool) -> tuple[str | None, int | None]:
        if not is_image or self.compression == "none":
            return None, None
        if self.compression == "lzf":
            return "lzf", None
        if self.compression == "gzip1":
            return "gzip", 1
        return "gzip", 4

    @staticmethod
    def _set_camera_attrs(
        camera_group: h5py.Group,
        value: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        if value.ndim == 3:
            camera_group.attrs["height"] = int(value.shape[0])
            camera_group.attrs["width"] = int(value.shape[1])
        camera_group.attrs["rgb_dtype"] = "uint8"
        camera_group.attrs["extrinsics"] = "T_world_camera_world and T_world_camera_ros"
        for key, item in metadata.items():
            camera_group.attrs[key] = _jsonable_attr(item)

    def _finalize_episode(
        self,
        episode_name: str,
        success: bool,
        termination_label: str,
        termination_names: tuple[str, ...],
        episode_metadata: dict[str, Any],
    ) -> None:
        if self._episodes is None or episode_name not in self._episodes:
            self._open_episode_metadata.pop(episode_name, None)
            return
        episode = self._episodes[episode_name]
        num_steps = int(episode["timestamp"].shape[0]) if "timestamp" in episode else 0
        episode.attrs["num_steps"] = num_steps
        episode.attrs["success"] = bool(success)
        episode.attrs["termination_label"] = termination_label
        episode.attrs["termination_names"] = json.dumps(list(termination_names))
        if num_steps:
            episode.attrs["start_timestamp"] = float(episode["timestamp"][0])
            episode.attrs["end_timestamp"] = float(episode["timestamp"][-1])
            episode.attrs["start_sim_time"] = float(episode["sim_time"][0])
            episode.attrs["end_sim_time"] = float(episode["sim_time"][-1])
        for key, value in episode_metadata.items():
            episode.attrs[key] = _jsonable_attr(value)
        termination = episode.require_group("termination")
        termination.attrs["names"] = json.dumps(list(termination_names))
        termination.attrs["label"] = termination_label
        for camera_name in episode.get("camera", {}):
            camera_group = episode["camera"][camera_name]
            if "rgb" in camera_group:
                camera_group.attrs["num_frames"] = int(camera_group["rgb"].shape[0])
        assert self._file is not None
        self._file.attrs["total_episodes"] = max(
            int(self._file.attrs.get("total_episodes", 0)),
            self._episode_count,
        )
        self._file.attrs["total_steps"] = int(self._file.attrs.get("total_steps", 0)) + num_steps
        self._open_episode_metadata.pop(episode_name, None)

    def __enter__(self) -> HDF5Writer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _jsonable_attr(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _frame_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_frame_to_numpy(item) for item in value]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_frame(frame: dict[str, Any], prefix: str = "") -> list[tuple[str, np.ndarray]]:
    flattened: list[tuple[str, np.ndarray]] = []
    for key, value in frame.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.extend(_flatten_frame(value, path))
        else:
            flattened.append((path, np.asarray(value)))
    return flattened


# Backward-compatible name for scripts and notebooks that predate the shared writer.
LowShelfHDF5Writer = HDF5Writer
