from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "rl_training"
    / "rl_training"
    / "datasets"
    / "hdf5.py"
)
SPEC = importlib.util.spec_from_file_location("hdf5_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(index: int, height: int = 8, width: int = 10) -> dict:
    rgb = np.full((height, width, 3), index, dtype=np.uint8)
    return {
        "timestamp": np.float64(index),
        "sim_time": np.float64(index * 0.1),
        "step_index": np.int64(index),
        "action": np.zeros(13, dtype=np.float32),
        "processed_action": np.zeros(13, dtype=np.float32),
        "ll_command": np.zeros(13, dtype=np.float32),
        "robot": {"root_pos_w": np.zeros(3, dtype=np.float32)},
        "object": {"root_pos_w": np.zeros(3, dtype=np.float32)},
        "target_pad": {"root_pos_w": np.zeros(3, dtype=np.float32)},
        "termination_flags": np.array([False], dtype=bool),
        "terminated": np.bool_(False),
        "truncated": np.bool_(False),
        "success": np.bool_(False),
        "camera": {
            "fisheye_cam0": {
                "rgb": rgb,
                "sensor_frame": np.int64(index),
            }
        },
    }


@pytest.mark.parametrize(
    ("compression", "expected"),
    [("none", None), ("lzf", "lzf"), ("gzip1", "gzip"), ("gzip4", "gzip")],
)
def test_async_writer_streams_extendable_camera_datasets(
    tmp_path: Path,
    compression: str,
    expected: str | None,
) -> None:
    path = tmp_path / f"stream_{compression}.hdf5"
    writer = MODULE.HDF5Writer(
        str(path),
        task="test",
        instruction="test instruction",
        source="pytest",
        camera_names=("fisheye_cam0",),
        compression=compression,
        async_write=True,
        queue_size=16,
    )
    metadata = {
        "env_id": 0,
        "camera_metadata": {
            "fisheye_cam0": {
                "projection_model": "fisheye_equidistant",
                "fisheye_fov_deg": 200.0,
            }
        },
    }
    episode_name = writer.start_episode(episode_metadata=metadata)
    try:
        for index in range(5):
            assert writer.append_frame(
                episode_name,
                _frame(index),
                episode_metadata=metadata,
            )
        writer.finalize_episode(
            episode_name,
            success=True,
            termination_label="manual_success",
            termination_names=("place_success",),
            episode_metadata={"env_id": 0, "auto_success": False},
        )
    finally:
        writer.close()

    with h5py.File(path, "r") as source:
        assert source.attrs["schema"] == "teleop_manipulation/v1"
        assert source.attrs["hdf5_compression"] == compression
        assert source.attrs["record_hz"] == 10.0
        episode = source[f"episodes/{episode_name}"]
        assert episode["action"].shape == (5, 13)
        assert episode["sim_time"].shape == (5,)
        rgb = episode["camera/fisheye_cam0/rgb"]
        assert rgb.shape == (5, 8, 10, 3)
        assert rgb.dtype == np.uint8
        assert rgb.compression == expected
        assert episode.attrs["num_steps"] == 5
        assert bool(episode.attrs["success"]) is True
        assert episode["camera/fisheye_cam0"].attrs["projection_model"] == "fisheye_equidistant"
        assert "intrinsics" not in episode["camera/fisheye_cam0"]

    assert writer.stats["queue_length"] == 0
