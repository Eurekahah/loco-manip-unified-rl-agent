from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from replay.hdf5_video import (  # noqa: E402
    ReplayRenderOptions,
    ordered_cameras,
    print_summary,
    render_episode,
    validate_episode,
)


def _write_dataset(
    tmp_path: Path,
) -> tuple[Path, str]:
    path = tmp_path / "generic_replay.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = "teleop_manipulation/v1"
        file.attrs["task"] = "Any-Task"
        file.attrs["task_profile"] = "any_task"
        file.attrs["source"] = "pytest"
        episode = file.create_group("episodes/episode_000000")
        episode.attrs["camera_names"] = json.dumps(["wide_camera", "small_camera"])
        episode.attrs["success"] = True
        episode.attrs["termination_label"] = "manual_success"
        frames = 5
        episode.create_dataset("timestamp", data=np.arange(frames) * 0.1)
        episode.create_dataset("sim_time", data=np.arange(frames) * 0.1)
        episode.create_dataset("step_index", data=np.arange(frames))
        episode.create_dataset("action", data=np.zeros((frames, 13), dtype=np.float32))
        termination = episode.create_group("termination")
        termination.create_dataset("success", data=np.ones(frames, dtype=bool))
        termination.attrs["names"] = json.dumps(["place_success"])

        wide = episode.create_group("camera/wide_camera")
        wide.attrs["projection_model"] = "fisheye_equidistant"
        wide.create_dataset(
            "rgb",
            data=np.full((frames, 50, 80, 3), 80, dtype=np.uint8),
        )
        wide.create_dataset(
            "depth",
            data=np.full((frames, 50, 80), 0.5, dtype=np.float32),
        )
        small = episode.create_group("camera/small_camera")
        small.attrs["projection_model"] = "pinhole"
        small.create_dataset(
            "rgb",
            data=np.full((frames, 40, 60, 3), 160, dtype=np.uint8),
        )
    return path, "episode_000000"


def test_generic_replay_handles_mixed_cameras_and_fisheye_metadata(
    tmp_path: Path,
    capsys,
) -> None:
    path, episode_name = _write_dataset(tmp_path)
    with h5py.File(path, "r") as file:
        episode = file[f"episodes/{episode_name}"]
        cameras = ordered_cameras(file, episode)
        assert cameras == ["wide_camera", "small_camera"]
        assert validate_episode(episode) == []
        print_summary(str(path), file, episode, cameras)
        summary = capsys.readouterr().out
        assert "projection=fisheye_equidistant" in summary
        assert "K=" not in summary

        video_path, frame_path = render_episode(
            str(path),
            episode,
            cameras,
            ReplayRenderOptions(
                cameras=tuple(cameras),
                fps=10.0,
                tile_width=80,
                tile_height=50,
                include_depth=True,
                output=str(tmp_path / "replay.mp4"),
                frame_output=str(tmp_path / "montage.png"),
                frame_step=2,
            ),
        )

    assert Path(video_path).is_file() and Path(video_path).stat().st_size > 0
    assert Path(frame_path).is_file() and Path(frame_path).stat().st_size > 0
