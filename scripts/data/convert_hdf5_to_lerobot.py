#!/usr/bin/env python3
"""Convert low-shelf HDF5 demonstrations to a LeRobot v3 dataset.

The output is created with the official ``LeRobotDataset.create`` API, so the
on-disk layout (parquet, videos, and metadata) matches the main framework and
can be read by ``lerobot-train`` without a custom loader.

Default mapping:

    HDF5 camera/workcell_camera/rgb -> observation.images.camera1
    HDF5 camera/wrist_camera/rgb    -> observation.images.camera2
    HDF5 robot/joint_pos            -> observation.state
    HDF5 action                     -> action

Keyboard and VR collection both write absolute offsets from a recalibrated
anchor, so the HDF5 ``action`` column has one meaning across input sources.

Example:

    python scripts/data/convert_hdf5_to_lerobot.py \
        --input recordings/low_shelf_place/demo.hdf5 \
        --output /data/lerobot/low_shelf_demo \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ACTION_LAYOUT = (
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

ACTION_SEMANTICS = "absolute_offset_from_recalibrated_anchor"

DEFAULT_CAMERA_MAP = {
    "workcell_camera": "camera1",
    "wrist_camera": "camera2",
}


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _json_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return json.loads(bytes(value).decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def _build_features(
    *,
    height: int,
    width: int,
    camera_suffixes: list[str],
    action_layout: list[str],
    state_names: list[str],
    fps: int,
    use_videos: bool,
) -> dict:
    image_dtype = "video" if use_videos else "image"
    image_info = {"is_depth_map": False}
    if use_videos:
        image_info.update(
            {
                "video.fps": float(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            }
        )

    def camera_feature() -> dict:
        return {
            "dtype": image_dtype,
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": dict(image_info),
        }

    features = {
        f"observation.images.{suffix}": camera_feature() for suffix in camera_suffixes
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(state_names)],
        "names": state_names,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [len(action_layout)],
        "names": action_layout,
    }
    return features


def _resolve_camera_map(source: h5py.File, episode_name: str, override: str | None) -> dict[str, str]:
    available = sorted(source[f"episodes/{episode_name}/camera"].keys())
    if override:
        mapping = json.loads(override)
        if not isinstance(mapping, dict):
            raise ValueError("--camera-map must be a JSON object")
        missing = sorted(set(mapping) - set(available))
        if missing:
            raise ValueError(f"--camera-map references missing HDF5 cameras: {missing}")
        return {str(key): str(value) for key, value in mapping.items()}

    mapping = {}
    for name in available:
        if name in DEFAULT_CAMERA_MAP:
            mapping[name] = DEFAULT_CAMERA_MAP[name]
        elif name.startswith("camera"):
            mapping[name] = name
        else:
            raise ValueError(
                f"No default LeRobot name for HDF5 camera '{name}'. "
                f"Pass --camera-map with one of: {available}"
            )
    return mapping


def _instruction(episode: h5py.Group, source: h5py.File) -> str:
    if "language_instruction" in episode:
        return _decode(episode["language_instruction"][()])
    if "instruction" in episode.attrs:
        return _decode(episode.attrs["instruction"])
    return _decode(source.attrs.get("instruction", ""))


def convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {input_path}")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    with h5py.File(input_path, "r") as source:
        schema = _decode(source.attrs.get("schema", ""))
        if schema != "low_shelf_pick_place/v1":
            raise ValueError(
                f"Unsupported HDF5 schema '{schema}'; expected 'low_shelf_pick_place/v1'."
            )
        action_semantics = _decode(source.attrs.get("action_semantics", ""))
        if action_semantics != ACTION_SEMANTICS:
            raise ValueError(
                f"Unsupported action_semantics '{action_semantics}'. "
                f"Expected '{ACTION_SEMANTICS}'. HDF5 files recorded with the old "
                "keyboard rolling-delta semantics must be re-collected."
            )

        episode_names = sorted(source["episodes"].keys())
        if not episode_names:
            raise ValueError(f"No episodes found in {input_path}")
        if args.max_episodes > 0:
            episode_names = episode_names[: args.max_episodes]

        first_episode = source[f"episodes/{episode_names[0]}"]
        camera_map = _resolve_camera_map(source, episode_names[0], args.camera_map)
        if not camera_map:
            raise ValueError(f"No camera groups found in {episode_names[0]}.")

        control_dt = float(source.attrs.get("control_dt", 0.02))
        source_fps = int(round(1.0 / control_dt))
        fps = int(args.fps or source_fps)
        if fps <= 0:
            raise ValueError(f"Invalid fps: {fps}")
        if fps > source_fps:
            raise ValueError(
                f"Output fps {fps} exceeds the HDF5 control rate {source_fps} Hz; "
                "cannot upsample."
            )
        fps_ratio = source_fps / fps
        stride = int(round(fps_ratio))
        if stride < 1 or abs(fps_ratio - stride) > 1e-6:
            raise ValueError(
                f"Output fps {fps} does not divide the HDF5 control rate {source_fps} Hz. "
                "Use an integer divisor such as 10, 25, or 50."
            )

        raw_action_layout = source.attrs.get("action_layout")
        action_layout = (
            [str(name) for name in _json_value(raw_action_layout)]
            if raw_action_layout is not None
            else list(DEFAULT_ACTION_LAYOUT)
        )

        first_action = first_episode["action"]
        if first_action.shape[-1] != len(action_layout):
            raise ValueError(
                f"action has {first_action.shape[-1]} columns but action_layout has "
                f"{len(action_layout)} names."
            )
        first_joint_pos = first_episode["robot/joint_pos"]
        state_dim = int(first_joint_pos.shape[-1])
        raw_joint_names = source.attrs.get("joint_names")
        if raw_joint_names is not None:
            state_names = [str(name) for name in _json_value(raw_joint_names)]
            if len(state_names) != state_dim:
                raise ValueError(
                    f"joint_names has {len(state_names)} entries but robot/joint_pos has "
                    f"{state_dim} columns."
                )
        else:
            state_names = [f"joint_{index}" for index in range(state_dim)]

        first_camera_name = next(iter(camera_map))
        first_rgb = first_episode[f"camera/{first_camera_name}/rgb"]
        height, width = int(first_rgb.shape[1]), int(first_rgb.shape[2])

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id or f"local/{output_path.name}",
            fps=fps,
            features=_build_features(
                height=height,
                width=width,
                camera_suffixes=list(camera_map.values()),
                action_layout=action_layout,
                state_names=state_names,
                fps=fps,
                use_videos=not args.no_videos,
            ),
            root=output_path,
            robot_type=args.robot_type,
            use_videos=not args.no_videos,
        )

        source_episodes = []
        for episode_name in episode_names:
            episode = source[f"episodes/{episode_name}"]
            actions = np.asarray(episode["action"][:], dtype=np.float32)
            joint_pos = np.asarray(episode["robot/joint_pos"][:], dtype=np.float32)
            num_steps = int(actions.shape[0])
            if num_steps == 0:
                print(f"[convert] skip empty episode {episode_name}")
                continue
            if joint_pos.shape != (num_steps, len(state_names)):
                raise ValueError(
                    f"{episode_name}: robot/joint_pos shape {joint_pos.shape} does not match "
                    f"({num_steps}, {len(state_names)})."
                )
            if not np.isfinite(actions).all() or not np.isfinite(joint_pos).all():
                raise ValueError(f"{episode_name}: action or joint_pos contains NaN/Inf.")

            task = _instruction(episode, source)
            camera_frames = {}
            for hdf5_name, lerobot_suffix in camera_map.items():
                rgb = np.asarray(episode[f"camera/{hdf5_name}/rgb"][:], dtype=np.uint8)
                if rgb.shape[0] != num_steps:
                    raise ValueError(
                        f"{episode_name}: camera '{hdf5_name}' has {rgb.shape[0]} frames, "
                        f"expected {num_steps}."
                    )
                if rgb.shape[1:] != (height, width, 3):
                    raise ValueError(
                        f"{episode_name}: camera '{hdf5_name}' frame shape {rgb.shape[1:]} "
                        f"does not match ({height}, {width}, 3)."
                    )
                camera_frames[f"observation.images.{lerobot_suffix}"] = rgb

            frame_indices = np.arange(0, num_steps, stride, dtype=np.int64)
            output_steps = int(frame_indices.size)
            if output_steps == 0:
                print(f"[convert] skip episode shorter than stride {stride}: {episode_name}")
                continue

            for index in frame_indices:
                frame = {
                    "observation.state": joint_pos[index],
                    "action": actions[index],
                    "task": task,
                }
                for key, frames in camera_frames.items():
                    frame[key] = frames[index]
                dataset.add_frame(frame)

            dataset.save_episode()
            source_episodes.append(
                {
                    "hdf5_episode": episode_name,
                    "num_steps": output_steps,
                    "source_steps": num_steps,
                    "stride": stride,
                    "success": bool(episode.attrs.get("success", False)),
                    "termination_label": _decode(episode.attrs.get("termination_label", "")),
                    "instruction": task,
                }
            )
            print(
                f"[convert] {episode_name}: {output_steps} frames "
                f"(from {num_steps} @ stride {stride}) -> {task}"
            )

        dataset.finalize()
        if not source_episodes:
            raise ValueError(f"No non-empty episodes found in {input_path}")

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "source_episodes.json").write_text(
        json.dumps(source_episodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (meta_dir / "conversion_info.json").write_text(
        json.dumps(
            {
                "source_hdf5": str(input_path),
                "repo_id": args.repo_id or f"local/{output_path.name}",
                "robot_type": args.robot_type,
                "source_fps": source_fps,
                "fps": fps,
                "stride": stride,
                "action_semantics": action_semantics,
                "camera_map": camera_map,
                "action_layout": action_layout,
                "state_layout": state_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[convert] wrote {len(source_episodes)} episode(s), {output_path} "
        f"(repo_id={args.repo_id or f'local/{output_path.name}'})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert low-shelf HDF5 demonstrations to a LeRobot v3 dataset."
    )
    parser.add_argument("--input", required=True, help="Input HDF5 file.")
    parser.add_argument("--output", required=True, help="Output LeRobot dataset directory.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo_id stored in metadata (default: local/<output directory name>).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "Output dataset fps (default: derived from the HDF5 control_dt attribute). "
            "Lower values are down-sampled by an integer stride, e.g. 50 -> 10 keeps "
            "every 5th frame."
        ),
    )
    parser.add_argument("--robot-type", default="m20_piper", help="Robot type stored in metadata.")
    parser.add_argument(
        "--camera-map",
        default=None,
        help=(
            'JSON object mapping HDF5 camera groups to LeRobot suffixes, e.g. '
            '\'{"workcell_camera":"camera1","wrist_camera":"camera2"}\'.'
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Only convert the first N episodes (0 converts all).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset.")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store camera frames as images instead of MP4 videos.",
    )
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
