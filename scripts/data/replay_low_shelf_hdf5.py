#!/usr/bin/env python3
"""Inspect a low-shelf HDF5 dataset and render a camera replay video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


parser = argparse.ArgumentParser(description="Inspect and replay low-shelf RGB-D HDF5 episodes.")
parser.add_argument("--input", type=str, required=True, help="Path to the .hdf5 file.")
parser.add_argument("--episode", type=int, default=0, help="Episode index, not the HDF5 group name.")
parser.add_argument("--camera", type=str, default=None, help="Camera name; defaults to the first camera.")
parser.add_argument("--check-only", action="store_true", help="Validate and print metadata without rendering.")
parser.add_argument("--output", type=str, default=None, help="Replay MP4 path.")
parser.add_argument("--fps", type=float, default=0.0, help="Replay FPS; 0 derives it from timestamps.")
parser.add_argument("--max_frames", type=int, default=0, help="Render at most N frames; 0 renders all.")
parser.add_argument("--no_depth", action="store_true", help="Render RGB only.")
args = parser.parse_args()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _episode_names(file: h5py.File) -> list[str]:
    return sorted(file["episodes"].keys())


def _require(group: h5py.Group, key: str, errors: list[str]) -> h5py.Dataset | None:
    if key not in group:
        errors.append(f"missing dataset: {group.name}/{key}")
        return None
    return group[key]


def _validate_episode(episode: h5py.Group) -> list[str]:
    errors: list[str] = []
    for key in ("timestamp", "sim_time", "step_index", "action", "processed_action", "ll_command"):
        dataset = _require(episode, key, errors)
        if dataset is not None and len(dataset.shape) < 1:
            errors.append(f"dataset has no time dimension: {dataset.name}")
    for key in ("root_pos_w", "root_quat_w", "joint_pos", "joint_vel", "ee_pos_w", "ee_quat_w"):
        group = episode.get("robot")
        if group is None:
            errors.append("missing group: /robot")
            break
        _require(group, key, errors)
    for key in ("flags", "terminated", "truncated", "success"):
        group = episode.get("termination")
        if group is None:
            errors.append("missing group: /termination")
            break
        _require(group, key, errors)
    if "camera" not in episode:
        errors.append("missing group: /camera")
    return errors


def _print_summary(file: h5py.File, episode: h5py.Group) -> None:
    print("=" * 78)
    print(f"file          : {os.path.abspath(args.input)}")
    print(f"schema        : {_text(file.attrs.get('schema', 'missing'))}")
    print(f"task          : {_text(file.attrs.get('task', 'missing'))}")
    print(f"source        : {_text(file.attrs.get('source', 'missing'))}")
    print(f"instruction   : {_text(episode.attrs.get('instruction', file.attrs.get('instruction', '')))}")
    print(f"episodes      : {len(file['episodes'])}")
    print(f"episode group : {episode.name}")
    print(f"steps         : {len(episode['timestamp'])}")
    print(f"success       : {bool(episode.attrs.get('success', False))}")
    print(f"termination   : {_text(episode.attrs.get('termination_label', 'missing'))}")
    action = episode["action"][:]
    print(f"action shape  : {action.shape}")
    print(f"action range  : min={action.min():.4f} max={action.max():.4f}")
    timestamps = episode["timestamp"][:]
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        print(f"timestamp dt  : median={np.median(dt):.4f}s min={dt.min():.4f}s max={dt.max():.4f}s")
    for camera_name, camera in episode["camera"].items():
        rgb = camera["rgb"]
        depth = camera["depth"]
        print(
            f"camera {camera_name:<16}: rgb={rgb.shape} depth={depth.shape} "
            f"K={camera['intrinsics'].shape} extrinsics={camera['T_world_camera_ros'].shape}"
        )
        depth_sample = depth[0]
        finite = depth_sample[np.isfinite(depth_sample)]
        if finite.size:
            print(
                f"  depth[0] m  : min={finite.min():.4f} median={np.median(finite):.4f} max={finite.max():.4f}"
            )
    print("=" * 78)


def _depth_preview(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scale = max(hi - lo, 1e-6)
    normalized = np.clip((depth - lo) / scale, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def _render_replay(episode: h5py.Group, camera_name: str) -> str:
    camera = episode["camera"][camera_name]
    rgb_data = camera["rgb"]
    depth_data = camera["depth"] if not args.no_depth else None
    num_frames = len(rgb_data)
    if args.max_frames > 0:
        num_frames = min(num_frames, args.max_frames)

    fps = args.fps
    if fps <= 0.0:
        timestamps = episode["timestamp"][:]
        dt = np.diff(timestamps[: max(num_frames, 2)])
        fps = 1.0 / float(np.median(dt)) if len(dt) and np.median(dt) > 0 else 5.0
    fps = float(np.clip(fps, 1.0, 60.0))

    depth_lo, depth_hi = 0.0, 1.0
    if depth_data is not None:
        depth_stack = depth_data[:num_frames]
        finite = depth_stack[np.isfinite(depth_stack)]
        if finite.size:
            depth_lo = float(np.percentile(finite, 2.0))
            depth_hi = float(np.percentile(finite, 98.0))

    first_rgb = rgb_data[0]
    height, width = first_rgb.shape[:2]
    output_width = width * (2 if depth_data is not None else 1)
    output_path = args.output
    if output_path is None:
        stem = Path(args.input).stem
        output_path = os.path.join(
            "recordings", "replay", f"{stem}_{episode.name.replace('/', '_')}_{camera_name}.mp4"
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    actions = episode["action"][:]
    success_flags = episode["termination"]["success"][:]
    for index in range(num_frames):
        rgb = rgb_data[index].astype(np.uint8)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if depth_data is not None:
            depth = depth_data[index].astype(np.float32)
            depth_view = _depth_preview(depth, depth_lo, depth_hi)
            frame = np.concatenate([frame, depth_view], axis=1)
        label = (
            f"ep={episode.name} step={index} success={bool(success_flags[index])} "
            f"gripper={actions[index, -1]:+.2f}"
        )
        cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    return os.path.abspath(output_path)


def main() -> None:
    with h5py.File(args.input, "r") as file:
        if "episodes" not in file:
            raise KeyError("HDF5 file has no /episodes group")
        names = _episode_names(file)
        if not names:
            raise RuntimeError("HDF5 file contains no episodes")
        if args.episode < 0 or args.episode >= len(names):
            raise IndexError(f"--episode must be in [0, {len(names) - 1}]")
        episode = file["episodes"][names[args.episode]]
        errors = _validate_episode(episode)
        _print_summary(file, episode)
        if errors:
            print("validation errors:")
            for error in errors:
                print(f"  - {error}")
            raise SystemExit(1)
        if "termination" in episode:
            names_attr = episode["termination"].attrs.get("names", "[]")
            print(f"termination terms: {json.loads(_text(names_attr))}")

        if not args.check_only:
            cameras = list(episode["camera"].keys())
            camera_name = args.camera or cameras[0]
            if camera_name not in episode["camera"]:
                raise KeyError(f"Unknown camera '{camera_name}', available: {cameras}")
            output_path = _render_replay(episode, camera_name)
            print(f"replay video: {output_path}")


if __name__ == "__main__":
    main()
