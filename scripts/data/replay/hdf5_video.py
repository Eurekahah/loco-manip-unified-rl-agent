"""Inspect and render task-agnostic HDF5 episodes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class ReplayRenderOptions:
    cameras: tuple[str, ...] = ()
    fps: float = 0.0
    stride: int = 1
    max_frames: int = 0
    include_depth: bool = True
    tile_width: int = 0
    tile_height: int = 0
    output: str | None = None
    frame_output: str | None = None
    frame_step: int = -1


def text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def episode_names(file: h5py.File) -> list[str]:
    if "episodes" not in file:
        raise KeyError("HDF5 file has no /episodes group")
    names = sorted(file["episodes"].keys())
    if not names:
        raise RuntimeError("HDF5 file contains no episodes")
    return names


def _json_list(value) -> list[str]:
    if value is None:
        return []
    try:
        parsed = json.loads(text(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def ordered_cameras(
    file: h5py.File,
    episode: h5py.Group,
    selected: tuple[str, ...] = (),
) -> list[str]:
    if "camera" not in episode:
        raise KeyError(f"Episode {episode.name} has no /camera group")
    available = list(episode["camera"].keys())
    if not available:
        raise RuntimeError(f"Episode {episode.name} has no camera groups")
    if selected:
        missing = [name for name in selected if name not in available]
        if missing:
            raise KeyError(f"Unknown cameras {missing}; available: {available}")
        return list(selected)

    metadata_order = _json_list(episode.attrs.get("camera_names"))
    if not metadata_order:
        metadata_order = _json_list(file.attrs.get("camera_names"))
    ordered = [name for name in metadata_order if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return ordered


def _require(group: h5py.Group, key: str, errors: list[str]) -> h5py.Dataset | None:
    if key not in group:
        errors.append(f"missing dataset: {group.name}/{key}")
        return None
    return group[key]


def validate_episode(episode: h5py.Group) -> list[str]:
    errors: list[str] = []
    for key in ("timestamp", "sim_time", "step_index", "action"):
        dataset = _require(episode, key, errors)
        if dataset is not None and len(dataset.shape) < 1:
            errors.append(f"dataset has no time dimension: {dataset.name}")

    if "camera" not in episode:
        errors.append("missing group: /camera")
        return errors
    cameras = list(episode["camera"].keys())
    if not cameras:
        errors.append("camera group is empty")
        return errors

    expected_frames = int(episode["timestamp"].shape[0]) if "timestamp" in episode else None
    if "action" in episode and expected_frames is not None:
        if int(episode["action"].shape[0]) != expected_frames:
            errors.append(
                f"action frames {episode['action'].shape[0]} != timestamp frames {expected_frames}"
            )

    for camera_name in cameras:
        camera = episode["camera"][camera_name]
        rgb = _require(camera, "rgb", errors)
        if rgb is None:
            continue
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            errors.append(f"{rgb.name} must have shape (T, H, W, 3), got {rgb.shape}")
        if expected_frames is not None and int(rgb.shape[0]) != expected_frames:
            errors.append(
                f"{rgb.name} frames {rgb.shape[0]} != timestamp frames {expected_frames}"
            )
        depth = camera.get("depth")
        if depth is not None and int(depth.shape[0]) != int(rgb.shape[0]):
            errors.append(
                f"{depth.name} frames {depth.shape[0]} != rgb frames {rgb.shape[0]}"
            )
    return errors


def print_summary(
    input_path: str,
    file: h5py.File,
    episode: h5py.Group,
    cameras: list[str],
) -> None:
    print("=" * 78)
    print(f"file          : {os.path.abspath(input_path)}")
    print(f"schema        : {text(file.attrs.get('schema', 'missing'))}")
    print(f"task          : {text(file.attrs.get('task', 'missing'))}")
    print(f"task profile  : {text(file.attrs.get('task_profile', 'missing'))}")
    print(f"source        : {text(file.attrs.get('source', 'missing'))}")
    print(
        "instruction   : "
        f"{text(episode.attrs.get('instruction', file.attrs.get('instruction', '')))}"
    )
    print(f"episodes      : {len(file['episodes'])}")
    print(f"episode group : {episode.name}")
    print(f"steps         : {len(episode['timestamp'])}")
    print(f"success       : {bool(episode.attrs.get('success', False))}")
    print(f"termination   : {text(episode.attrs.get('termination_label', 'missing'))}")
    action = episode["action"][:]
    print(f"action shape  : {action.shape}")
    if action.size:
        print(f"action range  : min={action.min():.4f} max={action.max():.4f}")
    timestamps = episode["timestamp"][:]
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        print(
            f"timestamp dt  : median={np.median(dt):.4f}s "
            f"min={dt.min():.4f}s max={dt.max():.4f}s"
        )
    print(f"cameras       : {cameras}")
    for camera_name in cameras:
        camera = episode["camera"][camera_name]
        rgb = camera["rgb"]
        depth = camera.get("depth")
        depth_text = f" depth={depth.shape}" if depth is not None else " depth=not-recorded"
        projection = text(camera.attrs.get("projection_model", "unknown"))
        details = [f"projection={projection}"]
        if "intrinsics" in camera:
            details.append(f"K={camera['intrinsics'].shape}")
        if "T_world_camera_ros" in camera:
            details.append(f"extrinsics={camera['T_world_camera_ros'].shape}")
        print(
            f"camera {camera_name:<18}: rgb={rgb.shape}{depth_text} "
            + " ".join(details)
        )
        if depth is not None:
            depth_sample = depth[0]
            finite = depth_sample[np.isfinite(depth_sample)]
            if finite.size:
                print(
                    f"  depth[0] m      : min={finite.min():.4f} "
                    f"median={np.median(finite):.4f} max={finite.max():.4f}"
                )
    if "termination" in episode and "names" in episode["termination"].attrs:
        print(f"termination terms: {_json_list(episode['termination'].attrs['names'])}")
    print("=" * 78)


def _depth_preview(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scale = max(hi - lo, 1e-6)
    normalized = np.clip((depth - lo) / scale, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def _label(image: np.ndarray, lines: tuple[str, ...]) -> np.ndarray:
    image = image.copy()
    line_height = 22
    header_height = line_height * len(lines)
    cv2.rectangle(image, (0, 0), (image.shape[1], header_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        scale = 0.40
        while scale > 0.22 and cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            1,
        )[0][0] > image.shape[1] - 10:
            scale -= 0.02
        cv2.putText(
            image,
            line,
            (5, 16 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _camera_tile(
    camera: h5py.Group,
    frame_index: int,
    *,
    width: int,
    height: int,
    include_depth: bool,
    depth_range: tuple[float, float] | None,
    label: tuple[str, ...],
) -> np.ndarray:
    rgb = camera["rgb"][frame_index].astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    if include_depth and "depth" in camera and depth_range is not None:
        depth = camera["depth"][frame_index].astype(np.float32)
        depth_view = _depth_preview(depth, *depth_range)
        if depth_view.shape[:2] != (height, width):
            depth_view = cv2.resize(
                depth_view,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        rgb = np.concatenate([rgb, depth_view], axis=1)
    return _label(rgb, label)


def render_episode(
    input_path: str,
    episode: h5py.Group,
    cameras: list[str],
    options: ReplayRenderOptions,
) -> tuple[str, str | None]:
    if options.stride < 1:
        raise ValueError("stride must be >= 1")
    total_frames = int(episode["timestamp"].shape[0])
    indices = list(range(0, total_frames, options.stride))
    if options.max_frames > 0:
        indices = indices[: options.max_frames]
    if not indices:
        raise RuntimeError("No frames selected for replay")

    fps = options.fps
    if fps <= 0.0:
        timestamps = episode["timestamp"][:]
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            positive = dt[dt > 0]
            fps = 1.0 / float(np.median(positive)) if positive.size else 5.0
        else:
            fps = 5.0
    fps = float(np.clip(fps, 1.0, 60.0))

    first_rgb = episode["camera"][cameras[0]]["rgb"][0]
    tile_height = options.tile_height or int(first_rgb.shape[0])
    tile_width = options.tile_width or int(first_rgb.shape[1])
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile dimensions must be positive")

    depth_ranges: dict[str, tuple[float, float] | None] = {}
    for camera_name in cameras:
        depth = episode["camera"][camera_name].get("depth") if options.include_depth else None
        if depth is None:
            depth_ranges[camera_name] = None
            continue
        finite = depth[:][np.isfinite(depth[:])]
        depth_ranges[camera_name] = (
            (float(np.percentile(finite, 2.0)), float(np.percentile(finite, 98.0)))
            if finite.size
            else (0.0, 1.0)
        )

    output_path = options.output
    if output_path is None:
        stem = Path(input_path).stem
        suffix = "-".join(cameras)
        output_path = os.path.join(
            "recordings",
            "replay",
            f"{stem}_{episode.name.replace('/', '_')}_{suffix}.mp4",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    actions = episode["action"][:]
    success = episode["termination"]["success"][:] if "termination" in episode and "success" in episode["termination"] else None

    sample_tiles = [
        _camera_tile(
            episode["camera"][camera_name],
            indices[0],
            width=tile_width,
            height=tile_height,
            include_depth=options.include_depth,
            depth_range=depth_ranges[camera_name],
            label=("",),
        )
        for camera_name in cameras
    ]
    output_width = sum(tile.shape[1] for tile in sample_tiles)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, tile_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame_index in indices:
            tiles = []
            for camera_name in cameras:
                success_value = (
                    "?"
                    if success is None
                    else str(int(bool(success[frame_index])))
                )
                label = (
                    f"{camera_name} t={frame_index} s={success_value} "
                    f"g={actions[frame_index, -1]:+.2f}",
                )
                tiles.append(
                    _camera_tile(
                        episode["camera"][camera_name],
                        frame_index,
                        width=tile_width,
                        height=tile_height,
                        include_depth=options.include_depth,
                        depth_range=depth_ranges[camera_name],
                        label=label,
                    )
                )
            writer.write(np.concatenate(tiles, axis=1))
    finally:
        writer.release()

    frame_output = None
    if options.frame_output:
        frame_index = options.frame_step if options.frame_step >= 0 else total_frames - 1
        frame_index = min(max(frame_index, 0), total_frames - 1)
        tiles = []
        for camera_name in cameras:
            success_value = "?" if success is None else str(int(bool(success[frame_index])))
            label = (f"{camera_name} t={frame_index} s={success_value}",)
            tiles.append(
                _camera_tile(
                    episode["camera"][camera_name],
                    frame_index,
                    width=tile_width,
                    height=tile_height,
                    include_depth=options.include_depth,
                    depth_range=depth_ranges[camera_name],
                    label=label,
                )
            )
        frame_output = os.path.abspath(options.frame_output)
        os.makedirs(os.path.dirname(frame_output), exist_ok=True)
        cv2.imwrite(frame_output, np.concatenate(tiles, axis=1))
    return os.path.abspath(output_path), frame_output
