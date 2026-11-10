#!/usr/bin/env python3
"""Render a three-camera side-by-side replay from a collection HDF5 episode."""

from __future__ import annotations

import argparse
import json
import os

import cv2
import h5py
import numpy as np


DEFAULT_CAMERAS = ("top_camera", "wrist_camera", "workcell_camera")


parser = argparse.ArgumentParser(description="Replay the three SmolVLA camera views side by side.")
parser.add_argument("--input", type=str, required=True, help="Path to the .hdf5 file.")
parser.add_argument("--episode", type=int, default=0, help="Episode index, not the HDF5 group name.")
parser.add_argument("--stride", type=int, default=5, help="Keep every Nth recorded frame (5 = 50 Hz -> 10 Hz).")
parser.add_argument("--fps", type=float, default=10.0, help="Replay video FPS.")
parser.add_argument("--output", type=str, default=None, help="Replay MP4 path.")
parser.add_argument("--frame", type=str, default=None, help="Optional PNG path for the last montage frame.")
parser.add_argument("--frame-step", type=int, default=-1, help="Recorded step to save as PNG; -1 uses the last frame.")
args = parser.parse_args()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    tile = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    with h5py.File(args.input, "r") as file:
        if "episodes" not in file:
            raise KeyError("HDF5 file has no /episodes group")
        names = sorted(file["episodes"].keys())
        if not names:
            raise RuntimeError("HDF5 file contains no episodes")
        if args.episode < 0 or args.episode >= len(names):
            raise IndexError(f"--episode must be in [0, {len(names) - 1}]")
        episode = file["episodes"][names[args.episode]]

        available = list(episode["camera"].keys())
        attr_names = episode.attrs.get("camera_names")
        ordered = list(DEFAULT_CAMERAS)
        if attr_names is not None:
            try:
                ordered = json.loads(_text(attr_names))
            except json.JSONDecodeError:
                ordered = list(DEFAULT_CAMERAS)
        cameras = [name for name in ordered if name in available]
        cameras += [name for name in available if name not in cameras]
        print(f"cameras: {cameras}")

        actions = episode["action"][:]
        success = episode["termination"]["success"][:]
        num_frames = len(actions)
        indices = list(range(0, num_frames, max(args.stride, 1)))
        if indices[-1] != num_frames - 1:
            indices.append(num_frames - 1)

        first = episode["camera"][cameras[0]]["rgb"][0]
        height, width = first.shape[:2]
        output_path = args.output
        if output_path is None:
            stem = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join("recordings", "replay", f"{stem}_{names[args.episode]}_three_view.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (width * len(cameras), height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        last_tile = None
        for frame_index in indices:
            tiles = []
            for camera_name in cameras:
                text = f"{camera_name} step={frame_index} succ={bool(success[frame_index])} grip={actions[frame_index, -1]:+.1f}"
                tiles.append(_label(episode["camera"][camera_name]["rgb"][frame_index], text))
            last_tile = np.concatenate(tiles, axis=1)
            writer.write(last_tile)
        writer.release()
        print(f"replay video: {os.path.abspath(output_path)}")

        if args.frame:
            if args.frame_step >= 0:
                frame_index = min(args.frame_step, num_frames - 1)
            else:
                frame_index = num_frames - 1
            tiles = []
            for camera_name in cameras:
                text = f"{camera_name} step={frame_index} succ={bool(success[frame_index])}"
                tiles.append(_label(episode["camera"][camera_name]["rgb"][frame_index], text))
            os.makedirs(os.path.dirname(os.path.abspath(args.frame)), exist_ok=True)
            cv2.imwrite(args.frame, np.concatenate(tiles, axis=1))
            print(f"montage frame: {os.path.abspath(args.frame)}")


if __name__ == "__main__":
    main()
