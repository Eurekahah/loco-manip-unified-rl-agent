"""Generic CLI for task-agnostic HDF5 inspection and video replay."""

from __future__ import annotations

import argparse

import h5py

from .hdf5_video import (
    ReplayRenderOptions,
    episode_names,
    ordered_cameras,
    print_summary,
    render_episode,
    validate_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and render any teleop_manipulation/v1 HDF5 episode."
    )
    parser.add_argument("--input", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names to render. Defaults to the order stored in HDF5 metadata.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate without rendering.")
    parser.add_argument("--output", default=None, help="Replay MP4 path.")
    parser.add_argument("--fps", type=float, default=0.0, help="Replay FPS; 0 derives from timestamps.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth recorded frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum rendered frames; 0 renders all.")
    parser.add_argument("--no-depth", action="store_true", help="Do not render recorded depth.")
    parser.add_argument("--tile-width", type=int, default=0, help="Normalize camera tile width.")
    parser.add_argument("--tile-height", type=int, default=0, help="Normalize camera tile height.")
    parser.add_argument("--frame", default=None, help="Optional output PNG montage path.")
    parser.add_argument("--frame-step", type=int, default=-1, help="Recorded frame for PNG; -1 uses last.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with h5py.File(args.input, "r") as file:
        names = episode_names(file)
        if args.episode < 0 or args.episode >= len(names):
            raise IndexError(f"--episode must be in [0, {len(names) - 1}]")
        episode = file["episodes"][names[args.episode]]
        cameras = ordered_cameras(file, episode, tuple(args.cameras or ()))
        print_summary(args.input, file, episode, cameras)

        errors = validate_episode(episode)
        if errors:
            print("validation errors:")
            for error in errors:
                print(f"  - {error}")
            raise SystemExit(1)
        if args.check_only:
            return

        video_path, frame_path = render_episode(
            args.input,
            episode,
            cameras,
            ReplayRenderOptions(
                cameras=tuple(cameras),
                fps=args.fps,
                stride=args.stride,
                max_frames=args.max_frames,
                include_depth=not args.no_depth,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                output=args.output,
                frame_output=args.frame,
                frame_step=args.frame_step,
            ),
        )
        print(f"replay video: {video_path}")
        if frame_path:
            print(f"montage frame: {frame_path}")
