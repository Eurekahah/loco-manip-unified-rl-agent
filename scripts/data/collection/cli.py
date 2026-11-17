"""Shared command-line interface for task-isolated HDF5 collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from fisheye_cli import add_fisheye_args, validate_fisheye_args  # noqa: E402


def build_collection_parser(
    *,
    description: str,
    default_task: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", type=str, default=default_task)
    parser.add_argument(
        "--profile",
        type=str,
        default="auto",
        help="Collection task profile key, or auto to resolve from --task.",
    )
    parser.add_argument(
        "--profile-module",
        action="append",
        default=[],
        help="Import a Python module that registers additional collection profiles.",
    )
    parser.add_argument("--input", choices=("keyboard", "vr"), default="keyboard")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after N environment steps; 0 runs until app exit.",
    )
    parser.add_argument("--camera_width", type=int, default=224)
    parser.add_argument("--camera_height", type=int, default=224)
    parser.add_argument("--camera_update_period", type=float, default=0.1)
    parser.add_argument(
        "--record_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record depth maps in HDF5. RGB-only is the default.",
    )
    parser.add_argument("--scene_camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist_camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top_camera", action=argparse.BooleanOptionalAction, default=True)
    add_fisheye_args(
        parser,
        include_panorama=True,
        width_default=480,
        height_default=300,
        update_period_default=0.1,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_collection_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if (
        not args.scene_camera
        and not args.wrist_camera
        and not args.top_camera
        and args.fisheye_rig == "none"
    ):
        parser.error("At least one camera must be enabled")
    validate_fisheye_args(parser, args, include_panorama=True)
    if args.num_envs != 1:
        parser.error("Interactive collection currently requires --num_envs 1")
    if args.show_wrist_camera and not args.wrist_camera:
        parser.error("--show-wrist-camera requires --wrist_camera")
    if args.max_steps < 0:
        parser.error("--max-steps must be >= 0")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera_width and --camera_height must be positive")
    if args.camera_update_period < 0.0:
        parser.error("--camera_update_period must be >= 0")


def apply_rendering_default(args: argparse.Namespace) -> None:
    if args.fisheye_rig != "none" and not any(
        item.startswith("--rendering_mode") for item in sys.argv[1:]
    ):
        args.rendering_mode = "balanced"
        print("[Collect] fisheye enabled; using rendering_mode=balanced unless explicitly overridden")
