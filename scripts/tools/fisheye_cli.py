"""Shared command-line options for optional fisheye teleoperation rigs."""

from __future__ import annotations

import argparse


FISHEYE_RIG_CHOICES = ("none", "dual", "four")


def add_fisheye_args(
    parser: argparse.ArgumentParser,
    *,
    include_panorama: bool,
    width_default: int = 480,
    height_default: int = 300,
    update_period_default: float = 0.0,
) -> None:
    """Add optional dual/four fisheye sensor arguments to a script parser."""
    parser.add_argument(
        "--fisheye-rig",
        choices=FISHEYE_RIG_CHOICES,
        default="none",
        help="Optional fisheye rig: none, dual front/back, or four-camera horizontal ring.",
    )
    parser.add_argument("--fisheye-width", type=int, default=width_default)
    parser.add_argument("--fisheye-height", type=int, default=height_default)
    parser.add_argument(
        "--fisheye-update-period",
        type=float,
        default=update_period_default,
        help="Sensor update period in seconds; 0 updates every render step.",
    )
    parser.add_argument(
        "--fisheye-fov-deg",
        type=float,
        default=200.0,
        help="Full field of view of each ideal circular fisheye camera.",
    )
    parser.add_argument(
        "--fisheye-mount-pos",
        type=float,
        nargs=3,
        default=(-0.44, 0.0, 0.34),
        metavar=("X", "Y", "Z"),
        help="Rig origin in the M20 base_link frame.",
    )
    parser.add_argument(
        "--fisheye-lens-offset",
        type=float,
        default=0.02,
        help="Half front/back lens spacing for the dual rig, in metres.",
    )

    if include_panorama:
        parser.add_argument(
            "--show-panorama",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show the stitched equirectangular panorama in a separate omni.ui window.",
        )
        parser.add_argument("--panorama-width", type=int, default=960)
        parser.add_argument("--panorama-height", type=int, default=480)


def validate_fisheye_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    include_panorama: bool,
) -> None:
    """Validate shared fisheye options before launching Isaac Sim."""
    if args.fisheye_width <= 0 or args.fisheye_height <= 0:
        parser.error("--fisheye-width and --fisheye-height must be positive")
    if args.fisheye_update_period < 0.0:
        parser.error("--fisheye-update-period must be >= 0")
    if not 0.0 < args.fisheye_fov_deg <= 220.0:
        parser.error("--fisheye-fov-deg must be in (0, 220]")
    if args.fisheye_lens_offset < 0.0:
        parser.error("--fisheye-lens-offset must be >= 0")
    if include_panorama:
        if args.show_panorama and args.fisheye_rig == "none":
            parser.error("--show-panorama requires --fisheye-rig dual or four")
        if args.show_panorama and args.headless:
            parser.error("--show-panorama requires GUI; remove --headless")
        if args.panorama_width <= 0 or args.panorama_height <= 0:
            parser.error("--panorama-width and --panorama-height must be positive")
