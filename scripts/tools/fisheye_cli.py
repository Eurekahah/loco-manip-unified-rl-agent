"""Shared command-line options for optional fisheye teleoperation rigs."""

from __future__ import annotations

import argparse


FISHEYE_RIG_CHOICES = ("none", "dual", "four")
PANORAMA_BACKEND_CHOICES = ("auto", "gpu", "cpu")
HDF5_COMPRESSION_CHOICES = ("none", "lzf", "gzip1", "gzip4")


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
        parser.add_argument(
            "--panorama-backend",
            choices=PANORAMA_BACKEND_CHOICES,
            default="auto",
            help="Panorama projection backend. auto uses CUDA when available and keeps CPU cv2.remap as fallback.",
        )
        parser.add_argument(
            "--panorama-display-hz",
            type=float,
            default=12.0,
            help="Maximum panorama UI refresh rate.",
        )
        parser.add_argument(
            "--panorama-window-pos",
            type=int,
            nargs=2,
            default=(980, 80),
            metavar=("X", "Y"),
            help="Desktop panorama window position.",
        )
        parser.add_argument(
            "--record-panorama",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Also save the stitched panorama. Off by default because raw fisheye RGB is preferred.",
        )

    parser.add_argument(
        "--show-wrist-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the existing wrist sensor in a throttled omni.ui window.",
    )
    parser.add_argument(
        "--wrist-display-hz",
        type=float,
        default=20.0,
        help="Maximum wrist UI refresh rate.",
    )
    parser.add_argument(
        "--wrist-window-width",
        type=int,
        default=320,
        help="Wrist preview width in desktop pixels.",
    )
    parser.add_argument(
        "--wrist-window-height",
        type=int,
        default=320,
        help="Wrist preview height in desktop pixels.",
    )
    parser.add_argument(
        "--wrist-window-pos",
        type=int,
        nargs=2,
        default=(640, 80),
        metavar=("X", "Y"),
        help="Desktop wrist window position.",
    )
    parser.add_argument(
        "--preview-hz",
        type=float,
        default=15.0,
        help="Reference GUI preview rate recorded in metadata; specific windows use their own rates.",
    )
    parser.add_argument(
        "--record-hz",
        type=float,
        default=10.0,
        help="Unified state/action/camera sampling rate for HDF5 collection.",
    )
    parser.add_argument(
        "--compression",
        choices=HDF5_COMPRESSION_CHOICES,
        default="lzf",
        help="HDF5 image compression. lzf is the collection default.",
    )
    parser.add_argument(
        "--async-write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write HDF5 frames from a bounded background writer queue.",
    )
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=8,
        help="Maximum queued HDF5 record frames before dropping samples.",
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        default=None,
        help="Write lightweight runtime profiling metrics to this JSON file.",
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=5.0,
        help="Seconds between profiling JSON snapshots.",
    )
    parser.add_argument(
        "--ui-snapshot-dir",
        type=str,
        default=None,
        help="Save the main viewport plus the last wrist/panorama UI images for GUI smoke tests.",
    )


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
    if args.record_hz <= 0.0:
        parser.error("--record-hz must be > 0")
    if args.preview_hz <= 0.0:
        parser.error("--preview-hz must be > 0")
    if args.wrist_display_hz <= 0.0:
        parser.error("--wrist-display-hz must be > 0")
    if args.wrist_window_width <= 0 or args.wrist_window_height <= 0:
        parser.error("--wrist-window-width and --wrist-window-height must be positive")
    if args.writer_queue_size <= 0:
        parser.error("--writer-queue-size must be positive")
    if args.profile_interval <= 0.0:
        parser.error("--profile-interval must be > 0")
    if args.show_wrist_camera and args.headless:
        parser.error("--show-wrist-camera requires GUI; remove --headless")
    if include_panorama:
        if args.show_panorama and args.fisheye_rig == "none":
            parser.error("--show-panorama requires --fisheye-rig dual or four")
        if args.show_panorama and args.headless:
            parser.error("--show-panorama requires GUI; remove --headless")
        if args.panorama_width <= 0 or args.panorama_height <= 0:
            parser.error("--panorama-width and --panorama-height must be positive")
        if args.panorama_display_hz <= 0.0:
            parser.error("--panorama-display-hz must be > 0")
        if args.record_panorama and args.fisheye_rig == "none":
            parser.error("--record-panorama requires --fisheye-rig dual or four")
