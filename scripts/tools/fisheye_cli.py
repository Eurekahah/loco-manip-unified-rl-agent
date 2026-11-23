"""Shared command-line options for optional fisheye teleoperation rigs."""

from __future__ import annotations

import argparse


FISHEYE_RIG_CHOICES = ("none", "two", "dual", "four")
PANORAMA_BACKEND_CHOICES = ("auto", "gpu", "cpu")
HDF5_COMPRESSION_CHOICES = ("none", "lzf", "gzip1", "gzip4")


def add_fisheye_args(
    parser: argparse.ArgumentParser,
    *,
    include_panorama: bool,
    width_default: int = 480,
    height_default: int = 300,
    update_period_default: float = 0.0,
    rig_default: str = "four",
    show_wrist_default: bool = True,
    show_panorama_default: bool = True,
    show_front_default: bool = True,
) -> None:
    """Add optional dual/four fisheye sensor arguments to a script parser."""
    parser.add_argument(
        "--fisheye-rig",
        choices=FISHEYE_RIG_CHOICES,
        default=rig_default,
        help=(
            "Fisheye rig: four (default horizontal ring), two/dual (front/back), "
            "or none to disable."
        ),
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
            default=show_panorama_default,
            help="Show the stitched equirectangular panorama in a separate omni.ui window.",
        )
        parser.add_argument("--panorama-width", type=int, default=960)
        parser.add_argument("--panorama-height", type=int, default=480)
        parser.add_argument(
            "--panorama-backend",
            choices=PANORAMA_BACKEND_CHOICES,
            default="gpu",
            help=(
                "Panorama projection backend. gpu falls back to CPU when CUDA is "
                "unavailable; auto also prefers CUDA."
            ),
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
            default=None,
            metavar=("X", "Y"),
            help="Optional fixed panorama window position; default is the bottom row.",
        )
        parser.add_argument(
            "--panorama-window-width",
            type=int,
            default=560,
            help="Panorama preview width in desktop pixels.",
        )
        parser.add_argument(
            "--panorama-window-height",
            type=int,
            default=280,
            help="Panorama preview height in desktop pixels.",
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
        default=show_wrist_default,
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
        default=240,
        help="Wrist preview width in desktop pixels.",
    )
    parser.add_argument(
        "--wrist-window-height",
        type=int,
        default=240,
        help="Wrist preview height in desktop pixels.",
    )
    parser.add_argument(
        "--wrist-window-pos",
        type=int,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed wrist window position; default is the bottom row.",
    )
    parser.add_argument(
        "--show-front-camera",
        action=argparse.BooleanOptionalAction,
        default=show_front_default,
        help="Show the forward/chest camera in a throttled omni.ui window.",
    )
    parser.add_argument(
        "--front-display-hz",
        type=float,
        default=20.0,
        help="Maximum front-camera UI refresh rate.",
    )
    parser.add_argument(
        "--front-window-width",
        type=int,
        default=320,
        help="Front-camera preview width in desktop pixels.",
    )
    parser.add_argument(
        "--front-window-height",
        type=int,
        default=240,
        help="Front-camera preview height in desktop pixels.",
    )
    parser.add_argument(
        "--front-window-pos",
        type=int,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed front-camera window position; default is the bottom row.",
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
    if args.front_display_hz <= 0.0:
        parser.error("--front-display-hz must be > 0")
    if args.front_window_width <= 0 or args.front_window_height <= 0:
        parser.error(
            "--front-window-width and --front-window-height must be positive"
        )
    if args.writer_queue_size <= 0:
        parser.error("--writer-queue-size must be positive")
    if args.profile_interval <= 0.0:
        parser.error("--profile-interval must be > 0")
    if args.headless:
        # GUI previews are meaningless without a window; disable them instead of
        # failing so that headless collection keeps working with the new defaults.
        args.show_wrist_camera = False
        args.show_front_camera = False
        if include_panorama:
            args.show_panorama = False
    if include_panorama:
        if args.fisheye_rig == "none":
            if args.show_panorama or args.record_panorama:
                print(
                    "[Teleop] --fisheye-rig none disables panorama preview/recording"
                )
            args.show_panorama = False
            args.record_panorama = False
        if args.panorama_width <= 0 or args.panorama_height <= 0:
            parser.error("--panorama-width and --panorama-height must be positive")
        if args.panorama_window_width <= 0 or args.panorama_window_height <= 0:
            parser.error(
                "--panorama-window-width and --panorama-window-height must be positive"
            )
        if args.panorama_display_hz <= 0.0:
            parser.error("--panorama-display-hz must be > 0")


def bottom_row_layout(
    panels: list[tuple[int, int, int]],
    *,
    app_width: int = 1280,
    app_height: int = 720,
    margin: int = 12,
    gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Lay out ``(content_width, content_height, title_height)`` panels in one row.

    The row is centred horizontally and anchored to the bottom edge of the app
    window. If the requested panels are too wide they are scaled down together
    so no two panels overlap.
    """
    if not panels:
        return []
    width = max(1, int(app_width))
    height = max(1, int(app_height))
    gap = max(0, int(gap))
    margin = max(0, int(margin))

    available = max(1, width - 2 * margin - gap * (len(panels) - 1))
    requested = sum(max(1, int(panel[0])) for panel in panels)
    scale = min(1.0, available / float(requested))

    sizes = [
        (
            max(1, int(round(panel[0] * scale))),
            max(1, int(round(panel[1] * scale))),
            int(panel[2]),
        )
        for panel in panels
    ]
    total_width = sum(item[0] for item in sizes) + gap * (len(sizes) - 1)
    max_height = max(item[1] + item[2] for item in sizes)
    x = max(margin, (width - total_width) // 2)
    y = max(margin, height - max_height - margin)

    layout: list[tuple[int, int, int, int]] = []
    for panel_width, panel_height, _ in sizes:
        layout.append((x, y, panel_width, panel_height))
        x += panel_width + gap
    return layout
