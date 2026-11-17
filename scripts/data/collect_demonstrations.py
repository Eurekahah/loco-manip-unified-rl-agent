#!/usr/bin/env python3
"""Collect HDF5 demonstrations through a task-isolated profile."""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from collection.cli import (  # noqa: E402
    apply_rendering_default,
    build_collection_parser,
    validate_collection_args,
)


def main(
    *,
    default_task: str = "",
    description: str = "Collect task-isolated demonstrations to HDF5.",
) -> None:
    parser = build_collection_parser(
        description=description,
        default_task=default_task,
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if not args_cli.task:
        parser.error("--task is required")

    apply_rendering_default(args_cli)
    validate_collection_args(parser, args_cli)
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        from collection.runner import run_collection

        run_collection(args_cli, parser, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
