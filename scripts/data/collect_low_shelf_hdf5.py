#!/usr/bin/env python3
"""Backward-compatible entry point for task-isolated HDF5 collection."""

from __future__ import annotations

import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from collect_demonstrations import main  # noqa: E402


if __name__ == "__main__":
    main(
        default_task="Isaac-M20-Piper-LowShelf-Collect-v0",
        description=(
            "Compatibility entry point for demonstration collection. "
            "Prefer scripts/data/collect_demonstrations.py for new tasks."
        ),
    )
