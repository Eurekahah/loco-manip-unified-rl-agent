#!/usr/bin/env python3
"""Task-agnostic HDF5 inspection and video replay entry point."""

from __future__ import annotations

import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from replay.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
