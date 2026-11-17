"""Shared teleoperation CLI defaults that must be applied before AppLauncher."""

from __future__ import annotations

import sys


def _has_explicit_device(argv: list[str]) -> bool:
    return any(item == "--device" or item.startswith("--device=") for item in argv)


def prefer_cpu_for_single_env_fast_path(
    args,
    *,
    enabled: bool,
    argv: list[str] | None = None,
) -> bool:
    """Use CPU physics for a single fast-path environment unless overridden.

    Isaac Lab documents that CPU physics can reduce latency for single-env
    teleoperation because GPU pipelines pay synchronization overhead without
    amortizing it across many environments.
    """
    if not enabled or int(getattr(args, "num_envs", 1)) != 1:
        return False
    argv = sys.argv[1:] if argv is None else argv
    if _has_explicit_device(argv):
        return False
    if str(getattr(args, "device", "")).startswith("cpu"):
        return False

    args.device = "cpu"
    print(
        "[Teleop] single-env fast path uses CPU physics by default; "
        "pass --device cuda:0 to override"
    )
    return True
