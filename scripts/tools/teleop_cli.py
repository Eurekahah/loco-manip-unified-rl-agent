"""Shared teleoperation CLI defaults that must be applied before AppLauncher."""

from __future__ import annotations

import argparse
import sys


def _has_explicit_device(argv: list[str]) -> bool:
    return any(item == "--device" or item.startswith("--device=") for item in argv)


def prefer_cpu_for_single_env_teleop(
    args,
    *,
    argv: list[str] | None = None,
) -> bool:
    """Use CPU physics for single-env teleoperation unless explicitly overridden.

    Isaac Lab documents that CPU physics can reduce latency for single-env
    teleoperation because GPU pipelines pay synchronization overhead without
    amortizing it across many environments.
    """
    if int(getattr(args, "num_envs", 1)) != 1:
        return False
    argv = sys.argv[1:] if argv is None else argv
    if _has_explicit_device(argv):
        return False
    if str(getattr(args, "device", "")).startswith("cpu"):
        return False

    args.device = "cpu"
    print(
        "[Teleop] single-env teleoperation uses CPU physics by default; "
        "pass --device cuda:0 to override"
    )
    return True


def add_standstill_filter_args(parser) -> None:
    """Add the standstill low-speed filter options shared by teleop scripts.

    The defaults match the M20-Piper deployment recommendation: a hard zero on
    the wheel action for an all-zero base command (the trained low-level policy
    carries a wheel bias at standstill), while lateral / yaw commands only get
    a small dead band so they stay responsive.
    """
    parser.add_argument(
        "--low-speed-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用低层腿/轮动作的静止低速滤波（默认开启）。",
    )
    parser.add_argument(
        "--standstill-hard-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "vx=vy=wz=0 时把轮子动作强制置零（默认开启）。"
            "关闭后退回“死区 + 低通”的软滤波。"
        ),
    )
    parser.add_argument(
        "--standstill-command-threshold",
        type=float,
        default=0.1,
        help="速度指令分量绝对值低于该阈值即视为 0（m/s 或 rad/s，默认 0.1）。",
    )
    parser.add_argument(
        "--standstill-leg-deadband",
        type=float,
        default=0.02,
        help="仅在 vx=0 时生效的腿部动作死区（归一化动作，默认 0.02）。",
    )
    parser.add_argument(
        "--standstill-leg-cutoff-hz",
        type=float,
        default=8.0,
        help="仅在 vx=0 时生效的腿部一阶低通截止频率（默认 8 Hz）。",
    )
    parser.add_argument(
        "--standstill-wheel-deadband",
        type=float,
        default=0.04,
        help="仅在 vx=0（vy/wz 可非零）时生效的轮子动作死区（默认 0.04）。",
    )
    parser.add_argument(
        "--standstill-wheel-cutoff-hz",
        type=float,
        default=2.0,
        help="仅在 vx=0 时生效的轮子一阶低通截止频率（默认 2 Hz）。",
    )
    parser.add_argument(
        "--standstill-wheel-all-zero-deadband",
        type=float,
        default=0.6,
        help=(
            "vx=vy=wz=0 时轮子动作死区（默认 0.6）。"
            "开启 --standstill-hard-zero 时该值不再影响全零命令，仅作为软滤波回退值。"
        ),
    )


def apply_standstill_filter_args(env_cfg, args) -> bool:
    """Write the teleop CLI filter options into the resolved low-level action cfg."""
    actions = getattr(env_cfg, "actions", None)
    term = getattr(actions, "pre_trained_pick_action", None)
    if term is None:
        return False
    applied = False
    specs = (
        (
            "low_level_leg_actions",
            args.standstill_leg_deadband,
            args.standstill_leg_deadband,
            args.standstill_leg_cutoff_hz,
            False,
        ),
        (
            "low_level_wheel_actions",
            args.standstill_wheel_deadband,
            args.standstill_wheel_all_zero_deadband,
            args.standstill_wheel_cutoff_hz,
            bool(args.standstill_hard_zero),
        ),
    )
    for name, deadband, all_zero_deadband, cutoff_hz, hard_zero in specs:
        action_cfg = getattr(term, name, None)
        filt = getattr(action_cfg, "low_speed_filter", None) if action_cfg is not None else None
        if filt is None:
            continue
        filt.enabled = bool(args.low_speed_filter)
        filt.command_threshold = float(args.standstill_command_threshold)
        filt.low_speed_threshold = float(deadband)
        filt.all_zero_low_speed_threshold = float(all_zero_deadband)
        filt.cutoff_hz = float(cutoff_hz)
        filt.all_zero_cutoff_hz = None
        filt.all_zero_hard_zero = bool(args.low_speed_filter and hard_zero)
        applied = True
    if applied:
        print(
            "[Teleop] standstill filter: "
            f"enabled={bool(args.low_speed_filter)} "
            f"cmd_thr={args.standstill_command_threshold} "
            f"leg={args.standstill_leg_deadband}/{args.standstill_leg_cutoff_hz}Hz "
            f"wheel={args.standstill_wheel_deadband}/{args.standstill_wheel_cutoff_hz}Hz "
            f"all_zero={args.standstill_wheel_all_zero_deadband} "
            f"hard_zero={bool(args.low_speed_filter and args.standstill_hard_zero)}"
        )
    return applied
