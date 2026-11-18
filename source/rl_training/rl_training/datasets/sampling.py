"""Deterministic record-clock helpers."""

from __future__ import annotations

import math


def align_period_to_step(period: float, step_dt: float) -> float:
    """Align a sensor period down to a whole number of control steps."""
    if period <= 0.0 or step_dt <= 0.0:
        raise ValueError("period and step_dt must be positive")
    steps = max(1, int(math.floor(period / step_dt + 1e-9)))
    return steps * step_dt


def advance_record_clock(
    elapsed: float,
    step_dt: float,
    period: float,
) -> tuple[float, bool]:
    """Advance a phase accumulator and return the remaining phase plus due flag."""
    elapsed += float(step_dt)
    if elapsed + 1e-9 >= float(period):
        return max(elapsed - float(period), 0.0), True
    return elapsed, False
