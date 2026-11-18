from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "rl_training"
    / "rl_training"
    / "datasets"
    / "sampling.py"
)
SPEC = importlib.util.spec_from_file_location("record_sampling_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SAMPLING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLING)


@pytest.mark.parametrize(
    ("record_hz", "expected_frames"),
    [
        (10.0, 40),
        (15.0, 60),
        (20.0, 80),
        (25.0, 100),
        (30.0, 120),
        (50.0, 200),
    ],
)
def test_record_clock_produces_requested_average_rate(
    record_hz: float,
    expected_frames: int,
) -> None:
    step_dt = 0.02
    period = 1.0 / record_hz
    elapsed = max(period - step_dt, 0.0)
    frames = 0

    for _ in range(200):
        elapsed, due = SAMPLING.advance_record_clock(elapsed, step_dt, period)
        frames += int(due)

    assert frames == expected_frames


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (0.10, 0.10),
        (0.0666666667, 0.06),
        (0.05, 0.04),
        (0.04, 0.04),
        (0.0333333333, 0.02),
    ],
)
def test_sensor_period_aligns_down_to_control_step(
    period: float,
    expected: float,
) -> None:
    assert SAMPLING.align_period_to_step(period, 0.02) == pytest.approx(expected)
