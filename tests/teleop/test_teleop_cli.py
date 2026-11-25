from __future__ import annotations

import argparse
from types import SimpleNamespace

from scripts.tools.teleop_cli import (
    add_standstill_filter_args,
    apply_standstill_filter_args,
    prefer_cpu_for_single_env_teleop,
)
from scripts.tools.fisheye_cli import add_fisheye_args, bottom_row_layout


def test_single_env_teleop_defaults_to_cpu() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_teleop(args, argv=[])

    assert changed
    assert args.device == "cpu"


def test_explicit_device_overrides_fast_default() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_teleop(
        args,
        argv=["--device", "cuda:0"],
    )

    assert not changed
    assert args.device == "cuda:0"


def test_multi_env_keeps_device() -> None:
    multi_env = SimpleNamespace(num_envs=2, device="cuda:0")

    assert not prefer_cpu_for_single_env_teleop(multi_env, argv=[])
    assert multi_env.device == "cuda:0"


def test_teleop_fisheye_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_fisheye_args(parser, include_panorama=True)
    parser.set_defaults(headless=False)

    args = parser.parse_args([])

    assert args.fisheye_rig == "four"
    assert args.panorama_backend == "gpu"
    assert args.show_panorama
    assert args.show_wrist_camera
    assert args.show_front_camera
    assert args.panorama_window_pos is None
    assert args.wrist_window_pos is None
    assert args.front_window_pos is None


def test_legacy_fisheye_defaults_stay_off() -> None:
    parser = argparse.ArgumentParser()
    add_fisheye_args(
        parser,
        include_panorama=True,
        rig_default="none",
        show_wrist_default=False,
        show_panorama_default=False,
        show_front_default=False,
    )
    parser.set_defaults(headless=False)

    args = parser.parse_args([])

    assert args.fisheye_rig == "none"
    assert not args.show_panorama
    assert not args.show_wrist_camera
    assert not args.show_front_camera


def test_bottom_row_layout_does_not_overlap() -> None:
    panels = [(560, 280, 52), (240, 240, 44), (320, 240, 44)]

    layout = bottom_row_layout(panels, app_width=1280, app_height=720)

    assert len(layout) == 3
    for (x, y, width, height), (_, _, title) in zip(layout, panels):
        assert x >= 0
        assert y >= 0
        assert x + width <= 1280
        assert y + height + title == 720 - 12
    for left, right in zip(layout, layout[1:]):
        assert left[0] + left[2] <= right[0]


def test_bottom_row_layout_scales_down_when_too_wide() -> None:
    panels = [(960, 480, 52), (640, 480, 44), (640, 480, 44)]

    layout = bottom_row_layout(panels, app_width=1280, app_height=720)

    assert sum(item[2] for item in layout) <= 1280
    for left, right in zip(layout, layout[1:]):
        assert left[0] + left[2] <= right[0]


def _standstill_args(**overrides):
    parser = argparse.ArgumentParser()
    add_standstill_filter_args(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _standstill_env_cfg():
    def _filter():
        return SimpleNamespace(
            enabled=True,
            command_threshold=0.1,
            low_speed_threshold=0.0,
            all_zero_low_speed_threshold=None,
            cutoff_hz=0.0,
            all_zero_cutoff_hz=None,
            all_zero_hard_zero=False,
        )

    leg = SimpleNamespace(low_speed_filter=_filter())
    wheel = SimpleNamespace(low_speed_filter=_filter())
    term = SimpleNamespace(low_level_leg_actions=leg, low_level_wheel_actions=wheel)
    return SimpleNamespace(actions=SimpleNamespace(pre_trained_pick_action=term)), leg, wheel


def test_standstill_hard_zero_defaults_to_leg_and_wheel() -> None:
    env_cfg, leg, wheel = _standstill_env_cfg()

    assert apply_standstill_filter_args(env_cfg, _standstill_args())

    assert leg.low_speed_filter.all_zero_hard_zero
    assert wheel.low_speed_filter.all_zero_hard_zero
    assert leg.low_speed_filter.enabled and wheel.low_speed_filter.enabled


def test_standstill_hard_zero_scope_wheel_keeps_leg_soft() -> None:
    env_cfg, leg, wheel = _standstill_env_cfg()

    assert apply_standstill_filter_args(
        env_cfg, _standstill_args(standstill_hard_zero_scope="wheel")
    )

    assert not leg.low_speed_filter.all_zero_hard_zero
    assert wheel.low_speed_filter.all_zero_hard_zero


def test_standstill_hard_zero_can_be_disabled() -> None:
    env_cfg, leg, wheel = _standstill_env_cfg()

    assert apply_standstill_filter_args(env_cfg, _standstill_args(standstill_hard_zero=False))

    assert not leg.low_speed_filter.all_zero_hard_zero
    assert not wheel.low_speed_filter.all_zero_hard_zero


def test_standstill_filter_disabled_turns_everything_off() -> None:
    env_cfg, leg, wheel = _standstill_env_cfg()

    assert apply_standstill_filter_args(env_cfg, _standstill_args(low_speed_filter=False))

    for action_cfg in (leg, wheel):
        assert not action_cfg.low_speed_filter.enabled
        assert not action_cfg.low_speed_filter.all_zero_hard_zero
