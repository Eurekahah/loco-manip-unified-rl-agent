from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "devices"
    / "keyboard_loco_manip_teleop.py"
)


@pytest.fixture()
def keyboard_module(monkeypatch):
    class DeviceBase:
        pass

    class DeviceCfg:
        sim_device = "cpu"

    key_press = object()
    key_release = object()

    class FakeInput:
        KeyboardEventType = types.SimpleNamespace(
            KEY_PRESS=key_press,
            KEY_RELEASE=key_release,
        )

        def acquire_input_interface(self):
            return self

        def subscribe_to_keyboard_events(self, keyboard, callback):
            self.callback = callback
            return 1

        def unsubscribe_to_keyboard_events(self, keyboard, subscription):
            return None

        def get_keyboard_name(self, keyboard):
            return "fake-keyboard"

    fake_carb = types.ModuleType("carb")
    fake_carb.input = FakeInput()
    fake_omni = types.ModuleType("omni")
    fake_appwindow = types.ModuleType("omni.appwindow")
    fake_appwindow.get_default_app_window = lambda: types.SimpleNamespace(
        get_keyboard=lambda: 1
    )
    fake_omni.appwindow = fake_appwindow
    fake_devices = types.ModuleType("isaaclab.devices")
    fake_devices.DeviceBase = DeviceBase
    fake_devices.DeviceCfg = DeviceCfg

    monkeypatch.setitem(sys.modules, "carb", fake_carb)
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.appwindow", fake_appwindow)
    monkeypatch.setitem(sys.modules, "isaaclab.devices", fake_devices)

    spec = importlib.util.spec_from_file_location("keyboard_teleop_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._KEY_PRESS = key_press
    module._KEY_RELEASE = key_release
    return module


def _event(module, name: str, event_type):
    return types.SimpleNamespace(
        type=event_type,
        input=types.SimpleNamespace(name=name),
    )


def test_repeated_press_counts_once_and_release_cancels(keyboard_module):
    cfg = keyboard_module.KeyboardLocoManipTeleopCfg(v_x_sensitivity=0.3)
    teleop = keyboard_module.KeyboardLocoManipTeleop(cfg)

    press = _event(keyboard_module, "UP", keyboard_module._KEY_PRESS)
    release = _event(keyboard_module, "UP", keyboard_module._KEY_RELEASE)
    teleop._on_keyboard_event(press)
    teleop._on_keyboard_event(press)
    teleop._on_keyboard_event(press)

    assert teleop.advance()[0] == pytest.approx(0.3)

    teleop._on_keyboard_event(release)
    assert teleop.advance()[0] == 0.0


def test_release_after_reset_does_not_create_opposite_command(keyboard_module):
    cfg = keyboard_module.KeyboardLocoManipTeleopCfg(v_x_sensitivity=0.3)
    teleop = keyboard_module.KeyboardLocoManipTeleop(cfg)

    teleop._on_keyboard_event(
        _event(keyboard_module, "UP", keyboard_module._KEY_PRESS)
    )
    teleop.reset()
    teleop._on_keyboard_event(
        _event(keyboard_module, "UP", keyboard_module._KEY_RELEASE)
    )

    assert teleop.advance()[0] == 0.0


def test_repeated_press_runs_callback_once(keyboard_module):
    cfg = keyboard_module.KeyboardLocoManipTeleopCfg()
    teleop = keyboard_module.KeyboardLocoManipTeleop(cfg)
    calls = []
    teleop.add_callback("L", lambda: calls.append(1))
    press = _event(keyboard_module, "L", keyboard_module._KEY_PRESS)

    teleop._on_keyboard_event(press)
    teleop._on_keyboard_event(press)

    assert calls == [1]


def test_string_input_event_is_supported(keyboard_module):
    cfg = keyboard_module.KeyboardLocoManipTeleopCfg(v_x_sensitivity=0.3)
    teleop = keyboard_module.KeyboardLocoManipTeleop(cfg)

    teleop._on_keyboard_event(
        types.SimpleNamespace(type=keyboard_module._KEY_PRESS, input="UP")
    )

    assert teleop.advance()[0] == pytest.approx(0.3)

    teleop._on_keyboard_event(
        types.SimpleNamespace(type=keyboard_module._KEY_RELEASE, input="UP")
    )
    assert teleop.advance()[0] == 0.0
