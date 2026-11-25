from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "low_speed_filter.py"
)


@pytest.fixture(scope="module")
def filter_module():
    spec = importlib.util.spec_from_file_location("low_speed_filter_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg(module, **overrides):
    values = {
        "command_name": None,
        "command_threshold": 0.1,
        "gate_on_all_zero": True,
        "gate_on_zero_vx": True,
        "low_speed_threshold": 0.05,
        "cutoff_hz": 0.0,
    }
    values.update(overrides)
    return module.LowSpeedFilterCfg(**values)


def _command(vx: float, vy: float, wz: float) -> torch.Tensor:
    return torch.tensor([[vx, vy, wz]], dtype=torch.float32)


def test_all_zero_command_is_standstill(filter_module):
    cfg = _cfg(filter_module)
    assert bool(filter_module.standstill_command_mask(_command(0.0, 0.0, 0.0), cfg)[0])


def test_zero_vx_with_lateral_and_yaw_is_standstill(filter_module):
    cfg = _cfg(filter_module, gate_on_all_zero=False, gate_on_zero_vx=True)
    assert bool(filter_module.standstill_command_mask(_command(0.0, 0.5, 0.5), cfg)[0])


def test_zero_vx_gate_can_be_disabled(filter_module):
    with pytest.raises(ValueError):
        filter_module.LowSpeedFilterCfg(
            command_name=None,
            gate_on_all_zero=False,
            gate_on_zero_vx=False,
        )
    cfg = _cfg(filter_module, gate_on_all_zero=True, gate_on_zero_vx=False)
    assert not bool(filter_module.standstill_command_mask(_command(0.0, 0.5, 0.5), cfg)[0])


def test_nonzero_vx_is_not_standstill(filter_module):
    cfg = _cfg(filter_module)
    assert not bool(filter_module.standstill_command_mask(_command(0.5, 0.0, 0.0), cfg)[0])


def test_wide_teleop_command_uses_first_three_columns(filter_module):
    cfg = _cfg(filter_module)
    command = torch.tensor(
        [[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert bool(filter_module.standstill_command_mask(command, cfg)[0])


def test_deadband_only_applies_while_gated(filter_module):
    cfg = _cfg(filter_module, low_speed_threshold=0.05)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=2, step_dt=0.02, device="cpu")
    actions = torch.tensor([[0.01, 0.5]], dtype=torch.float32)

    gated = flt(actions, command=_command(0.0, 0.0, 0.0))
    torch.testing.assert_close(gated, torch.tensor([[0.0, 0.5]]))

    passthrough = flt(actions, command=_command(0.5, 0.0, 0.0))
    torch.testing.assert_close(passthrough, actions)


def test_deadband_also_filters_zero_vx_case(filter_module):
    cfg = _cfg(filter_module, gate_on_all_zero=False, low_speed_threshold=0.05)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=2, step_dt=0.02, device="cpu")
    actions = torch.tensor([[0.01, 0.2]], dtype=torch.float32)
    out = flt(actions, command=_command(0.0, 0.4, 0.4))
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.2]]))


def test_all_zero_uses_stronger_deadband_than_zero_vx_only(filter_module):
    cfg = _cfg(filter_module, low_speed_threshold=0.05, all_zero_low_speed_threshold=0.5)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=2, step_dt=0.02, device="cpu")
    actions = torch.tensor([[0.3, 0.3]], dtype=torch.float32)

    out = flt(actions, command=_command(0.0, 0.0, 0.0))
    torch.testing.assert_close(out, torch.zeros_like(actions))
    assert bool(flt.last_all_zero_gate[0])

    out = flt(actions, command=_command(0.0, 0.5, 0.5))
    torch.testing.assert_close(out, actions)
    assert not bool(flt.last_all_zero_gate[0])


def test_all_zero_cutoff_override(filter_module):
    cfg = _cfg(
        filter_module,
        low_speed_threshold=0.0,
        cutoff_hz=0.0,
        all_zero_cutoff_hz=1.0,
    )
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=1, step_dt=0.02, device="cpu")
    alpha = float(torch.exp(torch.tensor(-2.0 * torch.pi * 1.0 * 0.02)))

    out = flt(torch.tensor([[1.0]]), command=_command(0.0, 0.0, 0.0))
    torch.testing.assert_close(out, torch.tensor([[1.0 - alpha]]), rtol=1e-5, atol=1e-7)

    out = flt(torch.tensor([[1.0]]), command=_command(0.0, 0.5, 0.5))
    torch.testing.assert_close(out, torch.tensor([[1.0]]))


def test_low_pass_uses_configured_cutoff(filter_module):
    cfg = _cfg(filter_module, low_speed_threshold=0.0, cutoff_hz=1.0)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=1, step_dt=0.02, device="cpu")
    alpha = float(torch.exp(torch.tensor(-2.0 * torch.pi * 1.0 * 0.02)))

    out = flt(torch.tensor([[1.0]]), command=_command(0.0, 0.0, 0.0))
    torch.testing.assert_close(out, torch.tensor([[1.0 - alpha]]), rtol=1e-5, atol=1e-7)

    out = flt(torch.zeros(1, 1), command=_command(0.0, 0.0, 0.0))
    torch.testing.assert_close(out, torch.tensor([[alpha * (1.0 - alpha)]]), rtol=1e-5, atol=1e-7)


def test_reset_clears_filter_state(filter_module):
    cfg = _cfg(filter_module, low_speed_threshold=0.0, cutoff_hz=1.0)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=1, step_dt=0.02, device="cpu")
    command = _command(0.0, 0.0, 0.0)

    first = flt(torch.tensor([[1.0]]), command=command)
    flt.reset()
    again = flt(torch.tensor([[1.0]]), command=command)
    torch.testing.assert_close(first, again)


def test_gate_entry_drops_previous_motion_state(filter_module):
    """Releasing the command must not coast from the last moving action."""
    moving = _command(0.5, 0.0, 0.0)
    still = _command(0.0, 0.0, 0.0)
    actions = torch.tensor([[0.4]], dtype=torch.float32)
    alpha = float(torch.exp(torch.tensor(-2.0 * torch.pi * 1.0 * 0.02)))

    cfg = _cfg(filter_module, low_speed_threshold=0.5, cutoff_hz=1.0)
    flt = filter_module.LowSpeedFilter(cfg, num_envs=1, action_dim=1, step_dt=0.02, device="cpu")
    torch.testing.assert_close(flt(actions, command=moving), actions)
    torch.testing.assert_close(flt(actions, command=still), torch.zeros_like(actions))

    cfg_hold = _cfg(
        filter_module,
        low_speed_threshold=0.5,
        cutoff_hz=1.0,
        reset_state_on_gate_enter=False,
    )
    flt_hold = filter_module.LowSpeedFilter(cfg_hold, num_envs=1, action_dim=1, step_dt=0.02, device="cpu")
    torch.testing.assert_close(flt_hold(actions, command=moving), actions)
    torch.testing.assert_close(
        flt_hold(actions, command=still), torch.tensor([[alpha * 0.4]], dtype=torch.float32), rtol=1e-5, atol=1e-7
    )
