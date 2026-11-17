from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


DATA_DIR = Path(__file__).resolve().parents[2] / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from collection.termination import (  # noqa: E402
    compute_fast_termination_status,
    validate_success_termination_term,
)

OUTCOMES_PATH = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "rl_training"
    / "rl_training"
    / "datasets"
    / "outcomes.py"
)
SPEC = importlib.util.spec_from_file_location("collection_outcomes_test", OUTCOMES_PATH)
assert SPEC is not None and SPEC.loader is not None
OUTCOMES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OUTCOMES)


class _FakeTerminationManager:
    def __init__(self) -> None:
        self.active_terms = ["time_out", "place_success"]
        self._cfgs = {
            "time_out": SimpleNamespace(time_out=True),
            "place_success": SimpleNamespace(time_out=False),
        }
        self.compute_calls = 0
        self.terminated = torch.tensor([False, True])
        self.time_outs = torch.tensor([True, False])

    def get_term_cfg(self, name: str):
        return self._cfgs[name]

    def compute(self):
        self.compute_calls += 1
        return self.terminated | self.time_outs


def test_compute_fast_termination_status_evaluates_terms_once() -> None:
    manager = _FakeTerminationManager()

    reset, terminated, truncated = compute_fast_termination_status(manager)

    assert manager.compute_calls == 1
    assert reset.tolist() == [True, True]
    assert terminated.tolist() == [False, True]
    assert truncated.tolist() == [True, False]


def test_success_term_must_exist_and_not_be_timeout() -> None:
    manager = _FakeTerminationManager()

    validate_success_termination_term(manager, "place_success")
    with pytest.raises(ValueError, match="not an active termination term"):
        validate_success_termination_term(manager, "missing")
    with pytest.raises(ValueError, match="must not be a timeout"):
        validate_success_termination_term(manager, "time_out")


def test_auto_success_survives_post_step_frame_consumption() -> None:
    last_outcome = {
        "termination_flags": torch.tensor([True]),
        "success": torch.tensor(True),
    }

    success, label, auto_success, fired = OUTCOMES.resolve_episode_outcome(
        last_outcome,
        manual_outcome=None,
        termination_names=("place_success",),
    )

    assert success
    assert auto_success
    assert label == "place_success"
    assert fired == ["place_success"]


def test_manual_outcome_overrides_automatic_success() -> None:
    success, label, auto_success, _ = OUTCOMES.resolve_episode_outcome(
        {"termination_flags": torch.tensor([True]), "success": torch.tensor(True)},
        manual_outcome=False,
        termination_names=("place_success",),
    )

    assert not success
    assert auto_success
    assert label == "manual_failure"
