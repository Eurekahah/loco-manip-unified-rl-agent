from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.teleoperation.fast_teleop import FastTeleopStepper


class _FakeAction:
    def to(self, device):
        return self


class _FakeActionManager:
    def __init__(self) -> None:
        self.processed = 0
        self.applied = 0

    def process_action(self, action) -> None:
        self.processed += 1

    def apply_action(self) -> None:
        self.applied += 1


class _FakeRobot:
    def __init__(self) -> None:
        self.written = 0
        self.updated = 0

    def write_data_to_sim(self) -> None:
        self.written += 1

    def update(self, dt: float) -> None:
        self.updated += 1


class _FakeScene:
    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot
        self.written = 0
        self.updated: list[float] = []

    def __getitem__(self, key: str):
        assert key == "robot"
        return self.robot

    def write_data_to_sim(self) -> None:
        self.written += 1

    def update(self, dt: float) -> None:
        self.updated.append(dt)


class _FakeSim:
    def __init__(self) -> None:
        self.stepped = 0
        self.rendered = 0

    def has_gui(self) -> bool:
        return False

    def has_rtx_sensors(self) -> bool:
        return False

    def step(self, render: bool = False) -> None:
        assert not render
        self.stepped += 1

    def render(self) -> None:
        self.rendered += 1


class _FakeEnv:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(decimation=4)
        self.physics_dt = 0.005
        self.step_dt = 0.02
        self.device = "cpu"
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.action_manager = _FakeActionManager()
        self.robot = _FakeRobot()
        self.scene = _FakeScene(self.robot)
        self.sim = _FakeSim()


def test_fast_step_advances_recording_counters_once_per_control_step() -> None:
    env = _FakeEnv()
    stepper = FastTeleopStepper(
        env,
        render_hz=15.0,
        robot_hot_loop=True,
        robot_hot_period_s=0.01,
    )

    stepper.step(_FakeAction())

    assert env._sim_step_counter == 4
    assert env.common_step_counter == 1
    assert env.episode_length_buf.tolist() == [1]
    assert env.action_manager.processed == 1
    assert env.action_manager.applied == 4
    assert env.scene.written == 1
    assert env.scene.updated == [env.step_dt]
    assert env.robot.written == 1
    assert env.robot.updated == 4
    assert env.sim.stepped == 4
    assert env.sim.rendered == 0


def test_robot_only_write_skips_scene_entity_write() -> None:
    env = _FakeEnv()
    stepper = FastTeleopStepper(
        env,
        robot_hot_loop=True,
        robot_hot_period_s=0.01,
        robot_only_write=True,
    )

    stepper.step(_FakeAction())

    assert env.scene.written == 0
    assert env.robot.written == 2
