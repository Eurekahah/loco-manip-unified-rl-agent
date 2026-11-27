"""Opt-in fast teleoperation stepping without changing training environments."""

from __future__ import annotations

import functools
import time


class DetailedStepProfiler:
    """Temporarily instrument the hot methods used by ManagerBasedEnv.step()."""

    def __init__(self, env, profiler, *, action_term_name: str = "pre_trained_pick_action") -> None:
        self.env = env
        self.profiler = profiler
        self.action_term_name = action_term_name
        self._originals: list[tuple[object, str, object]] = []
        self._action_term = None

    def install(self) -> None:
        self._wrap(self.env.action_manager, "process_action", "action_process_ms")
        self._wrap(self.env.action_manager, "apply_action", "action_apply_ms")
        self._wrap(self.env.scene, "write_data_to_sim", "scene_write_ms")
        self._wrap(self.env.scene, "update", "scene_update_ms")
        self._wrap(self.env.sim, "step", "sim_step_ms")
        self._wrap(self.env.sim, "render", "render_ms")
        self._wrap(self.env.observation_manager, "compute", "obs_manager_ms")
        self._wrap(self.env.recorder_manager, "record_pre_step", "recorder_pre_ms")
        self._wrap(self.env.recorder_manager, "record_post_step", "recorder_post_ms")
        self._wrap(self.env.event_manager, "apply", "event_manager_ms")
        try:
            self._action_term = self.env.action_manager.get_term(self.action_term_name)
            self._action_term.set_profile_hook(self.profiler.record)
        except (KeyError, AttributeError):
            self._action_term = None

    def uninstall(self) -> None:
        for target, name, original in reversed(self._originals):
            setattr(target, name, original)
        self._originals.clear()
        if self._action_term is not None and hasattr(self._action_term, "set_profile_hook"):
            self._action_term.set_profile_hook(None)

    def _wrap(self, target, name: str, metric: str) -> None:
        original = getattr(target, name, None)
        if original is None:
            return

        @functools.wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.profiler.record(metric, (time.perf_counter() - started) * 1000.0)

        self._originals.append((target, name, original))
        setattr(target, name, measured)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()
        return False


class FastTeleopStepper:
    """Minimal control loop using existing action terms and scene stepping."""

    def __init__(
        self,
        env,
        *,
        render_hz: float = 15.0,
        write_targets_each_tick: bool = False,
        robot_hot_loop: bool = False,
        robot_hot_period_s: float = 0.005,
        robot_only_write: bool = False,
        profiler=None,
    ) -> None:
        if render_hz <= 0.0:
            raise ValueError("render_hz must be positive")
        self.env = env
        self.render_hz = float(render_hz)
        self.write_targets_each_tick = bool(write_targets_each_tick)
        self.robot_hot_loop = bool(robot_hot_loop)
        self.robot_only_write = bool(robot_only_write)
        if robot_hot_period_s <= 0.0:
            raise ValueError("robot_hot_period_s must be positive")
        self.profiler = profiler
        self._last_render = 0.0
        self._decimation = int(env.cfg.decimation)
        self._physics_dt = float(env.physics_dt)
        self._robot = env.scene["robot"]
        self._robot_hot_decimation = max(
            1,
            int(round(float(robot_hot_period_s) / self._physics_dt)),
        )
        # Explicit actuator models (e.g. DelayedPDActuator) compute the joint
        # effort in Python and push it with ``set_dof_actuation_forces`` on
        # every ``write_data_to_sim``.  PhysX does not persist that effort
        # across physics steps, so skipping a write silently drops the motor
        # torque and makes the robot shake.  Never decimate writes in that case.
        self._requires_per_step_write = any(
            not actuator.is_implicit_model for actuator in self._robot.actuators.values()
        )
        if self._requires_per_step_write and self._robot_hot_decimation > 1:
            print(
                "[FastTeleop] explicit actuators detected; writing robot targets "
                f"every physics step (hot period {robot_hot_period_s:g}s -> "
                f"{self._physics_dt:g}s)."
            )
            self._robot_hot_decimation = 1

    def _measure(self, name: str, started: float) -> None:
        if self.profiler is not None:
            self.profiler.record(name, (time.perf_counter() - started) * 1000.0)

    def step(self, action) -> None:
        self.process_action(action)
        self.advance()

    def process_action(self, action) -> None:
        started = time.perf_counter()
        self.env.action_manager.process_action(action.to(self.env.device))
        self._measure("fast_action_process_ms", started)

    def advance(self) -> None:
        now = time.perf_counter()
        render_due = now - self._last_render >= 1.0 / self.render_hz
        render_enabled = self.env.sim.has_gui() or self.env.sim.has_rtx_sensors()
        should_render = render_due and render_enabled

        for index in range(self._decimation):
            self.env._sim_step_counter += 1

            # Always advance action terms so the low-level policy/history keep
            # their original 50 Hz cadence. Target writes may still be decimated.
            started = time.perf_counter()
            self.env.action_manager.apply_action()
            self._measure("fast_action_apply_ms", started)

            if self.write_targets_each_tick or index == 0:
                started = time.perf_counter()
                if self.robot_only_write:
                    self._robot.write_data_to_sim()
                else:
                    self.env.scene.write_data_to_sim()
                self._measure(
                    "fast_robot_only_write_ms"
                    if self.robot_only_write
                    else "fast_scene_write_ms",
                    started,
                )
            elif self.robot_hot_loop and index % self._robot_hot_decimation == 0:
                started = time.perf_counter()
                self._write_robot_hot()
                self._measure("fast_robot_write_ms", started)
            elif self._requires_per_step_write:
                # ``robot_hot_loop`` disabled but explicit actuators still need
                # their targets/forces written on every physics step.
                started = time.perf_counter()
                self._write_robot_hot()
                self._measure("fast_robot_write_ms", started)

            started = time.perf_counter()
            self.env.sim.step(render=False)
            self._measure("fast_sim_step_ms", started)

            if should_render and index == self._decimation - 1:
                started = time.perf_counter()
                self.env.sim.render()
                self._measure("fast_render_ms", started)

            if self.write_targets_each_tick:
                started = time.perf_counter()
                self.env.scene.update(dt=self._physics_dt)
                self._measure("fast_scene_update_ms", started)
            elif self.robot_hot_loop or self._requires_per_step_write:
                started = time.perf_counter()
                self._robot.update(dt=self._physics_dt)
                self._measure("fast_robot_update_ms", started)

        if not self.write_targets_each_tick:
            started = time.perf_counter()
            self.env.scene.update(dt=self.env.step_dt)
            self._measure("fast_scene_update_ms", started)

        if should_render:
            self._last_render = time.perf_counter()

        # Match the counters maintained by ManagerBasedRLEnv.step().  The
        # recorder uses common_step_counter to derive sim_time, and action
        # terms use episode_length_buf to detect episode boundaries.
        if hasattr(self.env, "episode_length_buf") and self.env.episode_length_buf is not None:
            self.env.episode_length_buf += 1
        if hasattr(self.env, "common_step_counter"):
            self.env.common_step_counter += 1

    def _write_robot_hot(self) -> None:
        self._robot.write_data_to_sim()
