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
        profiler=None,
    ) -> None:
        if render_hz <= 0.0:
            raise ValueError("render_hz must be positive")
        self.env = env
        self.render_hz = float(render_hz)
        self.write_targets_each_tick = bool(write_targets_each_tick)
        self.profiler = profiler
        self._last_render = 0.0
        self._decimation = int(env.cfg.decimation)
        self._physics_dt = float(env.physics_dt)

    def _measure(self, name: str, started: float) -> None:
        if self.profiler is not None:
            self.profiler.record(name, (time.perf_counter() - started) * 1000.0)

    def step(self, action) -> None:
        started = time.perf_counter()
        self.env.action_manager.process_action(action.to(self.env.device))
        self._measure("fast_action_process_ms", started)

        now = time.perf_counter()
        render_due = now - self._last_render >= 1.0 / self.render_hz
        render_enabled = self.env.sim.has_gui() or self.env.sim.has_rtx_sensors()
        should_render = render_due and render_enabled

        if not self.write_targets_each_tick:
            started = time.perf_counter()
            self.env.action_manager.apply_action()
            self._measure("fast_action_apply_ms", started)
            started = time.perf_counter()
            self.env.scene.write_data_to_sim()
            self._measure("fast_scene_write_ms", started)

        for index in range(self._decimation):
            self.env._sim_step_counter += 1

            if self.write_targets_each_tick:
                started = time.perf_counter()
                self.env.action_manager.apply_action()
                self._measure("fast_action_apply_ms", started)

                started = time.perf_counter()
                self.env.scene.write_data_to_sim()
                self._measure("fast_scene_write_ms", started)

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

        if not self.write_targets_each_tick:
            started = time.perf_counter()
            self.env.scene.update(dt=self.env.step_dt)
            self._measure("fast_scene_update_ms", started)

        if should_render:
            self._last_render = time.perf_counter()
