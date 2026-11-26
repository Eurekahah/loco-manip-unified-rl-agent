"""Shared low-speed filter for the M20 wheel/leg action path.

The filter is intentionally free of IsaacLab imports so it can be exercised by
plain unit tests and reused by the teleoperation / replay action terms.

Two pieces are provided:

* :func:`standstill_command_mask` -- command gating.  It covers both cases used
  by the M20 tasks: an all-zero command ``(0, 0, 0)`` and ``vx == 0`` while
  ``vy`` / ``wz`` may still be non-zero.
* :class:`LowSpeedFilter` -- a stateful per-environment filter.  While the gate
  is active it applies a small-action dead band and an optional first-order
  low-pass; outside the gate actions pass through unchanged.

The filter operates on *normalised policy actions* (the raw action vector),
before an action term applies its scale/offset.  That keeps training and the
test probes numerically identical without duplicating the per-joint gains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "LowSpeedFilter",
    "LowSpeedFilterCfg",
    "M20_STANDSTILL_PRESET",
    "StandstillFilterPreset",
    "leg_low_speed_filter_cfg",
    "standstill_command_mask",
    "wheel_low_speed_filter_cfg",
]


@dataclass
class LowSpeedFilterCfg:
    """Configuration for :class:`LowSpeedFilter`.

    Args:
        enabled: Master switch.
        command_name: Command term used to decide whether the robot is being
            asked to stand still.  ``None`` disables command-based gating; an
            external gate may still be supplied by a high-level action term
            that owns the velocity command itself.
        command_threshold: A command component below this magnitude counts as
            zero (m/s for ``vx`` / ``vy``, rad/s for ``wz``).
        gate_on_all_zero: Treat ``(vx, vy, wz) == (0, 0, 0)`` as standstill.
        gate_on_zero_vx: Treat ``vx == 0`` as standstill even when ``vy`` or
            ``wz`` are non-zero.  This is the "only vx is zero" case used by
            the keyboard / teleoperation interface.
        low_speed_threshold: Actions with ``|a|`` below this value are snapped
            to zero while the gate is active.  Units are normalised policy
            actions; ``<= 0`` disables the dead band.
        cutoff_hz: First-order low-pass cutoff in Hz while the gate is active.
            ``<= 0`` disables smoothing (dead band only).
        all_zero_low_speed_threshold: Dead band used only for the all-zero
            command ``(vx, vy, wz) == (0, 0, 0)``.  Falls back to
            ``low_speed_threshold`` when ``None``.  The wheel policy has a
            persistent DC bias at zero command, so this can be much larger
            than the "only vx == 0" threshold without affecting lateral/yaw.
        all_zero_cutoff_hz: Low-pass cutoff used only for the all-zero
            command.  Falls back to ``cutoff_hz`` when ``None``.
        reset_state_on_gate_enter: Clear the per-environment low-pass state on
            the step the standstill gate turns on.  Without this the filter
            would coast down from the last moving command (several tenths of a
            second of residual wheel target) after the command is released.
        all_zero_hard_zero: Force the filtered action to exactly zero whenever
            the all-zero command gate is active (still passes through when only
            ``vx`` is zero).  Use this on wheel terms when an undertrained
            policy still emits a large standstill bias (e.g. > 0.6 normalised
            action) that a dead band cannot remove.  It gives an exact stop at
            the cost of dropping that policy's standstill wheel torque.
    """

    enabled: bool = True
    command_name: str | None = "base_velocity"
    command_threshold: float = 0.1
    gate_on_all_zero: bool = True
    gate_on_zero_vx: bool = True
    low_speed_threshold: float = 0.0
    cutoff_hz: float = 0.0
    all_zero_low_speed_threshold: float | None = None
    all_zero_cutoff_hz: float | None = None
    reset_state_on_gate_enter: bool = True
    all_zero_hard_zero: bool = False

    def __post_init__(self) -> None:
        if self.command_threshold < 0.0:
            raise ValueError(f"command_threshold must be >= 0, got {self.command_threshold}")
        if self.low_speed_threshold < 0.0:
            raise ValueError(f"low_speed_threshold must be >= 0, got {self.low_speed_threshold}")
        if self.cutoff_hz < 0.0:
            raise ValueError(f"cutoff_hz must be >= 0, got {self.cutoff_hz}")
        if self.all_zero_low_speed_threshold is not None and self.all_zero_low_speed_threshold < 0.0:
            raise ValueError(
                "all_zero_low_speed_threshold must be >= 0, "
                f"got {self.all_zero_low_speed_threshold}"
            )
        if self.all_zero_cutoff_hz is not None and self.all_zero_cutoff_hz < 0.0:
            raise ValueError(f"all_zero_cutoff_hz must be >= 0, got {self.all_zero_cutoff_hz}")
        if not self.gate_on_all_zero and not self.gate_on_zero_vx:
            raise ValueError("At least one of gate_on_all_zero / gate_on_zero_vx must be True.")


@dataclass(frozen=True)
class StandstillFilterPreset:
    """Verified M20 standstill numbers shared by training, PLAY and teleop.

    The preset only covers the *soft* filter.  ``all_zero_hard_zero`` is
    deliberately **not** part of it: hard zeroing the wheel action is a
    deployment choice (it removes the policy's standstill wheel bias), and
    hard zeroing the leg action hurts training (26% vs 7% fall rate at 5k),
    so it must stay an explicit opt-in.
    """

    command_threshold: float
    leg_deadband: float
    leg_cutoff_hz: float
    wheel_deadband: float
    wheel_cutoff_hz: float
    wheel_all_zero_deadband: float


M20_STANDSTILL_PRESET = StandstillFilterPreset(
    command_threshold=0.1,
    leg_deadband=0.02,
    leg_cutoff_hz=8.0,
    wheel_deadband=0.04,
    wheel_cutoff_hz=2.0,
    wheel_all_zero_deadband=0.6,
)


def leg_low_speed_filter_cfg(
    preset: StandstillFilterPreset = M20_STANDSTILL_PRESET,
    *,
    command_name: str | None = "base_velocity",
) -> LowSpeedFilterCfg:
    """Leg filter config for the M20 standard: tiny dead band, 8 Hz low-pass."""
    return LowSpeedFilterCfg(
        command_name=command_name,
        command_threshold=preset.command_threshold,
        low_speed_threshold=preset.leg_deadband,
        cutoff_hz=preset.leg_cutoff_hz,
    )


def wheel_low_speed_filter_cfg(
    preset: StandstillFilterPreset = M20_STANDSTILL_PRESET,
    *,
    command_name: str | None = "base_velocity",
) -> LowSpeedFilterCfg:
    """Wheel filter config for the M20 standard.

    Only ``vx == 0`` uses the small dead band / 2 Hz low-pass so lateral and
    yaw commands stay responsive; the all-zero command gets the wide 0.6 dead
    band as the soft-filter fallback.  ``all_zero_hard_zero`` stays False here
    -- deployment enables it explicitly.
    """
    return LowSpeedFilterCfg(
        command_name=command_name,
        command_threshold=preset.command_threshold,
        low_speed_threshold=preset.wheel_deadband,
        cutoff_hz=preset.wheel_cutoff_hz,
        all_zero_low_speed_threshold=preset.wheel_all_zero_deadband,
    )


def standstill_command_masks(
    command: torch.Tensor,
    cfg: LowSpeedFilterCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(all_zero_mask, zero_vx_mask)`` for a command tensor.

    The command tensor may be wider than three columns (e.g. the unified
    ``[vx, vy, wz, ee..., body...]`` teleoperation command); only the first
    three columns are read.

    The two masks are kept separate so the filter can use a stronger dead band
    for a fully stationary command while keeping lateral / yaw responsive when
    only ``vx`` is zero.
    """
    if command.ndim != 2 or command.shape[1] < 3:
        raise ValueError(f"command must have shape (N, >=3), got {tuple(command.shape)}")

    threshold = float(cfg.command_threshold)
    vx = command[:, 0]
    planar_norm = torch.linalg.vector_norm(command[:, :2], dim=-1)
    wz = command[:, 2]

    all_zero = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    if cfg.gate_on_all_zero:
        all_zero = (planar_norm <= threshold) & (wz.abs() <= threshold)
    zero_vx = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    if cfg.gate_on_zero_vx:
        zero_vx = vx.abs() <= threshold
    return all_zero, zero_vx


def standstill_command_mask(
    command: torch.Tensor,
    cfg: LowSpeedFilterCfg,
) -> torch.Tensor:
    """Return the union ``(N,)`` standstill mask (all-zero OR zero-``vx``)."""
    all_zero, zero_vx = standstill_command_masks(command, cfg)
    return all_zero | zero_vx


class LowSpeedFilter:
    """Stateful low-speed filter with an explicit standstill gate.

    The filter is applied to normalised policy actions.  It keeps one scalar
    state per action dimension and per environment so the low-pass behaves
    like a causal first-order filter during deployment as well.
    """

    def __init__(
        self,
        cfg: LowSpeedFilterCfg,
        num_envs: int,
        action_dim: int,
        step_dt: float,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or action_dim <= 0:
            raise ValueError(f"num_envs/action_dim must be positive, got {num_envs}/{action_dim}")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.step_dt = float(step_dt)
        self.device = torch.device(device)
        self._state = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._alpha = self._alpha_for(self.cfg.cutoff_hz)
        self._all_zero_alpha = self._alpha_for(
            self.cfg.cutoff_hz if self.cfg.all_zero_cutoff_hz is None else self.cfg.all_zero_cutoff_hz
        )
        self.last_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_all_zero_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _alpha_for(self, cutoff_hz: float) -> float:
        """Coefficient for ``state <- alpha * state + (1 - alpha) * target``."""
        return math.exp(-2.0 * math.pi * cutoff_hz * self.step_dt) if cutoff_hz > 0.0 else 0.0

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Clear the filter state (all envs when ``env_ids`` is ``None``)."""
        if env_ids is None:
            self._state.zero_()
            self.last_gate.zero_()
            self.last_all_zero_gate.zero_()
            self._prev_gate.zero_()
        else:
            self._state[env_ids] = 0.0
            self.last_gate[env_ids] = False
            self.last_all_zero_gate[env_ids] = False
            self._prev_gate[env_ids] = False

    def __call__(
        self,
        actions: torch.Tensor,
        *,
        command: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Filter ``actions`` and return a new tensor (input is not mutated).

        Pass either the raw ``command`` (preferred: the all-zero gate can use
        stronger thresholds than the zero-``vx`` gate) or an explicit ``gate``
        mask (legacy path: base thresholds for every gated environment).  When
        both are ``None`` the filter is a no-op.
        """
        if not self.cfg.enabled:
            self.last_gate.zero_()
            self.last_all_zero_gate.zero_()
            self._prev_gate.zero_()
            return actions
        if actions.shape != self._state.shape:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} does not match filter state "
                f"{tuple(self._state.shape)}"
            )

        num_envs = actions.shape[0]
        if command is not None:
            all_zero_gate, zero_vx_gate = standstill_command_masks(command, self.cfg)
            gate = all_zero_gate | zero_vx_gate
        else:
            all_zero_gate = None
            if gate is None:
                self.last_gate.zero_()
                self.last_all_zero_gate.zero_()
                self._prev_gate.zero_()
                return actions
            gate = gate.to(device=actions.device, dtype=torch.bool).reshape(-1)
        if gate.numel() != actions.shape[0]:
            raise ValueError(
                f"gate must have {num_envs} entries, got {gate.numel()}"
            )
        gate = gate.to(device=actions.device, dtype=torch.bool)
        self.last_gate = gate

        if all_zero_gate is None:
            all_zero_gate = torch.zeros(num_envs, dtype=torch.bool, device=actions.device)
            threshold = torch.full(
                (num_envs,), float(self.cfg.low_speed_threshold), device=actions.device, dtype=actions.dtype
            )
            alpha = torch.full((num_envs,), self._alpha, device=actions.device, dtype=actions.dtype)
        else:
            all_zero_gate = all_zero_gate.to(device=actions.device, dtype=torch.bool).reshape(-1)
            if all_zero_gate.numel() != num_envs:
                raise ValueError(
                    f"all_zero_gate must have {num_envs} entries, got {all_zero_gate.numel()}"
                )
            all_zero_threshold = (
                float(self.cfg.low_speed_threshold)
                if self.cfg.all_zero_low_speed_threshold is None
                else float(self.cfg.all_zero_low_speed_threshold)
            )
            threshold = torch.where(
                all_zero_gate,
                torch.full((num_envs,), all_zero_threshold, device=actions.device, dtype=actions.dtype),
                torch.full(
                    (num_envs,), float(self.cfg.low_speed_threshold), device=actions.device, dtype=actions.dtype
                ),
            )
            alpha = torch.where(
                all_zero_gate,
                torch.full((num_envs,), self._all_zero_alpha, device=actions.device, dtype=actions.dtype),
                torch.full((num_envs,), self._alpha, device=actions.device, dtype=actions.dtype),
            )
        self.last_all_zero_gate = all_zero_gate

        # Entering standstill: drop the low-pass memory of the previous moving
        # action so the wheels brake on the same tick instead of coasting.
        if self.cfg.reset_state_on_gate_enter:
            rising_gate = gate & ~self._prev_gate
            if bool(rising_gate.any()):
                self._state = torch.where(
                    rising_gate.unsqueeze(-1), torch.zeros_like(self._state), self._state
                )
        self._prev_gate = gate.clone()

        # Dead band only applies where the standstill gate is active.
        if bool((threshold > 0.0).any()):
            dead = actions.abs() < threshold.unsqueeze(-1)
            target = torch.where((gate.unsqueeze(-1) & dead), torch.zeros_like(actions), actions)
        else:
            target = actions
        if self.cfg.all_zero_hard_zero:
            target = torch.where(all_zero_gate.unsqueeze(-1), torch.zeros_like(target), target)

        if bool((alpha > 0.0).any()):
            smoothed = alpha.unsqueeze(-1) * self._state + (1.0 - alpha.unsqueeze(-1)) * target
            gated = gate.unsqueeze(-1)
            self._state = torch.where(gated, smoothed, target)
            return torch.where(gated, smoothed, actions)

        gated = gate.unsqueeze(-1)
        self._state = torch.where(gated, target, actions)
        return torch.where(gated, target, actions)
