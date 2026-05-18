# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) + body pose control."""

from __future__ import annotations

import numpy as np
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import carb
import omni

from isaaclab.devices import DeviceBase, DeviceCfg


class Se2KeyboardExtended(DeviceBase):
    r"""A keyboard controller for SE(2) velocity + body pose commands.

    Extends the base SE(2) keyboard controller with 3 additional pose dimensions,
    outputting a 6D command: :math:`(v_x, v_y, \omega_z, \Delta h, \Delta pitch, \Delta roll)`.

    The pose deltas are **incremental** — each frame the key is held down the
    corresponding delta is output; the caller is responsible for accumulation.

    Key bindings:
        ====================== ========================= =========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= =========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        Body height            C                         V
        Body pitch             B                         N
        Body roll              LEFT_BRACKET ([)           RIGHT_BRACKET (])
        Reset pose to zero     M
        Reset all commands     L
        ====================== ========================= =========================

    .. note::
        Keys Z, X are already used for yaw by the parent class, so pose controls
        use C/V/B/N/[/] which have no default IsaacSim viewport bindings.
    """

    def __init__(self, cfg: Se2KeyboardExtendedCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration for the extended keyboard controller.
        """
        # store sensitivities
        self.v_x_sensitivity     = cfg.v_x_sensitivity
        self.v_y_sensitivity     = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self.height_sensitivity  = cfg.height_sensitivity
        self.pitch_sensitivity   = cfg.pitch_sensitivity
        self.roll_sensitivity    = cfg.roll_sensitivity
        self._sim_device         = cfg.sim_device

        # acquire omniverse interfaces  (identical to Se2Keyboard)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input     = carb.input.acquire_input_interface()
        self._keyboard  = self._appwindow.get_keyboard()

        # single unified subscriber — same pattern as Se2Keyboard
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # build key → delta-vector mapping
        self._create_key_bindings()

        # command buffers
        # _base_command : (6,)  [vx, vy, wz, delta_h, delta_pitch, delta_roll]
        #   • vx / vy / wz      → toggled by press/release (same as Se2Keyboard)
        #   • delta_h/pitch/roll → toggled by press/release; caller accumulates
        self._base_command = np.zeros(6, dtype=np.float32)

        # additional user callbacks  (same API as Se2Keyboard.add_callback)
        self._additional_callbacks: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        msg  = f"Keyboard Controller for SE(2) + Pose: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands : L\n"
        msg += "\tReset pose to zero : M\n"
        msg += "\t--- Chassis velocity ---\n"
        msg += "\tMove forward   (vx+) : Numpad 8 / Arrow Up\n"
        msg += "\tMove backward  (vx-) : Numpad 2 / Arrow Down\n"
        msg += "\tMove right     (vy+) : Numpad 4 / Arrow Right\n"
        msg += "\tMove left      (vy-) : Numpad 6 / Arrow Left\n"
        msg += "\tYaw positively (wz+) : Numpad 7 / Z\n"
        msg += "\tYaw negatively (wz-) : Numpad 9 / X\n"
        msg += "\t--- Body pose (incremental delta per frame) ---\n"
        msg += "\tHeight up   (dh+) : C\n"
        msg += "\tHeight down (dh-) : V\n"
        msg += "\tPitch up    (dp+) : B\n"
        msg += "\tPitch down  (dp-) : N\n"
        msg += "\tRoll left   (dr+) : [\n"
        msg += "\tRoll right  (dr-) : ]"
        return msg

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all command buffers to zero."""
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Bind an extra function to a keyboard key (press only).

        Args:
            key: Carb key name string (e.g. ``"L"``, ``"RESET"``).
            func: Zero-argument callable to invoke on key press.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Return the current 6-D command vector.

        Returns:
            ``np.ndarray`` of shape ``(6,)`` —
            ``[vx, vy, wz, delta_height, delta_pitch, delta_roll]``.
        """
        return self._base_command.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Unified keyboard event handler — mirrors Se2Keyboard._on_keyboard_event exactly.

        KEY_PRESS   → add the mapped delta vector to _base_command
        KEY_RELEASE → subtract it back  (net effect: zero when not held)

        Special keys:
            L → full reset (velocity + pose)
            M → pose-only reset (height / pitch / roll back to 0)
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name == "M":
                # reset only the pose channels (indices 3-5)
                self._base_command[3:] = 0.0
                print("[Se2KeyboardExtended] Pose reset to zero")
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[event.input.name]

            # user-registered callbacks
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command -= self._INPUT_KEY_MAPPING[event.input.name]

        return True  # no error

    def _create_key_bindings(self):
        """Build the key-name → 6D-delta mapping.

        Layout of the 6D vector:
            [0] vx  [1] vy  [2] wz  [3] delta_h  [4] delta_pitch  [5] delta_roll
        """
        vx  = self.v_x_sensitivity
        vy  = self.v_y_sensitivity
        wz  = self.omega_z_sensitivity
        dh  = self.height_sensitivity
        dp  = self.pitch_sensitivity
        dr  = self.roll_sensitivity

        self._INPUT_KEY_MAPPING: dict[str, np.ndarray] = {
            # ---- chassis velocity (indices 0-2) ----
            # forward  (+vx)
            "NUMPAD_8" : np.array([ vx,  0.,  0.,  0.,  0.,  0.], dtype=np.float32),
            "UP"       : np.array([ vx,  0.,  0.,  0.,  0.,  0.], dtype=np.float32),
            # backward (-vx)
            "NUMPAD_2" : np.array([-vx,  0.,  0.,  0.,  0.,  0.], dtype=np.float32),
            "DOWN"     : np.array([-vx,  0.,  0.,  0.,  0.,  0.], dtype=np.float32),
            # right    (+vy)
            "NUMPAD_4" : np.array([ 0.,  vy,  0.,  0.,  0.,  0.], dtype=np.float32),
            "LEFT"     : np.array([ 0.,  vy,  0.,  0.,  0.,  0.], dtype=np.float32),
            # left     (-vy)
            "NUMPAD_6" : np.array([ 0., -vy,  0.,  0.,  0.,  0.], dtype=np.float32),
            "RIGHT"    : np.array([ 0., -vy,  0.,  0.,  0.,  0.], dtype=np.float32),
            # yaw +
            "NUMPAD_7" : np.array([ 0.,  0.,  wz,  0.,  0.,  0.], dtype=np.float32),
            "Z"        : np.array([ 0.,  0.,  wz,  0.,  0.,  0.], dtype=np.float32),
            # yaw -
            "NUMPAD_9" : np.array([ 0.,  0., -wz,  0.,  0.,  0.], dtype=np.float32),
            "X"        : np.array([ 0.,  0., -wz,  0.,  0.,  0.], dtype=np.float32),

            # ---- body pose (indices 3-5) ----
            # height up   (+dh)   → C
            "C"        : np.array([ 0.,  0.,  0.,  dh,  0.,  0.], dtype=np.float32),
            # height down (-dh)   → V
            "V"        : np.array([ 0.,  0.,  0., -dh,  0.,  0.], dtype=np.float32),
            # pitch up    (+dp)   → B
            "B"        : np.array([ 0.,  0.,  0.,  0.,  dp,  0.], dtype=np.float32),
            # pitch down  (-dp)   → N
            "N"        : np.array([ 0.,  0.,  0.,  0., -dp,  0.], dtype=np.float32),
            # roll left   (+dr)   → [
            "LEFT_BRACKET"  : np.array([ 0.,  0.,  0.,  0.,  0.,  dr], dtype=np.float32),
            # roll right  (-dr)   → ]
            "RIGHT_BRACKET" : np.array([ 0.,  0.,  0.,  0.,  0., -dr], dtype=np.float32),
        }


@dataclass
class Se2KeyboardExtendedCfg(DeviceCfg):
    """Configuration for the extended SE(2) + pose keyboard controller."""

    v_x_sensitivity:     float = 0.8
    v_y_sensitivity:     float = 0.4
    omega_z_sensitivity: float = 1.0
    height_sensitivity:  float = 0.05
    pitch_sensitivity:   float = 0.02
    roll_sensitivity:    float = 0.02
    class_type: type[DeviceBase] = Se2KeyboardExtended