# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) + body pose + arm end-effector + gripper control."""

from __future__ import annotations

import numpy as np
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import carb
import omni

from isaaclab.devices import DeviceBase, DeviceCfg


class Se2KeyboardFull(DeviceBase):
    r"""Keyboard controller outputting a 13-D command vector.

    Command layout (matches the ActionTerm's ``action_dim=13``):

    ============  =====  ================================================
    Index         Dim    Description
    ============  =====  ================================================
    ``[0]``       1      ``v_x``       – chassis forward velocity
    ``[1]``       1      ``v_y``       – chassis lateral velocity
    ``[2]``       1      ``ω_z``       – chassis yaw rate
    ``[3:6]``     3      ``Δee_pos``   – EE position increment (x, y, z)
    ``[6:9]``     3      ``Δee_orn``   – EE orientation increment (roll, pitch, yaw)
    ``[9]``       1      ``Δheight``   – body height increment
    ``[10]``      1      ``Δpitch``    – body pitch increment
    ``[11]``      1      ``Δroll``     – body roll increment
    ``[12]``      1      ``gripper``   – gripper state (0 = open, 1 = closed)
    ============  =====  ================================================

    Key bindings
    ------------

    **Chassis velocity** (arrow keys + Z/X):

    ==================  ==================  ===================
    Command             Key (+)             Key (−)
    ==================  ==================  ===================
    Forward  (v_x)      Arrow Up            Arrow Down
    Lateral  (v_y)      Arrow Left          Arrow Right
    Yaw      (ω_z)      Z                   X
    ==================  ==================  ===================

    **Body pose** (incremental, caller accumulates):

    ==================  ==================  ===================
    Command             Key (+)             Key (−)
    ==================  ==================  ===================
    Body height         C                   V
    Body pitch          B                   N
    Body roll           ``[``               ``]``
    ==================  ==================  ===================

    **Arm end-effector position** (numpad, incremental):

    ==================  ==================  ===================
    Command             Key (+)             Key (−)
    ==================  ==================  ===================
    EE Δx               Numpad 8            Numpad 2
    EE Δy               Numpad 4            Numpad 6
    EE Δz               Numpad 7            Numpad 9
    ==================  ==================  ===================

    **Arm end-effector orientation** (numpad, incremental):

    ==================  ==================  ===================
    Command             Key (+)             Key (−)
    ==================  ==================  ===================
    EE Δroll            Numpad 1            Numpad 3
    EE Δpitch           Numpad 0            Numpad Period (.)
    EE Δyaw             Numpad +            Numpad −
    ==================  ==================  ===================

    **Gripper** (toggle on press):

    ==================  ================================
    Key                 Action
    ==================  ================================
    G                   Toggle gripper (open ↔ closed)
    ==================  ================================

    **Global resets**:

    ==================  ================================
    Key                 Action
    ==================  ================================
    L                   Reset all commands to zero
    ==================  ================================

    .. note::
        All incremental commands (body pose, EE pos/orn) are **toggled** on
        press and released on key-up — the returned delta is non-zero only
        while the key is held.  Callers must accumulate the deltas to obtain
        absolute targets.  Velocity commands (v_x, v_y, ω_z) behave the same
        way (zero when no key held).
    """

    def __init__(self, cfg: Se2KeyboardFullCfg):
        """Initialise the keyboard interface.

        Args:
            cfg: Configuration dataclass for this controller.
        """
        # ── sensitivities ──────────────────────────────────────────────
        self.v_x_sensitivity      = cfg.v_x_sensitivity
        self.v_y_sensitivity      = cfg.v_y_sensitivity
        self.omega_z_sensitivity  = cfg.omega_z_sensitivity
        self.height_sensitivity   = cfg.height_sensitivity
        self.pitch_sensitivity    = cfg.pitch_sensitivity
        self.roll_sensitivity     = cfg.roll_sensitivity
        self.ee_pos_sensitivity   = cfg.ee_pos_sensitivity
        self.ee_orn_sensitivity   = cfg.ee_orn_sensitivity
        self._sim_device          = cfg.sim_device

        # ── omniverse interfaces ────────────────────────────────────────
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input     = carb.input.acquire_input_interface()
        self._keyboard  = self._appwindow.get_keyboard()

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # ── build key → delta vector mapping ───────────────────────────
        self._create_key_bindings()

        # ── command buffers ─────────────────────────────────────────────
        # _base_command : (13,)
        #   [0]    vx
        #   [1]    vy
        #   [2]    wz
        #   [3:6]  delta_ee_pos  (x, y, z)
        #   [6:9]  delta_ee_orn  (roll, pitch, yaw)
        #   [9]    delta_height
        #   [10]   delta_pitch
        #   [11]   delta_roll
        #   [12]   gripper  (0 or 1)
        self._base_command = np.zeros(13, dtype=np.float32)

        # gripper is a toggle, not a held key
        self._gripper_state: float = 1.0   # 1 = open, -1 = closed

        # ── user callbacks ──────────────────────────────────────────────
        self._additional_callbacks: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def __del__(self):
        """Release the keyboard event subscription."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        msg  = f"Keyboard Controller (SE2 + Body Pose + Arm EE + Gripper): {self.__class__.__name__}\n"
        msg += f"\tKeyboard: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t" + "─" * 52 + "\n"
        msg += "\tReset all commands : L\n"
        msg += "\t── Chassis velocity ──────────────────────────────\n"
        msg += "\tForward  (vx+) : Arrow Up       Backward (vx-) : Arrow Down\n"
        msg += "\tRight    (vy+) : Arrow Left      Left     (vy-) : Arrow Right\n"
        msg += "\tYaw+     (wz+) : Z               Yaw-     (wz-) : X\n"
        msg += "\t── Body pose (incremental Δ per frame) ───────────\n"
        msg += "\tHeight up   (Δh+) : C            Height down (Δh-) : V\n"
        msg += "\tPitch up    (Δp+) : B            Pitch down  (Δp-) : N\n"
        msg += "\tRoll left   (Δr+) : [            Roll right  (Δr-) : ]\n"
        msg += "\t── Arm EE position (incremental Δ per frame) ─────\n"
        msg += "\tΔx+  : Numpad 8     Δx-  : Numpad 2\n"
        msg += "\tΔy+  : Numpad 4     Δy-  : Numpad 6\n"
        msg += "\tΔz+  : Numpad 7     Δz-  : Numpad 9\n"
        msg += "\t── Arm EE orientation (incremental Δ per frame) ──\n"
        msg += "\tΔroll+  : Numpad 1     Δroll-  : Numpad 3\n"
        msg += "\tΔpitch+ : Numpad 0     Δpitch- : Numpad .\n"
        msg += "\tΔyaw+   : Numpad +     Δyaw-   : Numpad -\n"
        msg += "\t── Gripper (toggle) ──────────────────────────────\n"
        msg += "\tToggle open/close : G"
        return msg

    # ------------------------------------------------------------------
    # Public API (DeviceBase interface)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all command buffers (including gripper) to zero."""
        self._base_command.fill(0.0)
        self._gripper_state = 0.0

    def add_callback(self, key: str, func: Callable):
        """Bind an extra zero-argument callable to a key press.

        Args:
            key: Carb key name (e.g. ``"L"``).
            func: Callable invoked on each press of *key*.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Return a snapshot of the current 13-D command vector.

        Returns:
            ``np.ndarray`` of shape ``(13,)`` with layout::

                [vx, vy, wz,
                 Δee_x, Δee_y, Δee_z,
                 Δee_roll, Δee_pitch, Δee_yaw,
                 Δbody_height, Δbody_pitch, Δbody_roll,
                 gripper]
        """
        self._base_command[12] = self._gripper_state
        return self._base_command.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle a single keyboard event.

        KEY_PRESS   → ``_base_command += mapping[key]``
        KEY_RELEASE → ``_base_command -= mapping[key]``  (cancels the press)

        Special keys handled before the general mapping:
            L → :py:meth:`reset`
            G → toggle gripper state
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name == "G":
                self._gripper_state = -1.0 if self._gripper_state >= 0.0 else 1.0
                state_str = "OPEN" if self._gripper_state > 0 else "CLOSED"
                print(f"[Se2KeyboardFull] Gripper → {state_str}")
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
        """Build ``_INPUT_KEY_MAPPING``: key name → 13-D delta vector.

        13-D layout:
            [0]  vx
            [1]  vy
            [2]  wz
            [3]  Δee_x
            [4]  Δee_y
            [5]  Δee_z
            [6]  Δee_roll
            [7]  Δee_pitch
            [8]  Δee_yaw
            [9]  Δbody_height
            [10] Δbody_pitch
            [11] Δbody_roll
            [12] gripper   ← managed separately via toggle; always 0 here
        """
        vx   = self.v_x_sensitivity
        vy   = self.v_y_sensitivity
        wz   = self.omega_z_sensitivity
        dh   = self.height_sensitivity
        dp   = self.pitch_sensitivity
        dr   = self.roll_sensitivity
        ep   = self.ee_pos_sensitivity
        eo   = self.ee_orn_sensitivity

        def _v(idx: int, val: float) -> np.ndarray:
            """Return a 13-D zero vector with *val* at position *idx*."""
            vec = np.zeros(13, dtype=np.float32)
            vec[idx] = val
            return vec

        self._INPUT_KEY_MAPPING: dict[str, np.ndarray] = {

            # ── Chassis velocity ────────────────────────────────────────
            # v_x  (index 0)
            "UP"    : _v(0,  vx),
            "DOWN"  : _v(0, -vx),
            # v_y  (index 1)  note: left/right arrow feel natural for lateral
            "LEFT"  : _v(1,  vy),
            "RIGHT" : _v(1, -vy),
            # ω_z  (index 2)
            "Z"     : _v(2,  wz),
            "X"     : _v(2, -wz),

            # ── Body pose ───────────────────────────────────────────────
            # body height  (index 9)
            "C"             : _v(9,   dh),
            "V"             : _v(9,  -dh),
            # body pitch   (index 10)
            "B"             : _v(10,  dp),
            "N"             : _v(10, -dp),
            # body roll    (index 11)
            "LEFT_BRACKET"  : _v(11,  dr),
            "RIGHT_BRACKET" : _v(11, -dr),

            # ── Arm EE position (numpad rows 1-3) ───────────────────────
            # Δee_x  (index 3) : Numpad 8 / 2
            "NUMPAD_8" : _v(3,  ep),
            "NUMPAD_2" : _v(3, -ep),
            # Δee_y  (index 4) : Numpad 4 / 6
            "NUMPAD_4" : _v(4,  ep),
            "NUMPAD_6" : _v(4, -ep),
            # Δee_z  (index 5) : Numpad 7 / 9
            "NUMPAD_7" : _v(5,  ep),
            "NUMPAD_9" : _v(5, -ep),

            # ── Arm EE orientation (numpad rows 0 + ops) ────────────────
            # Δee_roll   (index 6) : Numpad 1 / 3
            "NUMPAD_1"      : _v(6,  eo),
            "NUMPAD_3"      : _v(6, -eo),
            # Δee_pitch  (index 7) : Numpad 0 / .
            "NUMPAD_0"      : _v(7,  eo),
            "NUMPAD_DEL" : _v(7, -eo),
            # Δee_yaw    (index 8) : Numpad + / -
            "NUMPAD_ADD"      : _v(8,  eo),
            "NUMPAD_SUBTRACT" : _v(8, -eo),
        }


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Se2KeyboardFullCfg(DeviceCfg):
    """Configuration for :class:`Se2KeyboardFull`.

    All ``*_sensitivity`` values scale the per-frame delta magnitude while the
    corresponding key is held down.  Tune these to match your sim timestep and
    action-term expectations.
    """

    # ── Chassis velocity ────────────────────────────────────────────────
    v_x_sensitivity:     float = 0.8
    """Magnitude of the v_x command (m/s or normalised, depending on action term)."""

    v_y_sensitivity:     float = 0.4
    """Magnitude of the v_y command."""

    omega_z_sensitivity: float = 1.0
    """Magnitude of the ω_z command (rad/s or normalised)."""

    # ── Body pose ────────────────────────────────────────────────────────
    height_sensitivity:  float = 1.0
    """Per-frame body height increment (m)."""

    pitch_sensitivity:   float = 1.0
    """Per-frame body pitch increment (rad)."""

    roll_sensitivity:    float = 1.0
    """Per-frame body roll increment (rad)."""

    # ── Arm EE ───────────────────────────────────────────────────────────
    ee_pos_sensitivity:  float = 0.02
    """Per-frame EE position increment (m)."""

    ee_orn_sensitivity:  float = 0.02
    """Per-frame EE orientation increment (rad)."""

    class_type: type[DeviceBase] = Se2KeyboardFull