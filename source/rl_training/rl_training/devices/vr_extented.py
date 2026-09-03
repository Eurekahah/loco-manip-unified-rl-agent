# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VR controller for SE(2) + body pose + arm end-effector control.

Hardware layout
---------------
Left  controller : thumbstick  → chassis SE(2) velocity (vx, vy, wz)
                   hand pose Δ → body pitch / roll increment
                   trigger     → body height up / down
Right controller : hand pose Δ → arm end-effector 6-DOF (position + quaternion)
                   trigger     → gripper open / close

Button bindings
---------------
Right B  → Start control + calibrate both controllers
Left  X  → Reset / fail   (calls "R" callback)
Left  Y  → Reset / success (calls "N" callback)

Output from advance()
---------------------
np.ndarray of shape (13,):
    [vx, vy, wz,                          # indices 0-2  chassis velocity
     arm_dx, arm_dy, arm_dz,              # indices 3-5  arm Cartesian delta (m)
     arm_dr, arm_dp, arm_dy,              # indices 6-8  arm rotation delta (quat)
     delta_h, delta_pitch, delta_roll,    # indices 9-11 body pose delta
     arm_gripper]                         # index   12    gripper [0=open, 1=closed]
"""

from __future__ import annotations

import asyncio
import http.server
import os
import socket
import ssl
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaaclab.devices import DeviceBase, DeviceCfg

# ---------------------------------------------------------------------------
# Lazy import guard – XLeVR may not be installed in all environments
# ---------------------------------------------------------------------------
# try:
from rl_training.xtrainer_utils.XLeVR.xlevr.config import XLeVRConfig
from rl_training.xtrainer_utils.XLeVR.xlevr.inputs.vr_ws_server import VRWebSocketServer

_current_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir    = os.path.dirname(os.path.dirname(_current_dir))
print(_current_dir)
print(f"[VR Device] Attempting to import XLeVR from {_base_dir}...")

XLEVR_PATH   = os.path.join(_base_dir,"rl_training", "xtrainer_utils", "XLeVR")
print(f"[VR Device] XLeVR path set to: {XLEVR_PATH}")
_XLEVR_AVAILABLE = True
# except ImportError:
#     _XLEVR_AVAILABLE = False
#     XLEVR_PATH       = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [w, x, y, z]

def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert [w,x,y,z] → [x,y,z,w] (scipy convention)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,w] → [w,x,y,z] (Isaac convention)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

# 90-degree rotation around X to align VR wrist frame → simulation frame
_R_ALIGN = R.from_euler("x", 90, degrees=True)


# ---------------------------------------------------------------------------
# Main device class
# ---------------------------------------------------------------------------

class Se2VRExtended(DeviceBase):
    r"""VR controller for quadruped robot with arm.

    Outputs a 13-D command:
        ``[vx, vy, wz, arm_dx, arm_dy, arm_dz,
           arm_dr, arm_dp, arm_dy, Δh, Δpitch, Δroll, gripper]``

    See module docstring for full mapping details.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: Se2VRExtendedCfg):
        if not _XLEVR_AVAILABLE:
            raise RuntimeError(
                "XLeVR package not found. Please install it before using Se2VRExtended."
            )

        self._cfg = cfg

        # ---- sensitivity ----
        self.v_x_sensitivity     = cfg.v_x_sensitivity
        self.v_y_sensitivity     = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self.height_sensitivity  = cfg.height_sensitivity
        self.pitch_sensitivity   = cfg.pitch_sensitivity
        self.roll_sensitivity    = cfg.roll_sensitivity
        self.arm_pos_sensitivity = cfg.arm_pos_sensitivity
        self.arm_rot_sensitivity = cfg.arm_rot_sensitivity
        self.pos_deadzone        = cfg.pos_deadzone
        self.rot_deadzone        = cfg.rot_deadzone
        # 手柄局部轴 -> EE roll/pitch/yaw 通道的固定映射表（由单轴实验确定）
        self.arm_rot_axis_map    = cfg.arm_rot_axis_map
        # sim/房间系位移 -> EE x/y/z 通道的固定映射表
        self.arm_pos_axis_map    = cfg.arm_pos_axis_map

        # ---- state flags ----
        self._started      = False
        self._reset_state  = False

        # ---- VR origin for incremental mode ----
        self._vr_origin: dict[str, dict] = {
            "left":  {"pos": None, "rot": None},
            "right": {"pos": None, "rot": None},
        }
        self._calibration_triggered = {"left": False, "right": False}

        # ---- latest raw VR pose per hand ----
        #   format: [x,y,z, qw,qx,qy,qz, trigger]  (sim-frame position, isaac quat)
        self._latest_vr_pose: dict[str, np.ndarray | None] = {
            "left":  None,
            "right": None,
        }

        # ---- button edge-detection ----
        self._last_buttons: dict[str, bool] = {
            "right_a": False, "right_b": False,
            "left_x":  False, "left_y":  False,
        }

        # ---- user callbacks (same API as Se2Keyboard) ----
        self._additional_callbacks: dict[str, Callable] = {}

        # ---- command buffer [13] ----
        # 第 9~11 维是"相对标定原点的手柄偏移"，初始全 0。
        # 注意：不再把 default_body_height 写进第 9 维——那是"绝对目标"语义的残留，
        # 在增量/偏移语义下会导致机身高度一步顶到上限。
        self._command = np.zeros(13, dtype=np.float64)
        # 扳机积分出的机身高度偏移（按住 trigger 持续升高），标定/reset 时归零
        self._body_height_offset = 0.0
        # 相邻 VR 数据包间隔，用于把扳机"速度"积分成高度偏移
        self._dt = 0.0
        self._last_packet_time = time.time()

        # ---- previous calibrated pose for delta computation ----
        self._prev_arm_pos:  np.ndarray | None = None
        self._prev_arm_rot:  R          | None = None
        self._prev_body_rot: R          | None = None

        # ---- XLeVR setup ----
        self._xlevr_cfg = XLeVRConfig()
        self._xlevr_cfg.enable_vr   = True
        self._xlevr_cfg.enable_https = True
        self._xlevr_cfg.certfile = os.path.join(XLEVR_PATH, "cert.pem")
        self._xlevr_cfg.keyfile  = os.path.join(XLEVR_PATH, "key.pem")

        self._command_queue = asyncio.Queue()
        self._vr_server     = VRWebSocketServer(
            command_queue=self._command_queue,
            config=self._xlevr_cfg,
            print_only=False,
        )

        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_vr_services, daemon=True)
        self._thread.start()

        time.sleep(1)
        self._display_info()

    # ------------------------------------------------------------------
    # DeviceBase interface
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        msg  = f"VR Controller for SE(2) + Body Pose + Arm: {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t[Right B]    → Start control + calibrate\n"
        msg += "\t[Left  X]    → Reset (fail)\n"
        msg += "\t[Left  Y]    → Reset (success / next episode)\n"
        msg += "\t--- Left  controller ---\n"
        msg += "\tThumbstick X → lateral velocity (vy)\n"
        msg += "\tThumbstick Y → forward velocity  (vx)\n"
        msg += "\tThumbstick rotate → yaw (wz)\n"
        msg += "\tTrigger pull/release → body height up/down\n"
        msg += "\tHand pitch Δ → body pitch\n"
        msg += "\tHand roll  Δ → body roll\n"
        msg += "\t--- Right controller ---\n"
        msg += "\t6-DOF pose Δ → arm end-effector Cartesian + rotation delta\n"
        msg += "\tTrigger [0-1] → gripper open/close"
        return msg

    def reset(self):
        """Reset all buffers and VR origins."""
        self._command[:] = 0.0
        self._body_height_offset = 0.0
        self._vr_origin = {
            "left":  {"pos": None, "rot": None},
            "right": {"pos": None, "rot": None},
        }
        self._calibration_triggered = {"left": False, "right": False}
        self._prev_arm_pos  = None
        self._prev_arm_rot  = None
        self._prev_body_rot = None
        self._last_packet_time = time.time()

    def add_callback(self, key: str, func: Callable):
        """Register a zero-argument callable for a named event key.

        Supported keys: ``"S"`` (start control + recalibrate),
        ``"R"`` (reset/fail), ``"N"`` (reset/success).
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Poll the VR queue and return the current 13-D command.

        Returns:
            np.ndarray shape (13,):
            ``[vx, vy, wz,
               arm_dx, arm_dy, arm_dz,
               arm_droll, arm_dpitch, arm_dyaw,
                Δh, Δpitch, Δroll,
               gripper]``
        """
        self._poll_queue()
        return self._command.copy()

    # ------------------------------------------------------------------
    # Additional convenience: structured dict (optional for callers)
    # ------------------------------------------------------------------

    def advance_dict(self) -> dict:
        """Return command as a named dictionary.

        Keys: ``chassis``, ``arm_pos``, ``arm_rot``, ``body_pose``, ``gripper``,
        ``started``, ``reset``.
        """
        cmd = self.advance()
        return {
            "chassis": cmd[0:3],
            "arm_pos": cmd[3:6],
            "arm_rpy": cmd[6:9],
            "body_pose": cmd[9:12],
            "gripper":    float(cmd[12]),
            "started":    self._started,
            "reset":      self._reset_state,
        }

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self):
        """Drain the async queue and update _command."""
        while not self._command_queue.empty():
            try:
                goal = self._command_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if goal.arm == "headset":
                continue

            # 记录相邻包间隔，供扳机高度积分使用
            now = time.time()
            self._dt = min(max(now - self._last_packet_time, 0.0), 0.1)
            self._last_packet_time = now

            # --- buttons ---
            if goal.metadata and "buttons" in goal.metadata:
                self._check_buttons(goal.metadata["buttons"], goal.metadata.get("hand", goal.arm))

            if not self._started or goal.target_position is None:
                continue

            # --- pose update ---
            sim_pos, r_sim = self._vr_to_sim_pose(goal)
            hand = goal.arm

            # calibration (triggered once on button-B press)
            # 按B键设置当前位姿为 origin，后续计算增量
            if self._calibration_triggered.get(hand, False):
                print(f"[VR Device] Calibrating {hand} controller origin...")
                self._vr_origin[hand]["pos"] = sim_pos.copy()
                self._vr_origin[hand]["rot"] = r_sim
                self._calibration_triggered[hand] = False

            if self._vr_origin[hand]["pos"] is None:
                continue

            delta_pos = sim_pos - self._vr_origin[hand]["pos"]
            r_diff    = r_sim * self._vr_origin[hand]["rot"].inv()
            # 手柄自身坐标系下的相对旋转：转轴固定在手柄上，不随标定时手势漂移
            # （r_diff 是世界/sim 系的，纯俯仰会投影成多个轴的分量）
            r_diff_local = self._vr_origin[hand]["rot"].inv() * r_sim

            trigger = 0.0
            if goal.metadata and "trigger" in goal.metadata:
                trigger = float(goal.metadata["trigger"])

            if hand == "right":
                self._update_arm(delta_pos, r_diff_local, trigger)
                self._update_yaw_from_thumbstick(goal)
            elif hand == "left":
                self._update_body(delta_pos, r_diff, trigger)
                self._update_chassis_from_thumbstick(goal)

    # ------------------------------------------------------------------
    # Command updaters
    # ------------------------------------------------------------------

    def _update_arm(self,
                delta_pos: np.ndarray,
                r_diff_local: R,
                trigger: float):
        # ------------------------------------------------------------------
        # 位置增量：sim/房间系位移 -> EE x/y/z 通道
        #
        # delta_pos 是手柄相对标定点的位移（sim 系）。位置不像姿态那样需要
        # “手柄局部系”：上/下/左/右是房间方向，与手柄自身朝向无关，所以只要
        # 一个固定的轴约定映射即可，由 arm_pos_axis_map 决定。
        #
        # 注意：执行端把该增量加到 body 系基准上，因此实际手感是“相对机器人
        # 当前朝向”；若以后想改成“相对房间方向”，需先把增量按机身偏航旋转。
        # ------------------------------------------------------------------
        for i, (src, sgn) in enumerate(self.arm_pos_axis_map):
            v = self._apply_deadzone(sgn * delta_pos[src], self.pos_deadzone)
            self._command[3 + i] = v * self.arm_pos_sensitivity

        # ------------------------------------------------------------------
        # Rotation：使用“手柄局部系”的相对旋转（r_diff_local），而不是世界系
        # 的 r_diff。世界系分解会把“纯俯仰”拆成 roll+pitch 多个分量（耦合），
        # 且耦合方式随 B 标定时手势变化；局部系则“俯仰就是俯仰”，与标定手势无关。
        #
        # 局部系欧拉角 -> EE 通道的轴映射由 self.arm_rot_axis_map 决定，
        # 每项 (src, sgn) 表示 command[6+i] = sgn * euler_local[src]。
        # ------------------------------------------------------------------
        euler_local = r_diff_local.as_euler("XYZ", degrees=False)

        for i, (src, sgn) in enumerate(self.arm_rot_axis_map):
            v = self._apply_deadzone(sgn * euler_local[src], self.rot_deadzone)
            self._command[6 + i] = v * self.arm_rot_sensitivity

        # --- gripper ---
        self._command[12] = 1.0 if trigger > 0.5 else -1.0

    def _update_yaw_from_thumbstick(self, goal):
        """Map right thumbstick left/right axis to chassis yaw rate (wz)."""
        if not (goal.metadata and "thumbstick" in goal.metadata):
            return
        stick = goal.metadata["thumbstick"]
        stick_x = -float(stick.get("x", 0.0))
        self._command[2] = stick_x * self.omega_z_sensitivity

    def _update_body(self, delta_pos: np.ndarray, r_diff: R, trigger: float):
        """Write body height / pitch / roll offsets from left controller.

        输出的是"相对标定原点"的纯偏移（不再混入 default_body_height 绝对量），
        由执行端在 absolute_commands 模式下叠加到标定基准上。
        """
        # height: Z 偏移 + 扳机积分（按住持续升高，松手保持，标定/reset 归零）
        if trigger > 0.5:
            self._body_height_offset += self.height_sensitivity * self._dt
            self._body_height_offset = float(np.clip(self._body_height_offset, -0.5, 0.5))
        self._command[9] = (
            self._apply_deadzone(delta_pos[2], self.pos_deadzone) * self.height_sensitivity
            + self._body_height_offset
        )

        # pitch / roll from rotation delta (Euler XYZ in sim frame)
        euler = r_diff.as_euler("XYZ", degrees=False)
        self._command[10] = -self._apply_deadzone(euler[0], self.rot_deadzone) * self.pitch_sensitivity
        self._command[11] =  self._apply_deadzone(euler[1], self.rot_deadzone) * self.roll_sensitivity

    @staticmethod
    def _apply_deadzone(v: float, threshold: float) -> float:
        """小于阈值的偏移直接置 0，抑制 VR 手柄抖动产生的微小漂移。"""
        return 0.0 if abs(v) < threshold else float(v)

    def _update_chassis_from_thumbstick(self, goal):
        """Map left thumbstick axes to (vx, vy, wz)."""
        if not (goal.metadata and "thumbstick" in goal.metadata):
            return
        stick = goal.metadata["thumbstick"]
        # Typical VR thumbstick: x=horizontal, y=vertical (forward)
        stick_x = float(stick.get("x", 0.0))
        stick_y = float(stick.get("y", 0.0))

        self._command[0] = - stick_y * self.v_x_sensitivity
        self._command[1] = - stick_x * self.v_y_sensitivity


    # ------------------------------------------------------------------
    # Coordinate conversion  (identical to XTrainerVR._convert_goal_to_pose)
    # ------------------------------------------------------------------

    def _vr_to_sim_pose(self, goal) -> tuple[np.ndarray, R]:
        """Convert a VR goal to simulation-frame position + Rotation.

        Returns:
            (sim_pos, r_sim)  where sim_pos is shape (3,) and r_sim is a
            scipy Rotation representing the controller orientation in sim
            coordinates.
        """
        raw = goal.target_position
        sim_pos = np.array([raw[0], -raw[2], raw[1]], dtype=np.float64)

        raw_q = np.array([0., 0., 0., 1.], dtype=np.float64)
        if goal.metadata and "quaternion" in goal.metadata:
            q = goal.metadata["quaternion"]
            raw_q = np.array([
                q.get("x", 0.), q.get("y", 0.),
                q.get("z", 0.), q.get("w", 1.),
            ])
        r_vr  = R.from_quat(raw_q)
        r_sim = _R_ALIGN * r_vr * _R_ALIGN.inv()

        return sim_pos, r_sim

    # ------------------------------------------------------------------
    # Button handling
    # ------------------------------------------------------------------

    def _check_buttons(self, buttons_dict: dict, hand: str):
        is_right = "right" in hand.lower()
        is_left  = "left"  in hand.lower()
        for key, pressed in buttons_dict.items():
            k = str(key).lower()
            if is_right:
                if k == "a": self._edge_button("right_a", bool(pressed))
                elif k == "b": self._edge_button("right_b", bool(pressed))
            elif is_left:
                if k == "a": self._edge_button("left_x", bool(pressed))
                elif k == "b": self._edge_button("left_y", bool(pressed))

    def _edge_button(self, uid: str, pressed: bool):
        was = self._last_buttons.get(uid, False)
        if pressed and not was:
            parts = uid.split("_")
            self._on_button_press(parts[1], parts[0])
        self._last_buttons[uid] = pressed

    def _on_button_press(self, btn: str, hand: str):
        if btn == "b":
            print("🟢 [VR] Button B → START + Calibrate")
            self._started           = True
            self._reset_state       = False
            # 重新标定：清空命令缓冲与扳机高度积分，避免旧偏移残留到新标定
            self._command[:]        = 0.0
            self._body_height_offset = 0.0
            self._calibration_triggered["left"]  = True
            self._calibration_triggered["right"] = True
            if "S" in self._additional_callbacks:
                self._additional_callbacks["S"]()

        elif btn == "x":
            print("🔴 [VR] Button X → RESET (Fail)")
            self._started     = False
            self._reset_state = True
            self.reset()
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()

        elif btn == "y":
            print("🔵 [VR] Button Y → RESET (Success)")
            self._started     = False
            self._reset_state = True
            self.reset()
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()

    # ------------------------------------------------------------------
    # Background services  (identical to XTrainerVR._run_vr_services)
    # ------------------------------------------------------------------

    def _run_vr_services(self):
        asyncio.set_event_loop(self._loop)

        # HTTPS static file server
        # try:
        handler          = _SimpleFileHandler
        handler.web_root = os.path.join(XLEVR_PATH, "web-ui")
        httpd = http.server.HTTPServer(
            (self._xlevr_cfg.host_ip, self._xlevr_cfg.https_port), handler
        )
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        print(f"[VR Device] Loading SSL cert from {self._xlevr_cfg.certfile} and key from {self._xlevr_cfg.keyfile}...")
        ctx.load_cert_chain(self._xlevr_cfg.certfile, self._xlevr_cfg.keyfile)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"🌐 HTTPS server on port {self._xlevr_cfg.https_port}")
        # except Exception as exc:
        #     print(f"❌ HTTPS server failed: {exc}")

        # WebSocket VR server
        try:
            self._loop.run_until_complete(self._vr_server.start())
            print(f"✅ VR WebSocket on port {self._xlevr_cfg.websocket_port}")
            self._loop.run_forever()
        except Exception as exc:
            print(f"❌ VR loop failed: {exc}")

    def _display_info(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "localhost"
        print("\n" + "=" * 50)
        print("🎧  Se2VRExtended ready!")
        print(f"👉  https://{ip}:{self._xlevr_cfg.https_port}")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Internal HTTPS file handler
# ---------------------------------------------------------------------------

class _SimpleFileHandler(http.server.SimpleHTTPRequestHandler):
    web_root: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_root, **kwargs)

    def log_message(self, fmt, *args):  # silence access log
        pass


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Se2VRExtendedCfg(DeviceCfg):
    """Configuration for :class:`Se2VRExtended`.

    All sensitivity values are simple linear scales applied to the raw
    delta before it is written into the command vector.
    """

    # -- chassis --
    v_x_sensitivity:     float = 5.0
    """Forward / backward velocity scale (left thumbstick Y)."""
    v_y_sensitivity:     float = 1.0
    """Lateral velocity scale (left thumbstick X)."""
    omega_z_sensitivity: float = 1.0
    """Yaw rate scale (left thumbstick twist / rotation)."""

    # -- body pose --
    default_body_height: float = 0.513
    """Default body height (m) restored on reset / init."""
    height_sensitivity:  float = 0.5
    """Body height delta scale (left controller Z + trigger)."""
    pitch_sensitivity:   float = 0.35
    """Body pitch delta scale (left controller rotation Y)."""
    roll_sensitivity:    float = 0.25
    """Body roll delta scale (left controller rotation X)."""

    # -- arm --
    arm_pos_sensitivity: float = 1.0
    """Arm Cartesian delta scale (right controller position)."""
    arm_rot_sensitivity: float = 1.0
    """Arm rotation delta scale applied to the rotation-vector magnitude."""

    arm_rot_axis_map: tuple = (
        (1, +1.0),  # command[6] = EE roll  <- 手柄局部 pitch（初始猜测，实验后改）
        (0, -1.0),  # command[7] = EE pitch <- 手柄局部 roll（初始猜测，实验后改）
        (2, +1.0),  # command[8] = EE yaw   <- 手柄局部 yaw
    )
    """手柄局部轴 -> EE roll/pitch/yaw 通道映射，每项 (euler_local下标, 符号)。

    下标含义：0=手柄局部 roll(x)，1=手柄局部 pitch(y)，2=手柄局部 yaw(z)。
    单轴实验方法：按 B 标定后，每次只做一种纯旋转（俯仰/偏航/滚转），
    看 euler_local 哪个分量被激活、末端实际绕哪根轴转，再把 (下标, 符号)
    填到对应 command 通道。初始值是按位置通道的 (x,y,z)->(y,-x,z) 映射猜的。
    """

    arm_pos_axis_map: tuple = (
        (1, +1.0),  # command[3] = EE x  <- delta_pos.y
        (0, -1.0),  # command[4] = EE y  <- -delta_pos.x
        (2, +1.0),  # command[5] = EE z  <- delta_pos.z
    )
    """sim/房间系位移 -> EE x/y/z 通道映射，每项 (delta_pos下标, 符号)。

    下标：0=sim x，1=sim y，2=sim z。默认值保持当前已验证的手感；
    若某个方向错位或反向，改对应 (下标, 符号) 即可，无需动逻辑。
    """

    # -- deadzone --
    pos_deadzone: float = 0.004
    """手柄位置偏移死区（m），小于该值的偏移视为 0，抑制抖动。"""
    rot_deadzone: float = 0.006
    """手柄旋转偏移死区（rad），小于该值的旋转视为 0，抑制抖动。"""

    # Required by DeviceCfg
    class_type: type[DeviceBase] = Se2VRExtended
