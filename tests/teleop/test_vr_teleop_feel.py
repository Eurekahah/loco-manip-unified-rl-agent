"""VR 遥操作信号整形 + 设备摇杆通路的离线单测（仅需 numpy，stdlib unittest）。

运行：
    python3 -m unittest discover -s tests/teleop -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEV_DIR = REPO / "source" / "rl_training" / "rl_training" / "devices"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


shaping = _load("rl_training.devices.vr_signal_shaping", DEV_DIR / "vr_signal_shaping.py")


# ---------------------------------------------------------------------------
# 纯函数 / 滤波器
# ---------------------------------------------------------------------------
class TestStickShaping(unittest.TestCase):
    def test_deadzone_zero_inside(self):
        self.assertTrue(np.all(shaping.radial_deadzone([0.08, 0.05], 0.12) == 0.0))

    def test_deadzone_continuous_at_edge(self):
        # 刚出死区时输出应接近 0（不跳变）
        out = shaping.radial_deadzone([0.1201, 0.0], 0.12)
        self.assertLess(np.linalg.norm(out), 1e-3)

    def test_deadzone_full_deflection_is_one(self):
        out = shaping.radial_deadzone([1.0, 0.0], 0.12)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0)

    def test_deadzone_clips_norm_over_one(self):
        out = shaping.radial_deadzone([1.0, 1.0], 0.12)  # 方形推到角落，模长 1.41
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0)

    def test_deadzone_preserves_direction(self):
        out = shaping.radial_deadzone([0.6, 0.3], 0.1)
        self.assertAlmostEqual(out[1] / out[0], 0.5)

    def test_expo_endpoints_and_monotonic(self):
        self.assertEqual(shaping.expo_curve(0.0, 0.5), 0.0)
        self.assertAlmostEqual(shaping.expo_curve(1.0, 0.5), 1.0)
        xs = np.linspace(0, 1, 50)
        ys = [shaping.expo_curve(x, 0.6) for x in xs]
        self.assertTrue(all(b >= a for a, b in zip(ys, ys[1:])))

    def test_expo_softer_than_linear_in_middle(self):
        self.assertLess(shaping.expo_curve(0.5, 0.5), 0.5)

    def test_shape_stick_odd_symmetric(self):
        a = shaping.shape_stick(np.array([0.4, -0.7]), 0.12, 0.5)
        b = shaping.shape_stick(np.array([-0.4, 0.7]), 0.12, 0.5)
        np.testing.assert_allclose(a, -b)

    def test_shape_stick_never_exceeds_one(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            v = rng.uniform(-1.5, 1.5, 2)
            self.assertLessEqual(np.linalg.norm(shaping.shape_stick(v, 0.12, 0.5)), 1.0 + 1e-9)


class TestSlew(unittest.TestCase):
    def test_accel_limited(self):
        s = shaping.SlewRateLimiter([4.0], [8.0])
        out = s.update([5.0], 0.1)
        self.assertAlmostEqual(out[0], 0.4)

    def test_decel_faster_than_accel(self):
        s = shaping.SlewRateLimiter([4.0], [8.0], initial=[2.0])
        out = s.update([0.0], 0.1)
        self.assertAlmostEqual(out[0], 2.0 - 0.8)

    def test_reaches_target_exactly(self):
        s = shaping.SlewRateLimiter([4.0], [8.0])
        for _ in range(100):
            out = s.update([1.0], 0.1)
        self.assertAlmostEqual(out[0], 1.0)

    def test_reversal_uses_decel_limit(self):
        s = shaping.SlewRateLimiter([1.0], [10.0], initial=[1.0])
        out = s.update([-1.0], 0.1)
        self.assertAlmostEqual(out[0], 0.0)  # 反向按减速限制穿过 0

    def test_zero_dt_no_change(self):
        s = shaping.SlewRateLimiter([4.0], [8.0], initial=[1.0])
        self.assertAlmostEqual(s.update([5.0], 0.0)[0], 1.0)


class TestPoseFilters(unittest.TestCase):
    def test_continuous_deadzone(self):
        self.assertEqual(shaping.continuous_deadzone(0.003, 0.004), 0.0)
        self.assertAlmostEqual(shaping.continuous_deadzone(0.0041, 0.004), 0.0001)
        self.assertAlmostEqual(shaping.continuous_deadzone(-0.014, 0.004), -0.010)
        out = shaping.continuous_deadzone(np.array([0.01, -0.01, 0.001]), 0.004)
        np.testing.assert_allclose(out, [0.006, -0.006, 0.0])

    def test_lowpass_converges_and_is_dt_independent(self):
        a = shaping.LowPass(0.1)
        b = shaping.LowPass(0.1)
        a.update([0.0], 0.01); b.update([0.0], 0.01)
        for _ in range(100):
            ya = a.update([1.0], 0.01)
        yb = b.update([1.0], 1.0)  # 同样 1s，一次走完
        self.assertAlmostEqual(ya[0], yb[0], places=3)
        self.assertGreater(ya[0], 0.99)

    def test_lowpass_zero_tau_passthrough(self):
        f = shaping.LowPass(0.0)
        f.update([0.0], 0.01)
        self.assertEqual(f.update([3.0], 0.01)[0], 3.0)

    def test_quat_rotvec_roundtrip(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            rv = rng.normal(size=3)
            rv *= rng.uniform(0.0, 3.0) / np.linalg.norm(rv)
            q = shaping.rotvec_to_quat(rv)
            np.testing.assert_allclose(shaping.quat_to_rotvec(q), rv, atol=1e-9)

    def test_quat_to_rotvec_shortest_path(self):
        q = shaping.rotvec_to_quat([0, 0, 0.3])
        np.testing.assert_allclose(shaping.quat_to_rotvec(-q), [0, 0, 0.3], atol=1e-9)

    def test_slerp_endpoints_and_midpoint(self):
        q0 = shaping.rotvec_to_quat([0, 0, 0])
        q1 = shaping.rotvec_to_quat([0, 0, 1.0])
        np.testing.assert_allclose(shaping.quat_to_rotvec(shaping.quat_slerp(q0, q1, 0.5)), [0, 0, 0.5], atol=1e-9)

    def test_slerp_filter_converges(self):
        f = shaping.QuatSlerpFilter(0.1)
        f.update(shaping.rotvec_to_quat([0, 0, 0]), 0.01)
        tgt = shaping.rotvec_to_quat([0, 0, 1.0])
        for _ in range(200):
            q = f.update(tgt, 0.01)
        np.testing.assert_allclose(shaping.quat_to_rotvec(q), [0, 0, 1.0], atol=1e-2)

    def test_clamp_vec_norm(self):
        out = shaping.clamp_vec_norm([3.0, 4.0], 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0)
        np.testing.assert_allclose(shaping.clamp_vec_norm([0.1, 0.1], 1.0), [0.1, 0.1])


# ---------------------------------------------------------------------------
# 设备摇杆通路（用桩替换 isaaclab / scipy / XLeVR）
# ---------------------------------------------------------------------------
def _load_device_module():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class DeviceBase:  # noqa: D401
        pass

    @dataclass
    class DeviceCfg:
        pass

    mod("isaaclab")
    mod("isaaclab.devices", DeviceBase=DeviceBase, DeviceCfg=DeviceCfg)
    mod("scipy")
    mod("scipy.spatial")
    class _Rot:
        @staticmethod
        def from_euler(*a, **k):
            return _Rot()
    mod("scipy.spatial.transform", Rotation=_Rot)
    mod("rl_training")
    mod("rl_training.devices")
    mod("rl_training.xtrainer_utils")
    mod("rl_training.xtrainer_utils.XLeVR")
    mod("rl_training.xtrainer_utils.XLeVR.xlevr")
    mod("rl_training.xtrainer_utils.XLeVR.xlevr.config", XLeVRConfig=type("XLeVRConfig", (), {}))
    mod("rl_training.xtrainer_utils.XLeVR.xlevr.inputs")
    mod("rl_training.xtrainer_utils.XLeVR.xlevr.inputs.vr_ws_server",
        VRWebSocketServer=type("VRWebSocketServer", (), {"__init__": lambda self, **k: None}))
    sys.modules["rl_training.devices.vr_signal_shaping"] = shaping
    m = _load("vr_loco_manip_teleop_under_test", DEV_DIR / "vr_loco_manip_teleop.py")
    m.VRLocoManipTeleop._run_vr_services = lambda self: None
    m.VRLocoManipTeleop._display_info = lambda self: None
    m.time.sleep = lambda s: None  # 跳过构造里的 sleep(1)
    return m


class _Goal:
    def __init__(self, x, y, arm="left"):
        self.arm = arm
        self.metadata = {"thumbstick": {"x": x, "y": y}}


class TestDeviceChassis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_device_module()

    def make(self, **cfg):
        dev = self.m.VRLocoManipTeleop(self.m.VRLocoManipTeleopCfg(**cfg))
        dev.set_step_dt(0.02)
        return dev

    def push(self, dev, x, y, steps=1, right_x=None):
        for _ in range(steps):
            dev._update_chassis_from_thumbstick(_Goal(x, y))
            if right_x is not None:
                dev._update_yaw_from_thumbstick(_Goal(right_x, 0, "right"))
            out = dev.advance()
        return out

    def test_inside_deadzone_stays_zero(self):
        dev = self.make()
        out = self.push(dev, 0.05, 0.08, steps=50)
        np.testing.assert_allclose(out[0:3], 0.0)

    def test_forward_sign_matches_legacy(self):
        # 旧行为：推前（y>0）→ vx 为负（原代码 command[0] = -y*sens）
        dev = self.make()
        out = self.push(dev, 0.0, 1.0, steps=200)
        self.assertLess(out[0], 0)
        legacy = self.make(stick_shaping_enabled=False)
        legacy._update_chassis_from_thumbstick(_Goal(0.0, 1.0))
        self.assertEqual(np.sign(legacy.advance()[0]), np.sign(out[0]))

    def test_full_stick_reaches_max_speed(self):
        dev = self.make()
        out = self.push(dev, 0.0, 1.0, steps=500)
        self.assertAlmostEqual(out[0], -dev.v_x_sensitivity)

    def test_acceleration_limited(self):
        dev = self.make()
        out = self.push(dev, 0.0, 1.0, steps=1)
        # 4 m/s² × 0.02 s = 0.08
        self.assertAlmostEqual(abs(out[0]), 4.0 * 0.02)

    def test_release_decelerates_but_not_instant(self):
        dev = self.make()
        self.push(dev, 0.0, 1.0, steps=500)
        out = self.push(dev, 0.0, 0.0, steps=1)
        self.assertAlmostEqual(abs(out[0]), dev.v_x_sensitivity - 8.0 * 0.02)
        out = self.push(dev, 0.0, 0.0, steps=200)
        self.assertEqual(out[0], 0.0)

    def test_diagonal_not_faster_than_axis(self):
        dev = self.make()
        out = self.push(dev, 1.0, 1.0, steps=800)  # 方形角落
        spd = np.hypot(out[0] / dev.v_x_sensitivity, out[1] / dev.v_y_sensitivity)
        self.assertLessEqual(spd, 1.0 + 1e-9)

    def test_yaw_sign_matches_legacy_and_slewed(self):
        dev = self.make()
        out = self.push(dev, 0.0, 0.0, steps=1, right_x=1.0)
        self.assertLess(out[2], 0)  # 原代码 command[2] = -x*sens
        self.assertAlmostEqual(abs(out[2]), 4.0 * 0.02)

    def test_watchdog_zeroes_target_on_timeout(self):
        dev = self.make()
        self.push(dev, 0.0, 1.0, steps=300)
        self.assertNotEqual(dev.advance()[0], 0.0)
        dev._stick_stamp["left"] -= 10.0  # 模拟 10 秒没收到摇杆数据
        for _ in range(300):
            out = dev.advance()
        self.assertEqual(out[0], 0.0)

    def test_reset_clears_state(self):
        dev = self.make()
        self.push(dev, 0.0, 1.0, steps=100)
        dev.reset()
        np.testing.assert_allclose(dev._command[0:3], 0.0)
        np.testing.assert_allclose(dev._stick_target, 0.0)
        np.testing.assert_allclose(dev._chassis_slew.value, 0.0)

    def test_legacy_mode_is_linear_and_instant(self):
        dev = self.make(stick_shaping_enabled=False)
        dev._update_chassis_from_thumbstick(_Goal(0.0, 0.02))  # 死区内的小抖动也会输出（旧行为）
        out = dev.advance()
        self.assertAlmostEqual(out[0], -0.02 * dev.v_x_sensitivity)


if __name__ == "__main__":
    unittest.main()
