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

    def test_rotvec_90deg_pitch_is_continuous(self):
        # 欧拉角在 pitch=90° 会奇异；旋转向量应保持有限且可逆。
        rv = np.array([0.0, np.pi / 2.0, 0.0])
        q = shaping.rotvec_to_quat(rv)
        out = shaping.quat_to_rotvec(q)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, rv, atol=1e-9)

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
    m._orig_display_info = m.VRLocoManipTeleop._display_info
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
        # 底盘整形单测显式打开整形；默认“无整形”由 TestArmSignal 单独覆盖。
        cfg.setdefault("stick_shaping_enabled", True)
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


class TestConnectInfo(unittest.TestCase):
    def test_display_info_prints_urls_and_adb_hint(self):
        import contextlib, io
        m = _load_device_module()
        dev = object.__new__(m.VRLocoManipTeleop)
        dev._xlevr_cfg = types.SimpleNamespace(https_port=8443, websocket_port=8442)
        m._list_local_ipv4 = lambda: [("192.168.137.1", False), ("192.168.3.193", True)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._orig_display_info(dev)
        out = buf.getvalue()
        self.assertIn("https://192.168.137.1:8443", out)
        self.assertIn("https://192.168.3.193:8443", out)
        self.assertIn("adb reverse tcp:8443 tcp:8443", out)
        self.assertIn("adb reverse tcp:8442 tcp:8442", out)
        self.assertIn("https://localhost:8443", out)

    def test_list_local_ipv4_filters_loopback(self):
        m = _load_device_module()
        for ip, _ in m._list_local_ipv4():
            self.assertFalse(ip.startswith("127.") or ip.startswith("169.254."))


class TestArmSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_device_module()

    def make(self, **cfg):
        return self.m.VRLocoManipTeleop(self.m.VRLocoManipTeleopCfg(**cfg))

    def feed(self, dev, pos, rotvec, n=1):
        for _ in range(n):
            dev._update_arm(np.asarray(pos, float), np.asarray(rotvec, float), 0.0)
        return dev._command.copy()

    def test_comfortable_defaults(self):
        cfg = self.m.VRLocoManipTeleopCfg()
        self.assertFalse(cfg.stick_shaping_enabled)
        self.assertTrue(cfg.arm_clutch_enabled)
        self.assertTrue(cfg.body_clutch_enabled)

    def test_hard_deadzone(self):
        dev = self.make(arm_clutch_enabled=False)
        out = self.feed(dev, [0.0, 0.01, 0.0], [0, 0, 0])
        # 默认轴映射: command[3]=delta.y, [4]=-delta.x, [5]=delta.z；0.01 > 0.004 死区 → 原值输出
        self.assertAlmostEqual(out[3], 0.01)
        out = self.feed(dev, [0.0, 0.003, 0.0], [0, 0, 0])
        self.assertEqual(out[3], 0.0)

    def test_rotvec_pitch_90_has_no_gimbal_jump(self):
        dev = self.make(arm_clutch_enabled=False)
        out = self.feed(dev, np.zeros(3), [0.0, np.pi / 2.0, 0.0])
        self.assertTrue(np.all(np.isfinite(out[6:9])))
        self.assertAlmostEqual(out[6], np.pi / 2.0)
        self.assertAlmostEqual(out[7], 0.0)
        self.assertAlmostEqual(out[8], 0.0)

    def test_reset_clears_clutch(self):
        dev = self.make()
        dev._grip["right"] = True
        self.feed(dev, [0.0, 0.1, 0.0], [0, 0, 0], n=50)
        dev.reset()
        dev._grip["right"] = True
        out = self.feed(dev, [0.0, 0.1, 0.0], [0, 0, 0])
        self.assertEqual(out[3], 0.0)


class TestClutchOffset(unittest.TestCase):
    def test_follows_only_while_engaged_and_freezes(self):
        c = shaping.ClutchOffset(1)
        self.assertEqual(c.update([5.0], False)[0], 0.0)      # 未按：不动，与手位置无关
        c.update([5.0], True)
        self.assertAlmostEqual(c.update([5.3], True)[0], 0.3)  # 相对按下瞬间的位移
        self.assertAlmostEqual(c.update([5.3], False)[0], 0.3) # 松开：冻结
        self.assertAlmostEqual(c.update([9.0], False)[0], 0.3) # 松开后手随便动
        c.update([1.0], True)                                  # 再按：从冻结值继续
        self.assertAlmostEqual(c.update([1.2], True)[0], 0.5)

    def test_limit_absorbs_overshoot_no_dead_travel(self):
        c = shaping.ClutchOffset(1, limit=0.1)
        c.update([0.0], True)
        self.assertAlmostEqual(c.update([0.5], True)[0], 0.1)  # 撞限位
        self.assertAlmostEqual(c.update([0.45], True)[0], 0.05)  # 回手立刻反向跟随

    def test_reset(self):
        c = shaping.ClutchOffset(2)
        c.update([0, 0], True); c.update([1, 1], True); c.reset()
        np.testing.assert_allclose(c.update([3, 3], False), 0.0)

    def test_release_freezes_at_last_output_without_new_pose(self):
        c = shaping.ClutchOffset(1)
        c.update([0.0], True)
        c.update([0.3], True)
        np.testing.assert_allclose(c.release(), [0.3])
        np.testing.assert_allclose(c.update([5.0], False), [0.3])


class TestArmBodyGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_device_module()

    def make(self, **cfg):
        return self.m.VRLocoManipTeleop(self.m.VRLocoManipTeleopCfg(**cfg))

    def test_arm_clutch_gate(self):
        dev = self.make(arm_clutch_enabled=True)
        rot = np.zeros(3)
        dev._grip["right"] = False
        dev._update_arm(np.array([0.0, 0.2, 0.0]), rot, 0.0)
        self.assertEqual(dev._command[3], 0.0)                 # 没按握持键：不跟随
        dev._grip["right"] = True
        dev._update_arm(np.array([0.0, 0.2, 0.0]), rot, 0.0)   # 按下瞬间锚定
        dev._update_arm(np.array([0.0, 0.3, 0.0]), rot, 0.0)
        self.assertAlmostEqual(dev._command[3], 0.1)
        dev._grip["right"] = False
        dev._update_arm(np.array([0.0, 0.3, 0.0]), rot, 0.0)
        dev._update_arm(np.array([0.0, 0.9, 0.0]), rot, 0.0)  # 松开后手随便动
        self.assertAlmostEqual(dev._command[3], 0.1)   # 冻结

    def test_arm_gates_off_ignores_grip(self):
        dev = self.make(arm_clutch_enabled=False)
        dev._grip["right"] = False
        dev._update_arm(np.array([0.0, 0.2, 0.0]), np.zeros(3), 0.0)
        self.assertAlmostEqual(dev._command[3], 0.2)

    def test_body_default_matches_legacy(self):
        dev = self.make(body_clutch_enabled=False)
        dev._update_body(np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.3, 0.0]), 0.0)
        self.assertAlmostEqual(dev._command[9], 0.1 * dev.height_sensitivity)
        self.assertAlmostEqual(dev._command[10], -0.2 * dev.pitch_sensitivity)
        self.assertAlmostEqual(dev._command[11], 0.3 * dev.roll_sensitivity)

    def test_body_clutch_ignores_wrist_tilt_when_not_gripping(self):
        dev = self.make()
        dev._update_body(np.zeros(3), np.array([0.3, 0.3, 0.0]), 0.0)  # 推摇杆导致手柄倾斜
        np.testing.assert_allclose(dev._command[9:12], 0.0)

    def test_body_clutch_follows_when_gripping(self):
        dev = self.make()
        dev._grip["left"] = True
        dev._update_body(np.zeros(3), np.zeros(3), 0.0)  # 按下瞬间锚定
        dev._update_body(np.array([0.0, 0.0, 0.1]), np.array([0.2, 0.3, 0.0]), 0.0)
        self.assertAlmostEqual(dev._command[9], 0.1 * dev.height_sensitivity)
        self.assertAlmostEqual(dev._command[10], -0.2 * dev.pitch_sensitivity)
        self.assertAlmostEqual(dev._command[11], 0.3 * dev.roll_sensitivity)

    def test_grip_state_parsed_from_queue(self):
        dev = self.make()
        g = types.SimpleNamespace(arm="right", target_position=None,
                                  metadata={"grip_active": True})
        dev._command_queue.put_nowait(g)
        dev._poll_queue()
        self.assertTrue(dev._grip["right"])
        dev._command_queue.put_nowait(types.SimpleNamespace(
            arm="right", target_position=None, metadata={"source": "vr_grip_release"}))
        dev._poll_queue()
        self.assertFalse(dev._grip["right"])


class TestQueueCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_device_module()

    def make(self):
        dev = self.m.VRLocoManipTeleop(self.m.VRLocoManipTeleopCfg())
        dev._started = True
        return dev

    @staticmethod
    def goal(x, grip):
        return types.SimpleNamespace(
            arm="right",
            target_position=np.array([x, 0.0, 0.0]),
            metadata={"grip_active": grip},
        )

    def test_same_grip_state_keeps_only_latest_pose(self):
        dev = self.make()
        seen = []
        dev._process_goal_pose = lambda goal: seen.append(float(goal.target_position[0]))
        for i in range(6):
            dev._command_queue.put_nowait(self.goal(float(i), True))
        dev._poll_queue()
        self.assertEqual(seen, [5.0])

    def test_grip_transition_keeps_last_pose_before_release(self):
        dev = self.make()
        seen = []
        dev._process_goal_pose = lambda goal: seen.append(float(goal.target_position[0]))
        for i in range(4):
            dev._command_queue.put_nowait(self.goal(float(i), True))
        dev._command_queue.put_nowait(self.goal(4.0, False))
        dev._poll_queue()
        self.assertEqual(seen, [3.0, 4.0])
        self.assertFalse(dev._grip["right"])


if __name__ == "__main__":
    unittest.main()
