# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VR 遥操作的信号整形工具（纯 numpy，不依赖 isaaclab / scipy，可离线单测）。

流水线（摇杆）::

    原始 (x, y) ──► 圆形死区(并重标定) ──► 响应曲线(expo) ──► 模长限幅 ≤ 1
                ──► × 各轴最大速度 ──► 斜率限制(SlewRateLimiter) ──► 指令

手柄位姿（机械臂 / 机体姿态）::

    原始位移/旋转 ──► 连续死区 ──► 低通(EMA / 四元数 slerp)

设计要点：
* 死区作用在向量模长上，并把 [dz, 1] 重映射到 [0, 1]，因此在死区边缘输出连续、不跳变。
* 曲线保留符号（奇对称），小推杆时更细腻，推满仍是 1。
* 斜率限制对"加速"和"减速"可分别设限（加速更温和，减速可更快以便急停）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "radial_deadzone",
    "expo_curve",
    "shape_stick",
    "continuous_deadzone",
    "SlewRateLimiter",
    "LowPass",
    "QuatSlerpFilter",
    "quat_mul",
    "quat_conj",
    "quat_slerp",
    "quat_to_rotvec",
    "rotvec_to_quat",
    "quat_normalize",
    "clamp_vec_norm",
]


# ---------------------------------------------------------------------------
# 摇杆整形
# ---------------------------------------------------------------------------

def radial_deadzone(vec: np.ndarray, deadzone: float) -> np.ndarray:
    """圆形死区：模长 < deadzone 输出 0；其余把 [dz, 1] 线性重映射到 [0, 1]。

    输出在死区边缘连续（从 0 平滑起步），模长被限制在 ≤ 1。
    """
    v = np.asarray(vec, dtype=np.float64)
    mag = float(np.linalg.norm(v))
    if mag <= deadzone or mag < 1e-12:
        return np.zeros_like(v)
    dz = min(max(deadzone, 0.0), 0.99)
    new_mag = min((mag - dz) / (1.0 - dz), 1.0)
    return v * (new_mag / mag)


def expo_curve(mag: float, expo: float) -> float:
    """响应曲线：``(1-expo)*m + expo*m^3``（m ∈ [0,1]，expo ∈ [0,1]）。

    expo=0 为线性；expo 越大，小推杆越细腻，而 m=1 时输出恒为 1。
    """
    m = min(max(float(mag), 0.0), 1.0)
    e = min(max(float(expo), 0.0), 1.0)
    return (1.0 - e) * m + e * m ** 3


def shape_stick(vec: np.ndarray, deadzone: float, expo: float) -> np.ndarray:
    """摇杆整形：圆形死区 → 曲线 → 模长限幅 ≤ 1，保持方向不变。"""
    v = radial_deadzone(vec, deadzone)
    mag = float(np.linalg.norm(v))
    if mag < 1e-12:
        return np.zeros_like(v)
    return v * (expo_curve(mag, expo) / mag)


class SlewRateLimiter:
    """逐轴斜率限制器（加速度限幅）。

    Args:
        max_accel: 逐轴"远离 0（加速）"时的最大变化率（单位/s）。
        max_decel: 逐轴"趋向 0（减速）"时的最大变化率（单位/s）。反向时按减速限制穿过 0。
    """

    def __init__(self, max_accel, max_decel, initial=None):
        self.max_accel = np.atleast_1d(np.asarray(max_accel, dtype=np.float64))
        self.max_decel = np.atleast_1d(np.asarray(max_decel, dtype=np.float64))
        n = self.max_accel.shape[0]
        self.value = np.zeros(n) if initial is None else np.asarray(initial, dtype=np.float64).copy()

    def reset(self, value=None) -> None:
        self.value[:] = 0.0 if value is None else value

    def update(self, target, dt: float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(target, dtype=np.float64))
        dt = max(float(dt), 0.0)
        speeding_up = np.abs(t) > np.abs(self.value)
        # 同号才算"加速"；反向（t 与 value 异号）按减速限制，避免瞬间反转
        same_sign = (t * self.value) >= 0.0
        rate = np.where(speeding_up & same_sign, self.max_accel, self.max_decel)
        step = rate * dt
        self.value = self.value + np.clip(t - self.value, -step, step)
        return self.value.copy()


# ---------------------------------------------------------------------------
# 手柄位姿整形
# ---------------------------------------------------------------------------

def continuous_deadzone(v, threshold: float):
    """连续死区：``sign(v) * max(|v| - threshold, 0)``。

    与"小于阈值置 0，否则原值输出"相比，边缘处不会产生 ``threshold`` 大小的跳变。
    支持标量与 ndarray。
    """
    a = np.asarray(v, dtype=np.float64)
    out = np.sign(a) * np.maximum(np.abs(a) - threshold, 0.0)
    return float(out) if np.ndim(out) == 0 else out


def clamp_vec_norm(v, max_norm: float) -> np.ndarray:
    """把向量模长限制在 max_norm 以内（方向不变）。"""
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n <= max_norm or n < 1e-12:
        return a.copy()
    return a * (max_norm / n)


class LowPass:
    """一阶低通（EMA），时间常数 tau 与采样间隔无关。tau<=0 表示直通。"""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self._y: np.ndarray | None = None

    def reset(self, value=None) -> None:
        self._y = None if value is None else np.asarray(value, dtype=np.float64).copy()

    def update(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._y is None or self.tau <= 0.0:
            self._y = x.copy()
            return self._y.copy()
        alpha = 1.0 - np.exp(-max(float(dt), 0.0) / self.tau)
        self._y = self._y + alpha * (x - self._y)
        return self._y.copy()


# ---------------------------------------------------------------------------
# 四元数（xyzw 约定，与 scipy 一致）
# ---------------------------------------------------------------------------

def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return np.array([0.0, 0.0, 0.0, 1.0]) if n < 1e-12 else q / n


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_rotvec(q) -> np.ndarray:
    """四元数 → 旋转向量（最短路径，角度 ∈ [0, π]）。"""
    q = quat_normalize(q)
    if q[3] < 0.0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return 2.0 * q[:3]
    angle = 2.0 * np.arctan2(s, q[3])
    return q[:3] / s * angle


def rotvec_to_quat(rv) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0])
    axis = rv / angle
    half = 0.5 * angle
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def quat_slerp(q0, q1, t: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:  # 近似共线，退化为线性插值
        return quat_normalize(q0 + t * (q1 - q0))
    th = np.arccos(min(d, 1.0))
    return quat_normalize((np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th))


class QuatSlerpFilter:
    """四元数低通：以 slerp 向目标靠拢，时间常数 tau（tau<=0 直通）。"""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self._q: np.ndarray | None = None

    def reset(self, q=None) -> None:
        self._q = None if q is None else quat_normalize(q)

    def update(self, q, dt: float) -> np.ndarray:
        q = quat_normalize(q)
        if self._q is None or self.tau <= 0.0:
            self._q = q
            return self._q.copy()
        alpha = 1.0 - np.exp(-max(float(dt), 0.0) / self.tau)
        self._q = quat_slerp(self._q, q, alpha)
        return self._q.copy()
