# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX Phase 0 geometry / workspace probe.

用法::

    conda run -n env_isaac_lab python scripts/reinforcement_learning/rsl_rl/probe_as2w_geometry.py \
        --headless --steps 200 --num_envs 1 --self_collision off

输出 ``logs/probe/as2w/<YYYYMMDD-HHMMSS>/``：

* ``summary.json``    机器可读的全部实测值
* ``report.md``       人读结论 + §11 建议值
* ``reachability.npz`` EE 球坐标可达栅格

本脚本不依赖新低层 cfg，只用 ``Visualize-AS2W-PiperX-v0`` 的 stand env +
URDF 的 pytorch_kinematics FK。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AS2W-PiperX Phase 0 probe")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--skip_env", action="store_true", help="只跑 FK 部分（不建 Isaac env）")
parser.add_argument("--skip_fk", action="store_true", help="只跑 env 部分")
parser.add_argument("--skip_crouch_dynamic", action="store_true")
parser.add_argument("--self_collision", choices=["off", "on"], default="off")
parser.add_argument("--fk_samples", type=int, default=200_000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = ROOT / "deep_robotics_model" / "AS2W_PiperX" / "urdf" / "AS2W_PiperX.urdf"
LEGS = ("FL", "FR", "RL", "RR")
ARM_JOINTS = tuple(f"arm_joint{i}" for i in range(1, 7))


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _urdf_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    import xml.etree.ElementTree as ET

    out = {}
    for joint in ET.parse(str(path)).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        lo = float(limit.get("lower", -1.0e9))
        hi = float(limit.get("upper", 1.0e9))
        out[joint.get("name")] = (lo, hi)
    return out


def _build_chain(urdf_bytes: bytes, ee: str, root: str):
    import pytorch_kinematics as pk

    return pk.build_serial_chain_from_urdf(urdf_bytes, ee, root_link_name=root).to(dtype=torch.float32)


def _fk(chain, q) -> np.ndarray:
    q = torch.as_tensor(q, dtype=torch.float32).reshape(1, -1)
    return chain.forward_kinematics(q).get_matrix()[0, :3, 3].detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# FK：EE 工作空间
# --------------------------------------------------------------------------- #
def probe_workspace(urdf_bytes: bytes, limits: dict, n_samples: int, out_npz: Path, seed: int = 42) -> dict:
    chain = _build_chain(urdf_bytes, "gripper_base", "arm_base_link")
    names = list(chain.get_joint_parameter_names())
    if tuple(names) != ARM_JOINTS:
        raise RuntimeError(f"arm chain joints {names} != expected {ARM_JOINTS}")
    lo = torch.tensor([limits[n][0] for n in names], dtype=torch.float32)
    hi = torch.tensor([limits[n][1] for n in names], dtype=torch.float32)

    generator = torch.Generator().manual_seed(int(seed))
    chunks = []
    done = 0
    batch = 20_000
    while done < n_samples:
        n = min(batch, n_samples - done)
        u = torch.rand(n, lo.numel(), generator=generator)
        q = lo + u * (hi - lo)
        chunks.append(chain.forward_kinematics(q).get_matrix()[:, :3, 3].detach())
        done += n
    pts = torch.cat(chunks, dim=0).cpu().numpy()

    l = np.linalg.norm(pts, axis=1)
    pitch = np.arcsin(np.clip(pts[:, 2] / np.maximum(l, 1e-8), -1.0, 1.0))
    yaw = np.arctan2(pts[:, 1], pts[:, 0])

    l_edges = np.linspace(0.0, 0.8, 65)
    pitch_edges = np.linspace(-math.pi / 2, math.pi / 2, 55)
    yaw_edges = np.linspace(-math.pi, math.pi, 73)
    counts, _ = np.histogramdd((l, pitch, yaw), bins=(l_edges, pitch_edges, yaw_edges))

    # 低重心锚点候选：**强制 yaw≈0**（左右对称），用过滤样本的**中位数**而不是直方图
    # argmax —— 后者在 20 万随机样本下每个 bin 只有几十个点，argmax 会随种子跳动。
    region = (np.abs(yaw) <= 0.15) & (l >= 0.28) & (l <= 0.55) & (pitch >= -0.50) & (pitch <= 0.05)
    if int(region.sum()) < 2000:
        region = (np.abs(yaw) <= 0.35) & (l >= 0.24) & (l <= 0.60) & (pitch >= -0.60) & (pitch <= 0.20)
    a_l = float(np.median(l[region]))
    a_p = float(np.median(pitch[region]))
    a_y = 0.0
    a_p_low = float(np.percentile(pitch[region], 10))
    a_l_low = float(np.median(l[region]))

    # 过滤后最低点（世界 z 只在 arm_base 系下看相对值）
    mask = (l >= 0.28) & (l <= 0.55) & (pitch >= -0.50) & (pitch <= 0.05) & (np.abs(yaw) <= 0.15) & (pts[:, 0] > 0.22)
    lowest = None
    if mask.any():
        idx = np.argmin(pts[mask, 2])
        p = pts[mask][idx]
        ll = l[mask][idx]
        pp = pitch[mask][idx]
        yy = yaw[mask][idx]
        lowest = {"l": float(ll), "pitch": float(pp), "yaw": float(yy),
                  "xyz_in_arm_base": p.tolist()}

    np.savez(
        out_npz,
        counts=counts.astype(np.int64),
        l_edges=l_edges,
        pitch_edges=pitch_edges,
        yaw_edges=yaw_edges,
        anchor=np.array([a_l, a_p, a_y], dtype=np.float64),
    )

    return {
        "n_samples": int(n_samples),
        "joint_order": names,
        "joint_limits": {n: [float(limits[n][0]), float(limits[n][1])] for n in names},
        "l_percentiles": dict(zip(["p01", "p05", "p50", "p95", "p99"],
                                  np.percentile(l, [1, 5, 50, 95, 99]).tolist())),
        "pitch_range": [float(pitch.min()), float(pitch.max())],
        "yaw_range": [float(yaw.min()), float(yaw.max())],
        "reachable_voxels": int((counts > 0).sum()),
        "anchor_dense": {"l": a_l, "pitch": a_p, "yaw": a_y,
                         "samples_in_region": int(region.sum())},
        "anchor_low_pitch": {"l": a_l_low, "pitch": a_p_low, "yaw": 0.0},
        "anchor_lowest_z": lowest,
        "grid_shape": list(counts.shape),
    }


# --------------------------------------------------------------------------- #
# FK：镜像符号复核
# --------------------------------------------------------------------------- #
def _base_leg(leg: str, joint_type: str) -> float:
    if joint_type == "hip":
        return 0.10 if leg in ("FL", "RL") else -0.10
    if joint_type == "thigh":
        return 0.44
    if joint_type == "calf":
        return -0.88
    raise KeyError(joint_type)


def probe_mirror(urdf_bytes: bytes, limits: dict) -> dict:
    chains = {leg: _build_chain(urdf_bytes, f"{leg}_foot", "base_link") for leg in LEGS}
    for leg, chain in chains.items():
        if list(chain.get_joint_parameter_names()) != [f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf", f"{leg}_foot"]:
            raise RuntimeError(f"{leg} chain joint order unexpected: {chain.get_joint_parameter_names()}")

    def fk_leg(leg: str, hip: float, thigh: float, calf: float) -> np.ndarray:
        return _fk(chains[leg], [hip, thigh, calf, 0.0])

    def mirrored(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], -p[1], p[2]])

    ds = np.linspace(-0.3, 0.3, 13)
    result = {"pairs": {}}
    for a, b in (("FL", "FR"), ("RL", "RR")):
        pair = {}
        for jt in ("hip", "thigh", "calf"):
            errors = {}
            for s in (1.0, -1.0):
                err = 0.0
                for d in ds:
                    qa = [_base_leg(a, "hip"), _base_leg(a, "thigh"), _base_leg(a, "calf")]
                    qb = [_base_leg(b, "hip"), _base_leg(b, "thigh"), _base_leg(b, "calf")]
                    k = {"hip": 0, "thigh": 1, "calf": 2}[jt]
                    qa[k] += float(d)
                    qb[k] += float(s * d)
                    pa = fk_leg(a, *qa)
                    pb = fk_leg(b, *qb)
                    err += float(np.sum((pa - mirrored(pb)) ** 2))
                errors[str(int(s))] = math.sqrt(err / len(ds)) * 1000.0  # mm RMS
            best_s = 1.0 if errors["1"] < errors["-1"] else -1.0
            pair[jt] = {
                "best_sign": best_s,
                "rms_err_mm_best": errors[str(int(best_s))],
                "rms_err_mm_opposite": errors[str(int(-best_s))],
            }
        pair["base_pose"] = {
            "a": {jt: _base_leg(a, jt) for jt in ("hip", "thigh", "calf")},
            "b": {jt: _base_leg(b, jt) for jt in ("hip", "thigh", "calf")},
        }
        pair["default_mirror_rms_mm"] = float(
            np.linalg.norm(
                fk_leg(a, _base_leg(a, "hip"), _base_leg(a, "thigh"), _base_leg(a, "calf"))
                - mirrored(fk_leg(b, _base_leg(b, "hip"), _base_leg(b, "thigh"), _base_leg(b, "calf")))
            )
            * 1000.0
        )
        result["pairs"][f"{a}<->{b}"] = pair
    return result


# --------------------------------------------------------------------------- #
# FK：蹲伏几何扫描（静态）
# --------------------------------------------------------------------------- #
def probe_crouch_fk(urdf_bytes: bytes, limits: dict, wheel_radius: float, targets: list[float]) -> dict:
    chain = _build_chain(urdf_bytes, "FL_foot", "base_link")
    thigh_lo, thigh_hi = limits["FL_thigh"]
    calf_lo, calf_hi = limits["FL_calf"]
    rows = []
    for th in np.linspace(max(0.30, thigh_lo + 0.05), min(1.70, thigh_hi - 0.05), 121):
        for ca in np.linspace(max(-1.90, calf_lo + 0.05), min(-0.85, calf_hi - 0.02), 106):
            x, y, z = _fk(chain, [0.10, float(th), float(ca), 0.0])
            rows.append((wheel_radius - z, x, y, float(th), float(ca)))
    rows.sort(key=lambda r: r[0])
    valid = [r for r in rows if 0.04 <= r[1] <= 0.28]
    picks = {}
    for target in targets:
        if not valid:
            break
        best = min(valid, key=lambda r: abs(r[0] - target))
        picks[f"{target:.2f}"] = {
            "target_base_h": target,
            "fk_base_h": float(best[0]),
            "foot_x_in_base": float(best[1]),
            "foot_y_in_base": float(best[2]),
            "thigh": best[3],
            "calf": best[4],
        }
    return {
        "wheel_radius_used": float(wheel_radius),
        "min_fk_base_h_valid": float(valid[0][0]) if valid else None,
        "max_fk_base_h_valid": float(valid[-1][0]) if valid else None,
        "picks": picks,
    }


# --------------------------------------------------------------------------- #
# Isaac env 实测
# --------------------------------------------------------------------------- #
def _make_env():
    env_cfg = parse_env_cfg(
        "Visualize-AS2W-PiperX-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.self_collision == "on":
        spawn = env_cfg.scene.robot.spawn
        if hasattr(spawn, "self_collision"):
            spawn.self_collision = True
            spawn.force_usd_conversion = True
            spawn.usd_dir = str(spawn.usd_dir) + "_selfcoll"
            if getattr(spawn, "articulation_props", None) is not None:
                spawn.articulation_props.enabled_self_collisions = True
    env = gym.make("Visualize-AS2W-PiperX-v0", cfg=env_cfg)
    return env


def _projected_tilt(quat: torch.Tensor) -> torch.Tensor:
    g = math_utils.quat_apply_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=quat.device).repeat(quat.shape[0], 1))
    return torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))


def probe_env(crouch_fk: dict | None) -> dict:
    env = _make_env()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    dev = unwrapped.device

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    actuators = {name: list(act.joint_names) for name, act in robot.actuators.items()}
    wheel_bodies = [b for b in body_names if b.endswith("_foot")]
    wheel_joints = [j for j in joint_names if j.endswith("_foot")]

    idx = {n: i for i, n in enumerate(body_names)}
    base_i = idx["base_link"]
    arm_i = idx["arm_base_link"]
    grip_i = idx["gripper_base"]
    wheel_ids = [idx[b] for b in wheel_bodies]

    env.reset()
    zero_action = torch.zeros(env.action_space.shape, device=dev)
    rec = {k: [] for k in ("base_z", "wheel_z", "arm_z", "grip_pos", "grip_quat", "tilt", "torque_abs_max")}
    contact_peak = 0.0
    settle = min(max(args_cli.steps // 2, 50), args_cli.steps)

    sensor = unwrapped.scene.sensors.get("contact_forces") if getattr(unwrapped.scene, "sensors", None) else None
    for t in range(args_cli.steps):
        with torch.no_grad():
            env.step(zero_action)
        rec["base_z"].append(robot.data.root_pos_w[:, 2].mean().item())
        rec["wheel_z"].append(robot.data.body_pos_w[:, wheel_ids, 2].mean().item())
        rec["arm_z"].append(robot.data.body_pos_w[:, arm_i, 2].mean().item())
        rec["grip_pos"].append(robot.data.body_pos_w[:, grip_i].mean(dim=0).cpu().numpy())
        rec["grip_quat"].append(robot.data.body_quat_w[:, grip_i].mean(dim=0).cpu().numpy())
        rec["tilt"].append(_projected_tilt(robot.data.root_quat_w).max().item())
        rec["torque_abs_max"].append(robot.data.applied_torque.abs().max().item())
        if sensor is not None and t >= settle:
            contact_peak = max(contact_peak, float(sensor.data.net_forces_w.norm(dim=-1).max().item()))

    base_z = np.asarray(rec["base_z"])
    wheel_z = np.asarray(rec["wheel_z"])
    arm_z = np.asarray(rec["arm_z"])
    tilt = np.asarray(rec["tilt"])
    torq = np.asarray(rec["torque_abs_max"])
    tail = slice(settle, None)
    stand_h = float(base_z[tail].mean())
    wheel_h = float(wheel_z[tail].mean())
    # 轮心在世界系的高度**就是**轮半径（轮子静止在地面上时，圆心离地 = 半径）。
    # 站高 - 轮心高 = base_link 到轮心的垂直距离（≈0.369 m），不是半径 —— 两者别混。
    eq_radius = wheel_h

    out = {
        "self_collision_mode": args_cli.self_collision,
        "n_joints": len(joint_names),
        "n_bodies": len(body_names),
        "joint_names": joint_names,
        "body_names": body_names,
        "actuators": actuators,
        "wheel_joint_names": wheel_joints,
        "wheel_body_names": wheel_bodies,
        "fl_foot_is_joint": "FL_foot" in joint_names,
        "fl_foot_is_body": "FL_foot" in body_names,
        "fl_foot_in_wheel_actuator": "FL_foot" in actuators.get("wheel", []),
        "stand_base_z_mean": stand_h,
        "stand_base_z_std": float(base_z[tail].std()),
        "stand_base_z_settle_max": float(base_z[:settle].max()),
        "wheel_center_z_mean": wheel_h,
        "base_above_wheel_center": stand_h - wheel_h,
        "equivalent_wheel_radius": eq_radius,
        "arm_base_z_mean": float(arm_z[tail].mean()),
        "arm_base_z_std": float(arm_z[tail].std()),
        "gripper_default_pos_w": np.mean(np.asarray(rec["grip_pos"])[settle:], axis=0).tolist(),
        "gripper_default_quat_w": np.mean(np.asarray(rec["grip_quat"])[settle:], axis=0).tolist(),
        "tilt_max_rad": float(tilt[tail].max()),
        "applied_torque_abs_max": float(torq[tail].max()),
        "contact_force_peak_N": contact_peak,
    }

    if crouch_fk and not args_cli.skip_crouch_dynamic:
        out["crouch_dynamic"] = _crouch_dynamic(env, unwrapped, robot, crouch_fk)

    env.close()
    return out


def _crouch_dynamic(env, unwrapped, robot, crouch_fk: dict) -> dict:
    names = list(robot.joint_names)
    jidx = {n: i for i, n in enumerate(names)}
    term = unwrapped.action_manager.get_term("joint_pos")
    scale = getattr(term, "_scale", None)
    if scale is None:
        scale = torch.full((len(names),), 0.25, device=unwrapped.device)
    scale = torch.as_tensor(scale, device=unwrapped.device, dtype=torch.float32)
    if scale.ndim == 0:
        scale = scale.expand(len(names))

    results = {}
    for key, pick in crouch_fk["picks"].items():
        env.reset()
        default = robot.data.default_joint_pos[0].clone()
        target = default.clone()
        for leg in LEGS:
            target[jidx[f"{leg}_hip"]] = 0.10 if leg in ("FL", "RL") else -0.10
            target[jidx[f"{leg}_thigh"]] = float(pick["thigh"])
            target[jidx[f"{leg}_calf"]] = float(pick["calf"])
            target[jidx[f"{leg}_foot"]] = 0.0
        action = ((target - default) / scale.clamp_min(1e-6)).unsqueeze(0).repeat(unwrapped.num_envs, 1)
        env.reset()
        zs, tilts = [], []
        for t in range(180):
            with torch.no_grad():
                env.step(action)
            if t >= 60:
                zs.append(robot.data.root_pos_w[:, 2].mean().item())
                tilts.append(_projected_tilt(robot.data.root_quat_w).max().item())
        zs = np.asarray(zs)
        tilts = np.asarray(tilts)
        results[key] = {
            "target_base_h": pick["target_base_h"],
            "thigh": pick["thigh"],
            "calf": pick["calf"],
            "base_z_mean": float(zs.mean()),
            "base_z_std": float(zs.std()),
            "tilt_max": float(tilts.max()),
            "stable": bool(zs.std() < 0.02 and zs.mean() > 0.28 and tilts.max() < 0.8),
        }
    return results


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _write_report(path: Path, summary: dict):
    env = summary.get("env", {})
    fk = summary.get("fk", {})
    ws = fk.get("workspace", {})
    mirror = fk.get("mirror", {})
    crouch = fk.get("crouch", {})
    dyn = env.get("crouch_dynamic", {})

    lines = ["# AS2W-PiperX Phase 0 探针报告", ""]
    lines.append(f"- 生成时间：`{summary['timestamp']}`")
    lines.append(f"- self-collision 模式：`{summary.get('self_collision_mode')}`")
    lines.append("")

    if env:
        lines += ["## 1. 名单 / 站高 / 轮半径 / 臂基座", ""]
        lines.append(f"- 关节数：**{env['n_joints']}**；body 数：**{env['n_bodies']}**")
        lines.append(f"- 轮关节：`{env['wheel_joint_names']}`")
        lines.append(f"- 轮 body：`{env['wheel_body_names']}`")
        lines.append(f"- `FL_foot` 是关节/body/wheel actuator："
                     f"{env['fl_foot_is_joint']} / {env['fl_foot_is_body']} / {env['fl_foot_in_wheel_actuator']}")
        lines.append(f"- 站高 `base_link` z：**{env['stand_base_z_mean']:.4f} ± {env['stand_base_z_std']:.4f} m**")
        lines.append(f"- 轮心 z 均值：**{env['wheel_center_z_mean']:.4f} m** ⇒ 等效轮半径 "
                     f"**{env['equivalent_wheel_radius']:.4f} m**（= 轮心世界高度）")
        lines.append(f"- base 到轮心的垂直距离：**{env['base_above_wheel_center']:.4f} m**")
        lines.append(f"- `arm_base_link` z：**{env['arm_base_z_mean']:.4f} ± {env['arm_base_z_std']:.4f} m**")
        lines.append(f"- `gripper_base` 默认位姿（world）：pos `{np.round(env['gripper_default_pos_w'], 4).tolist()}` "
                     f"quat `{np.round(env['gripper_default_quat_w'], 4).tolist()}`")
        lines.append(f"- 姿态倾角最大：{env['tilt_max_rad']:.4f} rad；力矩峰值：{env['applied_torque_abs_max']:.3f} N·m")
        lines.append("")

    if ws:
        lines += ["## 2. EE 工作空间（arm_base_link 球坐标）", ""]
        lines.append(f"- 采样：{ws['n_samples']}；可达体素：{ws['reachable_voxels']}")
        lines.append(f"- l 分位：{json.dumps(ws['l_percentiles'])}")
        lines.append(f"- pitch 范围：{ws['pitch_range']}；yaw 范围：{ws['yaw_range']}")
        a = ws["anchor_dense"]
        lines.append(f"- **密集低重心锚点候选**：`p_l={a['l']:.3f}`, `p_pitch={a['pitch']:.3f}`, "
                     f"`p_yaw={a['yaw']:.3f}`（过滤区 {a['samples_in_region']} 样本的中位数）")
        if ws.get("anchor_low_pitch"):
            lo_p = ws["anchor_low_pitch"]
            lines.append(f"- 更低仰角候选（pitch p10）：`p_l={lo_p['l']:.3f}`, "
                         f"`p_pitch={lo_p['pitch']:.3f}`, `p_yaw=0.0`")
        if ws.get("anchor_lowest_z"):
            lo = ws["anchor_lowest_z"]
            lines.append(f"- 最低点候选：`p_l={lo['l']:.3f}`, `p_pitch={lo['pitch']:.3f}`, "
                         f"`p_yaw={lo['yaw']:.3f}`，arm_base 系 xyz `{[round(v,4) for v in lo['xyz_in_arm_base']]}`")
        lines.append("")

    if mirror:
        lines += ["## 3. 镜像符号复核（FK 扫 θ ∈ [-0.3, 0.3]）", ""]
        lines.append("| 镜像对 | 关节 | 最优符号 | 最优 RMS (mm) | 反向 RMS (mm) | 默认姿态镜像误差 (mm) |")
        lines.append("|---|---|---|---|---|---|")
        for pair, d in mirror["pairs"].items():
            for jt, r in d.items():
                if jt in ("base_pose", "default_mirror_rms_mm"):
                    continue
                lines.append(f"| {pair} | {jt} | {int(r['best_sign']):+d} | "
                             f"{r['rms_err_mm_best']:.2f} | {r['rms_err_mm_opposite']:.2f} | "
                             f"{d['default_mirror_rms_mm']:.2f} |")
        lines.append("")

    if crouch:
        lines += ["## 4. 蹲伏/最低高度（FK + 动态保持）", ""]
        lines.append(f"- FK 可行区间（0.04≤foot_x≤0.28）：**{crouch['min_fk_base_h_valid']:.4f} ~ "
                     f"{crouch['max_fk_base_h_valid']:.4f} m**")
        if dyn:
            lines.append("")
            lines.append("| 目标高度 | thigh | calf | 实测 base_z | std | tilt_max | 稳定 |")
            lines.append("|---|---|---|---|---|---|---|")
            for k, r in sorted(dyn.items(), key=lambda kv: float(kv[0])):
                lines.append(f"| {r['target_base_h']:.2f} | {r['thigh']:.3f} | {r['calf']:.3f} | "
                             f"{r['base_z_mean']:.4f} | {r['base_z_std']:.4f} | {r['tilt_max']:.3f} | "
                             f"{'✅' if r['stable'] else '❌'} |")
            stable = [r["base_z_mean"] for r in dyn.values() if r["stable"]]
            if stable:
                lines.append("")
                lines.append(f"- **建议 `body_pose.height_range` 下界**：`{max(min(stable), 0.30):.2f}` 左右"
                             f"（实测最低稳定 {min(stable):.3f} m，留 2 cm 余量）")
                lines.append(f"- **建议 `root_height_below_minimum`**：`{max(min(stable) - 0.05, 0.22):.2f}` 左右")
        lines.append("")

    lines += ["## 5. §11 建议值汇总", "", "| 参数 | 建议值 |", "|---|---|"]
    if env:
        lines.append(f"| 轮半径 | `{env['equivalent_wheel_radius']:.4f}` |")
        lines.append(f"| 默认站高 | `{env['stand_base_z_mean']:.4f}` |")
    if crouch and crouch.get("picks"):
        lines.append(f"| `body_pose.height_range` | 下界见 §4；上界 `{crouch['max_fk_base_h_valid']:.2f}` |")
    if ws:
        a = ws["anchor_dense"]
        lines.append(f"| EE Stage 0 锚点 | `p_l={a['l']:.3f}` / `p_pitch={a['pitch']:.3f}` / `p_yaw={a['yaw']:.3f}` |")
    if env:
        lines.append(f"| EE `sampled_height` | `{env['arm_base_z_mean']:.3f}`（arm_base 世界 z） |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args_cli.out) if args_cli.out else ROOT / "logs" / "probe" / "as2w" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    urdf_bytes = URDF_PATH.read_bytes()
    limits = _urdf_joint_limits(URDF_PATH)

    summary = {
        "timestamp": stamp,
        "urdf": str(URDF_PATH),
        "self_collision_mode": args_cli.self_collision,
        "fk": {},
        "env": {},
    }

    crouch_fk = None
    if not args_cli.skip_fk:
        print("[probe] FK workspace ...", flush=True)
        summary["fk"]["workspace"] = probe_workspace(
            urdf_bytes, limits, args_cli.fk_samples, out_dir / "reachability.npz", seed=args_cli.seed
        )
        print("[probe] FK mirror ...", flush=True)
        summary["fk"]["mirror"] = probe_mirror(urdf_bytes, limits)
        r_hint = summary["fk"]["workspace"]["l_percentiles"]["p50"]
        crouch_fk = probe_crouch_fk(urdf_bytes, limits, 0.077, [0.44, 0.42, 0.40, 0.38, 0.36, 0.34])
        summary["fk"]["crouch"] = crouch_fk
        del r_hint

    if not args_cli.skip_env:
        print("[probe] env measure ...", flush=True)
        summary["env"] = probe_env(crouch_fk)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = out_dir / "report.md"
    _write_report(report_path, summary)

    print("=" * 78)
    print(f"output dir : {out_dir}")
    print(f"summary    : {summary_path}")
    print(f"report     : {report_path}")
    if summary["env"]:
        e = summary["env"]
        print(f"stand_h={e['stand_base_z_mean']:.4f}  wheel_r={e['equivalent_wheel_radius']:.4f}  "
              f"arm_base_z={e['arm_base_z_mean']:.4f}")
    if summary["fk"].get("workspace"):
        print(f"anchor={summary['fk']['workspace']['anchor_dense']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
    os._exit(0)

