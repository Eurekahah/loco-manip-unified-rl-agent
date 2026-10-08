# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""臂关节 PD 参数的**开环**验收：同一串"已知可达"的 EE 目标，逐档量跟踪与力矩。

为什么不能直接用 `policy_report` 的 EE 跟踪列做 A/B
--------------------------------------------------
`HeightInvariantEECommand` 的指令是**从"重采样那一刻的实际 EE 位姿"插值到随机新目标**的，
所以换一组 PD 参数后机械臂的**实际位姿**变了 ⇒ 记录到的"指令"也跟着变
（实测：硬增益 `|p_cmd|` 均值 0.58 m、软增益 0.29 m）⇒ 跨配置的"误差"根本不可比。

本探针把目标**钉死**：以复位后的默认 EE 位姿为原点，在 root 系里走一串固定偏移
（前/上/侧各 ±若干 cm），每档保持 `--hold-s` 秒，逐档统计：

* 稳态位置误差 / 姿态误差（对**同一个**指令）；
* `|tau|` 均值与峰值、**饱和时间占比**；
* `|qd|` p99 与超速时间占比。

底盘/轮子由训练好的低层策略接管（`--checkpoint`），保证机械臂测试期间机器人是站着的。

用法::

    python scripts/reinforcement_learning/rsl_rl/probe_arm_pd.py --headless --label now \
        --checkpoint logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt
    # 只改阻尼做 A/B
    ... --label c8 env.scene.robot.actuators.piper_arm.damping=8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="臂关节 PD 开环阶跃探针")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="History-Adaptation-Deeprobotics-M20-play-v0",
                    help="必须与 --checkpoint 的训练任务同族（obs 布局要一致）")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--policy", type=str, default=None,
                    help="低层策略路径；不传就用 --checkpoint（cli_args 里已有这个选项）")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hold-s", type=float, default=1.2, help="每个目标点保持时长（秒）")
parser.add_argument("--label", type=str, default="run")
parser.add_argument("--out", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerHis  # noqa: E402

import rl_training.tasks  # noqa: F401,E402

#: 目标序列（root 系偏移，m）。0 号点是复位后的默认 EE 位姿 ⇒ 一定可达。
WAYPOINTS = (
    (0.00, 0.00, 0.00),
    (+0.10, 0.00, 0.00), (0.00, 0.00, 0.00),
    (0.00, 0.00, +0.08), (0.00, 0.00, 0.00),
    (0.00, +0.10, 0.00), (0.00, 0.00, 0.00),
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.command_manager.get_term("ee_pose")
    dt = unwrapped.step_dt
    n = unwrapped.num_envs
    ee_idx = robot.find_bodies("gripper_base")[0][0]
    arm_ids = [int(i) for i in robot.find_joints(["arm_joint[1-6]"])[0]]
    arm_names = [robot.data.joint_names[i] for i in arm_ids]

    # 限幅/速度限幅从 actuator 实例读
    eff, vel = {}, {}
    for act in robot.actuators.values():
        jn = list(getattr(act, "joint_names", []) or [])
        if not jn:
            continue
        for attr, store in (("effort_limit", eff), ("velocity_limit", vel)):
            buf = getattr(act, attr, None)
            if buf is None:
                continue
            a = torch.as_tensor(buf, dtype=torch.float32).ravel()
            if a.numel() % len(jn) == 0:
                a = a[: len(jn)]
            if a.numel() == len(jn):
                store.update({k: float(v) for k, v in zip(jn, a)})

    gains = {}
    for act in robot.actuators.values():
        jn = list(getattr(act, "joint_names", []) or [])
        if not jn or not jn[0].startswith("arm_joint"):
            continue
        for attr in ("stiffness", "damping", "min_delay", "max_delay"):
            buf = getattr(act, attr, None)
            if buf is not None:
                v = torch.as_tensor(buf, dtype=torch.float32).flatten()
                gains[attr] = float(v[0]) if v.numel() else None
    # 从 data 里读一遍限幅（本资产 effort 是 1e9 占位值；顺带看 velocity 到底是多少）
    for tag, arr in (("data_joint_vel_limits", getattr(robot.data, "joint_vel_limits", None)),
                     ("data_joint_effort_limits", getattr(robot.data, "joint_effort_limits", None))):
        if arr is None:
            continue
        v = torch.as_tensor(arr, dtype=torch.float32).flatten()[: max(arm_ids) + 1]
        sel = [float(v[i]) for i in arm_ids] if v.numel() > max(arm_ids) else []
        if sel:
            gains[tag] = [round(x, 2) for x in sel]

    # 载入低层策略（让机器人站着）
    runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ckpt = args_cli.policy or args_cli.checkpoint
    if not ckpt:
        raise SystemExit("需要 --checkpoint（低层策略）")
    runner.load(os.path.abspath(ckpt))
    policy = runner.get_inference_policy(device=unwrapped.device)
    obs = env.get_observations()

    hold = max(int(round(args_cli.hold_s / dt)), 2)
    results: dict = {}
    for wi, off in enumerate(WAYPOINTS):
        # 每档开始前把 episode 顶到上限 ⇒ 全体复位（角度/位姿回到默认，可复现）
        unwrapped.episode_length_buf[:] = unwrapped.max_episode_length
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        pos_w = robot.data.body_pos_w[:, ee_idx]
        quat_w = robot.data.body_quat_w[:, ee_idx]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)
        tgt = torch.cat([pos_b + torch.tensor(off, device=unwrapped.device), quat_b], dim=-1)

        rec_err, rec_ori, rec_tau, rec_qd = [], [], [], []
        for k in range(hold):
            with torch.inference_mode():
                term.pose_start_b[:] = tgt
                term.pose_end_b[:] = tgt
                term.T_traj[:] = 1e-3
                term.elapsed_time[:] = 1.0
                obs, _, _, _ = env.step(policy(obs))
            if k < max(hold // 3, 1):      # 丢掉过渡段
                continue
            ee_p_w = robot.data.body_pos_w[:, 0, :]
            ee_p = robot.data.body_pos_w[0, ee_idx]
            ee_q = robot.data.body_quat_w[0, ee_idx]
            ee_p_b, ee_q_b = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w[0], robot.data.root_quat_w[0], ee_p, ee_q)
            rec_err.append(float(torch.norm(ee_p_b - tgt[0, :3])))
            dot = torch.clamp(torch.abs(torch.dot(ee_q_b, tgt[0, 3:])), 0.0, 1.0)
            rec_ori.append(float(2.0 * torch.acos(dot)))
            rec_tau.append(robot.data.applied_torque[0, arm_ids].abs().cpu().numpy())
            rec_qd.append(robot.data.joint_vel[0, arm_ids].abs().cpu().numpy())
        if not rec_err:
            continue
        tau = np.stack(rec_tau); qd = np.stack(rec_qd)
        lim = np.array([eff.get(x, np.nan) for x in arm_names])
        vlim = np.array([vel.get(x, np.nan) for x in arm_names])
        sat = np.nanmean((tau >= 0.99 * lim).mean(axis=0))
        ovr = np.nanmean((qd > vlim).mean(axis=0))
        results[f"wp{wi}_{off}"] = {
            "offset": list(off),
            "pos_err_cm": float(np.mean(rec_err) * 100),
            "ori_err_deg": float(np.degrees(np.mean(rec_ori))),
            "tau_mean": float(tau.mean()),
            "tau_peak": float(tau.max()),
            "sat_frac": float(sat),
            "qd_p99": float(np.percentile(qd, 99)),
            "qd_max": float(qd.max()),
            "over_vel_frac": float(ovr),
        }
    print(f"\n[arm] task={args_cli.task} num_envs={n} 臂关节={arm_names}")
    print(f"[arm] 臂 PD/延迟 = {gains}")
    print(f"{'目标偏移':<20}{'位置误差cm':>11}{'姿态°':>8}{'|tau|均':>9}{'|tau|峰':>9}{'饱和%':>8}{'|qd|p99':>9}{'超速%':>8}")
    for k, r in results.items():
        print(f"{str(tuple(r['offset'])):<20}{r['pos_err_cm']:>11.2f}{r['ori_err_deg']:>8.1f}"
              f"{r['tau_mean']:>9.2f}{r['tau_peak']:>9.1f}{100 * r['sat_frac']:>8.1f}"
              f"{r['qd_p99']:>9.2f}{100 * r['over_vel_frac']:>8.1f}")
    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as f:
            json.dump({"label": args_cli.label, "task": args_cli.task, "pd": gains,
                       "waypoints": results}, f, ensure_ascii=False, indent=2)
        print(f"[arm] 指标写到 {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
