# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""夹爪 PD 参数（刚度/阻尼）的**动态**验收：方波阶跃响应 + 力矩饱和统计。

为什么需要它（B2-④ / DEF-058）
------------------------------
`policy_report.py` 只能给"整段测试里的 |tau| 均值 / 饱和占比"，看不到**阶跃响应**
（上升时间、超调、是否到位）。而改夹爪 PD 参数最大的风险正是动态：太软 ⇒ 夹不到底、
来不及闭合；太硬 ⇒ 一直在力矩限幅上（假电流、假磨损）。

本探针直接给夹爪两个关节下发**方波目标**（闭 → 开 → 闭，各 `--hold-s` 秒），
逐步记录 `joint_pos / joint_vel / applied_torque`，逐段输出：

* **上升时间**（到目标 90%）、**超调**（相对行程）、**稳态误差**；
* `|tau|` 峰值与**饱和时间占比**（≥99% 限幅）；
* `|qd|` 峰值与超速时间占比。

用法（同一任务、同一 seed，只改执行器参数做 A/B）::

    python scripts/reinforcement_learning/rsl_rl/probe_gripper_response.py --headless --label now
    python scripts/reinforcement_learning/rsl_rl/probe_gripper_response.py --headless --label old \
        env.scene.robot.actuators.piper_gripper.stiffness=4000.0 \
        env.scene.robot.actuators.piper_gripper.damping=200.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="夹爪 PD 阶跃响应探针")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default="Flat-Deeprobotics-M20-Piper-WBC-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hold-s", type=float, default=1.5, help="每段保持时长（秒）")
parser.add_argument("--open-target", type=float, default=0.04,
                    help="开目标（高层 BinaryJointPositionAction 用 ±0.04，注意行程只有 ±0.035）")
parser.add_argument("--label", type=str, default="run")
parser.add_argument("--out", type=str, default=None, help="把指标写成 JSON")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def _seg_metrics(t, q, qd, tau, start_q, target, lim, vlim) -> dict:
    """一段方波的响应指标（`target` = 该段目标，`start_q` = 段初实际位置）。"""
    span = float(target) - float(start_q)
    sign = 1.0 if span >= 0 else -1.0
    reach = float(start_q) + 0.9 * span
    hit = np.nonzero(sign * (q - reach) >= 0)[0]
    rise = float(t[hit[0]]) if hit.size else float("nan")
    ideal = max(abs(span), 1e-9)
    over = float(max(0.0, sign * (q.max() if sign > 0 else -q.min()) - sign * target)) / ideal
    tail = q[t >= t[-1] - 0.2] if t.size else q
    a = np.abs(tau)
    return {
        "target": float(target),
        "start_q": float(start_q),
        "rise_s": rise,
        "overshoot_frac": float(over),
        "ss_error": float(abs(tail.mean() - float(target))),
        "tau_peak": float(a.max()),
        "tau_mean": float(a.mean()),
        "sat_frac": float((a >= 0.99 * lim).mean()),
        "qd_peak": float(np.abs(qd).max()),
        "over_vel_frac": float((np.abs(qd) > vlim).mean()) if vlim > 0 else float("nan"),
    }


def _travel_sign(robot, jid: int) -> float:
    """两指行程方向相反（gripper_joint1: [0,+0.035]、gripper_joint2: [-0.035,0]）。"""
    lim = robot.data.joint_pos_limits[0, jid]
    return 1.0 if float(lim[1]) > 0 else -1.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    dt = unwrapped.step_dt
    n = unwrapped.num_envs
    ids = [int(i) for i in robot.find_joints(["gripper_joint[1-2]"])[0]]
    names = [robot.data.joint_names[i] for i in ids]

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
    lim = float(np.nanmedian([eff.get(x, np.nan) for x in names]))
    vlim = float(np.nanmedian([vel.get(x, np.nan) for x in names]))

    # 打印当前 PD 参数（做 A/B 时要看得见）
    gains = {}
    for act in robot.actuators.values():
        if "gripper" in "".join(getattr(act, "joint_names", []) or []):
            for attr in ("stiffness", "damping"):
                buf = getattr(act, attr, None)
                if buf is not None:
                    gains[attr] = float(torch.as_tensor(buf, dtype=torch.float32).flatten()[0])
    print(f"[grip] task={args_cli.task} num_envs={n} 关节={names} "
          f"PD={gains} 限幅={lim:.1f} N·m/{vlim:.2f} rad/s")

    total_dim = unwrapped.action_manager.total_action_dim
    # 下肢/底盘动作锁零（本探针只看夹爪）；机械臂 IK 仍由命令项驱动
    zero = torch.zeros((n, total_dim), device=unwrapped.device)
    obs = env.get_observations()

    results = {}
    for jid, name in zip(ids, names):
        sign = _travel_sign(robot, jid)
        plan = [(sign * 0.0, args_cli.hold_s), (sign * args_cli.open_target, args_cli.hold_s),
                (sign * 0.0, args_cli.hold_s)]
        rec_t, rec_q, rec_qd, rec_tau, rec_tgt = [], [], [], [], []
        t = 0.0
        jid_t = torch.tensor([jid], device=unwrapped.device)
        for target, dur in plan:
            for _ in range(int(round(dur / dt))):
                robot.set_joint_position_target(
                    torch.full((n, 1), float(target), device=unwrapped.device), jid_t)
                with torch.inference_mode():
                    obs, _, _, _ = env.step(zero)
                rec_t.append(t); t += dt
                rec_tgt.append(float(target))
                rec_q.append(float(robot.data.joint_pos[0, jid]))
                rec_qd.append(float(robot.data.joint_vel[0, jid]))
                rec_tau.append(float(robot.data.applied_torque[0, jid]))

        t_a, q_a = np.array(rec_t), np.array(rec_q)
        qd_a, tau_a, tgt_a = np.array(rec_qd), np.array(rec_tau), np.array(rec_tgt)
        m = int(round(args_cli.hold_s / dt))
        print(f"\n[grip] {name}（env0，行程 {float(robot.data.joint_pos_limits[0, jid, 0]):+.3f}"
              f"~{float(robot.data.joint_pos_limits[0, jid, 1]):+.3f}）：")
        for label, i0 in (("闭→(初始)", 0), ("开(±0.04)", m), ("再闭(0)", 2 * m)):
            i1 = min(i0 + m, len(t_a))
            if i1 - i0 < 5:
                continue
            mm = _seg_metrics(t_a[i0:i1] - t_a[i0], q_a[i0:i1], qd_a[i0:i1], tau_a[i0:i1],
                              q_a[i0], tgt_a[i0], lim, vlim)
            print(f"   {label:<10} 目标{mm['target']:+.4f}（起 {mm['start_q']:+.4f}）  "
                  f"上升 {mm['rise_s']:.3f} s  超调 {100 * mm['overshoot_frac']:.0f}%  "
                  f"稳态误差 {mm['ss_error']:.5f}  |  |tau| 峰 {mm['tau_peak']:5.1f} 均 {mm['tau_mean']:5.2f} N·m  "
                  f"饱和 {100 * mm['sat_frac']:5.1f}%  |  |qd| 峰 {mm['qd_peak']:.2f} rad/s  "
                  f"超速 {100 * mm['over_vel_frac']:4.0f}%")
            results[f"{name}_{i0 // m}"] = mm

    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as f:
            json.dump({"label": args_cli.label, "task": args_cli.task,
                       "pd": gains, "metrics": results}, f, ensure_ascii=False, indent=2)
        print(f"[grip] 指标写到 {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
