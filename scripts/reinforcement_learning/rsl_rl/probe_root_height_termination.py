# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：`root_height_below_minimum` 终止到底在"惩罚什么"，以及"只降阈值够不够"。

背景（`docs/review/known_issues.md` #19、用户 2026-09-19 的 20k run）：
`bad_orientation_2` 已经降到 0.7%，但 `root_height_below_minimum` 还有 ~35%；同一份日志里
`Metrics/body_pose/height_error_bias` 长期是 **+0.08~+0.21 m**，而
`height_error = 命令 − 实际` ⇒ **机器人系统性比命令低 8~21 cm**。
命令区间是 (0.33, 0.60)，终止阈值是 0.30 —— 所以"命令贴近下界时必然掉到阈值以下"。

本探针用**训练好的策略**做 rollout，把三件事测出来：

1. **命令 → 实际高度** 的映射（按命令分桶给出偏差、可达下界、`root_z` 分布）；
2. **终止瞬间的画像**：那一刻的命令、实际高度、倾角 —— 区分"只是压低了"与"真摔了"；
3. **阈值反事实**：把高度/倾角终止都关掉、跑同一份策略，统计
   "如果阈值取 0.24/0.26/0.28/0.30/0.32，会有多少比例的环境在记录窗口内被判终止"，
   以及"命令需要抬到多少才安全" —— 直接回答"单纯下调阈值够不够"。

用法（必须 headless；一个进程只建一个 Isaac env）::

    python scripts/reinforcement_learning/rsl_rl/probe_root_height_termination.py \
        --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 512 --steps 600 \
        --policy logs/rsl_rl/history_adaptation/2026-09-19_09-02-50/exported_deploy_19999/policy.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="分析 root_height_below_minimum 终止的成因")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="History-Adaptation-Deeprobotics-M20-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=600, help="rollout 的环境步数（低层 env，step_dt 见打印）")
parser.add_argument("--policy", type=str, required=True, help="部署态低层策略（含 policy_layout.json）")
parser.add_argument("--keep_push", action="store_true", default=False,
                    help="保留 push/外力事件（默认关掉，便于隔离‘只是跟踪误差’）")
parser.add_argument("--action_noise_std", type=float, default=0.0,
                    help="给策略输出加高斯噪声（训练时 rsl_rl 用 noise_std≈1.0 采样动作；"
                         "0 = 确定性推理）")
parser.add_argument("--freeze_ee_preset", type=str, default="none", choices=("none", "default", "low"),
                    help="把 EE 目标锁死：none=按任务采样；default=默认（举起）位姿；"
                         "low=低位锚点（任务工作空间中心）。用来量机械臂对高度终止的贡献")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import (
    compute_base_height_rel_to_feet,
)  # noqa: E402


def read_policy_layout(policy_path: str) -> dict:
    """读导出产物旁边的 policy_layout.json（没有就返回空 dict）。

    这里不 import 高层的 low_level_replay：本探针要能在**低层训练分支**
    （codex/ll-history-flat-eegoal，没有那个模块）上直接跑。
    """
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(policy_path)), "policy_layout.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float().flatten(), q))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    # ── 关掉"终止"与"超时复位"，让同一份策略自然地跑完窗口，便于做阈值反事实 ──
    min_height_cfg = env_cfg.terminations.root_height_below_minimum
    min_height = 0.3 if min_height_cfg is None else float(min_height_cfg.params["minimum_height"])
    limit_angle_cfg = env_cfg.terminations.bad_orientation_2
    limit_angle = 0.8 if limit_angle_cfg is None else float(limit_angle_cfg.params.get("limit_angle", 0.8))
    for name in ("root_height_below_minimum", "bad_orientation_2", "terrain_out_of_bounds"):
        if getattr(env_cfg.terminations, name, None) is not None:
            setattr(env_cfg.terminations, name, None)
    # 曲线在 25k 步（≈1042 iter）就把 body_pose 区间放到 (0.33, 0.60) 了，而课程只在
    # **复位时**触发；本探针为了做阈值反事实要让"复位"不干扰，所以直接把终态区间写进 cfg，
    # 复现"训练结束时的命令分布"（否则会把 height_range 卡在初始的 (0.513, 0.513)）。
    env_cfg.commands.body_pose.height_range = (0.33, 0.60)
    env_cfg.commands.body_pose.pitch_range = (-0.35, 0.35)
    env_cfg.commands.body_pose.roll_range = (-0.25, 0.25)
    # 速度命令用 cfg 里声明的 (±1) 口径（不再叠加 lin_vel_x 课程到 ±5，避免引入无关变量）
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    if args_cli.freeze_ee_preset != "none":
        same = lambda v: (v, v)  # noqa: E731
        if args_cli.freeze_ee_preset == "default":
            # `probe_ee_default_pose.py` 实测的默认位姿：r0=0.4035 m、仰角 +1.2615 rad、
            # 方位 +0.1739 rad（机械臂是"举起"的：EE 在采样平面之上 0.32 m）
            env_cfg.commands.ee_pose.ranges.p_l = same(0.4035)
            env_cfg.commands.ee_pose.ranges.p_pitch = same(1.2615)
            env_cfg.commands.ee_pose.ranges.p_yaw = same(0.1739)
        else:  # low：任务工作空间的中心（低位、前伸），= 课程 s0 想锁的锚点
            env_cfg.commands.ee_pose.ranges.p_l = same(0.41)
            env_cfg.commands.ee_pose.ranges.p_pitch = same(-0.08)
            env_cfg.commands.ee_pose.ranges.p_yaw = same(0.0)
        env_cfg.commands.ee_pose.ranges.o_roll = same(0.0)
        env_cfg.commands.ee_pose.ranges.o_pitch = same(0.0)
        env_cfg.commands.ee_pose.ranges.o_yaw = same(0.0)
    env_cfg.episode_length_s = 1.0e6          # 不超时（反事实按 20 s 窗口另算）
    if not args_cli.keep_push:
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.randomize_push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    policy = torch.jit.load(args_cli.policy, map_location=env.device).to(env.device).eval()
    layout = read_policy_layout(args_cli.policy) or {}
    is_history = layout.get("kind") == "history"

    body_pose = env.command_manager.get_term("body_pose")
    robot = env.scene["robot"]
    feet_cfg = body_pose.cfg.feet_cfg

    print("\n" + "=" * 96)
    print(f"[probe] task={args_cli.task} num_envs={env.num_envs} steps={args_cli.steps} "
          f"step_dt={env.step_dt:.4f}s（= {args_cli.steps * env.step_dt:.1f}s 窗口）")
    print(f"[probe] policy={args_cli.policy} kind={layout.get('kind')}")
    print(f"[probe] body_pose.height_range={tuple(body_pose.cfg.height_range)} "
          f"resampling={tuple(body_pose.cfg.resampling_time_range)}")
    print(f"[probe] 本轮要分析的终止阈值：root_height_below_minimum={min_height} "
          f"bad_orientation_2.limit_angle={limit_angle}")

    env.reset()
    n = env.num_envs
    rec_hcmd, rec_hmeas, rec_rootz, rec_tilt = [], [], [], []
    first_hit = {thr: torch.full((n,), -1.0, device=env.device) for thr in (0.24, 0.26, 0.28, 0.30, 0.32)}
    below_frac = {thr: 0 for thr in first_hit}
    n_samples = 0
    first_tilt = torch.full((n,), -1.0, device=env.device)
    # 终止画像：按"当前命令高度是否贴近下界"分开统计，便于区分"压低"与"塌了"
    term_profile = {"h_cmd": [], "h_meas": [], "root_z": [], "tilt": [], "err": []}
    n_term_h = 0

    for step in range(args_cli.steps):
        obs = env.observation_manager.compute()
        with torch.no_grad():
            if is_history:
                actions = policy(obs["policy"], obs["history"])
            else:
                actions = policy(obs["policy"])
            if args_cli.action_noise_std > 0.0:
                actions = actions + args_cli.action_noise_std * torch.randn_like(actions)
        env.step(actions)

        h_cmd = body_pose.command[:, 0]
        h_meas = compute_base_height_rel_to_feet(env, body_pose.cfg.asset_cfg, feet_cfg)
        root_z = robot.data.root_pos_w[:, 2]
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        rec_hcmd.append(h_cmd.clone()); rec_hmeas.append(h_meas.clone())
        rec_rootz.append(root_z.clone()); rec_tilt.append(tilt.clone())
        hit = root_z < min_height
        n_term_h += int(hit.sum())
        for ids in (hit.nonzero(as_tuple=False).squeeze(-1),):
            if ids.numel() > 0:
                term_profile["h_cmd"].append(h_cmd[ids].clone())
                term_profile["h_meas"].append(h_meas[ids].clone())
                term_profile["root_z"].append(root_z[ids].clone())
                term_profile["tilt"].append(tilt[ids].clone())
        for thr, buf in first_hit.items():
            newly = (root_z < thr) & (buf < 0)
            buf[newly] = step
            below_frac[thr] += int((root_z < thr).sum())
        n_samples += int(root_z.numel())
        newly = (tilt > limit_angle) & (first_tilt < 0)
        first_tilt[newly] = step

    h_cmd = torch.cat(rec_hcmd); h_meas = torch.cat(rec_hmeas)
    root_z = torch.cat(rec_rootz); tilt = torch.cat(rec_tilt)
    err = h_cmd - h_meas          # 与 Metrics/body_pose/height_error 同号（命令 − 实际）

    print("\n" + "-" * 96)
    print("[probe] (1) 命令 vs 实际（height 用 compute_base_height_rel_to_feet，root_z 是终止项的判据）")
    print(f"    height_error = 命令 − 实际 : mean(偏差) = {float(err.mean()):+.4f} m  "
          f"MAE = {float(err.abs().mean()):.4f} m  p95(|误差|) = {_pct(err.abs(), 0.95):.4f}")
    print(f"    root_z : min={float(root_z.min()):.4f}  p01={_pct(root_z,0.01):.4f}  "
          f"p05={_pct(root_z,0.05):.4f}  mean={float(root_z.mean()):.4f} m")
    print(f"    倾角   : mean={math.degrees(float(tilt.mean())):.2f}°  "
          f"p99={math.degrees(_pct(tilt,0.99)):.2f}°  max={math.degrees(float(tilt.max())):.2f}°")
    lo, hi = body_pose.cfg.height_range
    edges = [lo + (hi - lo) * k / 6 for k in range(7)]
    print("\n    按命令分桶：")
    print(f"      {'命令区间':<16}{'样本占比':>9}{'平均偏差(命令-实际)':>20}"
          f"{'实际高度p05':>13}{'root_z p05':>12}{'root_z<0.30 比例':>17}")
    for k in range(6):
        m = (h_cmd >= edges[k]) & (h_cmd < (edges[k + 1] + (1e-6 if k == 5 else 0)))
        if int(m.sum()) == 0:
            continue
        frac = float(m.float().mean())
        print(f"      [{edges[k]:.2f},{edges[k+1]:.2f})  {frac:>8.1%}  {float(err[m].mean()):>+18.4f} m "
              f"{_pct(h_meas[m],0.05):>12.4f} {_pct(root_z[m],0.05):>11.4f} "
              f"{float((root_z[m] < 0.30).float().mean()):>16.1%}")

    print("\n" + "-" * 96)
    print("[probe] (2) 高度终止瞬间画像（root_z < %.2f，共 %d 个 env·step）" % (min_height, n_term_h))
    if term_profile["root_z"]:
        pz = torch.cat(term_profile["root_z"]); pc = torch.cat(term_profile["h_cmd"])
        pm = torch.cat(term_profile["h_meas"]); pt = torch.cat(term_profile["tilt"])
        print(f"    命令高度  : mean={float(pc.mean()):.4f}  p05={_pct(pc,0.05):.4f}  p95={_pct(pc,0.95):.4f}")
        print(f"    实际高度  : mean={float(pm.mean()):.4f}  （比命令低 {float((pc-pm).mean()):.4f} m）")
        print(f"    root_z    : mean={float(pz.mean()):.4f}  min={float(pz.min()):.4f}")
        print(f"    倾角      : mean={math.degrees(float(pt.mean())):.2f}°  "
              f"p90={math.degrees(_pct(pt,0.9)):.2f}°  "
              f"超过 {math.degrees(limit_angle):.1f}° 的比例={float((pt > limit_angle).float().mean()):.1%}")
        also_tilted = float((pt > limit_angle).float().mean())
        upright_low = float((pt <= math.radians(15)).float().mean())
        print(f"    -> 其中【倾角也超限】= {also_tilted:.1%}（真摔），"
              f"【倾角正常但压得低】= {upright_low:.1%}")
    else:
        print("    （窗口内没有触发）")

    print("\n" + "-" * 96)
    print("[probe] (3) 阈值反事实：关掉终止后跑同一份策略，看各阈值下被判终止的环境比例")
    print(f"      {'阈值':>6}{'瞬时占比':>12}{'20s 内触发过的环境比例':>26}{'平均首次触发(s)':>18}")
    for thr, buf in first_hit.items():
        frac_inst = below_frac[thr] / max(n_samples, 1)
        frac_ep = float((buf >= 0).float().mean())
        tt = buf[buf >= 0]
        tt_str = f"{float(tt.mean()) * env.step_dt:.2f}" if tt.numel() else "-"
        flag = "  ← 当前" if abs(thr - min_height) < 1e-9 else ""
        print(f"      {thr:>6.2f}{frac_inst:>11.1%}{frac_ep:>25.1%}{tt_str:>18}{flag}")
    print(f"      倾角>{math.degrees(limit_angle):.1f}°（bad_orientation_2）比例 = "
          f"{float((first_tilt >= 0).float().mean()):.1%}")
    print(f"\n    结论提示：若'root_z<0.26 的比例'仍与 0.30 接近，说明降低阈值只是把门槛往下挪、")
    print(f"    并不能减少'压得过低'这件事；反过来若 0.26 明显更小，说明阈值确实是主要瓶颈。")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
    os._exit(0)
