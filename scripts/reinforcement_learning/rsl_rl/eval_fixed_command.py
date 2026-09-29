# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""固定命令 eval：把 vx/vy/wz 钉死，测"同命令下"的回报、速度误差与终止构成。

为什么需要它
------------
`DEF-024` 的验收口径 (b) 是"固定命令 eval（固定 vx/vy/yaw）下 reward 与速度误差不劣于基线"。
不能拿训练日志里的 `Train/mean_reward` 当这个数字：那是**在随机命令课程上**的均值，
而命令范围随课程一路放宽（`0.1×range → vx ±2 → ±3 → ±4 → ±5 m/s`，见
`flat_env_wbc_cfg.py:WBCCurriculumCfg.base_velocity_lin_vel_x_s4..s7`）——
跨 run 比它等于"比当时采到哪些命令"，不是比策略。本脚本把命令钉在给定值上滚动，给出可比的：

* 每回合回报 / 回合长度 / 每秒回报（与 `Train/mean_reward[_/time]` 同口径，但命令固定）；
* 线速度 xy 误差与 yaw 角速度误差（与 `Metrics/base_velocity/error_vel_*` 同定义）；
* 终止构成（倾角 / 高度 / 出界 / 超时）。

怎么保证"命令真的固定"
----------------------
* `resampling_time_range=(1e9, 1e9)` **并且**把命令项的 `_resample_command` 换成 no-op
  ⇒ 回合 reset 时也不会重采样；每个 control step 前还会重写一次 `vel_command_b`（双保险）。
* 训练用的是 `heading_command=True`（yaw 由航向控制器生成），eval 改成**直接给 yaw 角速度**
  （= 部署语义）；`rel_standing_envs` / `rel_heading_envs` 置 0，避免某些 env 被标成"站立"。
* 命令切换处把 `episode_length_buf` 顶到上限，强制所有 env 在同一回合边界重开，
  这样"上一档命令的回合"不会混进这一档的统计。

难度口径
--------
默认用 `-play-v0` 任务：它的 cfg 直接把 EE 目标 / 身体姿态区间放到**完整任务**（≈ 训练 s3），
并关掉 EE 目标课程与扰动课程，避免 eval 时课程从 s0 重新爬（否则"最难段谁更稳"根本测不出来）。
要测别的难度就换 `--task`（例如训练任务本身 = s0 难度，或 `Rough-...` 版本）。
push / reset 外力**保留**（= 任务原始强度），只在观测侧关掉噪声（确定性推理）。

用法
----
    python scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py \
        --headless --num_envs 512 --steps 2000 --seed 42 \
        --checkpoint logs/rsl_rl/history_adaptation/2026-09-20_18-54-34_cap_noise_std/model_3999.pt

    # 多档命令（分号分隔）与 JSON 落盘
    ... --commands "1.0,0,0;3.0,0,0;0,0,0;0.8,0.6,0.6" --out /tmp/eval.json

默认三档：巡航 1.0 m/s、高速 3.0 m/s、站立 0（都在训练末段 vx ±4 的范围内）。
每个 checkpoint 一个进程（一个 Isaac env），要比较多个 checkpoint 就并行/串行跑多次，
再 `--out` 出来的 JSON 合并成表。**注意**：同 `--seed` 只保证地形/EE 目标采样序列一致，
策略终止时机不同会让后续 reset 顺序分叉 —— 所以要靠多 env × 多回合平均，而不是逐帧对齐。

**选命令前先看这个 checkpoint"训练时见过多大命令"**（课程是按环境步推进的，
`iter = 步数 / num_steps_per_env(24)`）：

| 训练迭代 | vx 训练范围（时间课程） |
|---|---|
| < 3125 | 由奖励驱动课程从 ±0.1 逐步放宽（`command_levels_vel`） |
| ≥ 3125（75k 步） | ±2 m/s |
| ≥ 4167（100k 步） | ±3 m/s |
| ≥ 5208（125k 步） | ±4 m/s |
| ≥ 6250（150k 步） | ±5 m/s |

命令超出这张表 ⇒ 测到的是**外推能力**而不是跟踪质量（实测：4000-iter 的 checkpoint 在
3.0 m/s 下 reward 掉到 8.5、摔倒 25.6%，而 20000-iter 的同一策略在 3.0 m/s 下是 25.2 / 11.0%）。
所以：**跨 checkpoint 比较必须用同一个 --num_envs / --seed / --commands**，
且命令要落在**被比较双方都已经历过**的范围内。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

# 本地 cli_args（--checkpoint / --seed / --logger 等 rsl_rl 参数）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

DEFAULT_COMMANDS = "1.0,0,0;3.0,0,0;0,0,0"

parser = argparse.ArgumentParser(description="固定命令 eval 探针（回报 / 速度误差 / 终止构成）")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="History-Adaptation-Deeprobotics-M20-play-v0",
    help="任务名；默认用 -play- 版本（完整难度、无课程漂移）",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--steps",
    type=int,
    default=1100,
    help="每档命令记录的 control step 数；默认 1100 ≥ max_episode_length(1000)，保证每个 env 至少跑完 1 个回合",
)
parser.add_argument("--warmup", type=int, default=100, help="每档命令开始时丢弃的步数（重填 history / 过度瞬态）")
parser.add_argument(
    "--commands",
    type=str,
    default=DEFAULT_COMMANDS,
    help='分号分隔的固定命令，元素是 "vx,vy,wz"（m/s, m/s, rad/s）',
)
parser.add_argument("--label", type=str, default=None, help="表里的策略名（默认取 checkpoint 的父目录名）")
parser.add_argument("--out", type=str, default=None, help="把结果 JSON 写到这里")
# append RSL-RL cli arguments / AppLauncher cli args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# --checkpoint 由 cli_args.add_rsl_rl_args 提供（默认 None），这里只做"必填"校验
if not args_cli.checkpoint:
    parser.error("必须给 --checkpoint <run 目录>/model_<iter>.pt（训练 checkpoint，不是 exported_deploy/）")
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerHis  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def parse_commands(spec: str) -> list[tuple[float, float, float]]:
    """``"1.0,0,0;0,0,0"`` → ``[(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]``。"""
    out: list[tuple[float, float, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x) for x in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError(f"命令格式应为 'vx,vy,wz'，收到 {chunk!r}")
        out.append((vals[0], vals[1], vals[2]))
    if not out:
        raise ValueError("至少要给一档命令")
    return out


def _fmt(value: float | None, spec: str = ".3f") -> str:
    """None（该档命令没有回合跑完）不报错，直接打 ``n/a``。"""
    return "n/a" if value is None else format(value, spec)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """固定命令滚动一批 env，打印并（可选）保存统计。"""
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- 固定命令：关重采样 + 直接给 yaw 角速度（部署语义）+ 观测去噪 ----
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False

    # ---- 地形：与 play.py 同一口径（关地形课程 + 固定 5x5 网格，同 seed 可复现）----
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    dt = unwrapped.step_dt
    n_envs = unwrapped.num_envs
    device = unwrapped.device

    ckpt = os.path.abspath(args_cli.checkpoint)
    label = args_cli.label or os.path.basename(os.path.dirname(ckpt))
    print(f"[INFO] task={task_name}  label={label}")
    print(f"[INFO] checkpoint={ckpt}")
    print(f"[INFO] num_envs={n_envs}  step_dt={dt:.4f}s  steps/cmd={args_cli.steps}  warmup={args_cli.warmup}")

    # load previously trained model（和 play.py 同一条路径：runner + 推理策略）
    runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)

    term = unwrapped.command_manager.get_term("base_velocity")
    # 回合 reset 也不重采样（每个 step 前还会重写一次命令缓冲，双保险）
    term._resample_command = lambda env_ids: None  # noqa: ARG005

    robot = unwrapped.scene["robot"]
    term_names = list(unwrapped.termination_manager.active_terms)
    print(f"[INFO] 终止项：{term_names}")

    commands = parse_commands(args_cli.commands)
    obs = env.get_observations()
    rows: list[dict] = []

    for cmd in commands:
        cmd_t = torch.tensor(cmd, device=device, dtype=torch.float32).repeat(n_envs, 1)

        def pin_and_step(actions: torch.Tensor | None):
            """写死命令 → （可选）施加动作 → 返回新 obs。"""
            term.vel_command_b[:] = cmd_t
            if actions is None:
                actions = torch.zeros(n_envs, env.num_actions, device=device)
            with torch.inference_mode():
                new_obs, _, _, _ = env.step(actions)
            return new_obs

        def force_episode_boundary(new_obs):
            """把 episode_length_buf 顶到上限 ⇒ 下一步所有 env 统一 reset（对齐回合边界）。"""
            unwrapped.episode_length_buf[:] = unwrapped.max_episode_length
            return pin_and_step(policy(new_obs))

        # 1) 清场（丢弃上一档命令留下的回合）→ 2) 预热（重填 history、让瞬态过去）→ 3) 再清场
        term.vel_command_b[:] = cmd_t
        obs = force_episode_boundary(obs)
        for _ in range(args_cli.warmup):
            obs = pin_and_step(policy(obs))
        obs = force_episode_boundary(obs)

        ret = torch.zeros(n_envs, device=device)
        ep_len = torch.zeros(n_envs, device=device)
        err_xy_sum = torch.zeros(n_envs, device=device)
        err_yaw_sum = torch.zeros(n_envs, device=device)
        ep_returns: list[float] = []
        ep_lengths: list[float] = []
        steps_xy: list[torch.Tensor] = []
        steps_yaw: list[torch.Tensor] = []
        term_counts = {name: 0 for name in term_names}

        for _ in range(args_cli.steps):
            # 误差用"动作施加前"的状态算，避免把 done env 的 reset 状态算进来
            lin_vel = robot.data.root_lin_vel_b[:, :2]
            ang_vel_z = robot.data.root_ang_vel_b[:, 2]
            err_xy = torch.norm(cmd_t[:, :2] - lin_vel, dim=-1)
            err_yaw = torch.abs(cmd_t[:, 2] - ang_vel_z)
            term.vel_command_b[:] = cmd_t
            with torch.inference_mode():
                obs, rew, dones, _ = env.step(policy(obs))
            ret += rew
            ep_len += 1.0
            err_xy_sum += err_xy
            err_yaw_sum += err_yaw
            steps_xy.append(err_xy.mean().detach())
            steps_yaw.append(err_yaw.mean().detach())

            done = dones.bool()
            if done.any():
                for name in term_names:
                    term_counts[name] += int(unwrapped.termination_manager.get_term(name)[done].sum())
                for i in done.nonzero(as_tuple=True)[0].tolist():
                    ep_returns.append(float(ret[i]))
                    ep_lengths.append(float(ep_len[i]))
                ret[done] = 0.0
                ep_len[done] = 0.0
                err_xy_sum[done] = 0.0
                err_yaw_sum[done] = 0.0

        n_ep = len(ep_returns)
        row = {
            "label": label,
            "checkpoint": ckpt,
            "task": task_name,
            "command": {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]},
            "episodes": n_ep,
            "ep_return": float(sum(ep_returns) / n_ep) if n_ep else None,
            "ep_length": float(sum(ep_lengths) / n_ep) if n_ep else None,
            "return_per_s": (
                float(sum(ep_returns) / n_ep) / (float(sum(ep_lengths) / n_ep) * dt) if n_ep else None
            ),
            "err_vel_xy": float(torch.stack(steps_xy).mean()),
            "err_vel_yaw": float(torch.stack(steps_yaw).mean()),
            "termination_rate": {k: (v / n_ep if n_ep else None) for k, v in term_counts.items()},
            "termination_counts": term_counts,
        }
        rows.append(row)
        print(
            f"[RESULT] cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  "
            f"episodes={n_ep}  ep_return={_fmt(row['ep_return'])}  ep_len={_fmt(row['ep_length'], '.1f')}  "
            f"err_xy={row['err_vel_xy']:.4f}  err_yaw={row['err_vel_yaw']:.4f}  "
            f"摔倒={sum(term_counts.get(k, 0) for k in ('bad_orientation_2', 'root_height_below_minimum')) / max(n_ep, 1):.4f}"
        )

    # ---------------- 汇总表（Markdown） ----------------
    print("\n### 固定命令 eval —— " + label)
    print()
    print("| 命令 (vx,vy,wz) | 回合数 | 每回合回报 | 回合长度 | 回报/秒 | `error_vel_xy` | `error_vel_yaw` | 摔倒率 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["command"]
        fall = sum(r["termination_counts"].get(k, 0) for k in ("bad_orientation_2", "root_height_below_minimum"))
        fall_rate = fall / r["episodes"] if r["episodes"] else float("nan")
        print(
            f"| ({c['vx']:+.2f},{c['vy']:+.2f},{c['wz']:+.2f}) | {r['episodes']} | "
            f"{_fmt(r['ep_return'])} | {_fmt(r['ep_length'], '.1f')} | {_fmt(r['return_per_s'])} | "
            f"{r['err_vel_xy']:.4f} | {r['err_vel_yaw']:.4f} | {fall_rate:.4f} |"
        )
    print()
    print("[INFO] 终止构成（次数）：")
    for r in rows:
        c = r["command"]
        print(f"  - ({c['vx']:+.2f},{c['vy']:+.2f},{c['wz']:+.2f}) {r['termination_counts']}")

    if args_cli.out:
        payload = {
            "label": label,
            "checkpoint": ckpt,
            "task": task_name,
            "num_envs": n_envs,
            "seed": args_cli.seed,
            "steps_per_command": args_cli.steps,
            "warmup": args_cli.warmup,
            "step_dt": dt,
            "results": rows,
        }
        out_path = os.path.abspath(args_cli.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已写入 {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
