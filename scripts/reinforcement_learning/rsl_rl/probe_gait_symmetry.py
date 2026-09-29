# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""步态对称性探针：把"撇腿"变成一个可比的数字。

为什么需要它
------------
用户反馈"训出的模型右后腿往右前方撇"，而训练日志里没有**任何**直接度量"左右对称"的指标
（`Episode_Reward/joint_mirror` 是被奖励权重污染过的标量，而且它自己原来就是错的，
见 docs/review/DEFECT_LOG_zh.md DEF-027）。本脚本固定命令滚动若干步，直接测三组与
"对称"有关的物理量：

1. **每腿关节角均值**（hipx / hipy / knee）—— 最直观的"哪条腿撇出去"；
2. **镜像误差 RMS**（4 个镜像对：对角 fl↔hr、fr↔hl + 左右 fl↔fr、hl↔hr）；
   符号约定取自本机型的真实镜像关系（关节名后缀 → ±1，理由见 DEF-027 §2）；
3. **足端 body 系坐标**（4 个轮的 x/y）+ 两个**左右不对称度**
   `a_front = y_fl + y_fr`、`a_hind = y_hl + y_hr`（完全对称时 = 0；
   一侧单独外撇时非零）。

用法
----
    python scripts/reinforcement_learning/rsl_rl/probe_gait_symmetry.py \
        --headless --num_envs 256 --steps 800 --commands "1.0,0,0;0,0,0" \
        --checkpoint <run>/model_19999.pt --label baseline_20k \
        --out logs/smoke/gait_baseline_20k.json

* 每个 checkpoint 一个进程（一个 Isaac env），要对比就串行/并行跑多次再比 `--out` 的 JSON。
* 命令写在 `--commands`（分号分隔的 `vx,vy,wz`）；观测侧噪声关闭（确定性推理），
  push / reset 外力保留（与训练口径一致）。
* **注意**：本脚本不判"好坏"，只给数字；判据是"同一条命令下、同一 num_envs/seed 下
  镜像误差 RMS 与不对称度是否下降"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

# 本地 cli_args（--checkpoint / --seed / --logger 等 rsl_rl 参数）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

LEG_NAMES = ["fl", "fr", "hl", "hr"]
JOINT_TYPES = ["hipx", "hipy", "knee"]

# 本机型的镜像关系（DEF-027 §2 有三重证据）：
#   对角对（绕 z 轴 180°）：hipx/hipy/knee 全部取负
#   左右对：hipx 取负、hipy/knee 不变
MIRROR_PAIRS = [
    ("fl", "hr", {"hipx": -1.0, "hipy": -1.0, "knee": -1.0}),
    ("fr", "hl", {"hipx": -1.0, "hipy": -1.0, "knee": -1.0}),
    ("fl", "fr", {"hipx": -1.0, "hipy": 1.0, "knee": 1.0}),
    ("hl", "hr", {"hipx": -1.0, "hipy": 1.0, "knee": 1.0}),
]

DEFAULT_COMMANDS = "1.0,0,0;0,0,0"

parser = argparse.ArgumentParser(description="步态对称性探针（关节角 / 镜像误差 / 足端坐标）")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="History-Adaptation-Deeprobotics-M20-play-v0",
    help="任务名；默认用 -play- 版本（完整难度、无课程漂移）",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=800, help="每档命令记录的 control step 数")
parser.add_argument("--warmup", type=int, default=200, help="每档命令开始前丢弃的步数")
parser.add_argument("--commands", type=str, default=DEFAULT_COMMANDS, help='分号分隔的固定命令 "vx,vy,wz"')
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--out", type=str, default=None, help="把结果 JSON 写到这里")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("必须给 --checkpoint <run 目录>/model_<iter>.pt")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerHis  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def parse_commands(spec: str) -> list[tuple[float, float, float]]:
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    n_envs = unwrapped.num_envs
    device = unwrapped.device

    ckpt = os.path.abspath(args_cli.checkpoint)
    label = args_cli.label or os.path.basename(os.path.dirname(ckpt))
    print(f"[INFO] task={task_name}  label={label}")
    print(f"[INFO] checkpoint={ckpt}")
    print(f"[INFO] num_envs={n_envs}  steps/cmd={args_cli.steps}  warmup={args_cli.warmup}")

    runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)

    term = unwrapped.command_manager.get_term("base_velocity")
    term._resample_command = lambda env_ids: None  # noqa: ARG005
    robot = unwrapped.scene["robot"]

    # 关节 / 足端索引（按关节名映射，别用下标猜：本仓库有三套顺序）
    joint_ids: dict[str, list[int]] = {}
    for leg in LEG_NAMES:
        for jt in JOINT_TYPES:
            ids, names = robot.find_joints(f"{leg}_{jt}_joint")
            if len(ids) != 1:
                raise RuntimeError(f"找不到唯一关节 {leg}_{jt}_joint（匹配到 {names}）")
            joint_ids[f"{leg}_{jt}"] = ids[0]
    # 逐条解析（`find_bodies` 返回的是**按 articulation 顺序**排的，不是传入顺序，
    # 所以不能一次传 4 个名字再按下标对应；必须一条一条解析）
    foot_ids: list[int] = []
    foot_names: list[str] = []
    for leg in LEG_NAMES:
        ids, names = robot.find_bodies(f"{leg}_wheel")
        if len(ids) != 1:  # 兜底：有些资产把轮 body 命名成 fl_wheel_link 之类
            ids, names = robot.find_bodies(f"^{leg}_wheel")
        if len(ids) != 1:
            ids, names = robot.find_bodies(f"{leg}.*wheel")
        if len(ids) != 1:
            raise RuntimeError(f"找不到唯一轮 body {leg}_wheel（匹配到 {names}）")
        foot_ids.append(ids[0])
        foot_names.append(names[0])
    print(f"[INFO] 轮 body：{foot_names}")

    idx_all = torch.tensor([joint_ids[f"{leg}_{jt}"] for leg in LEG_NAMES for jt in JOINT_TYPES], device=device)
    # idx_all 的列序 = [fl_hipx, fl_hipy, fl_knee, fr_*, hl_*, hr_*] ⇒ 列号 = leg_idx*3 + type_idx

    commands = parse_commands(args_cli.commands)
    obs = env.get_observations()
    rows: list[dict] = []

    def step_with(actions_cmd: tuple[float, float, float], obs):
        cmd_t = torch.tensor(actions_cmd, device=device, dtype=torch.float32).repeat(n_envs, 1)
        term.vel_command_b[:] = cmd_t
        with torch.inference_mode():
            new_obs, _, _, _ = env.step(policy(obs))
        return new_obs

    def force_episode_boundary(actions_cmd: tuple[float, float, float], obs):
        """把 ``episode_length_buf`` 顶到上限 ⇒ 下一步所有 env 统一 reset（对齐回合边界）。"""
        term.vel_command_b[:] = torch.tensor(actions_cmd, device=device, dtype=torch.float32).repeat(n_envs, 1)
        unwrapped.episode_length_buf[:] = unwrapped.max_episode_length
        with torch.inference_mode():
            unwrapped.episode_length_buf[:] = unwrapped.max_episode_length
            new_obs, _, _, _ = env.step(policy(obs))
        return new_obs

    for cmd in commands:
        cmd_t = torch.tensor(cmd, device=device, dtype=torch.float32).repeat(n_envs, 1)
        # 清场（丢掉上一档命令留下的回合）→ 预热（重填 history、让瞬态过去）→ 再清场
        obs = force_episode_boundary(cmd, obs)
        for _ in range(args_cli.warmup):
            obs = step_with(cmd, obs)
        obs = force_episode_boundary(cmd, obs)

        joint_acc = torch.zeros(12, device=device, dtype=torch.float64)
        mirror_acc = {f"{a}~{b}": torch.zeros(3, device=device, dtype=torch.float64)
                      for a, b, _ in MIRROR_PAIRS}
        foot_acc = torch.zeros(4, 2, device=device, dtype=torch.float64)
        n = 0
        for _ in range(args_cli.steps):
            q = robot.data.joint_pos[:, idx_all].double()          # (N, 12)
            pos_w = robot.data.body_pos_w[:, foot_ids, :]          # (N, 4, 3)
            root_p = robot.data.root_pos_w
            root_q = robot.data.root_quat_w
            # 注意：`quat_apply_inverse` 是 TorchScript，**只吃 float32**（Double 会报
            # "Found dtype Double but expected Float"），所以这里在 fp32 里变换、
            # 累加时再升到 fp64 求均值。
            foot_b = torch.zeros(n_envs, 4, 3, device=device, dtype=torch.float32)
            for i in range(4):
                foot_b[:, i, :] = math_utils.quat_apply_inverse(root_q, pos_w[:, i, :] - root_p)
            joint_acc += q.mean(dim=0)
            foot_acc += foot_b[:, :, :2].double().mean(dim=0)
            n += 1
            for a, b, signs in MIRROR_PAIRS:
                diff = torch.zeros(3, device=device, dtype=torch.float64)
                for k, jt in enumerate(JOINT_TYPES):
                    ia = LEG_NAMES.index(a) * 3 + k
                    ib = LEG_NAMES.index(b) * 3 + k
                    diff[k] = ((q[:, ia] - signs[jt] * q[:, ib]) ** 2).mean()
                mirror_acc[f"{a}~{b}"] += diff
            obs = step_with(cmd, obs)

        joint_mean = (joint_acc / n).cpu().numpy()
        mirror_mse = {k: (v / n).cpu().numpy() for k, v in mirror_acc.items()}
        foot_mean = (foot_acc / n).cpu().numpy()

        row = {
            "label": label,
            "checkpoint": ckpt,
            "task": task_name,
            "command": {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]},
            "num_envs": n_envs,
            "steps": args_cli.steps,
            # 每腿关节角均值（rad）
            "joint_mean": {f"{leg}_{jt}": float(joint_mean[i * 3 + k])
                           for i, leg in enumerate(LEG_NAMES) for k, jt in enumerate(JOINT_TYPES)},
            # 镜像误差 RMS（rad）：sqrt(mean((θ_a − s·θ_b)²))
            "mirror_rms": {k: [float(np.sqrt(x)) for x in v] for k, v in mirror_mse.items()},
            # 足端 body 系 (x, y) 均值（m）
            "foot_xy_body": {leg: [float(foot_mean[i, 0]), float(foot_mean[i, 1])]
                             for i, leg in enumerate(LEG_NAMES)},
            # 左右不对称度：完全对称时应为 0
            "lateral_asymmetry": {
                "front_fl_plus_fr": float(foot_mean[0, 1] + foot_mean[1, 1]),
                "hind_hl_plus_hr": float(foot_mean[2, 1] + foot_mean[3, 1]),
            },
            # 轮距（同侧前后、左右）
            "stance_width_front": float(foot_mean[0, 1] - foot_mean[1, 1]),
            "stance_width_hind": float(foot_mean[2, 1] - foot_mean[3, 1]),
        }
        rows.append(row)

        print(f"\n### 步态对称性 —— {label}  命令 ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})")
        print("\n| 腿 | hipx | hipy | knee | foot x_b | foot y_b |")
        print("|---|---|---|---|---|---|")
        for i, leg in enumerate(LEG_NAMES):
            print(f"| {leg} | {joint_mean[i*3]:+.3f} | {joint_mean[i*3+1]:+.3f} | "
                  f"{joint_mean[i*3+2]:+.3f} | {foot_mean[i,0]:+.3f} | {foot_mean[i,1]:+.3f} |")
        print("\n| 镜像对 | hipx RMS | hipy RMS | knee RMS |")
        print("|---|---|---|---|")
        for k, v in row["mirror_rms"].items():
            print(f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |")
        print(f"\n左右不对称度：前 `y_fl+y_fr` = {row['lateral_asymmetry']['front_fl_plus_fr']:+.4f} m，"
              f"后 `y_hl+y_hr` = {row['lateral_asymmetry']['hind_hl_plus_hr']:+.4f} m；"
              f"轮距 前 {row['stance_width_front']:+.3f} / 后 {row['stance_width_hind']:+.3f} m")

    print("\n### 汇总（镜像误差 RMS，rad；越小越对称）")
    print("\n| checkpoint | 命令 | fl~hr | fr~hl | fl~fr | hl~hr | 前不对称 | 后不对称 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["command"]
        m = r["mirror_rms"]
        print(f"| {r['label']} | ({c['vx']:+.2f},{c['vy']:+.2f},{c['wz']:+.2f}) | "
              f"{np.mean(m['fl~hr']):.4f} | {np.mean(m['fr~hl']):.4f} | "
              f"{np.mean(m['fl~fr']):.4f} | {np.mean(m['hl~hr']):.4f} | "
              f"{r['lateral_asymmetry']['front_fl_plus_fr']:+.4f} | "
              f"{r['lateral_asymmetry']['hind_hl_plus_hr']:+.4f} |")

    if args_cli.out:
        out_path = os.path.abspath(args_cli.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"label": label, "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已写入 {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
