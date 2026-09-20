# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：把「部署侧要照抄的那张表」一次性打印出来（sim2sim / sim2real 用）。

为什么需要它
------------
部署脚本（MuJoCo / 真机）要复刻三件事，而且三件的**关节顺序互不相同**：

1. **动作**：16 维 = 12 腿（``joint_pos`` action 的 ``joint_names`` 顺序）+ 4 轮
   （``joint_vel`` action 的 ``joint_names`` 顺序）；
2. **观测** ``joint_pos`` / ``joint_vel``：24 维，按 **articulation 原生顺序**（``[".*"]``），
   其中 ``joint_pos`` 的轮子列被置零（``mdp.observations.joint_pos_rel_without_wheel``）；
3. **history**：每步 70 维 = ``[base_ang_vel(3), projected_gravity(3), joint_pos(24),
   joint_vel(24), last_action(16)]``，用**原生顺序**的 24 维（与 ② 同一份，但不置零）。

历史教训：``known_issues`` #1 / DEF-016 —— 把"列的下标"和"原生 id"混用会静默清错关节。
所以部署前先在部署机上跑一遍本脚本，拿实测输出当权威，别照抄文档里的表。

用法（必须 headless）::

    python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py \
        --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 2
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

parser = argparse.ArgumentParser(description="打印部署侧需要照抄的布局（关节顺序/缩放/观测）")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="History-Adaptation-Deeprobotics-M20-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
#  Isaac Sim 起好之后的 import
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402  触发任务注册


def _fmt_rng(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(f"{float(x):.4g}" for x in v) + ")"
    return f"{float(v):.4g}"


def _dim_of(d) -> int:
    if hasattr(d, "__len__"):
        n = 1
        for x in d:
            n *= int(x)
        return n
    return int(d)


def _term_scale(term, native_idx: dict, num_joints: int) -> str:
    """动作项的 scale：可能是 float，也可能是按关节名给的 dict（IsaacLab 已编译成张量）。"""
    raw = getattr(term, "scale", None)
    compiled = getattr(term, "_scale", None)
    return f" cfg={raw}" if raw is not None else ""


def _obs_group_cfg(env, gname: str):
    """从 env.cfg 上取观测组配置（scale/clip/noise 都写在这里）。"""
    return getattr(env.cfg.observations, gname, None)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos[0].detach().cpu()
    soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu()
    hard_limits = robot.data.joint_pos_limits[0].detach().cpu()
    natural_idx = {n: i for i, n in enumerate(joint_names)}

    print("\n" + "=" * 92)
    print(f"[deploy-layout] task={args_cli.task}  num_envs={env.num_envs}")
    print(
        f"[deploy-layout] sim.dt={env.cfg.sim.dt:.4f} s, decimation={env.cfg.decimation}, "
        f"策略周期 step_dt={env.step_dt:.4f} s → 策略频率 {1.0 / env.step_dt:.1f} Hz, "
        f"episode_length_s={getattr(env.cfg, 'episode_length_s', None)}"
    )

    # ---------------- 1. 原生关节顺序（观测用的就是它） ----------------
    print("\n[1] articulation 原生关节顺序（观测 joint_pos/joint_vel 的列序 = 这个顺序）")
    header = f"    {'idx':>3}  {'joint name':<20} {'默认角(rad)':>11} {'软下限':>8} {'软上限':>8}  {'硬限位':>20}"
    print(header)
    for i, name in enumerate(joint_names):
        print(
            f"    {i:>3}  {name:<20} {float(default_pos[i]):>11.4f} "
            f"{float(soft_limits[i, 0]):>8.3f} {float(soft_limits[i, 1]):>8.3f}  "
            f"[{float(hard_limits[i, 0]):>7.3f}, {float(hard_limits[i, 1]):>7.3f}]"
        )

    # ---------------- 2. 动作分块（policy 输出顺序） ----------------
    act_mgr = env.action_manager
    print(f"\n[2] 动作项（policy 输出顺序 = 各 term 的拼接顺序；总维度 {act_mgr.total_action_dim}）")
    offset = 0
    for tname, term in act_mgr._terms.items():  # noqa: SLF001 - 只用于打印
        jn = list(getattr(term, "_joint_names", None) or [])
        dim = int(term.action_dim)
        print(f"  - term '{tname}'：dim={dim}{_term_scale(term, natural_idx, len(joint_names))}")
        if dim == 0:
            continue
        if jn:
            print(f"      动作槽位 {offset}..{offset + dim - 1} → 关节名 / 原生下标：")
            for k, name in enumerate(jn):
                print(f"        a[{offset + k:>2}] {name:<20} 原生下标 {natural_idx.get(name, '?'):>3}")
        else:
            print(f"      （关节名未知，动作槽位 {offset}..{offset + dim - 1}）")
        offset += dim

    # ---------------- 2b. 实测增益（动作 1.0 → 关节目标） ----------------
    print("\n[2b] 实测增益：把该 term 的动作全设成 1.0，看它输出的关节目标（部署时照抄这两个数）")
    try:
        act_mgr.process_action(torch.ones(env.num_envs, act_mgr.total_action_dim, device=env.device))
        for tname, term in act_mgr._terms.items():  # noqa: SLF001
            dim = int(term.action_dim)
            if dim == 0:
                continue
            jn = list(getattr(term, "_joint_names", None) or [])
            proc = term.processed_actions[0].detach().cpu()
            jids = getattr(term, "_joint_ids", None)
            if isinstance(jids, torch.Tensor):
                jids = jids.tolist()
            default_pos = (
                robot.data.default_joint_pos[0, jids].detach().cpu()
                if jids is not None and not isinstance(jids, slice)
                else None
            )
            print(f"  - term '{tname}'：")
            for k in range(dim):
                name = jn[k] if k < len(jn) else f"joint#{k}"
                p = float(proc[k])
                if default_pos is not None:
                    d = float(default_pos[k])
                    print(f"      动作[{k:>2}] {name:<20} → 目标={p:>9.4f}  默认角={d:>8.4f}  增益={p - d:>8.4f}")
                else:
                    print(f"      动作[{k:>2}] {name:<20} → 目标={p:>9.4f}")
            print(f"      （动作 0.0 时目标应回到默认角；轮子是速度目标，默认 0.0）")
    except Exception as e:  # noqa: BLE001 - 探针不该因为打印失败而中断
        print(f"    （跳过：{type(e).__name__}: {e}）")

    obs_mgr = env.observation_manager
    print("\n[3] 观测组（IsaacLab 处理顺序：compute → noise → clip → scale）")
    for gname in obs_mgr.active_terms.keys():
        names = obs_mgr.active_terms[gname]
        dims = obs_mgr.group_obs_term_dim[gname]
        total = int(obs_mgr.group_obs_dim[gname][0])
        print(f"  - 组 '{gname}'：{total} 维")
        gcfg = _obs_group_cfg(env, gname)
        for tname, d in zip(names, dims):
            tcfg = getattr(gcfg, tname, None) if gcfg is not None else None
            scale = _fmt_rng(getattr(tcfg, "scale", None)) if tcfg is not None else "?"
            clip = getattr(tcfg, "clip", None) if tcfg is not None else None
            noise = getattr(tcfg, "noise", None) if tcfg is not None else None
            if noise is None:
                noise_s = "None"
            else:
                noise_s = (
                    f"U({getattr(noise, 'n_min', '?')},{getattr(noise, 'n_max', '?')})"
                    if type(noise).__name__ == "Unoise"
                    else type(noise).__name__
                )
            print(f"      {tname:<20} dim={_dim_of(d):>4}  scale={scale:<10} clip={clip} noise={noise_s}")

    # ---------------- 4. 部署关键的 body 下标 ----------------
    body_names = list(robot.data.body_names)
    print("\n[4] 部署关键 body 下标（IK / 接触归因用）")
    print(f"    body 总数 = {len(body_names)}")
    for want in ("base_link", "arm_base_link", "gripper_base"):
        print(f"    {want:<16} → {body_names.index(want) if want in body_names else '（不在 body 列表里）'}")
    wheels = [n for n in body_names if "wheel" in n]
    print(f"    足端 body（名字含 wheel，共 {len(wheels)} 个）= {wheels}")

    # ---------------- 4b. 默认位姿下的几何量（MuJoCo 建模后第一件事就是对照这些数） ----------------
    import isaaclab.utils.math as math_utils  # noqa: E402

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_idx = body_names.index("gripper_base") if "gripper_base" in body_names else None
    print("\n[4b] 默认位姿（qpos = 默认关节角、零根速度）下的几何量 —— MuJoCo 里要能对上")
    print(f"    root_pos_w (env0) = {[round(float(x), 4) for x in root_pos_w[0].cpu()]}"
          f"  ← 训练时的 spawn 高度 = 0.55 m")
    if ee_idx is not None:
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx],
        )
        print(f"    gripper_base 在 root 系: pos={[round(float(x), 4) for x in ee_pos_b[0].cpu()]} "
              f"quat(wxyz)={[round(float(x), 4) for x in ee_quat_b[0].cpu()]}")
        print("      ← 部署时 ee_goal 的第 3..9 维与 IK 目标都在这套坐标里")
    if wheels:
        wheel_idx = [body_names.index(n) for n in wheels]
        print(f"    足端在世界系 z = {[round(float(robot.data.body_pos_w[0, i, 2]), 4) for i in wheel_idx]}")
        print(f"    base 相对足端高度 = {round(float(root_pos_w[0, 2]) - float(robot.data.body_pos_w[0, wheel_idx, 2].mean()), 4)} m"
              f"（body_pose 的 height 命令用的是它 + 轮半径 0.09）")

    # ---------------- 5. 命令区间 ----------------
    print("\n[5] 命令项当前区间（这是**课程 s0 的初值**；跑满 20k iter 后实际生效的是 s3 终值）")
    for cname in env.command_manager.active_terms:
        term = env.command_manager.get_term(cname)
        cfg = term.cfg
        print(f"  - '{cname}' resampling={getattr(cfg, 'resampling_time_range', None)}")
        for attr in ("height_range", "pitch_range", "roll_range", "ranges"):
            val = getattr(cfg, attr, None)
            if val is None:
                continue
            if attr == "ranges":
                for r in sorted(vars(val)):
                    if not r.startswith("_"):
                        print(f"        ranges.{r} = {_fmt_rng(getattr(val, r))}")
            else:
                print(f"        {attr} = {_fmt_rng(val)}")
    print("\n    当前命令值（第 0 个 env）：")
    for cname in env.command_manager.active_terms:
        cmd = env.command_manager.get_command(cname)
        if cmd is not None:
            print(f"      {cname:<14} = {[round(float(x), 4) for x in cmd[0].detach().cpu()]}")
    print("\n    课程 s3 终值见 flat_env_wbc_cfg.py::WBCCurriculumCfg")
    print("    （ee_pose p_l (0.30,0.52) / body_pose height (0.33,0.55) / base_velocity vx (-5,5)）")
    print("=" * 92 + "\n")
    # os._exit(0) 不会 flush stdout：重定向到文件时必须手动 flush，否则日志是空的
    sys.stdout.flush()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]  # hydra_task_config 注入 cfg
    # 注意：Isaac 脚本里不要用 exit()/sys.exit()（会抛 SystemExit）
    os._exit(0)
