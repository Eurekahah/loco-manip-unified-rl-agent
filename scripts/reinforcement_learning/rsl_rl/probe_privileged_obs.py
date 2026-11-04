# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：特权观测（teacher 的输入）到底有没有信息 —— 缓存时序 / 跨环境方差 / 覆盖缺口。

起因（2026-10-16，用户提出的两个怀疑）：
  1) `velocity/mdp/observations.py` 里的 `_PrivilegedCachedTerm` 为了省开销把"不随 step 变化的
     环境参数"缓存了下来。**缓存是否会把教师输入冻结成随机化之前的常数？**（若冻结 → 教师
     输入跨环境完全相同 → 教师必然塌成常数，和探针观测到的 jac≈3% 一致。）
  2) **特权信息的选取是否既不全也不精？**（有些被随机化的量没进特权向量；有些进了但没用。）

本探针用真 env 量四件事（必须 headless）：

  A. 缓存时序：打印每个特权项的 `count`/`buf`，分别在
     [gym.make 之后（= ObservationManager 构造完成，早于 startup 随机化）] →
     [第一次 env.reset() 之后] → [连续 step] → [强制全体 reset] 四个阶段。
     判定标准：随机化之后 buf 必须**跨环境**有差异（std>0）。
  B. 缓存 = 真值：把 `count/buf` 归零强制重算，与缓存值逐位比较，应严格为 0。
  C. 逐项方差：89 维特权向量逐维的跨环境 std，统计"常量维"（std==0）占比；
     再按 step 统计"每步都在变的维"（应当≈0，因为域随机化只在 startup/reset 生效）。
  D. 覆盖清单：把 `EventCfg` 里所有随机化事件（mode / 范围 / 作用 body）列出来，
     并与特权项逐条对照，指出"被随机化但没被观测"的量。

用法::

    python scripts/reinforcement_learning/rsl_rl/probe_privileged_obs.py \
        --task History-Adaptation-V2-Deeprobotics-M20-v0 --headless --num_envs 64 --steps 32
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="特权观测健康度探针（缓存时序 + 方差 + 覆盖）")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="History-Adaptation-V2-Deeprobotics-M20-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=32, help="连续 step 的步数（统计'逐步变化'）")
parser.add_argument("--group", type=str, default="privileged", help="要检查的观测组名")
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


def _dim_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (跨环境 mean, 跨环境 std, 跨环境 min/max 幅值)。x: [N, D]"""
    return x.mean(dim=0), x.std(dim=0, unbiased=False), x.abs().amax(dim=0)


def _fmt_vec(v: torch.Tensor, k: int = 8) -> str:
    vals = v.flatten()[:k].tolist()
    s = "[" + ", ".join(f"{q:+.4f}" for q in vals) + (", ...]" if v.numel() > k else "]")
    return s


def _term_instances(om, group: str):
    """返回 [(term_name, term_cfg, instance_or_None)]；实例只有类式 term 才有。"""
    from isaaclab.managers.manager_base import ManagerTermBase

    names = om._group_obs_term_names[group]
    cfgs = om._group_obs_term_cfgs[group]
    out = []
    for n, c in zip(names, cfgs):
        inst = c.func if isinstance(c.func, ManagerTermBase) else None
        out.append((n, c, inst))
    return out


def _describe_events(env) -> None:
    print("\n" + "-" * 78)
    print("[probe] (D) EventCfg 里的随机化清单（真 env 上读到的，即课程覆盖之后的终态）")
    events_cfg = env.cfg.events
    from isaaclab.managers import EventTermCfg

    for name, term in vars(events_cfg).items() if not isinstance(events_cfg, dict) else events_cfg.items():
        if isinstance(term, EventTermCfg):
            if isinstance(term.func, str):
                fn = term.func
            else:
                fn = getattr(term.func, "__name__", str(term.func))
            params = dict(term.params)
            asset_cfg = params.pop("asset_cfg", None)
            bodies = getattr(asset_cfg, "body_names", None)
            joints = getattr(asset_cfg, "joint_names", None)
            scope = []
            if bodies is not None:
                scope.append(f"bodies={bodies}")
            if joints is not None:
                scope.append(f"joints={joints}")
            extra = " ".join(f"{k}={v}" for k, v in params.items())
            print(f"    [{term.mode:<10}] {name:<38} {fn:<38} {' '.join(scope)} {extra}")
    print("    可用 mode:", env.event_manager.available_modes)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    om = env.observation_manager
    group = args_cli.group

    print("\n" + "=" * 78)
    print(f"[probe] task={args_cli.task}  num_envs={env.num_envs}  seed={agent_cfg.seed}")
    print(f"[probe] 观测组维度 = {{" + ", ".join(f"{k}: {v}" for k, v in om.group_obs_dim.items()) + "}")
    print(f"[probe] '{group}' 组 = {om._group_obs_term_names[group]}")
    print(f"[probe] '{group}' 逐项维度 = {om._group_obs_term_dim[group]}")
    robot = env.scene["robot"]
    print(f"[probe] robot bodies({len(robot.body_names)}) = {robot.body_names}")
    print(f"[probe] robot joints({len(robot.joint_names)}) = {robot.joint_names}")
    if om.group_obs_concatenate[group]:
        total = sum(int(torch.tensor(d).prod()) for d in om._group_obs_term_dim[group])
        print(f"[probe] '{group}' 拼接总维度 = {total}（顺序 = {om._group_obs_term_names[group]}）")

    terms = _term_instances(om, group)
    print(f"[probe] '{group}' 各项的 asset_cfg 解析结果：")
    for name, cfg, inst in terms:
        ac = cfg.params.get("asset_cfg", None)
        if ac is None:
            print(f"    {name:<26} (无 asset_cfg)")
        else:
            print(
                f"    {name:<26} body_names={getattr(ac, 'body_names', None)} "
                f"body_ids={getattr(ac, 'body_ids', None)} "
                f"joint_names={getattr(ac, 'joint_names', None)} "
                f"joint_ids={getattr(ac, 'joint_ids', None)}"
            )

    # ── A. gym.make 之后（startup 随机化已完成，但还没 compute 过）────────────────
    print("\n" + "-" * 78)
    print("[probe] (A0) gym.make 之后 / 第一次 compute 之前：")
    print("     注意：ObservationManager 构造时（早于 startup 随机化）已经调过 1 次 term 来量维度，")
    print("     所以此时 buf 里装的是**随机化之前**的默认值 —— 这正是要检查的'冻结'风险。")
    pre = {}
    for name, cfg, inst in terms:
        if inst is None:
            continue
        buf = inst.buf
        if buf is None:
            print(f"    {name:<26} count={inst.count} buf=None")
            continue
        mean, std, _ = _dim_stats(buf)
        pre[name] = buf.detach().clone()
        print(
            f"    {name:<26} count={inst.count} dim={buf.shape[-1]} "
            f"跨env std_mean={std.mean():.4e} 常量维={int((std == 0).sum())}/{buf.shape[-1]} "
            f"值样例={_fmt_vec(mean)}"
        )

    # ── 第一次 reset：startup 之后的第一次 compute 会把缓存写成"真值" ────────────
    env.reset()
    om.compute()
    print("\n" + "-" * 78)
    print("[probe] (A1) 第一次 env.reset()+compute 之后（startup 随机化已生效）：")
    post = {}
    for name, cfg, inst in terms:
        if inst is None:
            continue
        buf = inst.buf
        mean, std, _ = _dim_stats(buf)
        post[name] = buf.detach().clone()
        changed = 0
        if name in pre:
            changed = int((post[name] != pre[name]).any(dim=-1).sum())
        print(
            f"    {name:<26} count={inst.count} dim={buf.shape[-1]} "
            f"跨env std_mean={std.mean():.4e} 常量维={int((std == 0).sum())}/{buf.shape[-1]} "
            f"与构造期相比变化的环境数={changed}/{buf.shape[0]} 值样例={_fmt_vec(mean)}"
        )

    # ── B. 缓存 == 真值（强制重算后逐位比较）────────────────────────────────────
    print("\n" + "-" * 78)
    print("[probe] (B) 缓存 vs 强制重算（把 count 归零 + buf 置 None，再调一次 __call__）：")
    for name, cfg, inst in terms:
        if inst is None:
            continue
        cached = inst.buf.detach().clone()
        inst.count = 0
        inst.buf = None
        fresh = inst(env, **cfg.params).detach().clone()
        maxdiff = float((cached - fresh).abs().max())
        print(f"    {name:<26} max|缓存 − 重算| = {maxdiff:.3e}  {'一致' if maxdiff == 0 else '**不一致**'}")

    # ── C1. 逐步是否变化（域随机化只在 startup/reset 生效 ⇒ 应≈0）───────────────
    print("\n" + "-" * 78)
    print(f"[probe] (C1) 连续 {args_cli.steps} 步：每项有多少维在变（期望 0，除非该随机化是 reset 模式）")
    prev = {name: inst.buf.detach().clone() for name, _, inst in terms if inst is not None}
    ever_changed = {
        name: torch.zeros(inst.buf.shape[-1], dtype=torch.bool, device=inst.buf.device)
        for name, _, inst in terms
        if inst is not None
    }
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args_cli.steps):
        env.step(actions)
        for name, cfg, inst in terms:
            if inst is None:
                continue
            cur = inst.buf.detach()
            ever_changed[name] |= (cur != prev[name]).any(dim=0)
            prev[name] = cur.clone()
    for name, cfg, inst in terms:
        if inst is None:
            continue
        n = int(ever_changed[name].sum())
        print(f"    {name:<26} 变过的维 = {n}/{inst.buf.shape[-1]}   (buf 是否被换对象: id={id(inst.buf)})")

    # ── C2. 强制全体 reset：只有 reset 模式的随机化对应的项应该刷新 ───────────────
    print("\n" + "-" * 78)
    print("[probe] (C2) 强制全体 reset 后（episode_length_buf=max → 下一步全体 _reset_idx）：")
    before_reset = {name: inst.buf.detach().clone() for name, _, inst in terms if inst is not None}
    env.episode_length_buf[:] = env.max_episode_length
    env.step(actions)
    for name, cfg, inst in terms:
        if inst is None:
            continue
        cur = inst.buf.detach()
        diff = (cur != before_reset[name]).any(dim=-1)
        # buf 内容变化的环境数（注意：判断"值变了"，而非 id 变了）
        print(f"    {name:<26} reset 后取值变化的环境数 = {int(diff.sum())}/{cur.shape[0]}")

    # ── C3. 合并后的完整特权向量：逐维跨环境方差 ────────────────────────────────
    print("\n" + "-" * 78)
    print("[probe] (C3) 合并后的特权向量（critic/teacher 真正吃到的那个）逐维诊断：")
    vec = om.compute()[group].detach()
    mean, std, mx = _dim_stats(vec)
    names = om._group_obs_term_names[group]
    dims = [int(torch.tensor(d).prod()) for d in om._group_obs_term_dim[group]]
    off = 0
    for name, d in zip(names, dims):
        s = std[off : off + d]
        m = mean[off : off + d]
        dead = (s == 0).nonzero().flatten().tolist()
        print(
            f"    {name:<26} 维[{off:>3}:{off + d:>3}]  std: mean={s.mean():.4e} max={s.max():.4e} "
            f"常量维={len(dead)}/{d}  |mean|max={m.abs().max():.4e}"
            + (f" 常量维(组内下标)={dead}" if dead else "")
        )
        off += d
    print(
        f"    {'合计':<24} 总维={vec.shape[-1]} 常量维={int((std == 0).sum())}/{vec.shape[-1]} "
        f"（常量维 = 教师不可能从中学到任何区分环境的信号）"
    )
    # 若 term 自己声明了通道切片（V4 的 privileged_env_params_v4 会声明），按块再统计一遍
    for name, cfg, inst in terms:
        blocks = getattr(inst, "block_slices", None)
        if not blocks:
            continue
        print(f"\n    [{name}] 通道块统计（每个通道都应按各自 DR 范围映射到 [-1,1]）：")
        for blk, (a, b) in blocks.items():
            s = std[a:b]
            m = mean[a:b]
            print(
                f"      {blk:<16} 维[{a:>3}:{b:>3}] std mean={s.mean():.3e} max={s.max():.3e} "
                f"常量维={int((s == 0).sum())}/{b - a} |mean|max={m.abs().max():.3f} "
                f"|值|最小={vec[:, a:b].abs().min():.3f}"
            )

    # ── E. gain_scale 的维序来源：actuator 组 × (k_scale, d_scale) ────────────────
    print("\n" + "-" * 78)
    print("[probe] (E) gain_scale 的维序 = 按 actuator 组顺序 [k_scale, d_scale] 拼接：")
    for actuator_name, actuator in robot.actuators.items():
        jidx = list(range(len(actuator.joint_names)))
        k = actuator.cfg.stiffness
        d = actuator.cfg.damping
        print(
            f"    {actuator_name:<16} joints={list(actuator.joint_names)}  默认 stiffness={k} damping={d}"
        )
    print("    （gain_scale 只统计 included_actuators = {'joint','wheel','piper_arm','piper_gripper'}）")

    # ── D. 随机化清单 vs 特权项覆盖 ─────────────────────────────────────────────
    _describe_events(env)

    print("\n[probe] 提示：把上面的'随机化清单'与特权组逐项对照 —— 凡是 mode=startup/reset 且改变了")
    print("        动力学、但既不在策略本体观测里、也不在特权向量里的量，就是'学生不可能辨识'的信息。")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    os._exit(0)
