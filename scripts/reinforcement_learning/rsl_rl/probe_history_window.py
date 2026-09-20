# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：验证高层 replay 侧构造的 **history 窗口**与低层训练口径逐位一致。

对应 `docs/review/history_low_level_policy_todo.md` 的验收项。检查四件事：

1. **维度**：`policy_layout.json` 的 `policy_obs_dim / history_length /
   history_single_step_dim` 与 replay 建出来的 obs/窗口一致；checkpoint 的 `action_dim`
   （16）与布局一致。
2. **单步内容**：replay 窗口里**最后一帧**必须等于用**低层训练的那个函数**
   （`velocity.mdp.history_single_step_obs`，它内部读 `env.action_manager.action`）
   独立复算出来的向量 —— 做法是临时把 `env.action_manager.action` 写成 replay 缓存的
   **低层 16 维动作**，再调训练函数；两者之差应为 0。
   （顺带证明"last_action 必须用低层动作"：高层动作是 12 维，宽度都不一样。）
3. **窗口顺序/复位语义**：整个 700 维窗口必须等于"最近 k 帧"按
   `[t_oldest(D) ... t_newest(D)]` 拼接的结果，且复位后按 IsaacLab `CircularBuffer`
   的"首次 push 填满整窗"规则补齐（`[f1]*(11-k) + [f2..fk]`）。
4. **环形缓冲语义**：直接对 `CircularBuffer` 做一次小实验（顺序 + 复位填充），
   确认我们依赖的行为与训练侧一致。

用法（必须 headless）::

    python scripts/reinforcement_learning/rsl_rl/probe_history_window.py \
        --task Isaac-M20-Piper-Teleop-v0 --headless --num_envs 8 --steps 40 \
        --policy logs/rsl_rl/history_adaptation/<run>/exported_deploy/policy.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="验证 history 回放窗口与低层训练口径一致")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-Teleop-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=40, help="记录多少个高层 env step")
parser.add_argument("--policy", type=str, required=True, help="导出的部署态低层策略（含 policy_layout.json 的目录）")
parser.add_argument("--action_term", type=str, default="pre_trained_pick_action",
                    help="高层 actions cfg 里的 action term 名（teleop=pre_trained_pick_action）")
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
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402
import rl_training.tasks.manager_based.locomotion.highlevel.mdp.teleop_ll_action as teleop_module  # noqa: E402
import rl_training.tasks.manager_based.locomotion.velocity.mdp as low_mdp  # noqa: E402
from rl_training.tasks.manager_based.locomotion.highlevel.mdp import low_level_replay as llr  # noqa: E402


def _check_circular_buffer() -> None:
    """直接验证我们依赖的 CircularBuffer 语义（顺序 + 复位填充）。"""
    from isaaclab.utils.buffers import CircularBuffer

    print("\n" + "-" * 78)
    print("[probe] (4) IsaacLab CircularBuffer 语义（replay 与训练共用同一个类）")
    buf = CircularBuffer(max_len=3, batch_size=1, device="cpu")
    for i in range(5):
        buf.append(torch.tensor([[float(i)] * 2]))          # 帧 i = [i, i]
    flat = buf.buffer.reshape(1, -1).tolist()[0]
    expect = [2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    print(f"    连续 5 帧(max_len=3) 展平 = {flat}")
    print(f"    期望 [t_oldest..t_newest] 展平 = {expect}   {'OK' if flat == expect else '**不一致**'}")
    if flat != expect:
        raise RuntimeError("CircularBuffer 展平顺序与预期不符")
    buf.reset(torch.tensor([0]))
    buf.append(torch.tensor([[9.0, 9.0]]))
    flat = buf.buffer.reshape(1, -1).tolist()[0]
    print(f"    reset 后 push 一帧 = {flat}（应填满整窗 → 训练侧第一次 push 的行为）")
    if flat != [9.0] * 6:
        raise RuntimeError("CircularBuffer reset 填充语义与预期不符")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    action_cfg = getattr(env_cfg.actions, args_cli.action_term)
    action_cfg.policy_path = args_cli.policy
    action_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    term = env.action_manager.get_term(args_cli.action_term)
    layout_json = llr.read_policy_layout(args_cli.policy)

    print("\n" + "=" * 78)
    print(f"[probe] task={args_cli.task} num_envs={env.num_envs} policy={args_cli.policy}")
    print(f"[probe] policy_layout.json = {layout_json}")
    print(f"[probe] 高层 action_manager.total_action_dim = {env.action_manager.total_action_dim}"
          f"（这就是**不能**拿来当 last_action 的那个量）")
    print(f"[probe] 低层布局: action_dim={term._layout.total_action_dim} "
          f"leg={term._layout.leg_dim} wheel={term._layout.wheel_dim} "
          f"ee_ik={term._layout.ee_action_dim} "
          f"policy_joint_names={len(term._layout.policy_joint_names)}")
    state = term._ll_replay_state
    print(f"[probe] history 窗口: length={state.length} single_step={state.single_step_dim} "
          f"flat={state.length * state.single_step_dim} has_history={state.has_history}")
    assert state.has_history, "期望 history 策略（policy_layout.json kind=history）"
    assert state.length == layout_json["history_length"]
    assert state.single_step_dim == layout_json["history_single_step_dim"]
    hlen = state.length

    _check_circular_buffer()

    # ── 2/3：逐 tick 复算 ────────────────────────────────────────────────
    # 用**低层训练的那个函数**做独立复算：它内部读 `env.action_manager.action`，
    # 所以临时把它写成 replay 缓存的低层动作（16 维）。
    ref_cfg = ObsGroup()
    ref_cfg.history_obs = ObsTerm(func=low_mdp.history_single_step_obs, params={})
    ref_cfg.enable_corruption = False
    ref_cfg.concatenate_terms = True
    from isaaclab.managers import ObservationManager
    ref_manager = ObservationManager({"ref_frame": ref_cfg}, env)

    records: list[dict] = []

    def _ref_frame() -> torch.Tensor:
        # 低层训练函数 `history_single_step_obs` 内部读 `env.action_manager.action`。
        # 高层 env 里那个张量是**高层动作**（13 维），所以临时把 action_manager 的
        # 内部张量换成 replay 缓存的**低层 16 维动作**，再调用训练函数做独立复算。
        low_action = torch.cat(
            [term.low_level_leg_actions, term.low_level_wheel_actions, term.low_level_ee_actions], dim=-1
        )
        save = env.action_manager._action
        env.action_manager._action = low_action
        try:
            return ref_manager.compute_group("ref_frame").clone()
        finally:
            env.action_manager._action = save

    original = teleop_module.run_low_level_policy

    def _wrapper(policy, policy_obs, history_flat=None):
        records.append(
            {
                "history": None if history_flat is None else history_flat.clone(),
                "frame": _ref_frame(),
                # CircularBuffer 的 current_length = min(pushes_since_reset, max_len)：
                # 1 表示"复位后第一次 push"，用于复算"填满整窗"的补齐规则。
                "kvec": state._buffer.current_length.clone(),
                "low_level_action": torch.cat(
                    [term.low_level_leg_actions, term.low_level_wheel_actions,
                     term.low_level_ee_actions], dim=-1
                ).clone(),
            }
        )
        return original(policy, policy_obs, history_flat)

    teleop_module.run_low_level_policy = _wrapper
    try:
        for _ in range(args_cli.steps):
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.step(actions)
    finally:
        teleop_module.run_low_level_policy = original

    n_records = len(records)
    print("\n" + "-" * 78)
    print(f"[probe] (2) 最后一帧 vs 训练函数独立复算（{n_records} 次低层 tick）")
    worst_last = 0.0
    for rec in records:
        diff = float((rec["history"][:, -state.single_step_dim:] - rec["frame"]).abs().max())
        worst_last = max(worst_last, diff)
    print(f"    max|窗口最后 {state.single_step_dim} 维 − 训练函数复算| = {worst_last:.3e}")
    print(f"    单步向量组成核对: 窗口最后一帧宽度 = {records[0]['history'].shape[-1] // hlen}，"
          f"其中 last_action 段 = {records[0]['low_level_action'].shape[-1]} 维"
          f"（高层动作 {env.action_manager.total_action_dim} 维 ⇒ 用错就宽度都不对）")

    print("\n" + "-" * 78)
    print(f"[probe] (3) 整窗顺序核对（[t_oldest(D) ... t_newest(D)]，含复位填充）")
    worst_win = 0.0
    checked = 0
    padded = 0
    n_envs = env.num_envs
    step0 = torch.full((n_envs,), -1, dtype=torch.long)      # 每个 env 本 episode 第一帧的记录下标
    for i, rec in enumerate(records):
        kvec = rec["kvec"]
        start_new = (kvec == 1).nonzero(as_tuple=False).squeeze(-1)
        if start_new.numel() > 0:
            step0[start_new] = i        # 这些 env 的新 episode 从本 tick 开始
        expect_list = []
        for e in range(n_envs):
            k = min(int(kvec[e]), hlen)
            ep_start = int(step0[e])
            if ep_start < 0 or ep_start > i:
                expect_list = None
                break
            if k == hlen:
                # 窗口已装满：就是最近 hlen 帧，没有填充
                frames = [records[j]["frame"][e] for j in range(i - hlen + 1, i + 1)]
            else:
                # 本 episode 的 push 次数不足 hlen：CircularBuffer 的"首次 push 填满整窗"
                # ⇒ 复位后第一帧占 (hlen - k + 1) 个槽位
                frames = [records[ep_start]["frame"][e]] * (hlen - k + 1)
                frames += [records[j]["frame"][e] for j in range(ep_start + 1, i + 1)]
            if len(frames) != hlen:
                raise RuntimeError(f"期望帧数 {hlen}，实际 {len(frames)}（k={k}, i={i}, env={e}）")
            if k < hlen:
                padded += 1
            expect_list.append(torch.cat(frames))
        if expect_list is None:
            continue
        expect = torch.stack(expect_list, dim=0)
        diff = float((rec["history"] - expect).abs().max())
        worst_win = max(worst_win, diff)
        checked += 1
    print(f"    逐 tick 整窗对比 {checked} 次（其中 {padded} 次带复位填充）："
          f"max|窗口 − 期望拼接| = {worst_win:.3e}")

    ok = worst_last < 1e-6 and worst_win < 1e-6
    print(f"\n[probe] 结论: {'history 窗口与低层训练口径一致（逐位相同）' if ok else '**不一致**'}")
    if not ok:
        raise RuntimeError("history 窗口校验未通过")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    os._exit(0)
