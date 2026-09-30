# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""核对高层 action term 的「复位重锚」时机：`_on_reset` 读到的是不是**复位后**的状态。

背景
----
`LowLevelPolicyActionBase` 提供了一个 `_on_reset(env_ids)` 钩子，由基类的 `reset()`
（= `ActionManager.reset` 在 `ManagerBasedRLEnv._reset_idx` 里调用）转发。Pick 系用它把
增量目标标成"未初始化"（下一步 `process_actions` 重锚到当前 EE 位姿），Teleop 用它
`recalibrate()` / `_capture_default_ee_pose()`（**当场**读 `robot.data`）。

这两者都隐含一个假设：`_on_reset` 被调用的那一刻，`robot.data` 已经是**复位后**的状态。
旧实现在 `apply_actions` 里用 `episode_length_buf == 0` 检测（复位后下一个 env step），
所以本探针要把它验证掉 —— 打印 `_on_reset` / 首次重锚 / 若干 step 后的
`episode_length_buf`、`root_z`、`ee_z`，看锚点用的是复位位姿还是终止（摔倒）位姿。

用法::

    python scripts/reinforcement_learning/rsl_rl/probe_reset_anchor_timing.py \
        --task Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0 \
        --headless --num_envs 4 --steps 60
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="复位重锚时机探针")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--action_term", type=str, default="pre_trained_pick_action")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=60, help="最多跑多少 env step")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.0,
    help="常量动作幅值（>0 用来把机器人推翻，制造「终止复位」而不是「超时复位」）",
)
parser.add_argument("--print_every", type=int, default=1, help="每多少步打印一次状态")
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

import rl_training.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.action_manager.get_term(args_cli.action_term)
    ee_idx = robot.find_bodies(getattr(term.cfg, "ee_body_name", "gripper_base"))[0][0]

    print(f"[INFO] task={args_cli.task}  num_envs={unwrapped.num_envs}")
    print(f"[INFO] action term = {type(term).__name__}（{args_cli.action_term}）")
    print(f"[INFO] ee body idx = {ee_idx} ({getattr(term.cfg, 'ee_body_name', 'gripper_base')})")

    # 记录每个 env step 的序号，便于把打印串起来
    state = {"step": -1, "events": []}

    def snap(tag: str, i: int = 0) -> str:
        ep = unwrapped.episode_length_buf[i].item()
        return (
            f"[{tag}] step={state['step']:>3} ep_buf[{i}]={ep:>4} "
            f"root_z={robot.data.root_pos_w[i, 2].item():+.4f} "
            f"ee_z={robot.data.body_pos_w[i, ee_idx, 2].item():+.4f}"
        )

    # ── 包装子类钩子 ────────────────────────────────────────────────
    orig_on_reset = term._on_reset

    def wrapped_on_reset(env_ids):
        out = snap("_on_reset(before)")
        if hasattr(term, "_target_initialized"):
            out += f" target_init={bool(term._target_initialized[0].item())}"
        print(out)
        state["events"].append(("_on_reset", snap("", 0)))
        return orig_on_reset(env_ids)

    term._on_reset = wrapped_on_reset

    for name in ("_reset_target_to_current_ee", "_capture_default_ee_pose", "recalibrate"):
        orig = getattr(term, name, None)
        if orig is None:
            continue

        def make(fn, fname):
            def wrapper(env_ids=None):
                print(snap(f"{fname}(before)"))
                out = fn(env_ids)
                if hasattr(term, "_target_ee_pos_b"):
                    print(f"[{fname}(after)] target_ee_pos_b[0] = "
                          f"{term._target_ee_pos_b[0].tolist()}")
                elif hasattr(term, "_target_pos_w"):
                    print(f"[{fname}(after)] target_pos_w[0] = {term._target_pos_w[0].tolist()}")
                return out
            return wrapper

        setattr(term, name, make(orig, name))

    zero_action = torch.full(
        (unwrapped.num_envs, unwrapped.action_space.shape[-1]),
        args_cli.action_scale,
        device=unwrapped.device,
    )

    # 起始（未 reset）状态
    print(snap("before_any_reset"))
    unwrapped.reset()
    print(snap("after_reset"))

    seen_reset = False
    for i in range(args_cli.steps):
        state["step"] = i
        with torch.inference_mode():
            unwrapped.step(zero_action)
        if i < 3 or (i % args_cli.print_every == args_cli.print_every - 1):
            print(snap(f"post_step{i}"))
        if state["events"]:
            seen_reset = True

    print(f"[INFO] 观测到的复位事件数 = {len(state['events'])}；{'OK' if seen_reset else '未触发复位'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
