# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""EE 位姿命令的"初始化 / reset 首帧"检查。

背景（低层 known_issues ⑪⑫，清单见 docs/review/TODO_zh.md P1-3）：

* ⑪ 说 `HeightInvariantEECommand._update_command` 覆盖了父类却没调 `super()`，
  于是 `pose_command_w` **永不更新**；
* ⑫ 说 reset 之后第一帧 `pose_command_b` 全是 0（观测/奖励会看到"零位姿目标"）。

这两条都只能靠**实测**判定：本脚本把环境建起来、跑若干步，直接打印
`pose_command_b / pose_command_w / pose_start_b / pose_end_b` 与
`Metrics/ee_pose/position_error`，并断言：

1. `pose_command_b` 的四元数部分必须是**单位四元数**（不是全 0）；
2. `pose_command_w` 必须等于 root → command 的坐标变换（即父类 `_update_metrics` 真的在跑）；
3. reset 之后第一帧的 `pose_command_b` 与"复位那一刻的真实 EE 位姿(root 系)"的偏差应为 0
   （`_resample_command` 把 `pose_start_b` 设成真实位姿、`_update_command` 的 alpha=0 输出它）。

用法::

    python scripts/reinforcement_learning/rsl_rl/probe_ee_command_init.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 2 --steps 6
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="EE 位姿命令初始化/reset 首帧检查")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Flat-Deeprobotics-M20-Piper-WBC-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=6, help="step 数（前几步看 init，之后看 reset 首帧）")
parser.add_argument(
    "--force_reset",
    action="store_true",
    default=True,
    help="把 episode_length_buf 顶到上限以强制 reset，专门看 reset 首帧（默认开）",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def _fmt(t: torch.Tensor, i: int = 0) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in t[i].tolist()) + "]"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    term = unwrapped.command_manager.get_term("ee_pose")
    print(f"[INFO] task={args_cli.task}  num_envs={unwrapped.num_envs}")
    print(f"[INFO] command term = {type(term).__name__}")

    ee_idx = robot.find_bodies("gripper_base")[0][0]

    def dump(tag: str):
        b = term.pose_command_b
        w = term.pose_command_w
        # 1) 四元数部分必须是单位四元数
        quat_norm = b[:, 3:].norm(dim=-1)
        # 2) pose_command_w 必须 = root 位姿 ∘ pose_command_b
        w_expected_pos, w_expected_quat = math_utils.combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, b[:, :3], b[:, 3:]
        )
        d_pos = (w[:, :3] - w_expected_pos).abs().max().item()
        d_quat = (w[:, 3:] - w_expected_quat).abs().max().item()
        # 3) 与实际 EE 位姿（root 系）的偏差
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx],
        )
        d_ee = (b[:, :3] - ee_pos_b).abs().max().item()
        print(f"[{tag}] pose_command_b[0] = {_fmt(b)}")
        print(f"[{tag}] pose_command_w[0] = {_fmt(w)}")
        print(f"[{tag}] quat_norm(command_b) max|1-|q|| = {((quat_norm - 1).abs().max().item()):.3e}")
        print(f"[{tag}] |pose_command_w - transform(root, command_b)| = pos {d_pos:.3e} quat {d_quat:.3e}")
        print(f"[{tag}] |pose_command_b - 真实EE位姿(root系)| = {d_ee:.3e}")
        return quat_norm, d_pos, d_quat

    # 注意：**建完环境还没 reset 时**（即 gym.make 之后、第一次 reset 之前）命令必然是
    # 父类 __init__ 里的初值 (0,0,0,1,0,0,0) —— 但训练循环第一步永远是 reset()，
    # 所以这里按真实顺序做：先 reset，再看首帧、再看 step、最后看"强制 reset 后那一帧"。
    dump("before_any_reset")
    unwrapped.reset()
    dump("after_reset")

    zero_action = torch.zeros(unwrapped.num_envs, unwrapped.action_space.shape[-1], device=unwrapped.device)
    for i in range(args_cli.steps):
        if args_cli.force_reset and i == args_cli.steps - 1:
            unwrapped.episode_length_buf[:] = unwrapped.max_episode_length
            print("[INFO] 强制 reset（把 episode_length_buf 顶到上限）")
        with torch.inference_mode():
            unwrapped.step(zero_action)
        dump(f"step{i}")
        if i == args_cli.steps - 1:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
