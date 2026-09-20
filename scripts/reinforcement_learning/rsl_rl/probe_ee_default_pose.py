# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：测「默认位姿」在 height-invariant 坐标系下的球坐标（半径/仰角/方位）。

为什么需要这个探针
------------------
``HeightInvariantEECommandCfg.ranges`` 是在 **height-invariant 坐标系**（原点 = arm_base_link
的 XY + 固定 Z=``sampled_height``，朝向只保留 yaw）里按球坐标采样 EE 目标的：

    (p_l, p_pitch, p_yaw) -> 笛卡尔 -> 世界系 -> root 系 -> IK

EE 目标课程（``WBCCurriculumCfg`` 的 s0 阶段）要把目标**锁在默认位姿**上，
而"锁"是通过把采样区间收成一个点实现的（``p_l=(r0, r0)`` 这种退化区间）——
所以必须知道默认位姿对应的 ``(r0, pitch0, yaw0)``，否则 s0 表达不出"默认位姿"
（这正是 `docs/review/DEFECT_LOG_zh.md` DEF-006 里留待下次 session 第一步做的事）。

本脚本还会回答两个相关的问题：

1. ``o_roll=o_pitch=o_yaw=(0,0)`` 时的姿态（= 把局部 +z 对齐到位置方向的 ``q_align``）
   与默认 EE 姿态差多少度？如果差得大，说明"只锁位置区间 + 姿态区间收 0"并不等于
   "姿态也锁在默认位姿"。
2. 默认位姿的半径 ``r0`` 是否落在当前 ``p_l`` 上界（0.52）之内 —— 如果不在，
   s1/s3 的区间就需要延伸，否则课程的起点在最终任务范围之外。

用法（必须 headless，一个进程只建一个 Isaac env）::

    python scripts/reinforcement_learning/rsl_rl/probe_ee_default_pose.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="测默认位姿在 height-invariant 坐标系下的球坐标")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Flat-Deeprobotics-M20-Piper-WBC-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=3, help="reset 后再 step 几步（让命令/IK 跑起来）")
parser.add_argument("--ee_body", type=str, default=None, help="EE body 名；默认取命令项 cfg.body_name")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
#  Isaac Sim 起好之后的 import
# --------------------------------------------------------------------------- #
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402

import rl_training.tasks  # noqa: F401,E402  触发任务注册
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import (  # noqa: E402
    cart2sphere,
)


def _stats(x: torch.Tensor) -> str:
    x = x.detach().flatten().float().cpu()
    return (f"mean={x.mean():+.4f} std={x.std():.4f} "
            f"min={x.min():+.4f} max={x.max():+.4f}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 观测噪声/扰动关掉，保证"默认位姿"是干净的初始状态
    env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg, "events", None) is not None:
        for name in ("randomize_apply_external_force_torque", "push_robot"):
            if getattr(env_cfg.events, name, None) is not None:
                setattr(env_cfg.events, name, None)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    command_term = env.command_manager.get_term("ee_pose")
    ee_body_name = args_cli.ee_body or command_term.cfg.body_name
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    arm_base_idx = robot.find_bodies(command_term.cfg.arm_base_link_name)[0][0]
    all_ids = torch.arange(env.num_envs, device=env.device)

    print("\n" + "=" * 78)
    print(f"[probe] task={args_cli.task} num_envs={env.num_envs} "
          f"command_term={type(command_term).__name__} ee_body={ee_body_name}")
    rng = command_term.cfg.ranges
    print(f"[probe] 当前 cfg.ranges: p_l={tuple(rng.p_l)} p_pitch={tuple(rng.p_pitch)} "
          f"p_yaw={tuple(rng.p_yaw)}")
    print(f"[probe]                  o_roll={tuple(rng.o_roll)} o_pitch={tuple(rng.o_pitch)} "
          f"o_yaw={tuple(rng.o_yaw)} T_traj={tuple(rng.T_traj)}")
    print(f"[probe] sampled_height={command_term.cfg.sampled_height} "
          f"arm_base_link={command_term.cfg.arm_base_link_name} "
          f"step_dt={env.step_dt:.4f} decimation={env.cfg.decimation}")

    # ── 1. reset 后的"默认位姿"（此时还没被任何命令拉动过）──────────────
    origin_pos, quat_yaw = command_term.get_height_invariant_base_frame(env, all_ids)
    ee_pos_w = robot.data.body_pos_w[all_ids, ee_idx]
    ee_quat_w = robot.data.body_quat_w[all_ids, ee_idx]
    arm_base_pos_w = robot.data.body_pos_w[all_ids, arm_base_idx]

    # 投到 height-invariant 坐标系：(p - origin) 先反 yaw，再转球坐标
    quat_yaw_inv = math_utils.quat_conjugate(quat_yaw)
    ee_pos_local = math_utils.quat_apply(quat_yaw_inv, ee_pos_w - origin_pos)
    ee_sphere = cart2sphere(ee_pos_local)              # (N, 3) = (r, pitch, yaw)
    ee_quat_local = math_utils.quat_mul(quat_yaw_inv, ee_quat_w)
    roll_l, pitch_l, yaw_l = math_utils.euler_xyz_from_quat(ee_quat_local)

    print("\n" + "-" * 78)
    print("[probe] 默认位姿（reset 后、未 step）：height-invariant 坐标系下")
    print(f"  r0     : {_stats(ee_sphere[:, 0])}")
    print(f"  pitch0 : {_stats(ee_sphere[:, 1])}  (deg {_stats(torch.rad2deg(ee_sphere[:, 1]))})")
    print(f"  yaw0   : {_stats(ee_sphere[:, 2])}  (deg {_stats(torch.rad2deg(ee_sphere[:, 2]))})")
    print(f"  z(origin)={origin_pos[0, 2].item():.4f} (sampled_height)  "
          f"arm_base z={arm_base_pos_w[0, 2].item():.4f}  ee z={ee_pos_w[0, 2].item():.4f}")
    print(f"  默认姿态（invariant 系, euler xyz）: roll={_stats(roll_l)} pitch={_stats(pitch_l)} "
          f"yaw={_stats(yaw_l)}   (deg roll {math.degrees(roll_l.mean().item()):+.2f} "
          f"pitch {math.degrees(pitch_l.mean().item()):+.2f} yaw {math.degrees(yaw_l.mean().item()):+.2f})")

    # ── 2. 若 s0 用 "区间收成一点（p_l=(r0,r0), o_*=0）"，姿态会是 q_align(位置方向)
    #      把局部 +z 对齐到位置方向 —— 和默认姿态差多少？
    pos_dir = math_utils.normalize(ee_pos_local)
    ref_axis = torch.zeros_like(pos_dir)
    ref_axis[:, 2] = 1.0
    q_align = command_term._quat_from_two_vectors(ref_axis, pos_dir)
    # 角度差 = 2*acos(|<q_a, q_b>|)
    dot = torch.abs((q_align * ee_quat_local).sum(dim=-1)).clamp(0.0, 1.0)
    angle_diff = 2.0 * torch.acos(dot)
    print(f"\n[probe] s0（p_l 收成一点 + o_*=0）的姿态 vs 默认姿态：")
    print(f"  夹角: {_stats(angle_diff)} rad  (deg {_stats(torch.rad2deg(angle_diff))})")
    print("  -> 角度大说明 `o_*=(0,0)` 并不等于'姿态锁默认'，s0 若要严格锁默认姿态，")
    print("     需要给命令项加一个显式的固定姿态（见 docs/review/DEFECT_LOG_zh.md DEF-006 的方案 A）。")

    # ── 3. step 几步，看命令项自己采样出来的目标（用于确认采样范围真的生效）──
    zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args_cli.steps):
        env.step(zero_actions)
    sampled = command_term.pose_end_cart                      # (N, 7) invariant 系
    sampled_sphere = cart2sphere(sampled[:, :3])
    sampled_quat_local = sampled[:, 3:]
    s_roll, s_pitch, s_yaw = math_utils.euler_xyz_from_quat(sampled_quat_local)
    print("\n" + "-" * 78)
    print(f"[probe] step {args_cli.steps} 步后命令项采样的**目标**（pose_end_cart, invariant 系）:")
    print(f"  r     : {_stats(sampled_sphere[:, 0])}")
    print(f"  pitch : {_stats(sampled_sphere[:, 1])} (deg {_stats(torch.rad2deg(sampled_sphere[:, 1]))})")
    print(f"  yaw   : {_stats(sampled_sphere[:, 2])} (deg {_stats(torch.rad2deg(sampled_sphere[:, 2]))})")
    print(f"  目标姿态 euler: roll={_stats(s_roll)} pitch={_stats(s_pitch)} yaw={_stats(s_yaw)}")
    # 当前实际生效的命令（插值中）与实际 EE 位姿
    cmd_b = command_term.pose_command_b
    origin_pos, quat_yaw = command_term.get_height_invariant_base_frame(env, all_ids)
    ee_pos_w = robot.data.body_pos_w[all_ids, ee_idx]
    ee_pos_local_now = math_utils.quat_apply(
        math_utils.quat_conjugate(quat_yaw), ee_pos_w - origin_pos
    )
    print(f"  pose_command_b[:3] (root 系) 当前位置半径 = {_stats(torch.linalg.norm(cmd_b[:, :3], dim=-1))}")
    print(f"  当前 EE 在 invariant 系的位置半径 = {_stats(torch.linalg.norm(ee_pos_local_now, dim=-1))}")

    print("\n[probe] 结论（把这三行抄进 WBCCurriculumCfg）：")
    print(f"  p_l     s0 = ({ee_sphere[:, 0].mean().item():.4f}, {ee_sphere[:, 0].mean().item():.4f})"
          f"       # 默认位姿半径 r0（当前 cfg 上界 {max(rng.p_l):.2f}）")
    print(f"  p_pitch s0 = ({ee_sphere[:, 1].mean().item():+.4f}, {ee_sphere[:, 1].mean().item():+.4f})"
          f"  # 默认位姿仰角 (deg {math.degrees(ee_sphere[:, 1].mean().item()):+.2f})")
    print(f"  p_yaw   s0 = ({ee_sphere[:, 2].mean().item():+.4f}, {ee_sphere[:, 2].mean().item():+.4f})"
          f"  # 默认位姿方位 (deg {math.degrees(ee_sphere[:, 2].mean().item()):+.2f})")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    # 注意：Isaac 脚本里不要用 exit()/sys.exit()（会抛 SystemExit）
    os._exit(0)
