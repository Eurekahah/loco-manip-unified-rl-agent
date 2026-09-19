# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""probe：验证 EE 目标课程（s0→s3）与姿态 slerp 修复，并实测「臂扰动底盘」的机制。

本脚本一次跑完四件事（都在同一个 Isaac env 里，进程只建一个 env）：

1. **课程阶段**：把 ``common_step_counter`` 依次设成 0 / 25001 / 50001 / 75001 后调用
   ``curriculum_manager.compute()``，打印并断言 ``commands.ee_pose.target_blend_pos/_orn``
   是否按 s0→s1→s2→s3 变化。
2. **slerp 数值**：``quat_slerp_batch``（批量化实现）与官方单样本
   ``isaaclab.utils.math.quat_slerp`` 逐样本对比，并检查 tau=0/1 的端点。
3. **姿态命令跳变 A/B**：用 monkeypatch 把 ``_update_command`` 换回「姿态直接取终点」的
   旧实现，与 slerp 版对比**命令姿态的单步变化**（即"重采样瞬间给关节一个尖峰"的度量）。
4. **臂扰动机制 A/B**：把 ``target_blend_*`` 设成 0（s0：目标=默认位姿，臂不动）与
   1（s3：完整任务），各跑同样步数，对比臂关节速度 RMS / 底盘角速度 RMS / 最大倾角 /
   EE 跟踪误差 —— 直接对应 `bad_orientation_analysis_zh.md` 里"臂把扰动传给底盘"的结论。

用法（必须 headless）::

    python scripts/reinforcement_learning/rsl_rl/probe_ee_curriculum.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="验证 EE 目标课程 + 姿态 slerp + 臂扰动机制")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Flat-Deeprobotics-M20-Piper-WBC-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--steps", type=int, default=150, help="每组 A/B 记录的环境步数")
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
    HeightInvariantEECommand,
    quat_slerp_batch,
)


def _check_curriculum_stages(env) -> None:
    """把 common_step_counter 推到各阶段阈值之后，检查 blend 是否按预期变化。"""
    print("\n" + "-" * 78)
    print("[probe] (1) EE 目标课程阶段检查（curriculum_manager.compute 手动触发）")
    term_cfg = env.command_manager.get_term("ee_pose").cfg
    expect = [
        (0, 0.0, 0.0),
        (25_001, 0.35, 0.0),
        (50_001, 0.35, 0.35),
        (75_001, 1.0, 1.0),
    ]
    all_ids = torch.arange(env.num_envs, device=env.device)
    ok = True
    for counter, exp_pos, exp_orn in expect:
        env.common_step_counter = counter
        env.curriculum_manager.compute(env_ids=all_ids)
        got_pos = float(term_cfg.target_blend_pos)
        got_orn = float(term_cfg.target_blend_orn)
        good = abs(got_pos - exp_pos) < 1e-9 and abs(got_orn - exp_orn) < 1e-9
        ok &= good
        print(f"    counter={counter:>7}  target_blend_pos={got_pos:.2f} "
              f"target_blend_orn={got_orn:.2f}  (期望 {exp_pos:.2f}/{exp_orn:.2f})  "
              f"{'OK' if good else '**不一致**'}")
    print(f"[probe] (1) 结论: {'全部符合预期' if ok else '存在不一致（见上）'}")
    if not ok:
        raise RuntimeError("EE 目标课程阶段不符合预期")


def _check_slerp() -> None:
    """`quat_slerp_batch` vs 官方单样本 `quat_slerp`。"""
    print("\n" + "-" * 78)
    print("[probe] (2) quat_slerp_batch 数值检查")
    torch.manual_seed(0)
    n = 64
    q0 = math_utils.normalize(torch.randn(n, 4))
    q1 = math_utils.normalize(torch.randn(n, 4))
    worst = 0.0
    for tau in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        ours = quat_slerp_batch(q0, q1, tau)
        ref = torch.stack([math_utils.quat_slerp(q0[i], q1[i], tau) for i in range(n)])
        # q 与 -q 等价，取两者中较小的偏差
        diff = torch.minimum((ours - ref).abs().max(dim=-1).values,
                             (ours + ref).abs().max(dim=-1).values)
        worst = max(worst, float(diff.max()))
        print(f"    tau={tau:<5} max|ours-ref| = {float(diff.max()):.3e}")
    e0 = float((quat_slerp_batch(q0, q1, 0.0) - q0).abs().max())
    e1 = float((quat_slerp_batch(q0, q1, 1.0) - q1).abs().max())
    same = float((quat_slerp_batch(q0, q0, 0.5) - q0).abs().max())
    print(f"    端点 tau=0: max|d|={e0:.3e}   tau=1: max|d|={e1:.3e}   q0==q1: max|d|={same:.3e}")
    # 顺带证明"官方实现会就地改输入"（这是我们不直接用它做批量插值的原因）
    qa = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    qb = math_utils.normalize(torch.tensor([[-1.0, 0.01, 0.0, 0.0]]))
    qb_before = qb.clone()
    math_utils.quat_slerp(qa[0], qb[0], 0.5)
    print(f"    官方 quat_slerp 就地修改输入: {'是' if not torch.allclose(qb, qb_before) else '否'}"
          f"（q2 从 {qb_before.tolist()[0]} 变成 {qb.tolist()[0]}）")
    print(f"[probe] (2) 结论: 与官方单样本实现最大偏差 {worst:.3e}"
          f"（{'一致' if worst < 1e-5 else '**偏差过大**'}）")
    if worst > 1e-5 or e0 > 1e-6 or e1 > 1e-6:
        raise RuntimeError("quat_slerp_batch 数值检查未通过")


def _old_update_command(self):
    """旧实现（仅用于 A/B 对比）：姿态不插值，直接取终点。"""
    dt = self._env.step_dt
    self.elapsed_time += dt
    alpha = (self.elapsed_time / self.T_traj).clamp(0.0, 1.0).unsqueeze(-1)
    self.pose_command_b[:, :3] = (
        self.pose_start_b[:, :3] + alpha * (self.pose_end_b[:, :3] - self.pose_start_b[:, :3])
    )
    self.pose_command_b[:, 3:] = self.pose_end_b[:, 3:]


def _angle_between(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """两个四元数（wxyz）之间的最小旋转角（rad）。"""
    dot = (qa * qb).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _check_update_command_slerp(env, term) -> None:
    """受控测试 `_update_command`：姿态是否按 T_traj 插值（而不是一步跳到终点）。

    做法：直接给命令项设定 ``pose_start_b = q_start``、``pose_end_b = q_end``、
    ``T_traj = 1 s``、``elapsed_time = 0``，然后反复调用 ``_update_command()``：

    * 新实现（slerp）：每个 sim step 只走 ``dt/T_traj`` 比例，并与**官方单样本
      ``quat_slerp`` 逐样本对比**（应一致）；
    * 旧实现（monkeypatch 回来）：第一步就跳到 ``q_end``（单步跳变 = 全部夹角）。
    """
    print("\n" + "-" * 78)
    print("[probe] (3) _update_command 姿态插值受控测试（不依赖复位/重采样时机）")
    torch.manual_seed(0)
    n = env.num_envs
    # 构造"夹角明显"的起止姿态对（30°~170°）：
    # 若两端几乎共线（|dot|≈1），官方 quat_slerp 会走"直接返回 q1"的退化分支
    # （`abs(abs(d)-1) < eps*4`），此时两者本来就不该一样，会掩盖真正的实现差异。
    q_start = math_utils.normalize(torch.randn(n, 4, device=env.device))
    axis = math_utils.normalize(torch.randn(n, 3, device=env.device))
    delta_angle = torch.empty(n, device=env.device).uniform_(math.radians(30.0), math.radians(170.0))
    q_end = math_utils.quat_mul(q_start, math_utils.quat_from_angle_axis(delta_angle, axis))
    # 保证 s0 的 "blend=0" 语义不干扰：这里直接测插值本身
    term.cfg.target_blend_pos = 1.0
    term.cfg.target_blend_orn = 1.0
    dt = env.step_dt
    total_angle = float(_angle_between(q_start, q_end).max())

    def _run(use_old: bool):
        original = HeightInvariantEECommand._update_command
        if use_old:
            HeightInvariantEECommand._update_command = _old_update_command
        try:
            term.pose_start_b[:, :3] = 0.0
            term.pose_end_b[:, :3] = 0.0
            term.pose_start_b[:, 3:] = q_start
            term.pose_end_b[:, 3:] = q_end
            term.T_traj[:] = 1.0
            term.elapsed_time[:] = 0.0
            prev = q_start.clone()
            max_step, max_err = 0.0, 0.0
            for step_i in range(12):
                term._update_command()
                cur = term.pose_command_b[:, 3:]
                max_step = max(max_step, float(_angle_between(cur, prev).max()))
                alpha = (term.elapsed_time / term.T_traj).clamp(0.0, 1.0)
                # 用 clone 传给官方实现：它内部会 `q2 *= -1.0` 就地改输入，
                # 直接传 pose_end_b 的切片会把命令项自己的目标翻转（另一处坑）。
                ref = torch.stack([
                    math_utils.quat_slerp(q_start[i].clone(), q_end[i].clone(), float(alpha[i]))
                    for i in range(n)
                ])
                # 注意：这里用**逐分量**偏差而不是夹角。两个四元数几乎相等时，
                # `2*acos(|dot|)` 的 float32 分辨率只有 ~1e-3 rad（0.05°），
                # 会把"实现完全一致"误判成"有偏差"。
                diff = torch.minimum((cur - ref).abs().max(dim=-1).values,
                                     (cur + ref).abs().max(dim=-1).values)
                err = float(diff.max())
                if not use_old:
                    print(f"      step {step_i:>2}  alpha={float(alpha[0]):.4f}  "
                          f"max|ours-ref|={err:.3e}  "
                          f"单步变化={math.degrees(float(_angle_between(cur, prev).max())):.4f} deg")
                max_err = max(max_err, err)
                prev = cur.clone()
            return max_step, max_err
        finally:
            HeightInvariantEECommand._update_command = original

    new_step, new_err = _run(use_old=False)
    old_step, _ = _run(use_old=True)
    print(f"    T_traj=1.0s, step_dt={dt:.4f}s, 起点到终点最大夹角 = {math.degrees(total_angle):.2f}°")
    print(f"    新实现（slerp）: 单步最大姿态变化 = {math.degrees(new_step):.3f}°"
          f"   |与官方 quat_slerp 的最大分量偏差| = {new_err:.3e}")
    print(f"    旧实现（直取终点）: 单步最大姿态变化 = {math.degrees(old_step):.3f}°")
    print(f"    -> 重采样瞬间的姿态跳变从 {math.degrees(old_step):.1f}° 降到 "
          f"{math.degrees(new_step):.3f}°（约 1/{math.degrees(old_step)/max(math.degrees(new_step),1e-9):.0f}）")
    if new_err > 1e-5 or new_step > old_step:
        raise RuntimeError("_update_command 的 slerp 插值不符合预期")


def _arm_disturbance_stats(env, term, steps: int, blend_pos: float, blend_orn: float,
                           velocity_ranges: tuple) -> dict:
    """给定 blend 设置跑 steps 步，统计臂关节速度 / 底盘角速度 / 倾角 / EE 跟踪误差。"""
    term.cfg.target_blend_pos = blend_pos
    term.cfg.target_blend_orn = blend_orn
    vel_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    vel_ranges.lin_vel_x, vel_ranges.lin_vel_y, vel_ranges.ang_vel_z = velocity_ranges

    env.reset()
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints("arm_joint[1-6]")
    ee_idx = term.ee_body_idx
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    arm_vel_sq, base_ang_sq, ee_err, tilts = [], [], [], []
    settle = max(1, steps // 3)          # 前 1/3 步只用于过渡（复位瞬态），不计入统计
    prev_len = env.episode_length_buf.clone()
    for k in range(steps):
        env.step(actions)
        buf_len = env.episode_length_buf
        fresh = buf_len < prev_len              # 这一步有 env 复位 → 其瞬态不计入
        prev_len = buf_len.clone()
        if k < settle:
            continue
        valid = ~fresh
        if not bool(valid.any()):
            continue
        arm_vel_sq.append(robot.data.joint_vel[valid][:, arm_ids].pow(2).mean().item())
        base_ang_sq.append(robot.data.root_ang_vel_b[valid].pow(2).sum(dim=-1).mean().item())
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[valid, 2], -1.0, 1.0))
        tilts.append(tilt)
        ee_pos_b, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w[valid], robot.data.root_quat_w[valid],
            robot.data.body_pos_w[valid, ee_idx], robot.data.body_quat_w[valid, ee_idx],
        )
        ee_err.append((ee_pos_b - term.pose_command_b[valid, :3]).norm(dim=-1).mean().item())
    tilt_all = torch.cat(tilts)
    return {
        "arm_vel_rms": math.sqrt(sum(arm_vel_sq) / len(arm_vel_sq)),
        "base_ang_vel_rms": math.sqrt(sum(base_ang_sq) / len(base_ang_sq)),
        "tilt_mean_deg": math.degrees(float(tilt_all.mean())),
        "tilt_p99_deg": math.degrees(float(tilt_all.quantile(0.99))),
        "tilt_max_deg": math.degrees(float(tilt_all.max())),
        "ee_pos_err_mean": sum(ee_err) / len(ee_err),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    # 关掉外力/推力事件：否则机器人会被推倒，测量到的全是"摔倒"而不是"臂扰动底盘"
    for _name in ("randomize_apply_external_force_torque", "push_robot"):
        if getattr(env_cfg.events, _name, None) is not None:
            setattr(env_cfg.events, _name, None)
    # 让重采样足够频繁，便于观察"重采样瞬间"的姿态跳变
    env_cfg.commands.ee_pose.resampling_time_range = (0.20, 0.30)
    env_cfg.commands.ee_pose.ranges.T_traj = (0.10, 0.20)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    term = env.command_manager.get_term("ee_pose")
    # 记录 base_velocity 的原始区间：课程阶段检查会把 common_step_counter 推到 75k 之后，
    # 那些 base_velocity_lin_vel_x_s4/s5... 课程项会顺带改掉它，测量前要复原。
    vel_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    orig_vel_ranges = (tuple(vel_ranges.lin_vel_x), tuple(vel_ranges.lin_vel_y),
                       tuple(vel_ranges.ang_vel_z))

    print("\n" + "=" * 78)
    print(f"[probe] task={args_cli.task} num_envs={env.num_envs} steps={args_cli.steps} "
          f"ee resampling={env_cfg.commands.ee_pose.resampling_time_range} "
          f"T_traj={env_cfg.commands.ee_pose.ranges.T_traj}")
    print(f"[probe] 初始 target_blend_pos={float(term.cfg.target_blend_pos)} "
          f"target_blend_orn={float(term.cfg.target_blend_orn)}   （Stage 0）")

    _check_curriculum_stages(env)
    _check_slerp()


    _check_update_command_slerp(env, term)

    print("\n" + "-" * 78)
    print("[probe] (4) 臂扰动机制 A/B（同样的零动作，只改 EE 目标混合比例）")
    # 恢复真实的重采样节奏（part 3 用了 0.2~0.3 s 的快速重采样）
    env_cfg.commands.ee_pose.resampling_time_range = (5.0, 5.0)
    env_cfg.commands.ee_pose.ranges.T_traj = (1.0, 3.0)
    term.cfg.resampling_time_range = (5.0, 5.0)
    term.cfg.ranges.T_traj = (1.0, 3.0)
    zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    for vel_cmd, title in ((zero, "站立（速度命令恒 0）"),
                           (orig_vel_ranges, f"速度命令 {orig_vel_ranges[0]}")):
        s0 = _arm_disturbance_stats(env, term, args_cli.steps, 0.0, 0.0, vel_cmd)
        s3 = _arm_disturbance_stats(env, term, args_cli.steps, 1.0, 1.0, vel_cmd)
        print(f"    [{title}]")
        for name, key in (("臂关节速度 RMS", "arm_vel_rms"),
                          ("底盘角速度 RMS", "base_ang_vel_rms"),
                          ("平均倾角 (deg)", "tilt_mean_deg"),
                          ("倾角 p99 (deg)", "tilt_p99_deg"),
                          ("最大倾角 (deg)", "tilt_max_deg"),
                          ("EE 位置跟踪误差 (m)", "ee_pos_err_mean")):
            print(f"      {name:<20} s0(blend=0) = {s0[key]:.4f}   s3(blend=1) = {s3[key]:.4f}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    # 注意：Isaac 脚本里不要用 exit()/sys.exit()（会抛 SystemExit）
    os._exit(0)
