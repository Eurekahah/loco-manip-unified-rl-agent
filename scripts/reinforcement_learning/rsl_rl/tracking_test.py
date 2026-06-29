"""
跟踪性能测试脚本
用于测试搭载机械臂的机器狗低层运动控制策略的速度/姿态跟踪性能。

使用方法（与原始 play.py 保持一致的参数风格）:
    python play_tracking_test.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-play-v0 \
        --checkpoint logs/rsl_rl/<run>/model_<iter>.pt \
        [--num_resample 10] \
        [--resample_interval 5.0] \
        [--env_id 0] \
        [--save_fig tracking_result.png] \
        [--num_envs 1] \
        [--headless]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------- #
#  命令行参数（与原始 play.py 保持一致，追加测试专用参数）
# --------------------------------------------------------------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402  （与原始 play.py 相同的本地模块）

parser = argparse.ArgumentParser(description="跟踪性能测试脚本")
# ── 原始 play.py 已有的参数 ──────────────────────────────────────────────────
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ── 测试专用参数 ─────────────────────────────────────────────────────────────
parser.add_argument("--num_resample", type=int, default=10,
                    help="命令重采样次数（默认 10）")
parser.add_argument("--resample_interval", type=float, default=5.0,
                    help="每段命令持续时间（秒，默认 5.0）")
parser.add_argument("--env_id", type=int, default=0,
                    help="记录哪个环境的数据（默认 0）")
parser.add_argument("--save_fig", type=str, default=None,
                    help="图像保存路径（默认根据 checkpoint 路径自动生成 tracking_result.png）")
# ── RSL-RL CLI 参数（checkpoint 等由此注入）──────────────────────────────────
cli_args.add_rsl_rl_args(parser)
# ── AppLauncher 参数（headless、device 等）───────────────────────────────────
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args   # 清理 hydra 不认识的参数

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
#  正式 import（Isaac Sim 启动后才能 import）
# --------------------------------------------------------------------------- #
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无显示器时也能保存图片
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner, DistillationRunner, OnPolicyRunnerHis

from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnvCfg,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401  触发任务注册


# --------------------------------------------------------------------------- #
#  辅助：读取当前命令值
# --------------------------------------------------------------------------- #
def get_commands(raw_env, env_id: int):
    """
    从 command_manager 读取 base_velocity / body_pose 命令。
    返回 dict，value 为 numpy 1-D array，读取失败时为 None。
    """
    cmd = {}
    cmd_manager = raw_env.command_manager
    for name in ["base_velocity", "body_pose"]:
        try:
            tensor = cmd_manager.get_command(name)   # (num_envs, dim)
            cmd[name] = tensor[env_id].cpu().numpy()
        except Exception:
            cmd[name] = None
    return cmd


# --------------------------------------------------------------------------- #
#  辅助：读取底盘实际状态
# --------------------------------------------------------------------------- #
def get_obs_state(raw_env, env_id: int):
    """
    返回:
        vel  : np.ndarray [v_x, v_y, w_z]        body frame
        pose : np.ndarray [height, pitch, roll]   世界系高度 + body 姿态
    """
    from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat

    robot = raw_env.scene["robot"]

    root_lin_vel_w = robot.data.root_lin_vel_w[env_id]   # (3,) world
    root_ang_vel_w = robot.data.root_ang_vel_w[env_id]   # (3,) world
    root_quat_w    = robot.data.root_quat_w[env_id]       # (4,) w,x,y,z

    # 转到 body frame
    lin_vel_b = quat_apply_inverse(root_quat_w.unsqueeze(0),
                                    root_lin_vel_w.unsqueeze(0)).squeeze(0)
    ang_vel_b = quat_apply_inverse(root_quat_w.unsqueeze(0),
                                    root_ang_vel_w.unsqueeze(0)).squeeze(0)

    v_x = lin_vel_b[0].item()
    v_y = lin_vel_b[1].item()
    w_z = ang_vel_b[2].item()

    height = robot.data.root_pos_w[env_id, 2].item()

    roll_t, pitch_t, _ = euler_xyz_from_quat(root_quat_w.unsqueeze(0))
    pitch = pitch_t[0].item()
    roll  = roll_t[0].item()

    vel  = np.array([v_x, v_y, w_z],      dtype=np.float32)
    pose = np.array([height, pitch, roll], dtype=np.float32)
    return vel, pose


# --------------------------------------------------------------------------- #
#  主函数（与原始 play.py 相同，用 hydra_task_config 装饰）
# --------------------------------------------------------------------------- #
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    env_id         = args_cli.env_id
    n_resample     = args_cli.num_resample
    dt_segment     = args_cli.resample_interval

    # ── 1. 与原始 play.py 完全相同的环境配置逻辑 ─────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    env_cfg.seed       = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # 关闭观测噪声、外力扰动、课程
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.command_levels = None

    # ── 测试专用：将重采样间隔固定为 dt_segment，防止环境自动打断测试节奏 ────
    for attr_name in dir(env_cfg.commands):
        cmd_term_cfg = getattr(env_cfg.commands, attr_name)
        if hasattr(cmd_term_cfg, "resampling_time_range"):
            cmd_term_cfg.resampling_time_range = (dt_segment, dt_segment)
            print(f"[INFO] 已将命令 '{attr_name}' 的重采样间隔固定为 {dt_segment}s")
        # if hasattr(cmd_term_cfg, "debug_vis"):
        #     cmd_term_cfg.debug_vis = False

    # ── 2. 与原始 play.py 相同：构建环境 + 可选 MARL 转换 ───────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── 3. 与原始 play.py 相同：用 RslRlVecEnvWrapper 包装 ──────────────────
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 4. 与原始 play.py 相同：定位 checkpoint ──────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    if args_cli.save_fig is not None:
        save_fig_path = args_cli.save_fig
    else:
        save_fig_path = os.path.join(os.path.dirname(args_cli.checkpoint), "tracking_result.png")
    print(f"[INFO] 跟踪结果图像将保存至: {save_fig_path}")

    # ── 5. 与原始 play.py 相同：用 Runner 加载模型 ───────────────────────────
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                                log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(),
                                    log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerHis":
        runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 6. 计算步数 ───────────────────────────────────────────────────────────
    # RslRlVecEnvWrapper 套了两层，通过 .unwrapped 一路拿到 ManagerBasedRLEnv
    raw_env = env.unwrapped
    sim_dt  = raw_env.step_dt
    steps_per_segment = max(1, int(dt_segment / sim_dt))
    total_steps       = steps_per_segment * n_resample

    print(f"[INFO] sim_dt={sim_dt:.4f}s | 每段步数={steps_per_segment} | "
          f"总步数={total_steps} | 总时长={total_steps * sim_dt:.1f}s")

    # ── 7. 数据容器 ───────────────────────────────────────────────────────────
    times    = []
    act_vel  = []   # [v_x, v_y, w_z]
    cmd_vel  = []
    act_pose = []   # [height, pitch, roll]
    cmd_pose = []

    # ── 8. Reset ──────────────────────────────────────────────────────────────
    obs = env.get_observations()
    current_time = 0.0

    # ── 9. 主采集循环 ─────────────────────────────────────────────────────────
    for seg in range(n_resample):
        
        print(f"[段 {seg+1:02d}/{n_resample}] 开始，时刻={current_time:.2f}s")

        for _ in range(steps_per_segment):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            obs =obs.clone()  # 避免后续被环境修改

            # 记录实际状态与命令
            a_vel, a_pose = get_obs_state(raw_env, env_id)
            cmds          = get_commands(raw_env, env_id)

            c_vel  = cmds["base_velocity"] if cmds["base_velocity"] is not None \
                     else np.zeros(3)
            c_pose = cmds["body_pose"]     if cmds["body_pose"]     is not None \
                     else np.zeros(3)

            times.append(current_time)
            act_vel.append(a_vel)
            cmd_vel.append(c_vel)
            act_pose.append(a_pose)
            cmd_pose.append(c_pose)

            current_time += sim_dt

    print("[INFO] 数据采集完毕，开始绘图…")

    # ── 10. 转 numpy ──────────────────────────────────────────────────────────
    times    = np.array(times)
    act_vel  = np.array(act_vel)
    cmd_vel  = np.array(cmd_vel)
    act_pose = np.array(act_pose)
    cmd_pose = np.array(cmd_pose)

    # ── 11. 绘图 ──────────────────────────────────────────────────────────────
    # 设置中文字体，并解决负号显示异常的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

    vel_labels  = ["v_x (m/s)", "v_y (m/s)", "ω_z (rad/s)"]
    pose_labels = ["height (m)", "pitch (rad)", "roll (rad)"]

    # 改为6行1列，每个子图宽度占满
    fig = plt.figure(figsize=(12, 18))  # 宽度12英寸，高度18英寸（6*3）
    fig.suptitle("机器狗低层控制跟踪性能测试", fontsize=16, fontweight="bold", y=0.995)

    # 重采样时刻竖线
    seg_times = [i * steps_per_segment * sim_dt for i in range(n_resample + 1)]

    def add_segment_lines(ax):
        for st in seg_times[1:-1]:
            ax.axvline(st, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    # 创建6个子图，垂直排列
    for i in range(6):
        ax = fig.add_subplot(6, 1, i+1)  # 6行1列，第i+1个子图
        
        if i < 3:  # 前3个是速度
            ax.plot(times, cmd_vel[:, i], color="royalblue", linewidth=1.5,
                    label="命令", linestyle="--")
            ax.plot(times, act_vel[:, i], color="tomato", linewidth=1.2,
                    label="实际")
            add_segment_lines(ax)
            ax.set_ylabel(vel_labels[i], fontsize=10)
            ax.set_title(f"底盘速度 — {vel_labels[i]}", fontsize=11, pad=10)
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3)
            
        else:  # 后3个是姿态
            idx = i - 3
            ax.plot(times, cmd_pose[:, idx], color="mediumseagreen", linewidth=1.5,
                    label="命令", linestyle="--")
            ax.plot(times, act_pose[:, idx], color="darkorange", linewidth=1.2,
                    label="实际")
            add_segment_lines(ax)
            ax.set_ylabel(pose_labels[idx], fontsize=10)
            ax.set_title(f"底盘姿态 — {pose_labels[idx]}", fontsize=11, pad=10)
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3)
        
        # 只给最后一个子图设置x轴标签
        if i == 5:
            ax.set_xlabel("时间 (s)", fontsize=10)
        else:
            ax.set_xlabel("")  # 其他子图不显示x轴标签

    # 自动调整子图间距
    plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.05)

    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] 图像已保存至: {save_fig_path}")

    # ── 12. 打印平均跟踪误差 ──────────────────────────────────────────────────
    vel_err  = np.abs(act_vel  - cmd_vel)
    pose_err = np.abs(act_pose - cmd_pose)

    print("\n" + "=" * 55)
    print("          平均跟踪误差统计（MAE）")
    print("=" * 55)
    print(f"  {'通道':<22} {'MAE':>10}")
    print("-" * 55)
    for i, lbl in enumerate(vel_labels):
        print(f"  {lbl:<22} {vel_err[:, i].mean():>10.4f}")
    for i, lbl in enumerate(pose_labels):
        print(f"  {lbl:<22} {pose_err[:, i].mean():>10.4f}")
    print("-" * 55)
    print(f"  {'速度综合 MAE':<22} {vel_err.mean():>10.4f}")
    print(f"  {'姿态综合 MAE':<22} {pose_err.mean():>10.4f}")
    print("=" * 55 + "\n")

    # ── 13. 关闭环境 ──────────────────────────────────────────────────────────
    cmd_manager = raw_env.command_manager
    for term_name in cmd_manager.active_terms:
        term = cmd_manager.get_term(term_name)
        if hasattr(term, "set_debug_vis"):
            term.set_debug_vis(False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()