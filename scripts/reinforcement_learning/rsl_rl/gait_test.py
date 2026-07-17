"""
步态检测测试脚本
用于记录搭载机械臂的机器狗低层运动控制策略在推理时，四条腿（轮）的
触地/腾空时序，并绘制步态图（gait diagram）、计算占空比/相位等指标。

使用方法（与原始 play.py 保持一致的参数风格）:
    python play_gait_test.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-play-v0 \
        --checkpoint logs/rsl_rl/<run>/model_<iter>.pt \
        --record_time 8.0 \
        --cmd_vx 0.5 --cmd_vy 0.0 --cmd_wz 0.0 \
        [--terrain flat] \
        [--env_id 0] \
        [--num_envs 1] \
        [--method air_time] \
        [--save_fig gait_diagram.png] \
        [--save_data gait_log.npz] \
        [--headless]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------- #
#  命令行参数
# --------------------------------------------------------------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

TERRAIN_NAMES = ["flat", "stairs", "stairs_inv", "boxes",
                  "random_rough", "slope", "slope_inv", "mixed"]

parser = argparse.ArgumentParser(description="步态检测测试脚本")
# ── 原始 play.py 已有的参数 ──────────────────────────────────────────────────
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ── 步态测试专用参数 ─────────────────────────────────────────────────────────
parser.add_argument("--record_time", type=float, default=8.0,
                    help="记录总时长（秒），默认 8.0")
parser.add_argument("--warmup_time", type=float, default=2.0,
                    help="记录前的预热时间（秒），用于让步态进入稳态，默认 2.0")
parser.add_argument("--cmd_vx", type=float, default=5.0, help="固定前向速度命令 (m/s)")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="固定侧向速度命令 (m/s)")
parser.add_argument("--cmd_wz", type=float, default=1.0, help="固定角速度命令 (rad/s)")
parser.add_argument("--env_id", type=int, default=0, help="记录哪个环境的数据（默认 0）")
parser.add_argument("--method", type=str, default="air_time", choices=["air_time", "force"],
                    help="接触判定方式：air_time(用current_contact_time) 或 force(用力阈值)")
parser.add_argument("--force_threshold", type=float, default=1.0,
                    help="method=force 时的接触力阈值 (N)，默认 1.0")
parser.add_argument("--debounce_steps", type=int, default=2,
                    help="去抖动最小持续步数，短于该值的翻转会被滤除，默认 2")
parser.add_argument("--wheel_body_regex", type=str, default=".*_wheel",
                    help="四条腿(轮)在contact sensor中对应的body名正则，默认 .*_wheel")
parser.add_argument("--wheel_joint_regex", type=str, default=".*_wheel_joint",
                    help="四个轮子对应的关节名正则（用于记录转速），默认 .*_wheel_joint")
parser.add_argument("--save_fig", type=str, default=None, help="步态图保存路径")
parser.add_argument("--save_data", type=str, default=None, help="原始接触数据保存路径(.npz)")
parser.add_argument("--terrain", type=str, default="flat", choices=TERRAIN_NAMES,
                    help=f"测试使用的地形类型，可选: {TERRAIN_NAMES}（默认 flat）")
# ── RSL-RL CLI 参数（checkpoint 等由此注入）──────────────────────────────────
cli_args.add_rsl_rl_args(parser)
# ── AppLauncher 参数（headless、device 等）───────────────────────────────────
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
#  正式 import（Isaac Sim 启动后才能 import）
# --------------------------------------------------------------------------- #
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from rl_training.tasks.manager_based.locomotion.velocity.mdp.terrains import (
    TERRAIN_CFGS,
)

_missing = set(TERRAIN_NAMES) - set(TERRAIN_CFGS.keys())
if _missing:
    raise KeyError(
        f"TERRAIN_NAMES 中的 {_missing} 在 terrains.py 的 TERRAIN_CFGS 里不存在，"
        f"请检查两边名字是否一致。TERRAIN_CFGS 现有 keys: {list(TERRAIN_CFGS.keys())}"
    )


# --------------------------------------------------------------------------- #
#  步态后处理辅助函数
# --------------------------------------------------------------------------- #
def debounce(contact_bool: np.ndarray, min_steps: int) -> np.ndarray:
    """滤除短于 min_steps 的翻转抖动（对每条腿的 1D 布尔序列操作）。"""
    arr = contact_bool.copy()
    n = len(arr)
    i = 0
    while i < n:
        j = i
        while j < n and arr[j] == arr[i]:
            j += 1
        if (j - i) < min_steps and i > 0:
            arr[i:j] = arr[i - 1]
        i = j
    return arr


def bool_to_segments(contact_bool: np.ndarray, time_arr: np.ndarray):
    """把 0/1 序列转成 (start_time, duration) 的 True 区间列表。"""
    segments = []
    start = None
    for k in range(len(contact_bool)):
        if contact_bool[k] and start is None:
            start = time_arr[k]
        elif (not contact_bool[k]) and start is not None:
            segments.append((start, time_arr[k] - start))
            start = None
    if start is not None:
        segments.append((start, time_arr[-1] - start))
    return segments


def rising_edges(contact_bool: np.ndarray, time_arr: np.ndarray):
    """返回从 0->1 跳变(触地开始)的时刻列表。"""
    edges = []
    for k in range(1, len(contact_bool)):
        if contact_bool[k] and not contact_bool[k - 1]:
            edges.append(time_arr[k])
    return edges


# --------------------------------------------------------------------------- #
#  主函数
# --------------------------------------------------------------------------- #
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    env_id = args_cli.env_id

    # ── 1. 环境配置（沿用原始逻辑）───────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    selected_terrain_cfg = TERRAIN_CFGS[args_cli.terrain]
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator = selected_terrain_cfg
    print(f"[INFO] 本次测试使用地形: '{args_cli.terrain}'")

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # 关闭观测噪声、外力扰动、课程，保证步态尽量干净、稳定
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.command_levels = None

    total_time = args_cli.warmup_time + args_cli.record_time
    env_cfg.episode_length_s = total_time + 1.0  # 留一点余量防止中途 reset

    # ── 固定速度命令：把 resample 区间拉长到覆盖整个记录窗口，
    #    并把命令范围收窄到我们指定的固定值，从而得到稳定周期的步态 ──────────
    for attr_name in dir(env_cfg.commands):
        cmd_term_cfg = getattr(env_cfg.commands, attr_name)
        if hasattr(cmd_term_cfg, "resampling_time_range"):
            cmd_term_cfg.resampling_time_range = (total_time + 10.0, total_time + 10.0)
            print(f"[INFO] 已将命令 '{attr_name}' 的重采样间隔固定为覆盖整个记录窗口")
        if hasattr(cmd_term_cfg, "ranges"):
            ranges = cmd_term_cfg.ranges
            if hasattr(ranges, "lin_vel_x"):
                ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
            if hasattr(ranges, "lin_vel_y"):
                ranges.lin_vel_y = (args_cli.cmd_vy, args_cli.cmd_vy)
            if hasattr(ranges, "ang_vel_z"):
                ranges.ang_vel_z = (args_cli.cmd_wz, args_cli.cmd_wz)
            print(f"[INFO] 已将命令 '{attr_name}' 固定为 "
                  f"vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}, wz={args_cli.cmd_wz}")

    # ── 2/3. 构建 + 包装环境（沿用原始逻辑）─────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 4. 定位 checkpoint（沿用原始逻辑）────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    if args_cli.save_fig is not None:
        save_fig_path = args_cli.save_fig
    else:
        save_fig_path = os.path.join(os.path.dirname(args_cli.checkpoint),
                                      f"gait_diagram_{args_cli.terrain}.png")
    if args_cli.save_data is not None:
        save_data_path = args_cli.save_data
    else:
        save_data_path = os.path.join(os.path.dirname(args_cli.checkpoint),
                                       f"gait_log_{args_cli.terrain}.npz")
    print(f"[INFO] 步态图将保存至: {save_fig_path}")
    print(f"[INFO] 原始接触数据将保存至: {save_data_path}")

    # ── 5. 加载模型（沿用原始逻辑）────────────────────────────────────────────
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerHis":
        runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 6. 定位四条腿(轮)在 contact sensor 中的 body index ──────────────────
    raw_env = env.unwrapped
    sim_dt = raw_env.step_dt

    contact_sensor = raw_env.scene["contact_forces"]
    print(f"[INFO] contact sensor 跟踪的全部 body: {contact_sensor.body_names}")

    leg_ids, leg_names = contact_sensor.find_bodies(args_cli.wheel_body_regex)
    if len(leg_ids) == 0:
        raise RuntimeError(
            f"未能用正则 '{args_cli.wheel_body_regex}' 在 contact sensor 的 body 中找到匹配项，"
            f"请核对实际 body 命名（见上方打印的 body_names 列表）后通过 --wheel_body_regex 指定。"
        )
    print(f"[INFO] 用于步态记录的腿(轮) body: {leg_names} (ids={leg_ids})")

    # ── 6b. 定位四个轮子在机器人关节中的 joint index（用于记录转速）─────────
    robot = raw_env.scene["robot"]
    print(f"[INFO] 机器人全部关节: {robot.joint_names}")

    wheel_joint_ids, wheel_joint_names = robot.find_joints(args_cli.wheel_joint_regex)
    if len(wheel_joint_ids) == 0:
        raise RuntimeError(
            f"未能用正则 '{args_cli.wheel_joint_regex}' 在机器人关节中找到匹配项，"
            f"请核对实际关节命名（见上方打印的 joint_names 列表）后通过 --wheel_joint_regex 指定。"
        )
    print(f"[INFO] 用于转速记录的轮子关节: {wheel_joint_names} (ids={wheel_joint_ids})")

    warmup_steps = max(1, int(args_cli.warmup_time / sim_dt))
    record_steps = max(1, int(args_cli.record_time / sim_dt))
    print(f"[INFO] sim_dt={sim_dt:.4f}s | 预热步数={warmup_steps} | 记录步数={record_steps}")

    # ── 7. 预热 + 记录循环 ────────────────────────────────────────────────────
    obs = env.get_observations()

    for _ in range(warmup_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        obs = obs.clone()

    contact_log = []      # 每一帧一个 (num_legs,) 的 bool 数组
    force_log = []         # 每一帧一个 (num_legs,) 的力大小数组（备用/参考用）
    wheel_vel_log = []     # 每一帧一个 (num_wheels,) 的关节角速度数组 (rad/s)
    time_log = []
    current_time = 0.0

    for _ in range(record_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        obs = obs.clone()

        data = contact_sensor.data
        forces = data.net_forces_w[env_id, leg_ids, :]          # (num_legs, 3)
        force_norm = torch.norm(forces, dim=-1).cpu().numpy()   # (num_legs,)

        if args_cli.method == "air_time":
            contact_time = data.current_contact_time[env_id, leg_ids].cpu().numpy()
            in_contact = contact_time > 0.0
        else:  # force
            in_contact = force_norm > args_cli.force_threshold

        # 轮子关节角速度 (rad/s)
        joint_vel = robot.data.joint_vel[env_id, wheel_joint_ids].cpu().numpy()  # (num_wheels,)

        contact_log.append(in_contact)
        force_log.append(force_norm)
        wheel_vel_log.append(joint_vel)
        time_log.append(current_time)
        current_time += sim_dt

    contact_log = np.array(contact_log)      # (T, num_legs) bool
    force_log = np.array(force_log)          # (T, num_legs) float
    wheel_vel_log = np.array(wheel_vel_log)  # (T, num_wheels) float, rad/s
    time_log = np.array(time_log)            # (T,)

    # ── 8. 去抖动 ─────────────────────────────────────────────────────────────
    contact_clean = np.zeros_like(contact_log)
    for i in range(contact_log.shape[1]):
        contact_clean[:, i] = debounce(contact_log[:, i], args_cli.debounce_steps)

    # ── 9. 保存原始数据 ───────────────────────────────────────────────────────
    np.savez(save_data_path,
             contact=contact_clean, contact_raw=contact_log,
             force=force_log, time=time_log, leg_names=np.array(leg_names),
             wheel_vel=wheel_vel_log, wheel_joint_names=np.array(wheel_joint_names))
    print(f"[INFO] 原始接触数据已保存至: {save_data_path}")

    # ── 10. 计算步态量化指标 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"          步态量化指标 — 地形: {args_cli.terrain}, "
          f"命令: vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}, wz={args_cli.cmd_wz}")
    print("=" * 60)

    duty_factors = contact_clean.mean(axis=0)
    ref_edges = rising_edges(contact_clean[:, 0], time_log)
    if len(ref_edges) >= 2:
        period = float(np.median(np.diff(ref_edges)))
        freq = 1.0 / period if period > 0 else float("nan")
    else:
        period, freq = float("nan"), float("nan")

    print(f"  参考腿: {leg_names[0]} | 步态周期 ≈ {period:.3f} s | 频率 ≈ {freq:.3f} Hz")
    print("-" * 60)
    print(f"  {'腿名':<12} {'占空比':>8} {'相位差(相对参考腿)':>20}")
    print("-" * 60)
    for i, name in enumerate(leg_names):
        edges_i = rising_edges(contact_clean[:, i], time_log)
        if len(ref_edges) > 0 and len(edges_i) > 0 and not np.isnan(period) and period > 0:
            # 取第一个参考触地时刻之后最近的一次该腿触地，计算相位差
            ref_t0 = ref_edges[0]
            candidates = [e for e in edges_i if e >= ref_t0]
            phase = ((candidates[0] - ref_t0) % period) / period if candidates else float("nan")
        else:
            phase = float("nan")
        print(f"  {name:<12} {duty_factors[i]:>8.3f} {phase:>20.3f}")
    print("=" * 60 + "\n")

    # ── 10b. 打印轮速统计 ────────────────────────────────────────────────────
    print("=" * 60)
    print("          轮子转速统计 (rad/s)")
    print("=" * 60)
    print(f"  {'轮子关节名':<10} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print("-" * 60)
    wheel_vel_mean = wheel_vel_log.mean(axis=0)
    wheel_vel_std = wheel_vel_log.std(axis=0)
    wheel_vel_min = wheel_vel_log.min(axis=0)
    wheel_vel_max = wheel_vel_log.max(axis=0)
    for i, name in enumerate(wheel_joint_names):
        print(f"  {name:<10} {wheel_vel_mean[i]:>10.3f} {wheel_vel_std[i]:>10.3f} "
              f"{wheel_vel_min[i]:>10.3f} {wheel_vel_max[i]:>10.3f}")
    print("=" * 60 + "\n")

    # ── 11. 绘制步态图（甘特图）+ 轮速曲线 ───────────────────────────────────
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax_gait, ax_vel) = plt.subplots(
        2, 1, figsize=(12, 0.6 * len(leg_names) + 4.0), sharex=True,
        gridspec_kw={"height_ratios": [len(leg_names), 2.5]},
    )

    # 上半部分：步态甘特图
    for i, name in enumerate(leg_names):
        segments = bool_to_segments(contact_clean[:, i], time_log)
        ax_gait.broken_barh(segments, (i - 0.4, 0.4), facecolors="black")

    ax_gait.set_yticks(range(len(leg_names)))
    ax_gait.set_yticklabels(leg_names)
    ax_gait.set_title(
        f"步态图 (黑色=触地/stance) — 地形: {args_cli.terrain}, "
        f"vx={args_cli.cmd_vx} m/s | 周期≈{period:.2f}s 频率≈{freq:.2f}Hz",
        fontsize=12,
    )
    ax_gait.grid(True, axis="x", alpha=0.3)

    # 下半部分：四个轮子的转速曲线
    for i, name in enumerate(wheel_joint_names):
        ax_vel.plot(time_log, wheel_vel_log[:, i], label=name, linewidth=1.2)
    ax_vel.set_xlabel("时间 (s)", fontsize=11)
    ax_vel.set_ylabel("轮速 (rad/s)", fontsize=11)
    ax_vel.set_title("四个轮子的关节角速度", fontsize=12)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc="upper right", fontsize=9, ncol=min(4, len(wheel_joint_names)))

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] 步态图（含轮速曲线）已保存至: {save_fig_path}")

    # ── 12. 关闭环境 ──────────────────────────────────────────────────────────
    cmd_manager = raw_env.command_manager
    for term_name in cmd_manager.active_terms:
        term = cmd_manager.get_term(term_name)
        if hasattr(term, "set_debug_vis"):
            term.set_debug_vis(False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()