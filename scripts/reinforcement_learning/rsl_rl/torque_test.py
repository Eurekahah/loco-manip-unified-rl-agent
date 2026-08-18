"""
全身关节力矩记录测试脚本
用于记录搭载机械臂的机器狗低层运动控制策略在推理时，全身关节
（底盘 4 条腿 × 4 关节 = 16 个 + 机械臂 6 个关节 + 夹爪 1 个关节，共 23 个）
的实时力矩曲线，并绘图、统计均值/方差/最值。

使用方法（与原始 play.py / play_gait_test.py 保持一致的参数风格）:
    python play_joint_torque_test.py \
        --task Flat-Deeprobotics-M20-Piper-WBC-play-v0 \
        --checkpoint logs/rsl_rl/<run>/model_<iter>.pt \
        --record_time 8.0 \
        --cmd_vx 0.5 --cmd_vy 0.0 --cmd_wz 0.0 \
        [--terrain flat] \
        [--env_id 0] \
        [--num_envs 1] \
        [--leg_joint_types hipx,hipy,knee,wheel] \
        [--leg_joint_regex ".*_(hipx|hipy|knee|wheel)_joint"] \
        [--arm_joint_regex "arm_joint[1-6]$"] \
        [--gripper_joint_regex "gripper_joint[1-2]"] \
        [--save_fig joint_torque.png] \
        [--save_data joint_torque_log.npz] \
        [--headless]

说明：
    - 力矩数据来自 robot.data.applied_torque（关节实际被施加的力矩，单位 N·m）。
    - 腿部关节的绘图不再按“每条腿一个子图”，而是按“每种关节类型一个子图”
      （例如 hipx / hipy / knee / wheel），每个子图内画出该类型在 4 条腿上的
      力矩曲线，便于比较同类关节在不同腿上的差异。
    - 每个关节的力矩上限：脚本会尝试自动从 robot.data.joint_effort_limits 读取，
      若成功，会在图中以虚线画出 ±上限；若失败，会打印一个
      EFFORT_LIMITS_PLACEHOLDER 字典（key=关节名，value=None），
      你可以自行在代码中补充真实值后重新绘图。
"""

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------- #
#  命令行参数
# --------------------------------------------------------------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

TERRAIN_NAMES = ["flat", "stairs", "stairs_inv", "boxes",
                  "random_rough", "slope", "slope_inv", "mixed"]

parser = argparse.ArgumentParser(description="全身关节力矩记录测试脚本")
# ── 原始 play.py 已有的参数 ──────────────────────────────────────────────────
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# ── 力矩测试专用参数 ─────────────────────────────────────────────────────────
parser.add_argument("--record_time", type=float, default=8.0,
                    help="记录总时长（秒），默认 8.0")
parser.add_argument("--warmup_time", type=float, default=2.0,
                    help="记录前的预热时间（秒），用于让运动进入稳态，默认 2.0")
parser.add_argument("--cmd_vx", type=float, default=0.5, help="固定前向速度命令 (m/s)")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="固定侧向速度命令 (m/s)")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="固定角速度命令 (rad/s)")
parser.add_argument("--env_id", type=int, default=0, help="记录哪个环境的数据（默认 0）")
parser.add_argument("--leg_joint_types", type=str, default="hipx,hipy,knee,wheel",
                    help="底盘腿部关节的类型（逗号分隔），用于按类型分组绘图，"
                         "默认 hipx,hipy,knee,wheel（每种类型一个子图，"
                         "子图内包含该类型在 4 条腿上的曲线）")
parser.add_argument("--leg_joint_regex", type=str, default=".*_(hipx|hipy|knee|wheel)_joint",
                    help="底盘四条腿关节的正则（应匹配全部 16 个关节），"
                         "默认 .*_(hipx|hipy|knee|wheel)_joint")
parser.add_argument("--arm_joint_regex", type=str, default="arm_joint[1-6]$",
                    help="机械臂 6 个关节的正则，默认 arm_joint[1-6]$")
parser.add_argument("--gripper_joint_regex", type=str, default="gripper_joint[1-2]",
                    help="夹爪关节的正则，默认 gripper_joint[1-2]")
parser.add_argument("--save_fig", type=str, default=None, help="力矩图保存路径")
parser.add_argument("--save_data", type=str, default=None, help="原始力矩数据保存路径(.npz)")
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
#  若无法自动获取力矩上限，供你手动填写的占位字典
#  key: 关节名, value: 力矩上限 (N·m)，请在自动获取失败时填入真实值
# --------------------------------------------------------------------------- #
EFFORT_LIMITS_PLACEHOLDER = {
    # joint 组（髋关节x、髋关节y、膝关节）
    "hl_hipx_joint": 76.4,
    "hl_hipy_joint": 76.4,
    "hl_knee_joint": 76.4,
    "hr_hipx_joint": 76.4,
    "hr_hipy_joint": 76.4,
    "hr_knee_joint": 76.4,
    "fl_hipx_joint": 76.4,
    "fl_hipy_joint": 76.4,
    "fl_knee_joint": 76.4,
    "fr_hipx_joint": 76.4,
    "fr_hipy_joint": 76.4,
    "fr_knee_joint": 76.4,
    # wheel 组（轮子关节）
    "hl_wheel_joint": 21.6,
    "hr_wheel_joint": 21.6,
    "fl_wheel_joint": 21.6,
    "fr_wheel_joint": 21.6,
    "arm_joint1": 100.0,
    "arm_joint2": 100.0,
    "arm_joint3": 100.0,
    "arm_joint4": 100.0,
    "arm_joint5": 100.0,
    "arm_joint6": 100.0,
    "arm_joint7": 100.0,
    "arm_joint8": 100.0,
}


def get_joint_type_for_joint(joint_name: str, leg_joint_types: list[str]) -> str | None:
    """根据关节类型关键字（如 hipx/hipy/knee/wheel）判断某个关节名属于哪种类型，
    找不到则返回 None。匹配方式为在下划线分隔的 token 中查找类型关键字，
    避免像 'hipx' 误匹配到其他包含子串的类型。
    """
    tokens = joint_name.split("_")
    for jtype in leg_joint_types:
        if jtype in tokens:
            return jtype
    # 兜底：如果类型关键字不是完整 token（用户自定义了不规则关节名），退化为子串匹配
    for jtype in leg_joint_types:
        if jtype in joint_name:
            return jtype
    return None


def try_get_effort_limits(
    robot,
    env_id: int,
    joint_ids: list[int],
    joint_names: list[str],
) -> dict[str, float | None]:
    """获取指定关节的力矩上限。

    优先从 robot.data 中读取；如果读取失败、字段不存在或数值异常，
    则使用 EFFORT_LIMITS_PLACEHOLDER 中的默认值。
    """

    MAX_REASONABLE_EFFORT = 1000.0  # N·m

    # 优先从 robot.data 获取
    for attr_name in (
        "joint_effort_limits",
        "joint_effort_limits_sim",
        "effort_limit",
    ):
        if hasattr(robot.data, attr_name):
            try:
                limits_all = getattr(robot.data, attr_name)[env_id].cpu().numpy()
                result = {}

                for jid, name in zip(joint_ids, joint_names):
                    val = float(limits_all[jid])

                    if val > MAX_REASONABLE_EFFORT:
                        print(
                            f"[WARN] 关节 '{name}' 的力矩上限 {val:.2e} N·m 异常大，"
                            "使用默认值。"
                        )
                        # result[name] = EFFORT_LIMITS_PLACEHOLDER.get(name)
                        result[name] = None
                    else:
                        result[name] = val

                return result

            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 读取 robot.data.{attr_name} 失败: {e}")
                break

    # 回退到默认值
    print("[INFO] 使用 EFFORT_LIMITS_PLACEHOLDER 中的力矩上限。")
    return {
        name: EFFORT_LIMITS_PLACEHOLDER.get(name)
        for name in joint_names
    }

# --------------------------------------------------------------------------- #
#  主函数
# --------------------------------------------------------------------------- #
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):

    env_id = args_cli.env_id
    leg_joint_types = [t.strip() for t in args_cli.leg_joint_types.split(",") if t.strip()]

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

    # 关闭观测噪声、外力扰动、课程，保证运动尽量干净、稳定
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.command_levels = None

    total_time = args_cli.warmup_time + args_cli.record_time
    env_cfg.episode_length_s = total_time + 1.0  # 留一点余量防止中途 reset

    # ── 固定速度命令：把 resample 区间拉长到覆盖整个记录窗口，
    #    并把命令范围收窄到我们指定的固定值，从而得到稳定周期的运动 ──────────
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
                                      f"joint_torque_{args_cli.terrain}.png")
    if args_cli.save_data is not None:
        save_data_path = args_cli.save_data
    else:
        save_data_path = os.path.join(os.path.dirname(args_cli.checkpoint),
                                       f"joint_torque_log_{args_cli.terrain}.npz")
    print(f"[INFO] 力矩图将保存至: {save_fig_path}")
    print(f"[INFO] 原始力矩数据将保存至: {save_data_path}")

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

    # ── 6. 定位全身关节：底盘 16 个（4 腿 x 4）+ 机械臂 6 个 + 夹爪 1 个 ──────
    raw_env = env.unwrapped
    sim_dt = raw_env.step_dt

    robot = raw_env.scene["robot"]
    print(f"[INFO] 机器人全部关节 ({len(robot.joint_names)} 个): {robot.joint_names}")

    leg_joint_ids, leg_joint_names = robot.find_joints(args_cli.leg_joint_regex)
    arm_joint_ids, arm_joint_names = robot.find_joints(args_cli.arm_joint_regex)
    gripper_joint_ids, gripper_joint_names = robot.find_joints(args_cli.gripper_joint_regex)

    if len(leg_joint_ids) == 0:
        raise RuntimeError(
            f"未能用正则 '{args_cli.leg_joint_regex}' 匹配到底盘腿部关节，"
            f"请核对上方打印的关节名列表后通过 --leg_joint_regex 指定。"
        )
    if len(arm_joint_ids) == 0:
        raise RuntimeError(
            f"未能用正则 '{args_cli.arm_joint_regex}' 匹配到机械臂关节，"
            f"请核对上方打印的关节名列表后通过 --arm_joint_regex 指定。"
        )
    if len(gripper_joint_ids) == 0:
        raise RuntimeError(
            f"未能用正则 '{args_cli.gripper_joint_regex}' 匹配到夹爪关节，"
            f"请核对上方打印的关节名列表后通过 --gripper_joint_regex 指定。"
        )

    print(f"[INFO] 底盘腿部关节 ({len(leg_joint_ids)} 个): {leg_joint_names}")
    print(f"[INFO] 机械臂关节 ({len(arm_joint_ids)} 个): {arm_joint_names}")
    print(f"[INFO] 夹爪关节 ({len(gripper_joint_ids)} 个): {gripper_joint_names}")

    if len(leg_joint_ids) != 16:
        print(f"[WARN] 底盘腿部关节数量为 {len(leg_joint_ids)}，与预期的 16 个不一致，请检查正则是否正确。")
    if len(arm_joint_ids) != 6:
        print(f"[WARN] 机械臂关节数量为 {len(arm_joint_ids)}，与预期的 6 个不一致，请检查正则是否正确。")
    if len(gripper_joint_ids) != 2:
        print(f"[WARN] 夹爪关节数量为 {len(gripper_joint_ids)}，与预期的 2 个不一致，请检查正则是否正确。")

    # 记录顺序：腿部 -> 机械臂 -> 夹爪
    all_joint_ids = list(leg_joint_ids) + list(arm_joint_ids) + list(gripper_joint_ids)
    all_joint_names = list(leg_joint_names) + list(arm_joint_names) + list(gripper_joint_names)

    # 按关节类型分组（hipx / hipy / knee / wheel ...），用于绘图：
    # 每种类型一个子图，子图内包含该类型在所有腿上的曲线
    joint_type_groups = {t: [] for t in leg_joint_types}
    for name, jid in zip(leg_joint_names, leg_joint_ids):
        jtype = get_joint_type_for_joint(name, leg_joint_types)
        if jtype is None:
            print(f"[WARN] 腿部关节 '{name}' 未能匹配到任何类型 {leg_joint_types}，将不会分组显示，"
                  f"请通过 --leg_joint_types 调整。")
            continue
        joint_type_groups[jtype].append((name, jid))

    # ── 6b. 尝试自动获取各关节力矩上限 ───────────────────────────────────────
    effort_limits = try_get_effort_limits(robot, env_id, all_joint_ids, all_joint_names)
    if any(v is None for v in effort_limits.values()):
        print("\n[WARN] 未能自动获取全部关节的力矩上限，以下字典中值为 None 的项，"
              "请你在代码顶部的 EFFORT_LIMITS_PLACEHOLDER 中手动填入真实值（单位 N·m），"
              "填好后重新运行本脚本即可在图中看到虚线上限：")
        print(effort_limits)
        print()
    else:
        print(f"[INFO] 已自动获取全部 {len(effort_limits)} 个关节的力矩上限。")

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

    torque_log = []   # 每一帧一个 (num_all_joints,) 的力矩数组 (N·m)
    vel_log = []      # 每一帧一个 (num_all_joints,) 的关节速度数组 (rad/s)
    time_log = []
    current_time = 0.0

    for _ in range(record_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        obs = obs.clone()

        # applied_torque: 实际施加到关节上的力矩（经过限幅等处理后的真实值）
        torque = robot.data.applied_torque[env_id, all_joint_ids].cpu().numpy()  # (num_all_joints,)
        print(f" 关节力矩: {torque[-8:-2]}")
        vel = robot.data.joint_vel[env_id, all_joint_ids].cpu().numpy()  # (num_all_joints,)

        torque_log.append(torque)
        vel_log.append(vel)

        time_log.append(current_time)
        current_time += sim_dt

    torque_log = np.array(torque_log)  # (T, num_all_joints)
    vel_log = np.array(vel_log)        # (T, num_all_joints)
    time_log = np.array(time_log)      # (T,)

    # ── 8. 保存原始数据 ───────────────────────────────────────────────────────
    np.savez(save_data_path,
             torque=torque_log, vel=vel_log, time=time_log,
             joint_names=np.array(all_joint_names),
             leg_joint_names=np.array(leg_joint_names),
             arm_joint_names=np.array(arm_joint_names),
             gripper_joint_names=np.array(gripper_joint_names),
             effort_limits=np.array([effort_limits[n] if effort_limits[n] is not None else np.nan
                                      for n in all_joint_names]))
    print(f"[INFO] 原始力矩数据已保存至: {save_data_path}")

    # ── 9. 统计指标 ───────────────────────────────────────────────────────────
    torque_mean = torque_log.mean(axis=0)
    torque_std = torque_log.std(axis=0)
    torque_min = torque_log.min(axis=0)
    torque_max = torque_log.max(axis=0)
    torque_absmax = np.abs(torque_log).max(axis=0)

    vel_mean = vel_log.mean(axis=0)
    vel_std = vel_log.std(axis=0)
    vel_min = vel_log.min(axis=0)
    vel_max = vel_log.max(axis=0)
    vel_absmax = np.abs(vel_log).max(axis=0)

    print("\n" + "=" * 78)
    print(f"          全身关节力矩统计 (N·m) — 地形: {args_cli.terrain}, "
          f"命令: vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}, wz={args_cli.cmd_wz}")
    print("=" * 78)
    print(f"  {'关节名':<20} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'|力矩|最大':>10} {'力矩上限':>10}")
    print("-" * 78)
    for i, name in enumerate(all_joint_names):
        limit = effort_limits[name]
        limit_str = f"{limit:>10.2f}" if limit is not None else f"{'N/A':>10}"
        print(f"  {name:<20} {torque_mean[i]:>10.3f} {torque_std[i]:>10.3f} "
              f"{torque_min[i]:>10.3f} {torque_max[i]:>10.3f} {torque_absmax[i]:>10.3f} {limit_str}")
    print("=" * 78 + "\n")
    print(f"          全身关节速度统计 (rad/s) — 地形: {args_cli.terrain}, "
          f"命令: vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}, wz={args_cli.cmd_wz}")
    print("=" * 78)
    print(f"  {'关节名':<20} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'|速度|最大':>10}")
    print("-" * 78)
    for i, name in enumerate(all_joint_names):
        print(f"  {name:<20} {vel_mean[i]:>10.3f} {vel_std[i]:>10.3f} "
              f"{vel_min[i]:>10.3f} {vel_max[i]:>10.3f} {vel_absmax[i]:>10.3f}")
    print("=" * 78 + "\n")

    # ── 10. 绘图：每个分组同时显示力矩 + 速度 ────────────────────────────────

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 原来的行数保持不变
    num_rows = len(leg_joint_types) + 2   # 腿部类型 + 机械臂 + 夹爪

    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(13, 2.8 * num_rows),
        sharex=True
    )

    if num_rows == 1:
        axes = [axes]
    def plot_group(ax, joint_list, title):

        # 速度共享第二坐标轴
        ax_vel = ax.twinx()

        for name, jid in joint_list:

            idx = all_joint_names.index(name)

            # ==========================
            # 力矩
            # ==========================
            ax.plot(
                time_log,
                torque_log[:, idx],
                linewidth=1.2,
                label=f"{name}-Torque"
            )


            # ==========================
            # 速度
            # ==========================
            ax_vel.plot(
                time_log,
                vel_log[:, idx],
                linestyle="--",
                linewidth=1.2,
                label=f"{name}-Velocity"
            )


        # 左轴
        ax.set_ylabel(
            "力矩 (N·m)",
            fontsize=10
        )

        ax.grid(True, alpha=0.3)


        # 右轴
        ax_vel.set_ylabel(
            "速度 (rad/s)",
            fontsize=10
        )


        # 合并两个legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_vel.get_legend_handles_labels()

        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            fontsize=7,
            ncol=2
        )


        ax.set_title(
            title,
            fontsize=11
        )



    row = 0


    # ==========================
    # 腿部
    # ==========================
    for jtype in leg_joint_types:

        joint_list = joint_type_groups.get(
            jtype,
            []
        )

        plot_group(
            axes[row],
            joint_list,
            f"腿部 {jtype} 关节：力矩 + 速度（4腿对比）"
        )

        row += 1



    # ==========================
    # 机械臂
    # ==========================
    arm_joint_list = list(
        zip(
            arm_joint_names,
            arm_joint_ids
        )
    )

    plot_group(
        axes[row],
        arm_joint_list,
        "机械臂关节：力矩 + 速度"
    )

    row += 1



    # ==========================
    # 夹爪
    # ==========================
    gripper_joint_list = list(
        zip(
            gripper_joint_names,
            gripper_joint_ids
        )
    )

    plot_group(
        axes[row],
        gripper_joint_list,
        "夹爪关节：力矩 + 速度"
    )



    axes[-1].set_xlabel(
        "时间 (s)",
        fontsize=11
    )


    fig.suptitle(
        f"全身关节力矩与速度曲线\n"
        f"（腿部按关节类型分组）"
        f" terrain={args_cli.terrain}, "
        f"vx={args_cli.cmd_vx} m/s",
        fontsize=13
    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )


    plt.savefig(
        save_fig_path,
        dpi=150,
        bbox_inches="tight"
    )


    print(
        f"[INFO] 全身关节力矩+速度图已保存至: {save_fig_path}"
    )
    # ── 11. 关闭环境 ──────────────────────────────────────────────────────────
    cmd_manager = raw_env.command_manager
    for term_name in cmd_manager.active_terms:
        term = cmd_manager.get_term(term_name)
        if hasattr(term, "set_debug_vis"):
            term.set_debug_vis(False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()