# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--max_evaluation_steps", type=int, default=1000, help="Max evaluation steps")

# ── [METRICS] 新增参数：控制记录哪个方案的指标 ──────────────────────
parser.add_argument(
    "--scheme",
    type=str,
    default="1",
    choices=["1", "2", "3", "4"],  # [SCHEME4] 新增 "4"
    help=(
        "指定评测方案：\n"
        "  1 = 底盘速度跟踪（线速度 + 偏航角速度）\n"
        "  2 = 底盘姿态跟踪（高度 + 滚转角 + 俯仰角）\n"
        "  3 = 末端跟踪（末端位置 + 末端姿态）\n"
        "  4 = Termination 统计（各终止条件触发比例）"  # [SCHEME4]
    ),
)
# ────────────────────────────────────────────────────────────────────

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json                          # [SCHEME4]
import time
import torch
import numpy as np
from collections import defaultdict  # [SCHEME4]
from datetime import datetime        # [SCHEME4]
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner, DistillationRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_utils import camera_follow
import rl_training.tasks  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
#  指令跟踪误差记录工具类
# ══════════════════════════════════════════════════════════════════════

class TrackingMetricsRecorder:
    """
    收集仿真过程中每一步的跟踪误差，仿真结束后计算 RMSE 并写入 TensorBoard。

    支持四种方案：
      scheme="1"  底盘速度跟踪
      scheme="2"  底盘姿态跟踪（高度 / 滚转 / 俯仰）
      scheme="3"  末端位姿跟踪
      scheme="4"  Termination 统计（各终止条件触发比例）
    """

    def __init__(self, scheme: str, log_dir: str):
        self.scheme = scheme
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "play_metrics"))
        self.global_step = 0

        # 每个 key 存放逐步的 per-env 均值误差（标量列表）
        self.records: dict[str, list[float]] = {}
        # done 掩码：reset 帧误差会失真，记录后在 RMSE 计算时可选过滤
        self.done_steps: list[bool] = []
        self.episode_records = []  # list[dict[str, float]]
        self.current_episode = {k: [] for k in self.records.keys()}

        # ── [SCHEME4] Termination 统计专用字段 ──────────────────────
        # term_names 在第一次 step 时从 env 懒初始化
        self._term_names: list[str] | None = None
        # 累计每个 term 在 done env 上的触发次数
        self._term_counts: dict[str, int] = defaultdict(int)
        # 总 episode 数（done env 累计）
        self._total_episodes: int = 0
        # 保存 log_dir，供 finalize 写 JSON
        self._log_dir = log_dir
        # ────────────────────────────────────────────────────────────

        scheme_names = {
            "1": "底盘速度跟踪",
            "2": "底盘姿态跟踪",
            "3": "末端位姿跟踪",
            "4": "Termination 统计",  # [SCHEME4]
        }
        print(f"[Metrics] 评测方案 {scheme}：{scheme_names[scheme]}")
        print(f"[Metrics] TensorBoard 日志将写入: {os.path.join(log_dir, 'play_metrics')}")

    # ------------------------------------------------------------------
    #  每步调用：从 env 中读取指令与观测，计算并记录误差
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, env, dones: torch.Tensor):
        """
        Parameters
        ----------
        env   : RslRlVecEnvWrapper 包裹后的环境，通过 env.unwrapped 访问底层
        dones : (num_envs,) bool tensor，标记哪些 env 本步 reset 了
        """
        base_env = env.unwrapped  # ManagerBasedRLEnv
        robot = base_env.scene["robot"]  # ArticulationView

        if self.scheme == "1":
            vals = self._record_scheme1(base_env, robot)
        elif self.scheme == "2":
            vals = self._record_scheme1(base_env, robot)
            vals.update(self._record_scheme2(base_env, robot))
        elif self.scheme == "3":
            vals = self._record_scheme1(base_env, robot)
            vals.update(self._record_scheme3(base_env, robot))
        elif self.scheme == "4":
            # [SCHEME4] Termination 统计，不产生 vals，单独处理
            self._record_scheme4(base_env, dones)
            self.global_step += 1
            return  # 提前返回，跳过下方 records 逻辑
        else:
            vals = {}

        # 记录当前 step（仅 scheme 1/2/3 走此分支）
        for k, v in vals.items():
            self.records.setdefault(k, []).append(v)

        # episode 结束
        if dones.any():
            self._end_episode()
        self.global_step += 1

    # ── [SCHEME4] Termination 统计核心逻辑 ──────────────────────────
    def _record_scheme4(self, base_env, dones: torch.Tensor):
        """
        在每个 step 结束后读取 termination_manager，
        仅统计本步 done 的 env 对应的触发 term。

        注意：必须在 env.step() 之后、下一次隐式 reset 之前调用，
              因为 IsaacLab 的 TerminationManager 在下一步 reset 时会清零。
        """
        mgr = base_env.termination_manager

        # 懒初始化：第一次调用时从 env 读取 term 名称列表
        if self._term_names is None:
            self._term_names = list(mgr.active_terms)
            print(f"[Metrics/Scheme4] 检测到 Termination Terms: {self._term_names}")

        # 若本步没有任何 env done，直接跳过
        if not dones.any():
            return

        # 找出 done 的 env 索引
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)  # (K,)

        # 统计每个 term 在 done env 上的触发次数
        for name in self._term_names:
            term_buf = mgr.get_term(name)          # (num_envs,) bool
            count = term_buf[done_idx].sum().item()
            self._term_counts[name] += count

        # 累计 episode 数（按 done env 数量计）
        self._total_episodes += done_idx.shape[0]
    # ────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    #  方案一：底盘速度跟踪
    # ------------------------------------------------------------------

    def _record_scheme1(self, base_env, robot):
        cmd = base_env.command_manager.get_command("base_velocity")  # (N, 3)
        lin_vel_cmd = cmd[:, :2]
        lin_vel_obs = robot.data.root_lin_vel_b[:, :2]
        
        # 分别计算 x, y 方向的误差
        lin_err = lin_vel_cmd - lin_vel_obs  # (N, 2)
        lin_err_x_mean = lin_err[:, 0].abs().mean().item()
        lin_err_y_mean = lin_err[:, 1].abs().mean().item()
        
        # 保留原来的合并误差（可选）
        lin_err_mean = torch.norm(lin_err, dim=-1).mean().item()
        
        ang_vel_cmd = cmd[:, 2]
        ang_vel_obs = robot.data.root_ang_vel_b[:, 2]
        ang_err = (ang_vel_cmd - ang_vel_obs).abs()
        ang_err_mean = ang_err.mean().item()
        
        return {
            "lin_vel_error": lin_err_mean,
            "lin_vel_error_x": lin_err_x_mean,
            "lin_vel_error_y": lin_err_y_mean,
            "ang_vel_error": ang_err_mean
        }

    # ------------------------------------------------------------------
    #  方案二：底盘姿态跟踪
    # ------------------------------------------------------------------

    def _record_scheme2(self, base_env, robot):
        cmd = base_env.command_manager.get_command("body_pose")

        height_cmd = cmd[:, 0]
        height_obs = robot.data.root_pos_w[:, 2]
        height_err = (height_cmd - height_obs).abs().mean().item()

        roll_obs, pitch_obs, _ = euler_xyz_from_quat(robot.data.root_quat_w)

        pitch_cmd = cmd[:, 1]
        roll_cmd  = cmd[:, 2]

        roll_err  = (roll_cmd  - roll_obs ).abs().mean().item()
        pitch_err = (pitch_cmd - pitch_obs).abs().mean().item()

        return {
            "height_error": height_err,
            "roll_error": roll_err,
            "pitch_error": pitch_err
        }

    # ------------------------------------------------------------------
    #  方案三：末端位姿跟踪
    # ------------------------------------------------------------------

    EE_BODY_NAME = "arm_link6"

    def _record_scheme3(self, base_env, robot):
        cmd = base_env.command_manager.get_command("ee_pose")

        if not hasattr(self, "_ee_idx"):
            body_names = robot.data.body_names
            if self.EE_BODY_NAME not in body_names:
                raise ValueError(
                    f"[Metrics] 末端 body '{self.EE_BODY_NAME}' 不存在于机器人中。\n"
                    f"可用的 body 名称：{body_names}\n"
                    "请修改脚本中的 EE_BODY_NAME。"
                )
            self._ee_idx = body_names.index(self.EE_BODY_NAME)

        ee_idx = self._ee_idx

        pos_cmd = cmd[:, :3]
        pos_obs = robot.data.body_pos_w[:, ee_idx, :]
        pos_err = torch.norm(pos_cmd - pos_obs, dim=-1).mean().item()

        quat_cmd = cmd[:, 3:7]
        quat_obs = robot.data.body_quat_w[:, ee_idx, :]
        rot_err  = quat_error_magnitude(quat_cmd, quat_obs).mean().item()

        return {
            "ee_pos_error": pos_err,
            "ee_rot_error": rot_err
        }

    # ------------------------------------------------------------------
    #  内部工具
    # ------------------------------------------------------------------

    def _append_and_log(self, key: str, value: float):
        """记录到列表并实时写 TensorBoard。"""
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value)
        self.writer.add_scalar(f"play/{key}", value, self.global_step)

    def _end_episode(self):
        episode_rmse = {}

        for k, vals in self.records.items():
            arr = np.array(vals)
            rmse = float(np.sqrt(np.mean(arr ** 2)))
            episode_rmse[k] = rmse

        self.episode_records.append(episode_rmse)

        # reset buffer
        self.records = {}

    # ------------------------------------------------------------------
    #  仿真结束后：计算并打印结果，写入 TensorBoard / JSON
    # ------------------------------------------------------------------

    def finalize(self):
        if self.scheme == "4":
            self._finalize_scheme4()
        else:
            self._finalize_tracking()

    def _finalize_tracking(self):
        """scheme 1/2/3 的 RMSE 汇总。"""
        if not self.episode_records:
            print("[Metrics] 无 episode 数据，跳过汇总。")
            return

        print("\n" + "=" * 50)
        print("Episode-level RMSE mean:")

        keys = self.episode_records[0].keys()
        summary = {}
        for k in keys:
            vals = [ep[k] for ep in self.episode_records if k in ep]
            mean_rmse = float(np.mean(vals))
            summary[k] = mean_rmse
            print(f"  {k}: {mean_rmse:.6f}")
        print("=" * 50)

    # ── [SCHEME4] Termination 统计汇总 ──────────────────────────────
    def _finalize_scheme4(self):
        """
        汇总各 termination term 的触发次数与比例，
        打印到控制台，写入 TensorBoard，并保存为 JSON 文件。
        """
        if self._total_episodes == 0:
            print("[Metrics/Scheme4] 未记录到任何 episode，跳过汇总。")
            return

        total_triggers = sum(self._term_counts.values())

        print("\n" + "=" * 55)
        print(f"{'[Termination Statistics]':^55}")
        print("=" * 55)
        print(f"{'Term Name':<35} {'Count':>6}  {'Ratio':>7}")
        print("-" * 55)

        ratios: dict[str, float] = {}
        for name in (self._term_names or []):
            count = self._term_counts.get(name, 0)
            ratio = count / total_triggers if total_triggers > 0 else 0.0
            ratios[name] = ratio
            print(f"  {name:<33} {count:>6}  {ratio:>7.1%}")

            # 写入 TensorBoard（用固定 step=0 写 scalar，方便多次比较）
            self.writer.add_scalar(
                f"play/termination_ratio/{name}", ratio, global_step=0
            )

        print("-" * 55)
        print(f"  {'Total episodes':<33} {self._total_episodes:>6}")
        print("=" * 55)

        # 保存 JSON，便于后续离线分析
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_episodes": self._total_episodes,
            "total_triggers": int(total_triggers),
            "termination_counts": {k: int(v) for k, v in self._term_counts.items()},
            "termination_ratios": {k: float(v) for k, v in ratios.items()},
        }
        save_path = os.path.join(self._log_dir, "termination_stats.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n[Metrics/Scheme4] 统计结果已保存至: {save_path}")
    # ────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.step_dt

    # ── [METRICS] 初始化记录器 ────────────────────────────────────────
    recorder = TrackingMetricsRecorder(scheme=args_cli.scheme, log_dir=log_dir)
    max_evaluation_steps = args_cli.max_evaluation_steps
    episode_count = 0
    # ─────────────────────────────────────────────────────────────────

    # reset environment
    obs = env.get_observations()

    timestep = 0
    # simulate environment
    while simulation_app.is_running() and episode_count < max_evaluation_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # ── [METRICS] 每步记录误差 ────────────────────────────────────
        # 注意：必须在 env.step() 之后立即调用，TerminationManager 尚未被 reset 清零
        recorder.step(env, dones)

        # 统计 episode 数
        if dones.any():
            episode_count += dones.sum().item()  # 并行 env，要加所有 done
        # ─────────────────────────────────────────────────────────────

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ── [METRICS] 仿真结束，计算并输出结果 ───────────────────────────
    recorder.finalize()
    # ─────────────────────────────────────────────────────────────────

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()