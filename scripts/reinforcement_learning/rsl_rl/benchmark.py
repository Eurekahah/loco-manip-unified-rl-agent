"""
benchmark_ee_low_reach.py
=========================
低位EE到达能力的Benchmark脚本，集成到现有Isaac Lab eval/play流程。

使用方式：
    在你的play.py中，替换掉原有的command manager，或在eval loop中调用
    BenchmarkRunner.step() 和 BenchmarkRunner.log_results()

核心逻辑：
    1. 用固定的分层测试集覆盖低位空间（按pitch分4层）
    2. 每层跑 N 个 episode，记录成功率 / 误差分布 / 狗姿态
    3. 最终输出 CSV + 热力图数据
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1.  测试集定义
#     按照你的坐标系：
#       p_pitch > 0  → EE 在采样原点（arm_base 高度）以下，越大越低
#       sampled_height = 0.6m，所以 pitch=π/3 对应 EE 约在地面附近
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkTier:
    """一个测试层级的定义"""
    name: str                          # 层级名称，用于日志
    p_pitch_range: tuple[float, float] # pitch 采样范围（决定高度）
    p_l_range: tuple[float, float]     # 半径范围
    p_yaw_range: tuple[float, float]   # yaw 范围
    n_episodes: int = 50               # 该层跑多少个 episode
    description: str = ""


# 四个测试层级，从"稍低"到"极低"
BENCHMARK_TIERS: list[BenchmarkTier] = [
    BenchmarkTier(
        name="tier0_baseline",
        p_pitch_range=(-0.3, 0.0),       # 稍高于采样原点，baseline
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 3 / 5, math.pi * 3 / 5),
        n_episodes=40,
        description="EE高于arm_base，正常可达区域（baseline）",
    ),
    BenchmarkTier(
        name="tier1_slightly_low",
        p_pitch_range=(0.0, math.pi / 6),  # EE 略低于 arm_base
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 3 / 5, math.pi * 3 / 5),
        n_episodes=50,
        description="EE略低于arm_base高度，轻微前倾即可",
    ),
    BenchmarkTier(
        name="tier2_low",
        p_pitch_range=(math.pi / 6, math.pi / 3),  # EE 明显偏低
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 2 / 5, math.pi * 2 / 5),  # 正面为主
        n_episodes=50,
        description="EE明显低位，需要显著俯身/前倾",
    ),
    BenchmarkTier(
        name="tier3_very_low",
        p_pitch_range=(math.pi / 3, math.pi * 2 / 5),  # 接近地面
        p_l_range=(0.4, 0.60),
        p_yaw_range=(-math.pi / 3, math.pi / 3),        # 主要正前方
        n_episodes=50,
        description="EE接近地面，需要大幅度俯身",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  单条 Episode 的记录结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    tier_name: str
    # 目标
    target_pitch: float        # 目标点的 pitch（决定高度）
    target_l: float            # 目标点的半径
    target_yaw: float          # 目标点的 yaw
    target_z_world: float      # 目标点的世界坐标 z（绝对高度，便于分析）
    # 结果
    pos_error_final: float     # episode 结束时的位置误差 (m)
    rot_error_final: float     # episode 结束时的旋转误差 (rad)
    pos_error_min: float       # episode 内最小位置误差（衡量是否曾到达）
    success: bool              # pos_error < 阈值 且 rot_error < 阈值
    # 机器狗姿态（到达最近时的状态）
    base_pitch_at_best: float  # 最近时刻的机身pitch（正=前倾/俯身）
    base_roll_at_best: float
    base_height_at_best: float # 机身高度（相对初始）
    # 稳定性
    base_pitch_std: float      # 整个 episode 中机身 pitch 的标准差（稳定性）
    contact_force_std: float   # 四足接触力标准差（均匀性）
    # 时间
    time_to_reach: float       # 首次达到阈值的时间（未达到则为 inf）


# ─────────────────────────────────────────────────────────────────────────────
# 3.  主 Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    集成到 Isaac Lab eval loop 的 benchmark 运行器。

    典型用法（在你的 play.py 的主循环中）：
    ─────────────────────────────────────────
        runner = BenchmarkRunner(env, ee_command_manager, cfg)
        runner.start()

        while runner.is_running():
            # 正常 env step
            obs, reward, done, info = env.step(action)
            # 喂数据给 runner
            runner.record_step(env, done)

        runner.finish()
    ─────────────────────────────────────────
    """

    def __init__(
        self,
        env,
        ee_command: "HeightInvariantEECommand",  # 你的自定义 command 实例
        tiers: list[BenchmarkTier] = BENCHMARK_TIERS,
        pos_success_threshold: float = 0.05,    # 5cm
        rot_success_threshold: float = 0.3,     # ~17°
        save_dir: str = "./benchmark_results",
        env_idx: int = 0,                        # 只跟踪第0个环境（单env benchmark）
        hold_steps_required: int = 10,           # 需要连续保持多少步才算"到达"
    ):
        self.env = env
        self.ee_cmd = ee_command
        self.tiers = tiers
        self.pos_thresh = pos_success_threshold
        self.rot_thresh = rot_success_threshold
        self.save_dir = save_dir
        self.env_idx = env_idx
        self.hold_steps = hold_steps_required

        os.makedirs(save_dir, exist_ok=True)

        # 运行状态
        self._tier_idx = 0
        self._episode_idx = 0
        self._running = False
        self._step_count = 0

        # 当前 episode 的缓存数据
        self._cur_pos_errors: list[float] = []
        self._cur_rot_errors: list[float] = []
        self._cur_base_pitches: list[float] = []
        self._cur_base_rolls: list[float] = []
        self._cur_base_heights: list[float] = []
        self._cur_contact_stds: list[float] = []
        self._hold_count = 0
        self._time_to_reach = float("inf")

        # 结果汇总
        self.all_records: list[EpisodeRecord] = []

        # 当前 tier 的固定目标参数（每个 episode 独立采样一次）
        self._cur_target_pitch = 0.0
        self._cur_target_l = 0.5
        self._cur_target_yaw = 0.0
        self._cur_target_z_world = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # 公开 API
    # ──────────────────────────────────────────────────────────────────────

    def start(self):
        """开始 benchmark，重置到第一个 tier 第一个 episode"""
        print("\n" + "=" * 60)
        print("  EE Low-Reach Benchmark 开始")
        print(f"  共 {len(self.tiers)} 个测试层，"
              f"总 episode 数: {sum(t.n_episodes for t in self.tiers)}")
        print("=" * 60 + "\n")
        self._running = True
        self._tier_idx = 0
        self._episode_idx = 0
        self._inject_tier_command(self.tiers[0])
        print(f"[Benchmark] 开始 {self.tiers[0].name}: {self.tiers[0].description}")

    def is_running(self) -> bool:
        return self._running

    def record_step(self, env, done: torch.Tensor):
        """
        每个 env step 调用一次，喂入当前环境状态。

        参数:
            env:  Isaac Lab environment
            done: (num_envs,) bool tensor，episode 是否结束
        """
        if not self._running:
            return

        i = self.env_idx
        self._step_count += 1

        # ── 读取当前状态 ─────────────────────────────────────────────────
        pos_err = self.ee_cmd.metrics["position_error"][i].item()
        rot_err = self.ee_cmd.metrics["orientation_error"][i].item()

        # 机身姿态
        from isaaclab.utils import math as math_utils  # lazy import
        base_quat = env.scene["robot"].data.root_quat_w[i]  # (4,)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat.unsqueeze(0))
        base_pitch = pitch[0].item()
        base_roll = roll[0].item()
        base_height = env.scene["robot"].data.root_pos_w[i, 2].item()

        # 接触力标准差（如果有 contact sensor）
        contact_std = self._get_contact_force_std(env, i)

        # ── 记录到缓存 ────────────────────────────────────────────────────
        self._cur_pos_errors.append(pos_err)
        self._cur_rot_errors.append(rot_err)
        self._cur_base_pitches.append(base_pitch)
        self._cur_base_rolls.append(base_roll)
        self._cur_base_heights.append(base_height)
        self._cur_contact_stds.append(contact_std)

        # ── 判断是否到达（连续hold_steps步误差达标）───────────────────────
        if pos_err < self.pos_thresh and rot_err < self.rot_thresh:
            self._hold_count += 1
            if self._hold_count >= self.hold_steps and self._time_to_reach == float("inf"):
                self._time_to_reach = self._step_count * env.step_dt
        else:
            self._hold_count = 0

        # ── episode 结束时记录并切换 ─────────────────────────────────────
        if done[i].item():
            self._finalize_episode(env)
            self._next_episode(env)

    def finish(self):
        """结束 benchmark，保存结果"""
        self._running = False
        self._save_results()
        self._print_summary()

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _inject_tier_command(self, tier: BenchmarkTier):
        """
        将 command manager 的采样范围覆盖为当前 tier 的范围。
        直接修改 cfg.ranges 即可，下次 _resample_command 时生效。
        """
        cfg = self.ee_cmd.cfg
        cfg.ranges.p_pitch = tier.p_pitch_range
        cfg.ranges.p_l = tier.p_l_range
        cfg.ranges.p_yaw = tier.p_yaw_range
        # 姿态范围保持不变（或你也可以在 tier 中指定）

    def _sample_episode_target(self, tier: BenchmarkTier, env):
        """
        为当前 episode 采样一个固定的目标点参数（记录用）。
        实际目标由 ee_cmd._resample_command 决定，这里只是读取记录。
        """
        # 等待 resample 完成后，读取实际采样到的球坐标
        i = self.env_idx
        sphere = self.ee_cmd.ee_end_sphere[i]  # (3,) = [l, pitch, yaw]
        self._cur_target_l = sphere[0].item()
        self._cur_target_pitch = sphere[1].item()
        self._cur_target_yaw = sphere[2].item()
        # 世界坐标 z
        self._cur_target_z_world = self.ee_cmd.pose_end_w[i, 2].item()

    def _finalize_episode(self, env):
        """当前 episode 结束，整理记录"""
        pos_errors = np.array(self._cur_pos_errors)
        rot_errors = np.array(self._cur_rot_errors)
        base_pitches = np.array(self._cur_base_pitches)

        # 最近时刻的索引
        best_idx = int(np.argmin(pos_errors))

        # 成功判断：episode 内曾连续到达
        success = self._time_to_reach < float("inf")
        print(f"第{self._episode_idx}个episode,{'成功' if success else '失败'}")

        tier = self.tiers[self._tier_idx]
        record = EpisodeRecord(
            tier_name=tier.name,
            target_pitch=self._cur_target_pitch,
            target_l=self._cur_target_l,
            target_yaw=self._cur_target_yaw,
            target_z_world=self._cur_target_z_world,
            pos_error_final=pos_errors[-1] if len(pos_errors) > 0 else float("nan"),
            rot_error_final=rot_errors[-1] if len(rot_errors) > 0 else float("nan"),
            pos_error_min=float(np.min(pos_errors)) if len(pos_errors) > 0 else float("nan"),
            success=success,
            base_pitch_at_best=self._cur_base_pitches[best_idx] if len(self._cur_base_pitches) > 0 else 0.0,
            base_roll_at_best=self._cur_base_rolls[best_idx] if len(self._cur_base_rolls) > 0 else 0.0,
            base_height_at_best=self._cur_base_heights[best_idx] if len(self._cur_base_heights) > 0 else 0.0,
            base_pitch_std=float(np.std(base_pitches)) if len(base_pitches) > 0 else 0.0,
            contact_force_std=float(np.mean(self._cur_contact_stds)) if len(self._cur_contact_stds) > 0 else 0.0,
            time_to_reach=self._time_to_reach,
        )
        self.all_records.append(record)

        # 重置缓存
        self._cur_pos_errors.clear()
        self._cur_rot_errors.clear()
        self._cur_base_pitches.clear()
        self._cur_base_rolls.clear()
        self._cur_base_heights.clear()
        self._cur_contact_stds.clear()
        self._hold_count = 0
        self._time_to_reach = float("inf")
        self._step_count = 0

    def _next_episode(self, env):
        """推进到下一个 episode 或下一个 tier"""
        self._episode_idx += 1
        tier = self.tiers[self._tier_idx]

        if self._episode_idx >= tier.n_episodes:
            # 当前 tier 完成
            tier_records = [r for r in self.all_records if r.tier_name == tier.name]
            success_rate = sum(r.success for r in tier_records) / len(tier_records)
            print(f"\n[Benchmark] ✓ {tier.name} 完成 | "
                  f"成功率: {success_rate:.1%} | "
                  f"平均最小误差: {np.mean([r.pos_error_min for r in tier_records]):.4f}m")

            self._tier_idx += 1
            self._episode_idx = 0

            if self._tier_idx >= len(self.tiers):
                # 所有 tier 完成
                self.finish()
                return
            else:
                new_tier = self.tiers[self._tier_idx]
                self._inject_tier_command(new_tier)
                print(f"\n[Benchmark] 开始 {new_tier.name}: {new_tier.description}")

        # 读取新 episode 的目标（resample 会在 env reset 后自动触发）
        self._sample_episode_target(self.tiers[self._tier_idx], env)

    def _get_contact_force_std(self, env, env_idx: int) -> float:
        """
        读取四足接触力标准差。
        如果你的环境有 contact_forces sensor，从这里读取；否则返回 0。
        """
        try:
            # Isaac Lab 接触传感器示例（根据你的 sensor 名称修改）
            contact_sensor = env.scene.sensors.get("contact_forces", None)
            if contact_sensor is not None:
                # shape: (num_envs, num_feet, 3)
                forces = contact_sensor.data.net_forces_w[env_idx]  # (num_feet, 3)
                force_norms = torch.norm(forces, dim=-1)             # (num_feet,)
                return force_norms.std().item()
        except Exception:
            pass
        return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # 结果保存与分析
    # ──────────────────────────────────────────────────────────────────────

    def _save_results(self):
        """保存原始记录为 CSV"""
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.save_dir, f"benchmark_{timestamp}.csv")

        fields = list(EpisodeRecord.__dataclass_fields__.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self.all_records:
                writer.writerow(rec.__dict__)

        print(f"\n[Benchmark] 原始数据已保存: {csv_path}")

        # 同时保存热力图数据（pitch vs 成功率）
        self._save_heatmap_data(timestamp)

    def _save_heatmap_data(self, timestamp: str):
        """
        生成 pitch-yaw 二维成功率热力图数据（numpy .npz 格式）
        可以用 matplotlib 直接绘制
        """
        if not self.all_records:
            return

        # pitch bins（按 tier 分层）
        pitch_bins = np.array([-0.3, 0.0, math.pi / 6, math.pi / 3, math.pi * 2 / 5])
        yaw_bins = np.linspace(-math.pi * 3 / 5, math.pi * 3 / 5, 7)

        # 2D 网格：pitch_bin x yaw_bin 的成功率
        heatmap = np.full((len(pitch_bins) - 1, len(yaw_bins) - 1), np.nan)
        counts = np.zeros_like(heatmap)

        for rec in self.all_records:
            pi = np.searchsorted(pitch_bins, rec.target_pitch, side="right") - 1
            yi = np.searchsorted(yaw_bins, rec.target_yaw, side="right") - 1
            pi = np.clip(pi, 0, heatmap.shape[0] - 1)
            yi = np.clip(yi, 0, heatmap.shape[1] - 1)
            if np.isnan(heatmap[pi, yi]):
                heatmap[pi, yi] = 0.0
            heatmap[pi, yi] += float(rec.success)
            counts[pi, yi] += 1

        with np.errstate(invalid="ignore"):
            heatmap_rate = np.where(counts > 0, heatmap / counts, np.nan)

        npz_path = os.path.join(self.save_dir, f"heatmap_{timestamp}.npz")
        np.savez(
            npz_path,
            success_rate=heatmap_rate,
            counts=counts,
            pitch_bins=pitch_bins,
            yaw_bins=yaw_bins,
            pos_error_min_mean=self._compute_grid_stat("pos_error_min", pitch_bins, yaw_bins),
            base_pitch_mean=self._compute_grid_stat("base_pitch_at_best", pitch_bins, yaw_bins),
        )
        print(f"[Benchmark] 热力图数据已保存: {npz_path}")

    def _compute_grid_stat(self, field_name: str, pitch_bins, yaw_bins) -> np.ndarray:
        grid = np.full((len(pitch_bins) - 1, len(yaw_bins) - 1), np.nan)
        accum = {}
        for rec in self.all_records:
            pi = np.clip(np.searchsorted(pitch_bins, rec.target_pitch, side="right") - 1,
                         0, grid.shape[0] - 1)
            yi = np.clip(np.searchsorted(yaw_bins, rec.target_yaw, side="right") - 1,
                         0, grid.shape[1] - 1)
            key = (pi, yi)
            accum.setdefault(key, []).append(getattr(rec, field_name))
        for (pi, yi), vals in accum.items():
            grid[pi, yi] = np.mean(vals)
        return grid

    def _print_summary(self):
        """打印分层汇总统计"""
        print("\n" + "=" * 60)
        print("  Benchmark 汇总")
        print("=" * 60)
        print(f"{'Tier':<25} {'N':>4} {'成功率':>8} {'avg最小误差':>12} {'avg狗pitch':>12} {'avg到达时间':>12}")
        print("-" * 75)

        for tier in self.tiers:
            recs = [r for r in self.all_records if r.tier_name == tier.name]
            if not recs:
                continue
            n = len(recs)
            sr = sum(r.success for r in recs) / n
            avg_err = np.mean([r.pos_error_min for r in recs])
            avg_pitch = np.mean([r.base_pitch_at_best for r in recs])
            reached = [r.time_to_reach for r in recs if r.time_to_reach < float("inf")]
            avg_time = np.mean(reached) if reached else float("inf")
            print(f"{tier.name:<25} {n:>4} {sr:>8.1%} {avg_err:>12.4f}m "
                  f"{math.degrees(avg_pitch):>11.1f}° {avg_time:>11.2f}s")

        print("=" * 60)
        total = len(self.all_records)
        overall_sr = sum(r.success for r in self.all_records) / total if total > 0 else 0
        print(f"总体成功率: {overall_sr:.1%}  (共 {total} episodes)")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  可视化工具（离线使用，benchmark 跑完后调用）
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(npz_path: str, save_path: Optional[str] = None):
    """
    从保存的 .npz 文件绘制 benchmark 热力图。

    用法：
        python benchmark_ee_low_reach.py --plot ./benchmark_results/heatmap_xxx.npz
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    data = np.load(npz_path)
    success_rate = data["success_rate"]
    pos_error = data["pos_error_min_mean"]
    base_pitch = data["base_pitch_mean"]
    pitch_bins = data["pitch_bins"]
    yaw_bins = data["yaw_bins"]

    pitch_labels = [f"{math.degrees(p):.0f}°" for p in pitch_bins]
    yaw_labels = [f"{math.degrees(y):.0f}°" for y in yaw_bins]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EE Low-Reach Benchmark\n（行=EE目标pitch，越大越低；列=目标yaw方向）",
                 fontsize=13)

    # ── 图1: 成功率热力图 ────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(success_rate, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                   origin="lower")
    plt.colorbar(im, ax=ax, label="成功率")
    ax.set_title("EE到达成功率")
    ax.set_xlabel("目标 yaw (°)")
    ax.set_ylabel("目标 pitch (°) [越大越低]")
    ax.set_xticks(range(len(yaw_labels) - 1))
    ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1))
    ax.set_yticklabels(pitch_labels[:-1], fontsize=8)
    # 标注数值
    for i in range(success_rate.shape[0]):
        for j in range(success_rate.shape[1]):
            if not np.isnan(success_rate[i, j]):
                ax.text(j, i, f"{success_rate[i, j]:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="black" if 0.3 < success_rate[i, j] < 0.7 else "white")

    # ── 图2: 最小位置误差热力图 ─────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(pos_error, vmin=0, vmax=0.15, cmap="YlOrRd_r", aspect="auto",
                    origin="lower")
    plt.colorbar(im2, ax=ax, label="平均最小位置误差 (m)")
    ax.set_title("EE最小位置误差 (越绿越好)")
    ax.set_xlabel("目标 yaw (°)")
    ax.set_xticks(range(len(yaw_labels) - 1))
    ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1))
    ax.set_yticklabels(pitch_labels[:-1], fontsize=8)

    # ── 图3: 机身pitch热力图（俯身程度）────────────────────────────────
    ax = axes[2]
    pitch_deg = np.degrees(base_pitch) if not np.all(np.isnan(base_pitch)) else base_pitch
    vmax_pitch = max(30, np.nanmax(np.abs(pitch_deg))) if not np.all(np.isnan(pitch_deg)) else 30
    im3 = ax.imshow(pitch_deg, vmin=-vmax_pitch, vmax=vmax_pitch,
                    cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(im3, ax=ax, label="机身pitch (°) [正=前倾/俯身]")
    ax.set_title("到达时机身pitch（俯身程度）")
    ax.set_xlabel("目标 yaw (°)")
    ax.set_xticks(range(len(yaw_labels) - 1))
    ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1))
    ax.set_yticklabels(pitch_labels[:-1], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"热力图已保存: {save_path}")
    else:
        plt.show()


def plot_cdf(csv_path: str, save_path: Optional[str] = None):
    """
    绘制各 tier 的 EE 位置误差 CDF 曲线。
    比单一成功率信息量更大，论文常用。
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("需要安装 matplotlib 和 pandas")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for (tier_name, group), color in zip(df.groupby("tier_name"), colors):
        errors = np.sort(group["pos_error_min"].dropna().values)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=tier_name, color=color, linewidth=2)

    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.7, label="5cm阈值")
    ax.set_xlabel("最小EE位置误差 (m)")
    ax.set_ylabel("累积比例 (CDF)")
    ax.set_title("各难度层级 EE 到达误差分布")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  集成示例：如何插入到你的 play.py
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_EXAMPLE = '''
# ── 在你的 play.py 中集成 benchmark ─────────────────────────────────────────
#
# Step 1: 在 env 和 policy 初始化完成后，创建 runner
from benchmark_ee_low_reach import BenchmarkRunner

ee_command = env.command_manager.get_term("ee_pose")  # 根据你的 term 名称
runner = BenchmarkRunner(
    env=env,
    ee_command=ee_command,
    pos_success_threshold=0.05,    # 5cm
    rot_success_threshold=0.3,     # ~17 degrees
    save_dir="./benchmark_results",
    env_idx=0,                     # 如果多env并行，只跟踪第0个
    hold_steps_required=10,        # 需要连续10步（0.2s）误差达标
)
runner.start()

# Step 2: 在主循环中
while runner.is_running():
    with torch.no_grad():
        action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    runner.record_step(env, done)

# Step 3: runner.finish() 在内部自动调用，结果保存到 save_dir

# ── 离线绘图 ──────────────────────────────────────────────────────────────────
# from benchmark_ee_low_reach import plot_heatmap, plot_cdf
# plot_heatmap("./benchmark_results/heatmap_xxx.npz", save_path="heatmap.png")
# plot_cdf("./benchmark_results/benchmark_xxx.csv", save_path="cdf.png")
'''

# ─────────────────────────────────────────────────────────────────────────────
# 6.  命令行工具（离线分析）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EE Low-Reach Benchmark 离线分析工具")
    parser.add_argument("--plot", type=str, help="绘制热力图，传入 .npz 文件路径")
    parser.add_argument("--cdf", type=str, help="绘制CDF曲线，传入 .csv 文件路径")
    parser.add_argument("--save", type=str, default=None, help="图片保存路径")
    args = parser.parse_args()

    if args.plot:
        plot_heatmap(args.plot, save_path=args.save)
    elif args.cdf:
        plot_cdf(args.cdf, save_path=args.save)
    else:
        print("集成示例：")
        print(INTEGRATION_EXAMPLE)