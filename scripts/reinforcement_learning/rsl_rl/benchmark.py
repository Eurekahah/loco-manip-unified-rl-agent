"""
benchmark_ee_low_reach.py  (parallel edition)
=============================================
低位EE到达能力的 Benchmark，完整利用 Isaac Lab 的并行环境。

改造要点（相比单环境版本）：
  1. 所有缓存从 list 改为 (num_envs, T) 的向量化结构
  2. record_step() 用 tensor 批量读取，零 Python for-loop
  3. done mask 决定哪些 env 本轮 finalize，其余继续积累
  4. episode 计数器按 done.sum() 累加，tier 切换自动对齐
  5. 跨 tier 边界时，正在跑的 env 会在当前 episode 结束后才切换
     （避免半途强行 reset 打乱状态）

使用方式（play.py 中）：
    runner = BenchmarkRunner(env, ee_command_manager, cfg)
    runner.start()
    while runner.is_running():
        obs, reward, terminated, truncated, info = env.step(action)
        runner.record_step(env, terminated | truncated)
    # finish() 在内部自动调用
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
# 1.  测试层级定义（与原版相同，方便直接替换）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkTier:
    name: str
    p_pitch_range: tuple[float, float]
    p_l_range: tuple[float, float]
    p_yaw_range: tuple[float, float]
    n_episodes: int = 50
    description: str = ""


BENCHMARK_TIERS: list[BenchmarkTier] = [
    BenchmarkTier(
        name="tier0_baseline",
        p_pitch_range=(math.pi / 6, math.pi * 2 / 5),
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 3 / 5, math.pi * 3 / 5),
        n_episodes=40,
        description="EE高于arm_base，正常可达区域（baseline）",
    ),
    BenchmarkTier(
        name="tier1_slightly_low",
        p_pitch_range=(0.0, math.pi / 6),
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 3 / 5, math.pi * 3 / 5),
        n_episodes=50,
        description="EE略低于arm_base高度，轻微前倾即可",
    ),
    BenchmarkTier(
        name="tier2_low",
        p_pitch_range=(-1, 0),
        p_l_range=(0.4, 0.65),
        p_yaw_range=(-math.pi * 2 / 5, math.pi * 2 / 5),
        n_episodes=50,
        description="EE明显低位，需要显著俯身/前倾",
    ),
    # BenchmarkTier(
    #     name="tier3_very_low",
    #     p_pitch_range=(math.pi / 3, math.pi * 2 / 5),
    #     p_l_range=(0.4, 0.60),
    #     p_yaw_range=(-math.pi / 3, math.pi / 3),
    #     n_episodes=50,
    #     description="EE接近地面，需要大幅度俯身",
    # ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  单条 Episode 记录（与原版相同）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    tier_name: str
    env_idx: int               # 新增：来自哪个并行环境
    target_pitch: float
    target_l: float
    target_yaw: float
    target_z_world: float
    pos_error_final: float
    rot_error_final: float
    pos_error_min: float
    success: bool
    base_pitch_at_best: float
    base_roll_at_best: float
    base_height_at_best: float
    base_pitch_std: float
    contact_force_std: float
    time_to_reach: float


# ─────────────────────────────────────────────────────────────────────────────
# 3.  并行 Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    完全并行化的 Benchmark Runner。

    关键设计：
    ─────────────────────────────────────────────────────────
    · 所有缓存均为 (num_envs, max_steps) 的预分配 tensor，
      record_step() 只做一次向量化写入，无 Python for-loop。
    · 每步只 finalize done=True 的环境，done=False 的继续积累。
    · episode 计数 += done.sum()，tier 切换不打断正在运行的 env。
    · 跨 tier 边界：当某 env 的 done 触发时，才检查是否切换 tier。
    ─────────────────────────────────────────────────────────

    典型用法：
        runner = BenchmarkRunner(env, ee_command, cfg)
        runner.start()
        while runner.is_running():
            obs, rew, terminated, truncated, info = env.step(action)
            runner.record_step(env, terminated | truncated)
        # finish() 在所有 tier 跑完后自动调用
    """

    def __init__(
        self,
        env,
        ee_command,
        tiers: list[BenchmarkTier] = BENCHMARK_TIERS,
        pos_success_threshold: float = 0.05,
        rot_success_threshold: float = 0.3,
        save_dir: str = "./benchmark_results",
        hold_steps_required: int = 10,
        max_episode_steps: int = 1000,   # 每个 episode 最长步数，用于预分配缓存
    ):
        self.env = env
        self.ee_cmd = ee_command
        self.tiers = tiers
        self.pos_thresh = pos_success_threshold
        self.rot_thresh = rot_success_threshold
        self.save_dir = save_dir
        self.hold_steps = hold_steps_required
        self.max_steps = max_episode_steps

        self.num_envs: int = env.num_envs
        self.device = env.device if hasattr(env, "device") else "cpu"

        os.makedirs(save_dir, exist_ok=True)

        # ── 运行状态 ──────────────────────────────────────────────────────
        self._tier_idx: int = 0
        self._running: bool = False

        # 每个 env 已完成的 episode 数（用于统计，不影响 tier 切换逻辑）
        self._env_episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 全局已完成 episode 数（所有 env 累计）
        self._total_episodes_done: int = 0

        # 当前 tier 已完成 episode 数
        # self._tier_episodes_done: int = 0
        self._tier_episode_counts = torch.zeros(len(self.tiers), dtype=torch.int)

        # ── 每 env 的 episode 内缓存（预分配，避免动态 append）──────────
        # shape: (num_envs, max_steps)
        self._buf_pos_err     = torch.zeros(self.num_envs, max_episode_steps, device=self.device)
        self._buf_rot_err     = torch.zeros(self.num_envs, max_episode_steps, device=self.device)
        self._buf_base_pitch  = torch.zeros(self.num_envs, max_episode_steps, device=self.device)
        self._buf_base_roll   = torch.zeros(self.num_envs, max_episode_steps, device=self.device)
        self._buf_base_height = torch.zeros(self.num_envs, max_episode_steps, device=self.device)
        self._buf_contact_std = torch.zeros(self.num_envs, max_episode_steps, device=self.device)

        # 每个 env 当前 episode 已写入的步数
        self._buf_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 连续达标步数计数器（per env）
        self._hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 首次到达时间（per env）；未到达 = inf
        self._time_to_reach = torch.full((self.num_envs,), float("inf"), device=self.device)

        # 当前 episode 的目标参数（per env，resample 后从 ee_cmd 读取）
        self._cur_target_pitch  = torch.zeros(self.num_envs, device=self.device)
        self._cur_target_l      = torch.zeros(self.num_envs, device=self.device)
        self._cur_target_yaw    = torch.zeros(self.num_envs, device=self.device)
        self._cur_target_z_world = torch.zeros(self.num_envs, device=self.device)

        # 每个 env 当前属于哪个 tier（支持跨 tier 边界时的平滑切换）
        self._env_tier_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── 结果 ─────────────────────────────────────────────────────────
        self.all_records: list[EpisodeRecord] = []

    # ─────────────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────────────

    def start(self):
        total_eps = sum(t.n_episodes for t in self.tiers)
        print("\n" + "=" * 60)
        print(f"  EE Low-Reach Benchmark 开始  （并行环境数: {self.num_envs}）")
        print(f"  共 {len(self.tiers)} 个测试层，总 episode 数: {total_eps}")
        print(f"  预计实际 wall-time episodes: {math.ceil(total_eps / self.num_envs)} 轮")
        print("=" * 60 + "\n")
        self._running = True
        self._inject_tier_command(self.tiers[0])
        self._read_targets_from_cmd()   # 读取初始目标
        print(f"[Benchmark] 开始 {self.tiers[0].name}: {self.tiers[0].description}")

    def is_running(self) -> bool:
        return self._running

    def record_step(self, env, done: torch.Tensor):
        """
        每个 env.step() 后调用一次。

        参数:
            env:  Isaac Lab environment
            done: (num_envs,) bool tensor
        """
        if not self._running:
            return

        # ── 1. 读取本步所有 env 的状态（向量化）───────────────────────────
        pos_err, rot_err, base_pitch, base_roll, base_height, contact_std = \
            self._read_all_env_states(env)

        # ── 2. 写入缓存（clamp ptr 防越界）─────────────────────────────────
        ptr = self._buf_ptr.clamp(max=self.max_steps - 1)  # (num_envs,)
        env_idx = torch.arange(self.num_envs, device=self.device)

        self._buf_pos_err[env_idx, ptr]     = pos_err
        self._buf_rot_err[env_idx, ptr]     = rot_err
        self._buf_base_pitch[env_idx, ptr]  = base_pitch
        self._buf_base_roll[env_idx, ptr]   = base_roll
        self._buf_base_height[env_idx, ptr] = base_height
        self._buf_contact_std[env_idx, ptr] = contact_std

        self._buf_ptr = (self._buf_ptr + 1).clamp(max=self.max_steps)

        # ── 3. 判断连续达标，记录首次到达时间 ──────────────────────────────
        reached = (pos_err < self.pos_thresh) & (rot_err < self.rot_thresh)
        self._hold_count = torch.where(reached, self._hold_count + 1, torch.zeros_like(self._hold_count))

        newly_reached = (self._hold_count >= self.hold_steps) & (self._time_to_reach == float("inf"))
        if newly_reached.any():
            step_time = self._buf_ptr.float() * self._get_step_dt(env)
            self._time_to_reach = torch.where(newly_reached, step_time, self._time_to_reach)

        # ── 4. 处理本步 done 的 env ──────────────────────────────────────
        done = done.to(self.device)
        done_indices = done.nonzero(as_tuple=True)[0]
        if len(done_indices) > 0:
            self._finalize_envs(done_indices)
            self._reset_env_buffers(done_indices)
            self._advance_tier(env)

    def finish(self):
        self._running = False
        self._save_results()
        self._print_summary()

    # ─────────────────────────────────────────────────────────────────────
    # 内部：状态读取
    # ─────────────────────────────────────────────────────────────────────

    def _read_all_env_states(self, env):
        """
        向量化读取所有 env 的状态。
        返回 6 个 (num_envs,) 的 tensor。
        """
        # 位置 / 旋转误差
        pos_err = self.ee_cmd.metrics["position_error"].clone()     # (num_envs,)
        rot_err = self.ee_cmd.metrics["orientation_error"].clone()  # (num_envs,)

        # 机身姿态
        try:
            from isaaclab.utils import math as math_utils
            base_quat = env.scene["robot"].data.root_quat_w         # (num_envs, 4)
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat)
            base_pitch  = pitch                                     # (num_envs,)
            base_roll   = roll                                      # (num_envs,)
        except Exception:
            base_pitch = torch.zeros(self.num_envs, device=self.device)
            base_roll  = torch.zeros(self.num_envs, device=self.device)

        base_height = env.scene["robot"].data.root_pos_w[:, 2]     # (num_envs,)

        # 接触力标准差
        contact_std = self._get_contact_force_std_batch(env)       # (num_envs,)

        return pos_err, rot_err, base_pitch, base_roll, base_height, contact_std

    def _get_contact_force_std_batch(self, env) -> torch.Tensor:
        """读取所有 env 的四足接触力标准差，返回 (num_envs,) tensor。"""
        try:
            contact_sensor = env.scene.sensors.get("contact_forces", None)
            if contact_sensor is not None:
                forces = contact_sensor.data.net_forces_w          # (num_envs, num_feet, 3)
                force_norms = torch.norm(forces, dim=-1)           # (num_envs, num_feet)
                return force_norms.std(dim=-1)                     # (num_envs,)
        except Exception:
            pass
        return torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _get_step_dt(env) -> float:
        try:
            return env.step_dt
        except AttributeError:
            return 0.02

    # ─────────────────────────────────────────────────────────────────────
    # 内部：episode finalize / reset
    # ─────────────────────────────────────────────────────────────────────

    def _finalize_envs(self, done_indices: torch.Tensor):
        """
        对 done_indices 中的每个 env，从缓存中提取数据并生成 EpisodeRecord。
        使用 CPU numpy 做统计（缓存本身是 tensor，按需转换）。
        """
        for i in done_indices.tolist():
            n = int(self._buf_ptr[i].item())
            if n == 0:
                # 极少情况：env 在写入任何数据前就 done（例如初始化 reset）
                continue

            # 提取本 episode 的数据（CPU numpy）
            pos_errors  = self._buf_pos_err[i, :n].cpu().numpy()
            rot_errors  = self._buf_rot_err[i, :n].cpu().numpy()
            base_pitches = self._buf_base_pitch[i, :n].cpu().numpy()
            base_rolls  = self._buf_base_roll[i, :n].cpu().numpy()
            base_heights = self._buf_base_height[i, :n].cpu().numpy()
            contact_stds = self._buf_contact_std[i, :n].cpu().numpy()

            best_idx  = int(np.argmin(pos_errors))
            success   = self._time_to_reach[i].item() < float("inf")
            tier_idx  = int(self._env_tier_idx[i].item())
            tier_name = self.tiers[tier_idx].name

            record = EpisodeRecord(
                tier_name=tier_name,
                env_idx=i,
                target_pitch=self._cur_target_pitch[i].item(),
                target_l=self._cur_target_l[i].item(),
                target_yaw=self._cur_target_yaw[i].item(),
                target_z_world=self._cur_target_z_world[i].item(),
                pos_error_final=float(pos_errors[-1]),
                rot_error_final=float(rot_errors[-1]),
                pos_error_min=float(np.min(pos_errors)),
                success=success,
                base_pitch_at_best=float(base_pitches[best_idx]),
                base_roll_at_best=float(base_rolls[best_idx]),
                base_height_at_best=float(base_heights[best_idx]),
                base_pitch_std=float(np.std(base_pitches)),
                contact_force_std=float(np.mean(contact_stds)),
                time_to_reach=self._time_to_reach[i].item(),
            )
            self.all_records.append(record)

            tier_idx = int(self._env_tier_idx[i].item())
            self._tier_episode_counts[tier_idx] += 1

            status = "✓ 成功" if success else "✗ 失败"
            print(f"  [env {i:>2}] {tier_name} ep#{self._env_episode_counts[i].item():>3} "
                  f"{status}  min_err={record.pos_error_min:.4f}m")

        # 更新 per-env episode 计数
        self._env_episode_counts[done_indices] += 1

    def _reset_env_buffers(self, done_indices: torch.Tensor):
        """清零 done envs 的缓存和计数器。"""
        self._buf_ptr[done_indices]        = 0
        self._hold_count[done_indices]     = 0
        self._time_to_reach[done_indices]  = float("inf")
        # 缓冲区无需显式清零（ptr=0 后会被覆盖）
        self._env_tier_idx[done_indices]   = self._tier_idx
        self._read_targets_for_envs(done_indices)

        # 目标在 env reset 后由 ee_cmd resample，下一步读取即可
        # 这里用 after_reset_hook 或直接在下一步 record_step 开头读取
        self._read_targets_for_envs(done_indices)

    # ─────────────────────────────────────────────────────────────────────
    # 内部：tier 管理
    # ─────────────────────────────────────────────────────────────────────

    def _advance_tier(self, env):
        # 不再使用 n_done 累加全局计数，因为计数已在 _finalize_envs 中处理
        current_tier = self.tiers[self._tier_idx]
        # 检查当前 tier 的完成数是否已经达到要求
        if self._tier_episode_counts[self._tier_idx] >= current_tier.n_episodes:
            # 打印本 tier 统计
            tier_records = [r for r in self.all_records if r.tier_name == current_tier.name]
            if tier_records:
                sr = sum(r.success for r in tier_records) / len(tier_records)
                avg_err = np.mean([r.pos_error_min for r in tier_records])
                print(f"\n[Benchmark] ✓ {current_tier.name} 完成 | "
                    f"实际 episodes: {len(tier_records)} | "
                    f"成功率: {sr:.1%} | 平均最小误差: {avg_err:.4f}m\n")

            # 进入下一个 tier
            self._tier_idx += 1
            if self._tier_idx >= len(self.tiers):
                self.finish()
                return

            # 注入新 tier 参数
            new_tier = self.tiers[self._tier_idx]
            self._inject_tier_command(new_tier)
            print(f"[Benchmark] 开始 {new_tier.name}: {new_tier.description}")

        # 更新总完成 episode 数（可选，用于打印进度）
        # self._total_episodes_done = self._tier_episode_counts.sum().item()

    def _inject_tier_command(self, tier: BenchmarkTier):
        """覆盖 ee_command 的采样范围。"""
        cfg = self.ee_cmd.cfg
        cfg.ranges.p_pitch = tier.p_pitch_range
        cfg.ranges.p_l     = tier.p_l_range
        cfg.ranges.p_yaw   = tier.p_yaw_range

    def _read_targets_from_cmd(self):
        """从 ee_cmd 读取所有 env 当前 episode 的目标参数（全量）。"""
        try:
            sphere = self.ee_cmd.ee_end_sphere       # (num_envs, 3) = [l, pitch, yaw]
            self._cur_target_l     = sphere[:, 0].clone()
            self._cur_target_pitch = sphere[:, 1].clone()
            self._cur_target_yaw   = sphere[:, 2].clone()
            self._cur_target_z_world = self.ee_cmd.pose_end_w[:, 2].clone()
        except Exception:
            pass  # 初始化时 cmd 可能还未 resample，忽略

    def _read_targets_for_envs(self, env_indices: torch.Tensor):
        """从 ee_cmd 读取指定 env 的目标参数（reset 后调用）。"""
        try:
            sphere = self.ee_cmd.ee_end_sphere       # (num_envs, 3)
            self._cur_target_l[env_indices]      = sphere[env_indices, 0]
            self._cur_target_pitch[env_indices]  = sphere[env_indices, 1]
            self._cur_target_yaw[env_indices]    = sphere[env_indices, 2]
            self._cur_target_z_world[env_indices] = self.ee_cmd.pose_end_w[env_indices, 2]
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # 保存与可视化
    # ─────────────────────────────────────────────────────────────────────

    def _save_results(self):
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
        self._save_heatmap_data(timestamp)

    def _save_heatmap_data(self, timestamp: str):
        if not self.all_records:
            return
        pitch_bins = np.array([-0.3, 0.0, math.pi / 6, math.pi / 3, math.pi * 2 / 5])
        yaw_bins   = np.linspace(-math.pi * 3 / 5, math.pi * 3 / 5, 7)
        heatmap = np.full((len(pitch_bins) - 1, len(yaw_bins) - 1), np.nan)
        counts  = np.zeros_like(heatmap)
        for rec in self.all_records:
            pi = np.clip(np.searchsorted(pitch_bins, rec.target_pitch, side="right") - 1, 0, heatmap.shape[0] - 1)
            yi = np.clip(np.searchsorted(yaw_bins,   rec.target_yaw,   side="right") - 1, 0, heatmap.shape[1] - 1)
            if np.isnan(heatmap[pi, yi]):
                heatmap[pi, yi] = 0.0
            heatmap[pi, yi] += float(rec.success)
            counts[pi, yi]  += 1
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
        grid  = np.full((len(pitch_bins) - 1, len(yaw_bins) - 1), np.nan)
        accum: dict = {}
        for rec in self.all_records:
            pi = np.clip(np.searchsorted(pitch_bins, rec.target_pitch, side="right") - 1, 0, grid.shape[0] - 1)
            yi = np.clip(np.searchsorted(yaw_bins,   rec.target_yaw,   side="right") - 1, 0, grid.shape[1] - 1)
            accum.setdefault((pi, yi), []).append(getattr(rec, field_name))
        for (pi, yi), vals in accum.items():
            grid[pi, yi] = np.mean(vals)
        return grid

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("  Benchmark 汇总")
        print("=" * 70)
        print(f"{'Tier':<25} {'N':>5} {'成功率':>8} {'avg最小误差':>12} "
              f"{'avg狗pitch':>12} {'avg到达时间':>12}")
        print("-" * 70)
        for tier in self.tiers:
            recs = [r for r in self.all_records if r.tier_name == tier.name]
            if not recs:
                continue
            n       = len(recs)
            sr      = sum(r.success for r in recs) / n
            avg_err = np.mean([r.pos_error_min for r in recs])
            avg_pitch = np.mean([r.base_pitch_at_best for r in recs])
            reached = [r.time_to_reach for r in recs if r.time_to_reach < float("inf")]
            avg_time = np.mean(reached) if reached else float("inf")
            print(f"{tier.name:<25} {n:>5} {sr:>8.1%} {avg_err:>12.4f}m "
                  f"{math.degrees(avg_pitch):>11.1f}° {avg_time:>11.2f}s")
        print("=" * 70)
        total = len(self.all_records)
        overall_sr = sum(r.success for r in self.all_records) / total if total > 0 else 0
        print(f"总体成功率: {overall_sr:.1%}  (共 {total} episodes, {self.num_envs} 并行环境)")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  可视化工具（离线，benchmark 跑完后调用）
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(npz_path: str, save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib"); return

    data = np.load(npz_path)
    success_rate = data["success_rate"]
    pos_error    = data["pos_error_min_mean"]
    base_pitch   = data["base_pitch_mean"]
    pitch_bins   = data["pitch_bins"]
    yaw_bins     = data["yaw_bins"]

    pitch_labels = [f"{math.degrees(p):.0f}°" for p in pitch_bins]
    yaw_labels   = [f"{math.degrees(y):.0f}°" for y in yaw_bins]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EE Low-Reach Benchmark\n（行=EE目标pitch，越大越低；列=目标yaw方向）", fontsize=13)

    ax = axes[0]
    im = ax.imshow(success_rate, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="成功率")
    ax.set_title("EE到达成功率")
    ax.set_xlabel("目标 yaw (°)"); ax.set_ylabel("目标 pitch (°) [越大越低]")
    ax.set_xticks(range(len(yaw_labels) - 1)); ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1)); ax.set_yticklabels(pitch_labels[:-1], fontsize=8)
    for i in range(success_rate.shape[0]):
        for j in range(success_rate.shape[1]):
            if not np.isnan(success_rate[i, j]):
                ax.text(j, i, f"{success_rate[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="black" if 0.3 < success_rate[i, j] < 0.7 else "white")

    ax = axes[1]
    im2 = ax.imshow(pos_error, vmin=0, vmax=0.15, cmap="YlOrRd_r", aspect="auto", origin="lower")
    plt.colorbar(im2, ax=ax, label="平均最小位置误差 (m)")
    ax.set_title("EE最小位置误差 (越绿越好)")
    ax.set_xlabel("目标 yaw (°)")
    ax.set_xticks(range(len(yaw_labels) - 1)); ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1)); ax.set_yticklabels(pitch_labels[:-1], fontsize=8)

    ax = axes[2]
    pitch_deg = np.degrees(base_pitch) if not np.all(np.isnan(base_pitch)) else base_pitch
    vmax_p = max(30, np.nanmax(np.abs(pitch_deg))) if not np.all(np.isnan(pitch_deg)) else 30
    im3 = ax.imshow(pitch_deg, vmin=-vmax_p, vmax=vmax_p, cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(im3, ax=ax, label="机身pitch (°) [正=前倾/俯身]")
    ax.set_title("到达时机身pitch（俯身程度）")
    ax.set_xlabel("目标 yaw (°)")
    ax.set_xticks(range(len(yaw_labels) - 1)); ax.set_xticklabels(yaw_labels[:-1], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pitch_labels) - 1)); ax.set_yticklabels(pitch_labels[:-1], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"热力图已保存: {save_path}")
    else:
        plt.show()


def plot_cdf(csv_path: str, save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("需要安装 matplotlib 和 pandas"); return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for (tier_name, group), color in zip(df.groupby("tier_name"), colors):
        errors = np.sort(group["pos_error_min"].dropna().values)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=tier_name, color=color, linewidth=2)
    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.7, label="5cm阈值")
    ax.set_xlabel("最小EE位置误差 (m)"); ax.set_ylabel("累积比例 (CDF)")
    ax.set_title("各难度层级 EE 到达误差分布")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  集成示例
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_EXAMPLE = '''
# ── play.py 集成（并行版）──────────────────────────────────────────────────
from benchmark_ee_low_reach import BenchmarkRunner

ee_command = env.command_manager.get_term("ee_pose")

runner = BenchmarkRunner(
    env=env,
    ee_command=ee_command,
    pos_success_threshold=0.05,
    rot_success_threshold=0.3,
    save_dir="./benchmark_results",
    hold_steps_required=10,
    max_episode_steps=1000,      # ← 根据你的 episode 最大长度设置
)
runner.start()

while runner.is_running():
    with torch.no_grad():
        action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    runner.record_step(env, terminated | truncated)

# runner.finish() 在内部自动调用
# 离线绘图：
# from benchmark_ee_low_reach import plot_heatmap, plot_cdf
# plot_heatmap("./benchmark_results/heatmap_xxx.npz", save_path="heatmap.png")
# plot_cdf("./benchmark_results/benchmark_xxx.csv",   save_path="cdf.png")
'''

# ─────────────────────────────────────────────────────────────────────────────
# 6.  命令行（离线分析）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EE Low-Reach Benchmark 离线分析工具")
    parser.add_argument("--plot", type=str, help="绘制热力图，传入 .npz 路径")
    parser.add_argument("--cdf",  type=str, help="绘制CDF曲线，传入 .csv 路径")
    parser.add_argument("--save", type=str, default=None, help="图片保存路径")
    args = parser.parse_args()
    if args.plot:
        plot_heatmap(args.plot, save_path=args.save)
    elif args.cdf:
        plot_cdf(args.cdf, save_path=args.save)
    else:
        print("集成示例：")
        print(INTEGRATION_EXAMPLE)