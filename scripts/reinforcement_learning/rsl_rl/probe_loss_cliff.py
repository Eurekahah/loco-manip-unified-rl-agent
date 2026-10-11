# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""在 TensorBoard 标量上找"悬崖式"突降：定位 step、量出前后平台、给上下文。

为什么需要它
    "损失曲线有个悬崖"这种观察，靠肉眼看 TensorBoard 只能定性（大概第几千次迭代？），
    而不同的机制（课程台阶、约束系数 λ 开始爬、学习率被 KL 调度砍、dagger 模仿更新、
    latent 表示塌缩）会给出**不同的时间点**。把每个候选时间点和代码里的调度对齐，
    才能判断到底是哪一条。这个脚本负责把"时间点"钉死：

    * 对每条曲线做滑窗平滑（默认 51 点），再算"窗口内的相对变化"；
    * 取相对变化最负的前 N 个位置 ⇒ 那就是悬崖；
    * 打印悬崖前后的原始值、以及同期的 `Loss/learning_rate`、`Policy/mean_noise_std`
      和（如果给了 `--constraint-schedule`）由 `PPORoA` 公式算出的 λ。

用法::

    # 单 run：找最大 5 个突降
    python scripts/reinforcement_learning/rsl_rl/probe_loss_cliff.py \
        --run logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k

    # 指定曲线 / 窗口 / λ 调度（(start,end,delay,ramp)，与 cfg 里一致）
    python .../probe_loss_cliff.py --run <run> \
        --tags Loss/constraint Loss/latent_distance Loss/imitation \
        --window 200 --topk 3 --constraint-schedule 0.0 0.1 3000 7000

不启动 Isaac Sim（纯 tensorboard + numpy）。读一次 70MB 的 event 文件大约 20~60 s。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = [
    "Loss/constraint",
    "Loss/latent_distance",
    "Loss/imitation",
    "Loss/learning_rate",
    "Loss/surrogate",
    "Policy/mean_noise_std",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="定位 loss 曲线上的悬崖式突降")
    p.add_argument("--run", required=True, help="run 目录（含 events.out.tfevents.*）")
    p.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    p.add_argument("--window", type=int, default=51, help="平滑窗口（点数）")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--span", type=int, default=120, help="打印悬崖前后各多少点")
    p.add_argument(
        "--constraint-schedule", nargs=4, type=float, default=None,
        metavar=("START", "END", "DELAY", "RAMP"),
        help="PPORoA 的 constraint_coef_schedule（给了就按公式算 λ 打出来）",
    )
    return p.parse_args()


def load_series(run: str, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags()["scalars"])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        if tag not in avail:
            print(f"[warn] 该 run 没有 tag：{tag}")
            continue
        events = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in events], dtype=float),
            np.array([e.value for e in events], dtype=float),
        )
    return out


def moving_average(v: np.ndarray, w: int) -> np.ndarray:
    """中心滑窗均值；两端用缩短短窗，避免边缘假象。"""
    if w <= 1 or len(v) < 3:
        return v.copy()
    w = min(w, len(v) - (len(v) % 2 == 0))
    half = w // 2
    csum = np.concatenate([[0.0], np.cumsum(v)])
    out = np.empty_like(v)
    for i in range(len(v)):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def find_cliffs(steps: np.ndarray, values: np.ndarray, window: int, topk: int) -> list[dict]:
    """返回相对变化最负的若干位置（用平滑后的曲线 + 前 window 点的平台做基准）。"""
    sm = moving_average(values, window)
    rows: list[dict] = []
    for i in range(window, len(sm)):
        base = sm[i - window]
        if not np.isfinite(base) or abs(base) < 1e-12:
            continue
        rel = (sm[i] - base) / abs(base)
        rows.append({"i": i, "step": float(steps[i]), "rel": float(rel),
                     "before": float(base), "after": float(sm[i])})
    rows.sort(key=lambda r: r["rel"])
    return rows[:topk]


def lam(counter: float, schedule: list[float]) -> float:
    start, end, delay, ramp = schedule
    stage = min(max((counter - delay) / max(ramp, 1.0), 0.0), 1.0)
    return start + stage * (end - start)


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.run):
        raise SystemExit(f"目录不存在：{args.run}")
    series = load_series(args.run, args.tags)
    if not series:
        return 1
    print(f"# {os.path.basename(args.run.rstrip('/'))}")
    print(f"   平滑窗口 = {args.window} 点；突降 = 窗口内相对变化最负的位置\n")

    for tag, (steps, values) in series.items():
        cliffs = find_cliffs(steps, values, args.window, args.topk)
        print(f"## {tag}   （{len(steps)} 点，值域 {values.min():.4g} ~ {values.max():.4g}）")
        for k, c in enumerate(cliffs, 1):
            extra = ""
            if args.constraint_schedule is not None and tag.startswith("Loss/"):
                extra = f"，同期 λ={lam(c['step'], args.constraint_schedule):.4f}"
            print(f"  {k}) step {c['step']:.0f}: {c['before']:.4g} → {c['after']:.4g}"
                  f"  （相对 {c['rel'] * 100:+.1f}%）{extra}")
        # 打印最大那个悬崖附近的原始点
        if cliffs:
            c = cliffs[0]
            lo = max(0, int(np.searchsorted(steps, c["step"] - args.span)))
            hi = min(len(steps), int(np.searchsorted(steps, c["step"] + args.span)) + 1)
            print(f"  附近原始点（step: value，最多 12 个）：")
            idx = np.unique(np.linspace(lo, hi - 1, num=min(12, max(1, hi - lo))).round().astype(int))
            print("    " + "  ".join(f"{steps[i]:.0f}:{values[i]:.4g}" for i in idx))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
