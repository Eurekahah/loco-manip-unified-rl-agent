# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""EE 锚点对照一键脚本（`docs/review/TODO_zh.md` P3）。

"EE 锚点"是什么
---------------
低层训练早期（课程 s0）把机械臂的目标位姿**锁死在一个固定点**上，让机械臂先别乱动、
底盘专心学平衡，之后再逐步放开到完整工作空间（课程 s1→s3）。这个"锁死的固定点"就是锚点。
仓库里可选的三种锚点（由 `probe_root_height_termination.py --freeze_ee_preset` 实现）：

* `none`    —— 不锁（= 无课程），EE 命令按任务采样；
* `default` —— 锁在**默认姿态**（机械臂举起）；
* `low`     —— 锁在工作空间中心的**低位前伸**位姿（= 课程 s0 最终选的那个）。

DEF-006 实测过 20 s 内 `root_z < 0.30` 的触发率：none **25.8%** / default **55.5%**（更差，
举臂抬高重心）/ low **1.0%** ⇒ 最终选 `low`。本脚本把这组对照固化成一条命令，
以后改 EE 区间 / 换锚点前先跑一遍（512 envs × 600 步 ≈ 2~4 min/组）。

为什么必须"每组一个子进程"
--------------------------
一个进程只能建一个 Isaac env（同进程里反复 `gym.make` 会踩 PhysX/kit 的全局状态），
所以本脚本只负责**编排**：逐组起 `probe_root_height_termination.py`、收它写出的
`--json_out`、汇总成一张表。

用法::

    python scripts/reinforcement_learning/rsl_rl/sweep_ee_anchor.py \
        --task History-Adaptation-Deeprobotics-M20-v0 \
        --policy logs/rsl_rl/history_adaptation/<run>/exported_deploy/policy.pt \
        --num_envs 512 --steps 600 \
        --out logs/smoke/ee_anchor_sweep_<日期>.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_root_height_termination.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EE 锚点对照一键脚本（none / default / low）")
    p.add_argument("--task", default="History-Adaptation-Deeprobotics-M20-v0")
    p.add_argument("--policy", required=True, help="部署态低层策略 policy.pt（旁边要有 policy_layout.json）")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=600, help="rollout 步数（512 envs × 600 步 ≈ 2 min/组）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--presets",
        default="full,default,low,cfg",
        help="要跑的锚点，逗号分隔（probe 支持的取值：full / default / low / cfg，none=cfg 别名）",
    )
    p.add_argument("--python", default=sys.executable, help="用哪个解释器起探针（要能 import isaaclab）")
    p.add_argument("--keep_push", action="store_true", default=False, help="保留 push 事件（默认关，隔离高度因素）")
    p.add_argument("--timeout", type=float, default=900.0, help="单组超时（秒）")
    p.add_argument("--out", default=None, help="汇总 JSON 落盘路径（默认 logs/smoke/ee_anchor_sweep_<日期>.json）")
    p.add_argument("--log_dir", default=os.path.join(REPO_ROOT, "logs", "smoke"), help="每组日志落盘目录")
    return p.parse_args()


def run_one(args: argparse.Namespace, preset: str, log_path: str) -> dict | None:
    fd, json_path = tempfile.mkstemp(suffix=".json", prefix=f"ee_anchor_{preset}_")
    os.close(fd)
    cmd = [
        args.python, PROBE,
        "--task", args.task,
        "--headless",
        "--num_envs", str(args.num_envs),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--policy", args.policy,
        "--freeze_ee_preset", preset,
        "--json_out", json_path,
    ]
    if args.keep_push:
        cmd.append("--keep_push")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[sweep] ── preset={preset} ──> {os.path.basename(log_path)}")
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=env, timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"[sweep]    -> TIMEOUT（>{args.timeout:.0f}s），判 FAIL")
        return None

    if proc.returncode != 0:
        print(f"[sweep]    -> EXIT={proc.returncode}，判 FAIL（看 {log_path} 的尾部）")
        return None

    try:
        with open(json_path, encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep]    -> 解析 {json_path} 失败：{exc}")
        return None
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass

    thr = metrics["thresholds"][f"{metrics['min_height_threshold']:.2f}"]
    print(
        f"[sweep]    -> OK  root_z p05={metrics['root_z_p05']:.4f}  "
        f"height_err mean={metrics['height_error_mean']:+.4f}  "
        f"{metrics['window_s']:.0f}s 内触发(阈值 {metrics['min_height_threshold']:.2f})="
        f"{thr['episode_frac']:.1%}  "
        f"倾角>{metrics['limit_angle_rad']/3.141592653589793*180:.1f}° 比例={metrics['tilt_episode_frac']:.1%}"
    )
    return metrics


def fmt_anchor_note(preset: str) -> str:
    return {
        "full": "全任务分布（= 无课程，s3）",
        "default": "锁默认姿态（举臂）",
        "low": "锁低位前伸（课程 s0 选它）",
        "cfg": "不改 cfg（当前代码里 == low，用来验证这一点）",
    }.get(preset, "自定义")


def main() -> int:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    print(f"[sweep] task={args.task} presets={presets} num_envs={args.num_envs} steps={args.steps}")
    print(f"[sweep] policy={args.policy}")

    results: list[dict] = []
    for preset in presets:
        log_path = os.path.join(args.log_dir, f"{stamp}_ee_anchor_{preset}.log")
        metrics = run_one(args, preset, log_path)
        if metrics is not None:
            metrics["log"] = log_path
            results.append(metrics)

    ok = len(results)
    print(f"\n### EE 锚点对照 —— {stamp}")
    print(f"\n`task={args.task}` / `{args.num_envs} envs × {args.steps} steps` / `policy={args.policy}`\n")
    if results:
        win = results[0]["window_s"]
        print(
            f"| 锚点 | 说明 | root_z p05 (m) | height_error 均值 (m) | {win:.0f} s 内 root_z<阈值 "
            f"| 倾角超限比例 | root_z 最小 |"
        )
        print("|---|---|---|---|---|---|---|")
        for m in results:
            thr = m["thresholds"][f"{m['min_height_threshold']:.2f}"]
            limit_deg = m["limit_angle_rad"] / 3.141592653589793 * 180.0
            print(
                f"| `{m['preset']}` | {fmt_anchor_note(m['preset'])} | {m['root_z_p05']:.4f} | "
                f"{m['height_error_mean']:+.4f} | {thr['episode_frac']:.1%} "
                f"(阈值 {m['min_height_threshold']:.2f}) | {m['tilt_episode_frac']:.1%} "
                f"(>{limit_deg:.1f}°) | {m['root_z_min']:.4f} |"
            )
    else:
        print("（没有任何一组成功 —— 看日志）")

    if results:
        def _ep_frac(m: dict) -> float:
            return m["thresholds"][f"{m['min_height_threshold']:.2f}"]["episode_frac"]

        best = min(results, key=_ep_frac)
        print(
            f"\n**结论**：`{best['preset']}`（{fmt_anchor_note(best['preset'])}）在"
            f"「{best['min_height_threshold']:.2f} 阈值下 20 s 内触发率」这项上最好"
            f"（{_ep_frac(best):.1%}）。"
        )

    out_path = args.out or os.path.join(args.log_dir, f"{stamp}_ee_anchor_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stamp": stamp, "task": args.task, "policy": args.policy,
                   "num_envs": args.num_envs, "steps": args.steps, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[sweep] 汇总已写入 {out_path}；OK {ok}/{len(presets)}")
    return 0 if ok == len(presets) else 1


if __name__ == "__main__":
    sys.exit(main())
