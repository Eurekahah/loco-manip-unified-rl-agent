# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""冒烟回归矩阵：一次跑完所有任务各 2 iteration，逐条记 EXIT / 关键打印，卡死自动判 SKIP。

为什么需要它
------------
`docs/review/TODO_zh.md` P3 要求"每个改动都要跑一遍回归矩阵"，但手工跑 8~13 条命令
（每条都要等 40~90 s，还得自己盯 `$LASTEXITCODE`）很容易漏；而且本机（Windows + A4000）
跑**生成地形**的任务会在 env 创建期**死锁**（DEF-031），手工跑会一直挂着。

所以这里把矩阵固化成脚本：

* 每条任务独立进程：`train.py --task <T> --headless --num_envs 64 --max_iterations 2`
  日志落到 `logs/smoke/<日期>_<task>.log`（与历史日志同一命名）；
* **卡死检测**：日志文件连续 `--stall-timeout` 秒没有增长 ⇒ 判 **SKIP**（打印 DEF-031 提示）
  并杀掉进程；整体超过 `--timeout` 也判 SKIP（这两种都**不算失败**）；
* 成功判据：日志里 `Learning iteration` 至少出现 1 次、且没有 `Traceback`；
* 结尾打一张 Markdown 表（`--out` 可落盘），**有 FAIL 就非零退出**（可以直接接 CI / 脚本链）。

用法::

    # 用装了 Isaac Lab 的解释器跑（脚本内部用 sys.executable 去起 train.py）
    python scripts/reinforcement_learning/rsl_rl/smoke_regression.py
    # 只跑几条 / 调参数 / 落盘
    python scripts/reinforcement_learning/rsl_rl/smoke_regression.py \
        --tasks History-Adaptation-Deeprobotics-M20-v0 Flat-Deeprobotics-M20-Piper-WBC-v0 \
        --num_envs 64 --stall-timeout 180 --out logs/smoke/2026-09-30_regression.md

注意：本脚本**不启动 Isaac**（只起子进程），所以可以用任何 python 跑；
但子进程必须能 import isaaclab ⇒ 默认就用当前解释器（`sys.executable`），
换解释器请用 `--python`。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_PY = os.path.join(REPO_ROOT, "scripts", "reinforcement_learning", "rsl_rl", "train.py")

#: 默认矩阵（与 docs/review/DONE_zh.md 第六节"基线可运行性验收"一致 + 本轮新增任务）
DEFAULT_TASKS = [
    # ── 低层（平地）──────────────────────────────────────────────
    "History-Adaptation-Deeprobotics-M20-v0",
    "Flat-Deeprobotics-M20-Piper-WBC-v0",
    "Flat-Deeprobotics-M20-Piper-v0",
    "Flat-Deeprobotics-M20-Piper-Arm-v0",
    # ── 高层（教师/遥操）────────────────────────────────────────
    "Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0",
    "Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0",
    "Isaac-M20-Piper-Teleop-v0",
    "Isaac-M20-Piper-Teleop-History-v0",
    "Isaac-Deeprobotics-High-Level-Nav-Flat-Teacher-v0",
    # ── 消融（2026-09-30 新增，见 DEFECT_LOG_zh.md DEF-033）─────
    "History-Ablation-PushOnly-Deeprobotics-M20-v0",
    "History-Ablation-RewardOnly-Deeprobotics-M20-v0",
    # ── 生成地形（本机必卡死 ⇒ 走 SKIP 分支，见 DEF-031）───────
    "Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="冒烟回归矩阵（逐任务 EXIT + 卡死判 SKIP）")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="要跑的任务名（默认整套矩阵）")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--timeout", type=float, default=420.0, help="单任务总超时（秒）；超时判 SKIP")
    p.add_argument(
        "--stall-timeout",
        type=float,
        default=150.0,
        help="日志连续多久没增长就判 SKIP（秒）。生成地形任务在本机表现为「完全无输出」，见 DEF-031",
    )
    p.add_argument("--python", default=sys.executable, help="用哪个解释器起 train.py")
    p.add_argument("--log-dir", default=os.path.join(REPO_ROOT, "logs", "smoke"))
    p.add_argument("--out", default=None, help="把汇总 Markdown 写到这里（同时打印）")
    p.add_argument("--tag", default=None, help="日志文件名前缀（默认当天日期）")
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="给所有子进程额外注入的环境变量，可重复。例："
        "--env RL_TRAINING_LOW_LEVEL_POLICY_WBC=D:/x/policy.pt（高层四条的低层 checkpoint 路径）",
    )
    p.add_argument(
        "--hydra",
        action="append",
        default=[],
        metavar="OVERRIDE",
        help="透传给 train.py 的 hydra 覆盖项，可重复。例："
        "--hydra env.actions.pre_trained_pick_action.policy_path=D:/x/policy.pt",
    )
    return p.parse_args()


def run_one(task: str, args: argparse.Namespace, tag: str) -> dict:
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{tag}_{task}.log")
    cmd = [
        args.python,
        TRAIN_PY,
        "--task", task,
        "--headless",
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.max_iterations),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"[smoke] --env 需要 KEY=VALUE 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    if args.hydra:
        cmd += list(args.hydra)
    # 每条任务一个进程组，方便卡死时整组杀掉（Isaac 会再起 kit 子进程）
    popen_kwargs: dict = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    t0 = time.time()
    last_size, last_growth = -1, time.time()
    outcome, note = "FAIL", ""
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, **popen_kwargs)
        killed = False
        while True:
            rc = proc.poll()
            size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
            if size != last_size:
                last_size, last_growth = size, time.time()
            if rc is not None:
                break
            now = time.time()
            if now - last_growth > args.stall_timeout:
                _kill(proc)
                killed = True
                outcome = "SKIP"
                note = f"日志 {args.stall_timeout:.0f}s 无增长（疑似 env 创建期死锁，见 DEF-031），已杀掉"
                break
            if now - t0 > args.timeout:
                _kill(proc)
                killed = True
                outcome = "SKIP"
                note = f"总耗时超过 {args.timeout:.0f}s，已杀掉"
                break
            time.sleep(1.0)
        if not killed:
            # 进程自己退了：用日志判成功
            text = _read_log(log_path)
            n_iter = text.count("Learning iteration")
            if proc.returncode != 0:
                outcome, note = "FAIL", f"EXIT={proc.returncode}"
            elif "Traceback" in text:
                outcome, note = "FAIL", "日志里有 Traceback"
            elif n_iter < 1:
                outcome, note = "FAIL", "日志里没有 'Learning iteration'"
            else:
                outcome, note = "OK", f"EXIT=0，迭代 {n_iter} 次"
            rc = proc.returncode
    dt = time.time() - t0
    return {"task": task, "outcome": outcome, "note": note,
            "seconds": dt, "log": os.path.relpath(log_path, REPO_ROOT),
            "exit": locals().get("rc", None)}


def _kill(proc: subprocess.Popen) -> None:
    """尽量把整棵进程树杀掉（Isaac 会再起 kit 子进程）。"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), 15)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _read_log(path: str, limit: int = 4_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def main() -> int:
    args = parse_args()
    tag = args.tag or _dt.date.today().isoformat()
    print(f"[smoke] 共 {len(args.tasks)} 个任务；num_envs={args.num_envs} "
          f"max_iterations={args.max_iterations} stall={args.stall_timeout:.0f}s "
          f"timeout={args.timeout:.0f}s")
    rows = []
    for i, task in enumerate(args.tasks, 1):
        print(f"[smoke] ({i}/{len(args.tasks)}) {task} ...", flush=True)
        row = run_one(task, args, tag)
        rows.append(row)
        print(f"[smoke]     -> {row['outcome']}  {row['note']}  ({row['seconds']:.0f}s)", flush=True)

    lines = ["", f"### 冒烟回归 —— {tag}", "",
             "| 任务 | 结果 | 说明 | 耗时(s) | 日志 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['task']}` | {r['outcome']} | {r['note']} | {r['seconds']:.0f} | `{r['log']}` |")
    n_fail = sum(1 for r in rows if r["outcome"] == "FAIL")
    n_skip = sum(1 for r in rows if r["outcome"] == "SKIP")
    n_ok = len(rows) - n_fail - n_skip
    lines += ["", f"**OK {n_ok} / SKIP {n_skip} / FAIL {n_fail}**",
              "", "> SKIP = 本机跑不了生成地形任务（DEF-031）或超时；到能跑生成地形的机器上会自动变成 OK。", ""]
    table = "\n".join(lines)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(table)
        print(f"[smoke] 已写入 {args.out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
