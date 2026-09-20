# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""把 tensorboard run 的标量压成「关键指标 × 迭代 / 课程阶段」表，支持两个 run 对比。

为什么需要它
------------
``logs/rsl_rl/<experiment>/<run>/events.out.tfevents.*`` 单个 70 MB 左右，
用 ``EventAccumulator`` 手抠一次要 30~60 s；而 ``CommandTerm`` 的 metrics 是
「复位那一刻」的均值，单点读数抖动大、很容易误判趋势（见 TODO P1-1/P1-2）。
本脚本一次读盘、按「课程阶段 + 采样网格」聚合，直接给出可写进文档的数字。

用法
----
    # 1) 先看有哪些 tag（按前缀分组，避免手猜名字）
    python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
        --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 --list-tags

    # 2) 单个 run：阶段均值 + 采样网格 + 趋势
    python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
        --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 \
        --tags 'noise_std' --tags 'error_vel_xy' --tags 'root_height_below_minimum'

    # 3) 两个 run 对比（同迭代点对齐，A=--run，B=--baseline）
    python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
        --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 \
        --baseline logs/rsl_rl/history_adaptation/2026-09-19_09-02-50

不启动 Isaac Sim（纯 tensorboard + numpy）。解析结果默认缓存到
``<run>/.summary_cache.npz``（用 ``--no-cache`` 关掉、``--refresh`` 重读），
因为 70 MB 的事件文件读一次要几十秒。

关于「阶段」：课程只在 env reset 时推进，课程项按 **环境步数** 定长
（默认 25000 步/阶段），所以阶段边界要换算成迭代号
``ceil(k * stage_steps / num_steps_per_env)``；``num_steps_per_env`` 从
``<run>/params/agent.yaml`` 读（读不到用 24），可用 ``--stage-steps`` /
``--num-steps-per-env`` 覆盖。
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

# 默认关注的「关键指标」（正则，按顺序匹配到就收）——覆盖 P1 分析要看的量。
# tag 名以本仓库 rsl_rl 的 logger 为准（用 ``--list-tags`` 复核）。
DEFAULT_KEY_PATTERNS = [
    r"^Train/mean_reward$",
    r"^Train/mean_episode_length$",
    r"^Episode_Termination/",  # 终止构成（看**合计**，别只看单项）
    r"^Metrics/base_velocity/error_vel_(xy|yaw)$",
    r"^Metrics/base_velocity/end_error_(lin|ang)_vel$",
    r"^Policy/mean_noise_std$",
    r"^Metrics/body_pose/height_error_bias_steady$",
    r"^Metrics/body_pose/(roll|pitch)_error_bias$",
    r"^Metrics/ee_pose/",
    r"^Curriculum/",
    r"^Loss/(surrogate|value_function|entropy|learning_rate|latent_distance)$",
]


def apply_derives(data: dict, specs: list[str] | None) -> list[str]:
    """按 ``--derive 名称=tag1+tag2[-tag3]`` 生成派生指标（例如终止项求和），返回新 tag 名。"""
    if not specs:
        return []
    added: list[str] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--derive 需要 `名称=tag1+tag2` 形式，收到：{spec}")
        name, expr = spec.split("=", 1)
        terms = [t.strip() for t in expr.split("+") if t.strip()]
        if not terms:
            raise SystemExit(f"--derive 表达式为空：{spec}")
        base_steps, acc = None, None
        for term in terms:
            sign, tag = (-1.0, term[1:].strip()) if term.startswith("-") else (1.0, term)
            if tag not in data:
                raise SystemExit(f"--derive 里的 tag 不存在：{tag}（用 --list-tags 查）")
            steps, vals = data[tag]
            if base_steps is None:
                base_steps, acc = steps, sign * vals
            else:
                if len(steps) != len(base_steps) or not np.array_equal(steps, base_steps):
                    raise SystemExit(f"--derive 的 tag 迭代点不一致：{tag}（先各自 --tags 单独看）")
                acc = acc + sign * vals
        data[name] = (base_steps, acc)
        added.append(name)
    return added


# ----------------------------------------------------------------------------- 读盘
def find_event_files(run_dir: str) -> list[str]:
    """run 目录下的 tfevents 文件（只看顶层，忽略 exported_deploy/ 之类的子目录）。"""
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:  # 兜底：万一被放进子目录
        files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return files


def _read_scalars_raw(event_files: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict[str, tuple[list, list]] = {}
    # scalars=0 ⇒ 全部读入；其余类型只留 1 条，避免把 70 MB 事件里的图/直方图全塞进内存
    guidance = {"scalars": 0, "histograms": 1, "images": 1, "audio": 1, "tensors": 1}
    for path in event_files:
        try:
            acc = EventAccumulator(path, size_guidance=guidance, purge_orphaned_data=False)
        except TypeError:  # 老版本签名
            acc = EventAccumulator(path, size_guidance=guidance)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            steps, vals = out.setdefault(tag, ([], []))
            for ev in acc.Scalars(tag):
                steps.append(int(ev.step))
                vals.append(float(ev.value))
    return {t: (np.asarray(s, dtype=np.int64), np.asarray(v, dtype=np.float64)) for t, (s, v) in out.items()}


def load_run(run_dir: str, use_cache: bool = True, refresh: bool = False) -> dict:
    """读一个 run 的标量；返回 {tag: (steps, values)}（steps 已排序去重，重复取后者）。"""
    run_dir = os.path.abspath(run_dir)
    event_files = find_event_files(run_dir)
    if not event_files:
        raise SystemExit(f"没找到 events.out.tfevents.*：{run_dir}")

    meta = "|".join(f"{os.path.basename(p)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}" for p in event_files)
    cache_path = os.path.join(run_dir, ".summary_cache.npz")

    if use_cache and not refresh and os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["_meta"]) == meta:
                data = {k: (cached[k][:, 0].astype(np.int64), cached[k][:, 1]) for k in cached.files if k != "_meta"}
                print(f"[cache] 命中 {cache_path}（{len(data)} 个 tag）", file=sys.stderr)
                return data
            print("[cache] 事件文件已变化，重新解析", file=sys.stderr)
        except Exception as exc:  # 缓存坏了就当没有
            print(f"[cache] 忽略（{exc}）", file=sys.stderr)

    data = _read_scalars_raw(event_files)
    # 排序 + 同 step 去重（保留后写的那条）
    cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, (steps, vals) in data.items():
        order = np.argsort(steps, kind="stable")
        steps, vals = steps[order], vals[order]
        keep = np.ones(len(steps), dtype=bool)
        keep[:-1] = steps[1:] != steps[:-1]
        cleaned[tag] = (steps[keep], vals[keep])

    if use_cache:
        try:
            payload = {t: np.stack([s.astype(np.float64), v], axis=1) for t, (s, v) in cleaned.items()}
            payload["_meta"] = np.asarray(meta)
            np.savez_compressed(cache_path, **payload)
            print(f"[cache] 已写入 {cache_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[cache] 写入失败（忽略）：{exc}", file=sys.stderr)

    return cleaned


# ----------------------------------------------------------------------------- 选择 / 聚合
def select_tags(data: dict, patterns: list[str], explicit: list[str] | None = None) -> list[str]:
    """按用户给的正则（或默认清单）挑 tag；匹配不到的显式 tag 会报错提示。"""
    pats = explicit or patterns
    hits: list[str] = []
    for pat in pats:
        for tag in sorted(data):
            if re.search(pat, tag) and tag not in hits:
                hits.append(tag)
    if explicit:
        missing = [p for p in explicit if not any(re.search(p, t) for t in data)]
        if missing:
            print(f"[warn] 这些 pattern 没有匹配到任何 tag：{missing}", file=sys.stderr)
    return hits


def stage_bounds(num_steps_per_env: int, stage_steps: int, n_stages: int) -> list[int]:
    """阶段边界（迭代号，左闭右开）：[1042, 2083, 3125] 表示 s0<s1<s2<s3。"""
    return [math.ceil(k * stage_steps / num_steps_per_env) for k in range(1, n_stages)]


def stage_label(idx: int) -> str:
    return f"s{idx}"


def stage_rows(steps: np.ndarray, vals: np.ndarray, bounds: list[int]) -> list[dict]:
    """按阶段切分，返回每个阶段的均值/标准差/样本数（外加首尾两段）。"""
    rows = []
    edges = [0, *bounds]
    for i, lo in enumerate(edges):
        hi = bounds[i] if i < len(bounds) else math.inf
        mask = (steps >= lo) & (steps < hi)
        rows.append(
            {
                "label": stage_label(i),
                "lo": lo,
                "hi": None if math.isinf(hi) else int(hi),
                "n": int(mask.sum()),
                "mean": float(vals[mask].mean()) if mask.any() else float("nan"),
                "std": float(vals[mask].std()) if mask.any() else float("nan"),
            }
        )
    return rows


def tail_mean(steps: np.ndarray, vals: np.ndarray, last_n: int) -> float:
    if len(steps) == 0:
        return float("nan")
    lo = steps.max() - last_n + 1
    mask = steps >= lo
    return float(vals[mask].mean()) if mask.any() else float("nan")


def head_mean(steps: np.ndarray, vals: np.ndarray, first_n: int) -> float:
    if len(steps) == 0:
        return float("nan")
    mask = steps <= first_n
    return float(vals[mask].mean()) if mask.any() else float("nan")


def slope_per_1k(steps: np.ndarray, vals: np.ndarray, lo: int, hi: int = 10**9) -> float:
    """[lo, hi) 区间内的最小二乘斜率，单位 = 每 1000 迭代的值变化。"""
    mask = (steps >= lo) & (steps < hi)
    x, y = steps[mask].astype(np.float64), vals[mask]
    if len(x) < 2:
        return float("nan")
    a = np.polyfit(x, y, 1)[0]
    return float(a * 1000.0)


def value_at(steps: np.ndarray, vals: np.ndarray, step: int) -> float:
    """阶跃取值：迭代号 ≤ step 的最后一条（不存在则 nan）。"""
    idx = np.searchsorted(steps, step, side="right") - 1
    return float(vals[idx]) if idx >= 0 else float("nan")


# ----------------------------------------------------------------------------- 输出
def fmt(v: float, width: int = 12, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-".rjust(width)
    if v == 0:
        return "0".rjust(width)
    return f"{v:.{prec}g}".rjust(width)


def read_agent_yaml(run_dir: str) -> dict:
    """读 ``<run>/params/agent.yaml``（Isaac 写出来的 yaml 带 ``!!python/tuple``，
    这里给 SafeLoader 补一个构造器；实在读不动就退回正则抓 ``num_steps_per_env``）。"""
    import yaml

    for name in ("params/agent.yaml", "agent.yaml"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            text = open(path, "r", encoding="utf-8", errors="replace").read()
            try:
                loader = type("_TolerantLoader", (yaml.SafeLoader,), {})
                loader.add_constructor(
                    "tag:yaml.org,2002:python/tuple",
                    lambda l, node: tuple(l.construct_sequence(node, deep=True)),
                )
                return yaml.load(text, Loader=loader) or {}
            except Exception as exc:
                print(f"[warn] 解析 {path} 失败（{exc}），退回正则", file=sys.stderr)
                m = re.search(r"^num_steps_per_env:\s*(\d+)", text, re.MULTILINE)
                return {"num_steps_per_env": int(m.group(1))} if m else {}
    return {}


def print_tag_list(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    groups: dict[str, list[str]] = {}
    for tag in sorted(data):
        groups.setdefault(tag.split("/")[0], []).append(tag)
    for prefix in sorted(groups):
        print(f"\n## {prefix}/  ({len(groups[prefix])} tags)")
        for tag in groups[prefix]:
            steps, _ = data[tag]
            print(f"  {tag}   [{len(steps)} pts, {steps.min()}..{steps.max()}]")


def print_stage_table(title: str, data: dict, tags: list[str], bounds: list[int], last_n: int) -> list[str]:
    """阶段均值表（含首/末段），返回 Markdown 行。"""
    labels = [stage_label(i) for i in range(len(bounds) + 1)]
    header = f"| 指标 | {' | '.join(labels)} | 前{last_n} | 末{last_n} | 末段斜率/1k |"
    sep = "|" + "---|" * (len(labels) + 4)
    lines = [f"\n### {title}", "", header, sep]
    for tag in tags:
        steps, vals = data[tag]
        rows = stage_rows(steps, vals, bounds)
        cells = [fmt(r["mean"]) for r in rows]
        s3_lo = bounds[-1] if bounds else 0
        lines.append(
            f"| `{tag}` | {' | '.join(cells)} | {fmt(head_mean(steps, vals, last_n))} | "
            f"{fmt(tail_mean(steps, vals, last_n))} | {fmt(slope_per_1k(steps, vals, s3_lo))} |"
        )
    print("\n".join(lines))
    return lines


def print_grid_table(title: str, data: dict, tags: list[str], grid: list[int]) -> list[str]:
    header = "| 指标 | " + " | ".join(f"@iter={g}" for g in grid) + " |"
    sep = "|" + "---|" * (len(grid) + 1)
    lines = [f"\n### {title}", "", header, sep]
    for tag in tags:
        steps, vals = data[tag]
        cells = [fmt(value_at(steps, vals, g)) for g in grid]
        lines.append(f"| `{tag}` | {' | '.join(cells)} |")
    print("\n".join(lines))
    return lines


def print_compare_table(title: str, run_a: dict, run_b: dict, tags: list[str], grid: list[int], label_a: str, label_b: str) -> list[str]:
    lines = [f"\n### {title}", ""]
    for tag in tags:
        if tag not in run_a or tag not in run_b:
            continue
        sa, va = run_a[tag]
        sb, vb = run_b[tag]
        lines.append(f"\n`{tag}`")
        lines.append("")
        lines.append("| 迭代 | " + " | ".join(str(g) for g in grid) + " |")
        lines.append("|" + "---|" * (len(grid) + 1))
        lines.append(f"| A {label_a} | " + " | ".join(fmt(value_at(sa, va, g)) for g in grid) + " |")
        lines.append(f"| B {label_b} | " + " | ".join(fmt(value_at(sb, vb, g)) for g in grid) + " |")
        lines.append(
            "| Δ (A−B) | "
            + " | ".join(fmt(value_at(sa, va, g) - value_at(sb, vb, g)) for g in grid)
            + " |"
        )
    print("\n".join(lines))
    return lines


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="把 tensorboard run 压成关键指标表（支持两 run 对比）")
    p.add_argument("--run", required=True, help="run 目录（A）")
    p.add_argument("--baseline", default=None, help="对比用 run 目录（B）；给了就出对比表")
    p.add_argument("--tags", action="append", default=None, help="指标过滤正则（可多次）；不传用内置关键指标清单")
    p.add_argument(
        "--derive",
        action="append",
        default=None,
        help="派生指标：`名称=tag1+tag2`（tag 前加 `-` 表示相减），例如把两个高度/倾角终止项求和",
    )
    p.add_argument("--list-tags", action="store_true", help="只打印 tag 清单")
    p.add_argument("--grid", default=None, help="采样网格（逗号分隔的迭代号）；不传按 --stride 生成")
    p.add_argument("--stride", type=int, default=2500, help="采样网格间隔（默认 2500）")
    p.add_argument("--stage-steps", type=int, default=25000, help="每个课程阶段的环境步数（默认 25000）")
    p.add_argument("--num-steps-per-env", type=int, default=None, help="覆盖 agent.yaml 里的 num_steps_per_env")
    p.add_argument("--stages", type=int, default=4, help="课程阶段数（默认 4：s0~s3）")
    p.add_argument("--window", type=int, default=1000, help="首/末段的窗口长度（默认 1000 迭代）")
    p.add_argument("--out", default=None, help="把 Markdown 表写到这个文件（追加）")
    p.add_argument("--no-cache", action="store_true", help="不读写 .summary_cache.npz")
    p.add_argument("--refresh", action="store_true", help="忽略缓存重新解析事件文件")
    return p


def main() -> None:
    args = build_parser().parse_args()
    use_cache = not args.no_cache

    data = load_run(args.run, use_cache=use_cache, refresh=args.refresh)
    if args.list_tags:
        print(f"# {os.path.basename(os.path.abspath(args.run))} —— {len(data)} 个标量 tag")
        print_tag_list(data)
        return

    derived = apply_derives(data, args.derive)
    agent = read_agent_yaml(args.run)
    nspe = args.num_steps_per_env or int(agent.get("num_steps_per_env", 24))
    bounds = stage_bounds(nspe, args.stage_steps, args.stages)

    all_steps = np.concatenate([s for s, _ in data.values()]) if data else np.asarray([0])
    last_iter = int(all_steps.max())
    if args.grid:
        grid = [int(x) for x in args.grid.split(",") if x.strip()]
    else:
        grid = list(range(0, last_iter + 1, args.stride)) + [last_iter]
        grid = sorted(set(grid))

    selected = select_tags(data, DEFAULT_KEY_PATTERNS, args.tags)
    tags = derived + [t for t in selected if t not in derived]
    out_lines: list[str] = []
    out_lines.append(f"\n# summarize_run: {os.path.basename(os.path.abspath(args.run))}")
    out_lines.append("")
    out_lines.append(
        f"- 事件文件: {', '.join(os.path.basename(p) for p in find_event_files(args.run))}"
    )
    out_lines.append(f"- 迭代范围: 0 ~ {last_iter}（{len(data)} 个标量 tag，选中 {len(tags)} 个）")
    out_lines.append(
        f"- 课程阶段边界（num_steps_per_env={nspe}, stage_steps={args.stage_steps}）: "
        f"{bounds} ⇒ s0 <{bounds[0] if bounds else '-'}, ..."
    )
    out_lines += print_stage_table("1. 关键指标 × 课程阶段（阶段内均值）", data, tags, bounds, args.window)
    out_lines += print_grid_table("2. 关键指标 × 迭代（采样网格）", data, tags, grid)

    if args.baseline:
        base = load_run(args.baseline, use_cache=use_cache, refresh=args.refresh)
        apply_derives(base, args.derive)
        b_steps = np.concatenate([s for s, _ in base.values()]) if base else np.asarray([0])
        b_agent = read_agent_yaml(args.baseline)
        b_nspe = args.num_steps_per_env or int(b_agent.get("num_steps_per_env", nspe))
        b_bounds = stage_bounds(b_nspe, args.stage_steps, args.stages)
        b_tags = [t for t in tags if t in base]
        out_lines += print_stage_table(
            f"1b. baseline 关键指标 × 课程阶段（{os.path.basename(os.path.abspath(args.baseline))}）",
            base,
            b_tags,
            b_bounds,
            args.window,
        )
        common = [g for g in grid if g <= min(last_iter, int(b_steps.max()))]
        out_lines += print_compare_table(
            "3. 两 run 对比（同迭代点，步进取值）",
            data,
            base,
            tags,
            common,
            os.path.basename(os.path.abspath(args.run)),
            os.path.basename(os.path.abspath(args.baseline)),
        )

    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"\n[out] 已写入 {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
