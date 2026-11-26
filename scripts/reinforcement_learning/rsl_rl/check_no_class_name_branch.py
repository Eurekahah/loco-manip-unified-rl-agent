#!/usr/bin/env python
"""类名分支检查（交接单 §4 / §7 / D-09）。

文本扫描 ``as2w_piperx/**/*.py``，断言**没有** ``__class__.__name__`` 这种
"按类名分支"的写法 —— 新加子类会静默跳过设置，正是 M20 ``flat_env_wbc_cfg.py``
里 11 处 bug 的根源。

纯标准库、秒级，不需要启动 Isaac Sim：

    python scripts/reinforcement_learning/rsl_rl/check_no_class_name_branch.py

失败返回码 1，CI / 合并前必须通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: 禁止出现的文本。
FORBIDDEN_TOKEN = "__class__.__name__"


def main() -> int:
    parser = argparse.ArgumentParser(description="AS2W-PiperX 类名分支检查")
    parser.add_argument(
        "--path",
        default=None,
        help="as2w_piperx 目录（默认按当前脚本位置推断仓库内的标准路径）",
    )
    args = parser.parse_args()

    if args.path is not None:
        target_dir = Path(args.path).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3]
        target_dir = (
            repo_root
            / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/as2w_piperx"
        )

    if not target_dir.is_dir():
        print(f"[check_no_class_name_branch] 目标目录不存在：{target_dir}", file=sys.stderr)
        return 2

    files = sorted(target_dir.rglob("*.py"))
    problems: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FORBIDDEN_TOKEN in line:
                problems.append(f"{path}:{lineno}: {line.strip()}")

    print(f"[check_no_class_name_branch] 扫描 {len(files)} 个文件：{target_dir}")
    if problems:
        print(
            f"[check_no_class_name_branch] 发现按类名分支（{FORBIDDEN_TOKEN}），"
            "请改为显式字段 / 独立子类：",
            file=sys.stderr,
        )
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1

    print(f"[check_no_class_name_branch] OK：未发现 {FORBIDDEN_TOKEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())