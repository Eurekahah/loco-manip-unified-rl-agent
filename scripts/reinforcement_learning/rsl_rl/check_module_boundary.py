#!/usr/bin/env python
"""依赖边界检查（交接单 §7 / D-09）。

AST 扫描 ``as2w_piperx/**/*.py`` 的 import 语句，断言它们**从不**依赖：

* ``deeprobotics_m20``（任何层级）；
* ``velocity_env_cfg``（共享的 M20 基线配置）。

纯标准库、秒级，不需要启动 Isaac Sim：

    python scripts/reinforcement_learning/rsl_rl/check_module_boundary.py

失败返回码 1，CI / 合并前必须通过。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: 禁止出现在任何 import 目标里的子串。
FORBIDDEN_SUBSTRINGS = ("deeprobotics_m20", "velocity_env_cfg")


def _iter_import_targets(tree: ast.AST):
    """产出 (当前模块目标字符串, 行号)。覆盖 import / from ... import / importlib.import_module。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # level>0 的相对导入，module 可能为 None（from . import x）；用 current 的目标
            if node.module:
                yield node.module, node.lineno
        elif isinstance(node, ast.Call):
            func = node.func
            fname = None
            if isinstance(func, ast.Attribute):
                fname = func.attr
            elif isinstance(func, ast.Name):
                fname = func.id
            if fname == "import_module" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield arg.value, node.lineno


def scan_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    problems = []
    for target, lineno in _iter_import_targets(tree):
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in target:
                problems.append(f"{path}:{lineno}: 禁止 import {target!r}（命中 {bad!r}）")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="AS2W-PiperX 依赖边界检查")
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
        print(f"[check_module_boundary] 目标目录不存在：{target_dir}", file=sys.stderr)
        return 2

    files = sorted(target_dir.rglob("*.py"))
    problems: list[str] = []
    for path in files:
        problems.extend(scan_file(path))

    print(f"[check_module_boundary] 扫描 {len(files)} 个文件：{target_dir}")
    if problems:
        print("[check_module_boundary] 发现越界依赖：", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1

    print("[check_module_boundary] OK：未发现 deeprobotics_m20 / velocity_env_cfg 依赖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())