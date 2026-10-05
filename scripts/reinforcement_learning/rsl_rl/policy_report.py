# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""低层策略"体检报告"：一次滚动，出一整套诊断图 + 汇总表（取代 4 个老 test 脚本）。

为什么要重写
------------
仓库里原来的 ``gait_test.py`` / ``torque_test.py`` / ``tracking_test.py`` / ``test.py``
各写一份 Isaac 启动 + argparse + 环境构建 + 内联画图的样板，每份 20~28 KB，而且：

* 每次只能看**一个角度**（步态 / 力矩 / 跟踪三选一），要看全就得建 3~4 次 Isaac env；
* 每个脚本各有一套 ``--terrain`` / ``--cmd_*`` / ``--save_fig`` 语义，还互相不一致；
* 绘图逻辑与数据采集混在 ``main()`` 里，改一个指标要动一大段；
* 没有"同样口径下两个 checkpoint 对比"的能力（而 A/B 恰恰是这个项目最常见的动作）。

本脚本把"采集"与"画图"分开，**一次滚动**就输出 10 个角度（每个角度一张 PNG）+ 一份
Markdown 汇总 + 一份 NPZ 原始数据，并且支持 ``--compare`` 直接叠另一个 checkpoint。

十个角度
--------
1. ``fig01_tracking_timeseries`` 速度/角速度指令 vs 实际（逐档命令，含包络）
2. ``fig02_tracking_summary``   跟踪汇总：逐档误差柱状 + 指令-实际散点 + 终止构成
3. ``fig03_posture``            机身高度（相对足端）/ 俯仰 / 侧倾 / root_z + 误差直方图
4. ``fig04_gait_diagram``       四足触地时序（步态图）+ 占空比 + 步频(FFT) + 轮速
5. ``fig05_joints``             12 个腿关节位置/速度 + 4 个轮转速（网格图）
6. ``fig06_actuation``          力矩曲线 / 力矩占限幅比例 / 关节功率
7. ``fig07_symmetry``           左右镜像散点 + 镜像 RMS + 足端俯视图（"撇腿"角度）
8. ``fig08_arm_ee``             机械臂：EE 位置/姿态跟踪误差 + 臂关节位置速度（有臂才画）
9. ``fig09_terrain``            地形：高度扫描点云 / 足端离地间隙 / root 轨迹（有地形才画）
10. ``fig10_compare``           两个 checkpoint 的关键指标并排（``--compare`` 才画）

用法
----
::

    # 单个策略（平地上，"静止 + 0.5 + 1.0" 三档命令）
    python scripts/reinforcement_learning/rsl_rl/policy_report.py --headless \
        --task History-Adaptation-Deeprobotics-M20-play-v0 \
        --checkpoint logs/rsl_rl/history_adaptation/<run>/model_19999.pt \
        --commands "0,0,0;0.5,0,0;1.0,0,0" --num_envs 64 --steps 400 \
        --out-dir logs/smoke/report_cap12

    # A/B 对比（同一 env / 同一命令 / 同一 seed）
    ... --checkpoint A.pt --compare B.pt --out-dir logs/smoke/report_AB

    # 多地形（本机小环境数可以跑；多环境训练仍不行，见 DEF-031）
    ... --task Rough-Slopes-History-Adaptation-Deeprobotics-M20-play-v0 \
        --checkpoint <roughslopes run>/model_19999.pt --num_envs 2 --steps 600

约定
----
* 观测侧噪声关闭（确定性推理）；**push 事件默认保留**（与训练口径一致），
  要关掉加 ``--no-push``（那样只剩纯跟踪/步态特性）。
* 每档命令前都会强制一次 episode 边界 + warmup，避免上一档的瞬态污染。
* 数值都在 JSON/NPZ 里，PNG 只负责"看"；判好坏时看 ``report.md`` 的表。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# 画图依赖放在 AppLauncher 之前 import：这样才能支持 `--from-npz` 的"只画图"模式
# （云端只负责采集，图回本机画 —— 云端通常没有中文字体）。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 2026-10-05：**所有图里的文字改成纯英文**（用户需求 10）⇒ 默认 DejaVu Sans，
# 中文字体只当兜底（万一某处还留着中文，至少不画成方块）。
_CJK_FONTS = ["DejaVu Sans", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Micro Hei"]
matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# 云端一般没有中文字体（图里中文会变方块）⇒ 允许用环境变量塞一个字体文件进来，
# 例如把本机 C:\Windows\Fonts\simhei.ttf 传上去后：
#   POLICY_REPORT_FONT=/root/fonts/simhei.ttf python .../policy_report.py ...
_font_path = os.environ.get("POLICY_REPORT_FONT")
if _font_path and os.path.exists(_font_path):
    from matplotlib import font_manager  # noqa: PLC0415

    font_manager.fontManager.addfont(_font_path)
    _fam = font_manager.FontProperties(fname=_font_path).get_name()
    # 追加在 DejaVu Sans **之后**：只做兜底，图里英文仍走 DejaVu（字形稳定）
    matplotlib.rcParams["font.sans-serif"] = list(matplotlib.rcParams["font.sans-serif"]) + [_fam]
    print(f"[report] 兜底字体：{_fam} ← {_font_path}")

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# ─────────────────────────── 常量（本机型约定）───────────────────────────
LEGS = ("fl", "fr", "hl", "hr")
JOINT_TYPES = ("hipx", "hipy", "knee")
WHEEL_PREFIX = "Wheel"

#: 左右/对角镜像关系：符号取自 DEF-027（三重证据：MJCF 轴 / 关节限位 / 默认姿态）
MIRROR_PAIRS = (
    ("fl", "hr", {"hipx": -1.0, "hipy": -1.0, "knee": -1.0}),
    ("fr", "hl", {"hipx": -1.0, "hipy": -1.0, "knee": -1.0}),
    ("fl", "fr", {"hipx": -1.0, "hipy": 1.0, "knee": 1.0}),
    ("hl", "hr", {"hipx": -1.0, "hipy": 1.0, "knee": 1.0}),
)

#: 一档一条的固定命令（分地形 / 步态 / 汇总表用）。2026-10-05 扩充：
#: 原来只有 vx —— 而"轮腿在纯 vx 下轮子转就行、根本不需要迈步"，既看不出步态、
#: 也测不出侧向 / 偏航能力 ⇒ 现在把 vy / wz 也放进来（用户需求 5/9）。
DEFAULT_COMMANDS = ("0,0,0;0.3,0,0;0.8,0,0;1.5,0,0;0,0.4,0;0,-0.4,0;"
                    "0,0,0.6;0,0,-0.6;0.8,0.4,0.3")

#: 默认「整段滚动」schedule：(vx, vy, wz)；每段 --seg-s 秒（默认 0.8 s，13 段 ≈ 10.4 s）。
#: 一条连续轨迹里把三个速度维度都用上并来回切换 ⇒ fig01/02/05/06/07 不再是"只测了 vx"，
#: 顺带就能评价指令变换能力（用户需求 2/3/6）。
DEFAULT_SCHEDULE = (
    (0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.8, 0.0, 0.0), (1.5, 0.0, 0.0),
    (1.0, 0.5, 0.0), (1.0, -0.5, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, 0.0),
    (-0.6, 0.0, 0.0), (0.6, 0.0, 0.4), (0.6, 0.0, -0.4), (0.0, 0.0, 0.6), (0.0, 0.0, -0.6),
)
DEFAULT_SEG_S = 0.8

#: 姿态 schedule：(速度指令, 姿态关键字)；关键字在 main() 里按 env_cfg 的实际区间解成
#: (height, pitch, roll)。pitch 与 roll 分别单独出现 ⇒ fig03 可以把两者分两张子图画
#: （用户需求 4：不要挤在一张图里）。速度指令"少一点但成组出现"。
DEFAULT_POSTURE_SCHEDULE = (
    ((0.0, 0.0, 0.0), "h_hi"),
    ((0.6, 0.0, 0.0), "h_lo"),
    ((0.0, 0.0, 0.0), "pitch_hi"),
    ((0.6, 0.0, 0.0), "pitch_lo"),
    ((0.0, 0.0, 0.0), "roll_hi"),
    ((0.6, 0.0, 0.0), "roll_lo"),
    ((0.0, 0.0, 0.0), "h_mid"),
)

#: fig08 臂测试：一条轨迹里切几个末端目标（比原来"只切一次"能看出重复性/一致性）。
DEFAULT_ARM_TARGETS = 5

#: push 抗扰档位扫描：(力度倍数, 频率倍数)。1.0/1.0 = 训练口径
#: （±2 / ±1 / yaw ±0.52 m/s、间隔 5~10 s）。频率倍数 k ⇒ 间隔除以 k（推得更勤）；
#: 力度倍数 > 1 就是**外推**（训练分布之外），用来看泛化边界（用户需求 1）。
DEFAULT_PUSH_GRID = ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))


# ─────────────────────────────── 命令行 ─────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="低层策略体检报告（跟踪 / 步态 / 力矩 / 对称 / 地形，一次滚动全出）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--disable_fabric", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=64, help="环境数（地形任务建议 1~4）")
    p.add_argument(
        "--task",
        type=str,
        default="History-Adaptation-Deeprobotics-M20-play-v0",
        help="任务名；默认 -play-（完整难度、无课程漂移）",
    )
    p.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=400, help="每档命令记录的步数")
    p.add_argument("--warmup", type=int, default=100, help="每档命令开始前丢弃的步数")
    p.add_argument("--commands", type=str, default=DEFAULT_COMMANDS,
                   help='分号分隔的 "vx,vy,wz"（逐档固定滚动：步态 / 分地形 / 汇总表用）')
    p.add_argument("--schedule", choices=("full", "none"), default="full",
                   help="整段滚动 schedule：full = 一条 10 s+ 轨迹里把 vx/vy/wz（+机身姿态）"
                        "都切换一遍，fig01/02/03/05/06/07 用它；none = 不采集")
    p.add_argument("--seg-s", type=float, default=DEFAULT_SEG_S, help="schedule 每段秒数")
    p.add_argument("--gait-commands", type=str, default=None,
                   help="fig04 步态图专用命令档（默认：--commands 里所有 vy/wz 有非零的档 + 一档零速）")
    p.add_argument("--push-sweep", type=str, default=None,
                   help='push 抗扰扫描 "力度,频率;力度,频率"（如 "1,1;2,2;3,3"；1.0=训练口径）。'
                        "开启后额外出 fig13_push_robustness + 一张表（生还率 / 尖刺 / 恢复时间）")
    p.add_argument("--arm-targets", type=int, default=DEFAULT_ARM_TARGETS,
                   help="fig08 臂测试：一条轨迹里切几个末端目标（0 = 不切，只画原命令）")
    p.add_argument("--compare", type=str, default=None, help="第二个 checkpoint（A/B 对比）")
    p.add_argument("--label", type=str, default=None, help="A 的名称（默认取目录名）")
    p.add_argument("--label-b", type=str, default=None, help="B 的名称")
    p.add_argument("--env_id", type=int, default=0, help="画时序/步态图用哪个 env")
    p.add_argument("--out-dir", type=str, required=True, help="输出目录（PNG + report.md + data.npz）")
    p.add_argument("--no-push", action="store_true", default=False, help="关掉 push 事件（只看纯跟踪/步态）")
    p.add_argument(
        "--switch",
        type=float,
        default=0.0,
        help="（兼容旧写法）>0 时等价于 --seg-s 该值；更推荐直接用 --seg-s",
    )
    p.add_argument(
        "--terrain-grid",
        choices=("auto", "keep", "5x5"),
        default="auto",
        help="地形网格：auto = num_envs<=16 时缩成 5×5（生成快），否则保持训练网格；"
        "keep = 一定保持训练网格（分地形统计要用它，否则 5列 会丢掉部分地形）",
    )
    p.add_argument(
        "--no-height-scan",
        action="store_true",
        default=False,
        help="地形任务上不要临时加诊断用 height_scanner（加上才能画 fig09_terrain）",
    )
    p.add_argument("--dpi", type=int, default=140, help="PNG 分辨率")
    p.add_argument(
        "--from-npz",
        type=str,
        default=None,
        help="只画图：从 data.npz 读回数据渲染（**不启动 Isaac**）。用于『云端采集 + 本机画图』",
    )
    cli_args.add_rsl_rl_args(p)
    AppLauncher.add_app_launcher_args(p)
    return p


parser = build_parser()
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint and not args_cli.from_npz:
    parser.error("必须给 --checkpoint <run 目录>/model_<iter>.pt")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下在 Isaac 起来之后再 import。"""

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 同上（AppLauncher 之后那一份）；图已改成纯英文。
_CJK_FONTS = ["DejaVu Sans", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Micro Hei"]
matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerHis  # noqa: E402

import rl_training.tasks  # noqa: F401,E402
from rl_training.tasks.manager_based.locomotion.velocity.mdp.utils import (  # noqa: E402
    compute_base_height_rel_to_feet,
)


# ──────────────────────────── 小工具 ────────────────────────────
def parse_commands(spec: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x) for x in chunk.split(",")]
        if len(vals) != 3:
            raise ValueError(f"命令格式应为 'vx,vy,wz'，收到 {chunk!r}")
        out.append((vals[0], vals[1], vals[2]))
    if not out:
        raise ValueError("至少要给一档命令")
    return out


def _mean_std(x: torch.Tensor) -> tuple[float, float]:
    return float(x.mean()), float(x.std())


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), q))


def find_one(robot, kind: str, pattern: str) -> int:
    """按名字精确取一个关节/body 的索引。

    注意：`find_joints/find_bodies` 返回的是**按 articulation 顺序**排的结果，
    一次传多个名字不能按下标对应 —— 所以这里一条一条解析（历史踩坑，见 DEF-027/DEF-034）。
    """
    fn = robot.find_joints if kind == "joint" else robot.find_bodies
    ids, names = fn(pattern)
    if len(ids) != 1:
        raise RuntimeError(f"{kind} 名 {pattern!r} 匹配到 {len(ids)} 个（{names}），要求唯一")
    return int(ids[0])


@dataclass
class EpisodeData:
    """一次"命令档"的滚动结果（数组第一维都是 时间）。"""

    label: str
    command: tuple[float, float, float]
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cmd: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vel_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    yaw_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    root_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    root_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    height: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pitch: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(0))
    body_cmd: np.ndarray | None = None
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    torque: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    contact: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=bool))
    foot_xy_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2)))
    foot_z_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    base_vel_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    terrain_pts_w: np.ndarray | None = None
    term_counts: dict = field(default_factory=dict)
    n_done: int = 0
    #: 关节列名/索引/限幅/镜像误差由 harness 统一填（每个 episode 都一样）
    joint_names_all: list[str] = field(default_factory=list)
    col: dict[str, int] = field(default_factory=dict)
    torque_limit: np.ndarray | None = None
    mirror_rms: dict[str, list[float]] = field(default_factory=dict)
    #: 机械臂（有臂才有）
    ee_cmd_pos_b: np.ndarray | None = None
    ee_pos_b: np.ndarray | None = None
    ee_ori_err: np.ndarray | None = None
    #: 地形（有 height_scanner 才有）
    scan_min: np.ndarray | None = None
    scan_max: np.ndarray | None = None
    scan_mean: np.ndarray | None = None
    #: 逐地形的代表性高度图：``{地形名: (N,3) 世界系点}`` + 该 env 的 root 轨迹
    #: （fig09 用它画"每个地形一张稠密高度图 + 轨迹"，用户需求 8）
    terrain_maps: dict | None = None
    terrain_traj: dict | None = None
    #: 每个 env 落在哪种地形上 / 地形等级（多地形任务才有）
    env_terrain: list[str] | None = None
    env_terrain_level: np.ndarray | None = None
    #: 逐 env 的汇总（用于"分地形"统计，避免存 (T,N) 大数组）
    per_env: dict | None = None
    #: 指令切换模式：[(t0, t1, vel_cmd, body_cmd), ...]
    schedule: list | None = None

    @property
    def err_xy(self) -> np.ndarray:
        return np.linalg.norm(self.cmd[:, :2] - self.vel_b, axis=1)

    @property
    def err_yaw(self) -> np.ndarray:
        return np.abs(self.cmd[:, 2] - self.yaw_rate)


# ──────────────────────────── 采集器 ────────────────────────────
class Harness:
    """把"环境 + 策略 + 固定命令滚动 + 信号采集"包在一块，画图那边只管画。"""

    def __init__(self, env, unwrapped, label: str, env_id: int, contact_threshold: float = 1.0,
                 terrain_gen=None):
        self.env = env
        self.raw = unwrapped
        self.label = label
        self.env_id = env_id
        self.device = unwrapped.device
        self.n_envs = unwrapped.num_envs
        self.robot = unwrapped.scene["robot"]

        # 命令项：关掉重采样（每步手写命令）
        self.term = unwrapped.command_manager.get_term("base_velocity")
        self.term._resample_command = lambda env_ids: None  # noqa: ARG005
        self.dt = unwrapped.step_dt

        # ── 关节选择：12 腿 + 4 轮 +（有的话）机械臂/夹爪 ──
        self.joint_names: list[str] = []
        sel: list[int] = []
        for leg in LEGS:
            for jt in JOINT_TYPES:
                name = f"{leg}_{jt}_joint"
                sel.append(find_one(self.robot, "joint", name))
                self.joint_names.append(name)
        for leg in LEGS:
            name = f"{leg}_wheel_joint"
            sel.append(find_one(self.robot, "joint", name))
            self.joint_names.append(name)
        self.leg_slice = slice(0, 12)
        self.wheel_slice = slice(12, 16)
        self.joint_sel = torch.tensor(sel, device=self.device, dtype=torch.long)
        self.has_arm = False
        try:
            arm_ids, arm_names = self.robot.find_joints(["arm_joint[1-6]", "gripper_joint[12]"])
            if len(arm_ids) > 0:
                self.has_arm = True
                self.joint_names += list(arm_names)
                self.arm_slice = slice(len(sel), len(sel) + len(arm_ids))
                self.joint_sel = torch.cat(
                    [self.joint_sel, torch.tensor(arm_ids, device=self.device, dtype=torch.long)]
                )
        except Exception:  # noqa: BLE001 - 有些资产没有臂，直接跳过
            pass
        self.n_joints = self.joint_sel.numel()

        # ── 力矩限幅：USD 里常常是 0（本资产的 joint_effort_limits 全 0），
        #    依次退到 joint_effort_limits_sim → actuator cfg 的 effort_limit。
        self.torque_limit = _resolve_effort_limits(self.robot, self.joint_sel)
        if self.torque_limit is None:
            print("[report] 力矩限幅元数据不可用（IsaacLab 对本资产给的是 1e9 占位值）"
                  "⇒ 该角度改用『峰值因子 |tau|max/RMS』呈现")
        else:
            lim = self.torque_limit
            leg, wheel = lim[0:12], lim[12:16]
            arm = lim[16:22] if lim.size >= 22 else np.array([np.nan])
            print(f"[report] 力矩限幅：腿 {np.nanmin(leg):.1f}~{np.nanmax(leg):.1f} N·m / "
                  f"轮 {np.nanmin(wheel):.1f} / 臂 {np.nanmax(arm):.1f}")

        # ── 足端（轮）body ──
        self.foot_ids: list[int] = []
        self.foot_names: list[str] = []
        for leg in LEGS:
            for pat in (f"{leg}_wheel", f"^{leg}_wheel", f"{leg}.*wheel"):
                ids, names = self.robot.find_bodies(pat)
                if len(ids) == 1:
                    self.foot_ids.append(int(ids[0]))
                    self.foot_names.append(names[0])
                    break
            else:
                raise RuntimeError(f"找不到轮 body：{leg}_wheel")
        self.foot_sel = torch.tensor(self.foot_ids, device=self.device, dtype=torch.long)

        # ── 接触传感器（有的任务才有）＋ 高度扫描（多地形才有） ──
        self.contact_ids: list[int] | None = None
        try:
            sensor = unwrapped.scene["contact_forces"]
            ids, _ = sensor.find_bodies(".*_wheel")
            if len(ids) == 4:
                self.contact_ids = list(ids)
                self.contact_threshold = contact_threshold
        except Exception:  # noqa: BLE001
            pass
        self.height_sensor = None
        try:
            self.height_sensor = unwrapped.scene.sensors.get("height_scanner")
        except Exception:  # noqa: BLE001
            pass

        # ── 机身姿态命令项（有的任务才有）＋ EE 命令项 ──
        self.body_cmd = None
        try:
            self.body_cmd = unwrapped.command_manager.get_term("body_pose")
        except Exception:  # noqa: BLE001
            pass
        self.ee_cmd = None
        self.ee_body_idx = None
        try:
            self.ee_cmd = unwrapped.command_manager.get_term("ee_pose")
            # 注意：`gripper_base` 可能匹配到多个 body（find_one 会直接抛）⇒ 取第一个即可
            ids, names = self.robot.find_bodies("gripper_base")
            if len(ids) == 0:
                ids, names = self.robot.find_bodies(".*gripper.*")
            self.ee_body_idx = int(ids[0])
            print(f"[report] 末端 body = {names[0]} (#{self.ee_body_idx}) ⇒ fig08 可用")
        except Exception as exc:  # noqa: BLE001
            print(f"[report] 没有 EE 命令 / 末端 body（跳过 fig08）：{exc}")
            self.ee_cmd = None

        # 高度/足端 cfgs（与 body_pose 奖励同一口径）
        from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

        self.asset_cfg = SceneEntityCfg("robot")
        self.feet_cfg = SceneEntityCfg("robot", body_names=".*wheel")
        self.feet_cfg.resolve(self.raw.scene)

        # ── push 事件：抗扰扫描要按倍数改它的**幅度**与**间隔** ──────────────
        # `EventManager` 每次触发都**现读** `term_cfg.params` / `interval_range_s`
        # （isaaclab/managers/event_manager.py:217-231）⇒ 运行时改 cfg 立刻生效，
        # 不用重建环境。先把训练口径的基准值记下来，后面按倍数缩放。
        self.push_term = None
        self.push_base: dict | None = None
        em = getattr(self.raw, "event_manager", None)
        try:
            # ⚠️ `EventManager.active_terms` 是 ``{模式: [term 名]}`` 的 **dict** ——
            # 直接 `name in active_terms` 比的是模式名，永远 False（2026-10-05 踩到）。
            _terms = getattr(em, "active_terms", {}) if em is not None else {}
            if isinstance(_terms, dict):
                _names = {n for v in _terms.values() for n in v}
            else:
                _names = set(_terms)
            if "randomize_push_robot" in _names:
                cfg = em.get_term_cfg("randomize_push_robot")
                vr = dict(cfg.params.get("velocity_range", {}) or {})
                # 训练口径一般只给 x/y/yaw；z/roll/pitch 缺省即 0
                self.push_term = cfg
                self.push_base = {
                    "velocity_range": {k: (float(v[0]), float(v[1])) for k, v in vr.items()},
                    "interval_range_s": tuple(float(x) for x in cfg.interval_range_s),
                }
                print(f"[report] push 事件：interval={self.push_base['interval_range_s']} "
                      f"velocity_range={self.push_base['velocity_range']}")
            else:
                print("[report] 该任务没有 randomize_push_robot ⇒ --push-sweep 会自动跳过")
        except Exception as exc:  # noqa: BLE001
            print(f"[report] 读 push 事件配置失败（抗扰扫描会跳过）：{exc}")

        # ── 地形：每个 env 落在哪种地形上 ──────────────────────────────────
        # IsaacLab 把子地形**按列**分配：sub_indices[c] = min(i : c/num_cols+0.001 < cumsum(prop))
        # （见 isaaclab/terrains/terrain_generator.py:241），所以知道 terrain_types[env]
        # 就能反查名字。这里把映射和每 env 的名字都算好（拿不到就 None，后面跳过相关图）。
        self.env_terrain: list[str] | None = None
        self.env_terrain_level: np.ndarray | None = None
        terr = getattr(self.raw.scene, "terrain", None)
        # 地形生成器在运行时 importer 上常常是 None（生成完就丢了）⇒ 优先用调用方从 env_cfg 传进来的那份
        gen = terrain_gen if terrain_gen is not None else (
            getattr(terr, "terrain_generator", None) if terr is not None else None
        )
        if gen is not None and getattr(gen, "sub_terrains", None):
            try:
                props = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
                props = props / props.sum()
                cum = np.cumsum(props)
                names = list(gen.sub_terrains.keys())
                col2name = [
                    names[int(np.min(np.where(c / gen.num_cols + 0.001 < cum)[0]))]
                    for c in range(gen.num_cols)
                ]
                # 每 env 的列号：优先用 importer 的 terrain_types；没有就按 IsaacLab 的
                # `env_id % num_patches`（`_get_env_origins` 的分配方式）反推
                types = None
                if terr is not None and getattr(terr, "terrain_types", None) is not None:
                    types = np.asarray(terr.terrain_types.detach().cpu().numpy(), dtype=int).ravel()
                if types is None or types.size != self.n_envs:
                    n_patch = int(getattr(gen, "num_rows", 1)) * int(gen.num_cols)
                    types = (np.arange(self.n_envs) % n_patch) % int(gen.num_cols)
                self.env_terrain = [col2name[min(int(t), len(col2name) - 1)] for t in types]
                levels = getattr(terr, "terrain_levels", None)
                if levels is not None:
                    self.env_terrain_level = np.asarray(levels.detach().cpu().numpy(), dtype=int).ravel()
                uniq = {}
                for n in self.env_terrain:
                    uniq[n] = uniq.get(n, 0) + 1
                print(f"[report] 地形分布（按 env）：{uniq}")
            except Exception as exc:  # noqa: BLE001
                print(f"[report] 地形名映射失败（跳过分地形统计）：{exc}")

    # ---------------- 与环境交互 ----------------
    def _write_cmd(self, cmd: tuple[float, float, float]) -> None:
        self.term.vel_command_b[:] = torch.tensor(cmd, device=self.device, dtype=torch.float32)

    def _write_body_cmd(self, body) -> None:
        """写机身姿态指令。不同实现的字段名不同（`pose_command_b` 或 `command`），两个都试。"""
        if body is None or self.body_cmd is None:
            return
        t = torch.tensor(body, device=self.device, dtype=torch.float32).expand(self.n_envs, -1)
        for attr in ("pose_command_b", "command"):
            buf = getattr(self.body_cmd, attr, None)
            if isinstance(buf, torch.Tensor) and buf.shape[-1] >= 3:
                buf[:, :3] = t

    def _read_body_cmd(self) -> np.ndarray:
        """读回"策略实际看到的"机身姿态指令（优先 pose_command_b）。"""
        if self.body_cmd is None:
            return np.zeros(3)
        for attr in ("pose_command_b", "command"):
            buf = getattr(self.body_cmd, attr, None)
            if isinstance(buf, torch.Tensor) and buf.shape[-1] >= 3:
                return buf[self.env_id, :3].detach().cpu().numpy().copy()
        return np.zeros(3)

    def disable_body_resample(self) -> None:
        """切换测试里机身姿态指令由脚本手写 ⇒ 关掉这个命令项自己的重采样。"""
        if self.body_cmd is not None:
            self.body_cmd._resample_command = lambda env_ids: None  # noqa: ARG005

    def set_push(self, mag_scale: float = 1.0, freq_scale: float = 1.0) -> bool:
        """把 push 事件的**幅度** ×``mag_scale``、**频率** ×``freq_scale``（间隔 ÷ 后者）。

        相比训练口径（1.0/1.0）：频率 >1 = 推得更勤；幅度 >1 = 外推（训练分布之外），
        用来看模型在"更凶的扰动"下的泛化/退化（用户 2026-10-05 需求 1）。
        返回 False 表示这个任务没有 push 事件（或读不到基准值）。
        """
        if self.push_term is None or self.push_base is None:
            return False
        iv0, iv1 = self.push_base["interval_range_s"]
        self.push_term.interval_range_s = (iv0 / freq_scale, iv1 / freq_scale)
        self.push_term.params["velocity_range"] = {
            k: (v[0] * mag_scale, v[1] * mag_scale)
            for k, v in self.push_base["velocity_range"].items()
        }
        return True

    def _step(self, cmd: tuple[float, float, float], obs):
        self._write_cmd(cmd)
        with torch.inference_mode():
            obs, _, _, _ = self.env.step(self.policy(obs))
        return obs

    def _force_reset(self, cmd: tuple[float, float, float], obs):
        """把 episode 顶到上限 ⇒ 下一步所有 env 统一 reset（对齐回合边界）。"""
        self.raw.episode_length_buf[:] = self.raw.max_episode_length
        return self._step(cmd, obs)

    # ---------------- 采集 ----------------
    def collect(
        self,
        commands: list[tuple[float, float, float]],
        steps: int,
        warmup: int,
        policy,
        term_names: list[str],
        want_terrain: bool = True,
    ) -> list[EpisodeData]:
        self.policy = policy
        obs = self.env.get_observations()
        out: list[EpisodeData] = []
        for ci, cmd in enumerate(commands):
            t_wall = time.perf_counter()
            obs = self._force_reset(cmd, obs)
            for _ in range(warmup):
                obs = self._step(cmd, obs)
            obs = self._force_reset(cmd, obs)

            rec = {k: [] for k in (
                "t", "cmd", "vel_b", "yaw", "xy", "z", "h", "pitch", "roll",
                "qpos", "qvel", "tau", "contact", "foot_xy", "foot_z",
            )}
            rec["body_cmd"] = [] if self.body_cmd is not None else None
            rec["ee"] = [] if self.ee_cmd is not None else None
            rec["scan"] = [] if self.height_sensor is not None else None
            # 逐地形的"代表 env"（每种地形取第一个落上去的 env）⇒ fig09 每个地形
            # 画一张稠密高度图 + 该 env 的真实轨迹（用户 2026-10-05 需求 8）。
            warn_maps = bool(want_terrain and self.height_sensor is not None
                             and self.env_terrain is not None)
            rep_env: dict[str, int] = {}
            if warn_maps:
                for _ri, _rname in enumerate(self.env_terrain):
                    rep_env.setdefault(_rname, _ri)
            rep_traj: dict[str, list] = {nm: [] for nm in rep_env}
            term_counts = {n: 0 for n in term_names}
            n_done = 0
            # 逐 env 累加（"分地形"统计用；不存 (T,N) 大数组）
            _N, _dev = self.n_envs, self.device
            acc = {k: torch.zeros(_N, device=_dev) for k in
                   ("xy", "xy2", "yaw", "yaw2", "h", "h2", "z", "z2", "pitch2", "roll2", "tau", "done")}
            c_cnt = torch.zeros(_N, 4, device=_dev)
            cmd_t = torch.tensor(cmd, device=_dev, dtype=torch.float32).expand(_N, 3)
            for _ in range(steps):
                # 误差用"动作施加前"的状态算（与 eval_fixed_command 同口径）
                v_b = self.robot.data.root_lin_vel_b[:, :2].clone()
                w_z = self.robot.data.root_ang_vel_b[:, 2].clone()
                xy = self.robot.data.root_pos_w[:, :2].clone()
                z = self.robot.data.root_pos_w[:, 2].clone()
                qpos = self.robot.data.joint_pos[:, self.joint_sel].clone()
                qvel = self.robot.data.joint_vel[:, self.joint_sel].clone()
                tau = self.robot.data.applied_torque[:, self.joint_sel].clone()
                g_b = self.robot.data.projected_gravity_b
                pitch = torch.asin(torch.clamp(-g_b[:, 0], -1.0, 1.0))
                roll = torch.atan2(-g_b[:, 1], -g_b[:, 2])
                h = compute_base_height_rel_to_feet(self.raw, self.asset_cfg, self.feet_cfg)
                foot_xy_b, _ = _foot_xy_body(self.robot, self.foot_sel)
                foot_z = self.robot.data.body_pos_w[:, self.foot_sel, 2].clone()
                contact = self._contact_mask()

                self._write_cmd(cmd)
                with torch.inference_mode():
                    obs, _, dones, _ = self.env.step(self.policy(obs))

                i = self.env_id
                rec["t"].append(len(rec["t"]) * self.dt)
                rec["cmd"].append(np.array(self.term.vel_command_b[i].tolist(), dtype=float)[:3])
                rec["vel_b"].append(v_b[i].cpu().numpy())
                rec["yaw"].append(float(w_z[i]))
                rec["xy"].append(xy[i].cpu().numpy())
                rec["z"].append(float(z[i]))
                rec["h"].append(float(h[i]))
                rec["pitch"].append(float(pitch[i]))
                rec["roll"].append(float(roll[i]))
                rec["qpos"].append(qpos[i].cpu().numpy())
                rec["qvel"].append(qvel[i].cpu().numpy())
                rec["tau"].append(tau[i].cpu().numpy())
                rec["contact"].append(contact[i].cpu().numpy())
                rec["foot_xy"].append(foot_xy_b[i].cpu().numpy())
                rec["foot_z"].append(foot_z[i].cpu().numpy())
                if rec["body_cmd"] is not None:
                    rec["body_cmd"].append(self._read_body_cmd())
                if rec["ee"] is not None:
                    rec["ee"].append(self._ee_row(i))
                if rec["scan"] is not None:
                    hits = self.height_sensor.data.ray_hits_w[i]
                    z_scan = hits[:, 2]
                    rec["scan"].append(
                        np.array([float(z_scan.min()), float(z_scan.max()), float(z_scan.mean())])
                    )
                if rep_traj:
                    for _rname, _rid in rep_env.items():
                        rep_traj[_rname].append(
                            self.robot.data.root_pos_w[_rid, :2].detach().cpu().numpy().copy()
                        )

                done = dones.bool()
                if bool(done.any()):
                    for name in term_names:
                        term_counts[name] += int(self.raw.termination_manager.get_term(name)[done].sum())
                    n_done += int(done.sum())
                # ---- 逐 env 累加 ----
                # 注意：某些信号的长度可能和 num_envs 不一致（地形任务上见过 root_pos_w 短一截），
                # 所以每个信号都按"自己的长度"截断累加，缺的部分保持 0（NaN 不做数）以免整份报告挂掉。
                def _acc(key: str, t: torch.Tensor | None) -> None:
                    if t is None:
                        return
                    m = min(acc[key].numel(), t.shape[0])
                    acc[key][:m] += t[:m]
                    if f"{key}2" in acc:
                        acc[f"{key}2"][:m] += t[:m] ** 2

                n_ok = min(cmd_t.shape[0], v_b.shape[0], h.shape[0])
                _acc("xy", torch.norm(cmd_t[:n_ok, :2] - v_b[:n_ok], dim=-1))
                _acc("yaw", torch.abs(cmd_t[:n_ok, 2] - w_z[:n_ok]))
                _acc("h", h)
                _acc("z", z)
                m2 = min(acc["pitch2"].numel(), pitch.shape[0])
                acc["pitch2"][:m2] += pitch[:m2] ** 2
                m3 = min(acc["roll2"].numel(), roll.shape[0])
                acc["roll2"][:m3] += roll[:m3] ** 2
                t_rms = torch.sqrt((tau ** 2).mean(dim=-1))
                m4 = min(acc["tau"].numel(), t_rms.shape[0])
                acc["tau"][:m4] = torch.maximum(acc["tau"][:m4], t_rms[:m4])
                m5 = min(acc["done"].numel(), done.shape[0])
                acc["done"][:m5] += done[:m5].float()
                if contact.shape[0] >= c_cnt.shape[0]:
                    c_cnt += contact[: c_cnt.shape[0]].float()

            data = EpisodeData(label=self.label, command=cmd)
            data.t = np.array(rec["t"])
            data.cmd = np.array(rec["cmd"])
            data.vel_b = np.array(rec["vel_b"])
            data.yaw_rate = np.array(rec["yaw"])
            data.root_xy = np.array(rec["xy"])
            data.root_z = np.array(rec["z"])
            data.height = np.array(rec["h"])
            data.pitch = np.array(rec["pitch"])
            data.roll = np.array(rec["roll"])
            data.joint_pos = np.array(rec["qpos"])
            data.joint_vel = np.array(rec["qvel"])
            data.torque = np.array(rec["tau"])
            data.contact = np.array(rec["contact"], dtype=bool)
            data.foot_xy_b = np.array(rec["foot_xy"])
            data.foot_z_w = np.array(rec["foot_z"])
            if rec["body_cmd"] is not None:
                data.body_cmd = np.array(rec["body_cmd"])
            if rec["ee"] is not None:
                ee = np.array(rec["ee"])
                data.ee_cmd_pos_b, data.ee_pos_b, data.ee_ori_err = ee[:, :3], ee[:, 3:6], ee[:, 6]
            if rec["scan"] is not None:
                sc = np.array(rec["scan"])
                data.scan_min, data.scan_max, data.scan_mean = sc[:, 0], sc[:, 1], sc[:, 2]
            if rep_traj:
                hits_all = self.height_sensor.data.ray_hits_w
                data.terrain_maps = {
                    nm: np.asarray(hits_all[rid].detach().cpu().numpy(), dtype=float).reshape(-1, 3)
                    for nm, rid in rep_env.items()
                }
                data.terrain_traj = {nm: np.asarray(v, dtype=float) for nm, v in rep_traj.items()}
            data.term_counts = term_counts
            data.n_done = n_done
            data.joint_names_all = list(self.joint_names)
            data.col = {n: i for i, n in enumerate(self.joint_names)}
            data.torque_limit = self.torque_limit
            data.mirror_rms = mirror_rms(data.joint_pos, self.joint_names)
            _n = float(max(steps, 1))
            data.per_env = {
                "err_xy": (acc["xy"] / _n).cpu().numpy(),
                "err_yaw": (acc["yaw"] / _n).cpu().numpy(),
                "height_mean": (acc["h"] / _n).cpu().numpy(),
                "height_std": torch.sqrt(
                    torch.clamp(acc["h2"] / _n - (acc["h"] / _n) ** 2, min=0)
                ).cpu().numpy(),
                "root_z_mean": (acc["z"] / _n).cpu().numpy(),
                "pitch_std_deg": np.degrees(
                    torch.sqrt(torch.clamp(acc["pitch2"] / _n, min=0)).cpu().numpy()
                ),
                "roll_std_deg": np.degrees(
                    torch.sqrt(torch.clamp(acc["roll2"] / _n, min=0)).cpu().numpy()
                ),
                "tau_rms_max": acc["tau"].cpu().numpy(),
                "duty": (c_cnt / _n).cpu().numpy(),
                "done": acc["done"].cpu().numpy(),
            }
            data.env_terrain = self.env_terrain
            data.env_terrain_level = self.env_terrain_level
            if want_terrain and self.height_sensor is not None:
                hits = self.height_sensor.data.ray_hits_w[self.env_id].detach().cpu().numpy()
                data.terrain_pts_w = np.asarray(hits, dtype=float).reshape(-1, 3)
            out.append(data)
            print(f"[report]   档 {ci + 1}/{len(commands)} cmd={cmd} 完成"
                  f"（{steps} 步 / {time.perf_counter() - t_wall:.1f} s"
                  f" = {steps / max(time.perf_counter() - t_wall, 1e-6):.1f} 步/秒）")
        return out

    # ---------------- 指令切换（考察指令间的变换能力）----------------
    def collect_schedule(self, segments, policy, term_names):
        """在一次连续滚动里按 ``segments`` 切换「速度 + 机身姿态」指令。

        ``segments = [(秒数, (vx,vy,wz), (height,pitch,roll) | None), ...]``；
        返回单个 :class:`EpisodeData`（``.schedule`` 记着每段的 ``(t0, t1, vel, body)``），
        用来算"切换后的稳定时间 / 超调 / 稳态误差"。
        """
        self.policy = policy
        obs = self.env.get_observations()
        rec = {k: [] for k in ("t", "cmd", "vel_b", "yaw", "h", "pitch", "roll", "body_cmd")}
        for k in ("xy", "z", "qpos", "qvel", "tau", "contact", "foot_xy", "foot_z"):
            rec[k] = []
        # 末端执行器（有臂才有）——fig08 的臂测试就是走这条 schedule 路径
        rec["ee"] = [] if self.ee_cmd is not None else None
        schedule: list = []
        term_counts = {n: 0 for n in term_names}
        n_done = 0
        t = 0.0
        obs = self._force_reset(segments[0][1], obs)
        for dur_s, vel, body in segments:
            n_steps = int(round(dur_s / self.dt))
            t0 = t
            t_wall = time.perf_counter()
            for _ in range(n_steps):
                v_b = self.robot.data.root_lin_vel_b[:, :2].clone()
                w_z = self.robot.data.root_ang_vel_b[:, 2].clone()
                g_b = self.robot.data.projected_gravity_b
                h = compute_base_height_rel_to_feet(self.raw, self.asset_cfg, self.feet_cfg)
                xy = self.robot.data.root_pos_w[:, :2].clone()
                z = self.robot.data.root_pos_w[:, 2].clone()
                qpos = self.robot.data.joint_pos[:, self.joint_sel].clone()
                qvel = self.robot.data.joint_vel[:, self.joint_sel].clone()
                tau = self.robot.data.applied_torque[:, self.joint_sel].clone()
                foot_xy, _ = _foot_xy_body(self.robot, self.foot_sel)
                foot_z = self.robot.data.body_pos_w[:, self.foot_sel, 2].clone()
                contact = self._contact_mask()
                self._write_cmd(vel)
                self._write_body_cmd(body)
                bcmd = self._read_body_cmd()
                with torch.inference_mode():
                    obs, _, dones, _ = self.env.step(self.policy(obs))
                i = self.env_id
                rec["t"].append(t); t += self.dt
                rec["cmd"].append(np.asarray(vel, dtype=float))
                rec["vel_b"].append(v_b[i].cpu().numpy())
                rec["yaw"].append(float(w_z[i]))
                rec["h"].append(float(h[i]))
                rec["pitch"].append(float(torch.asin(torch.clamp(-g_b[i, 0], -1.0, 1.0))))
                rec["roll"].append(float(torch.atan2(-g_b[i, 1], -g_b[i, 2])))
                rec["body_cmd"].append(bcmd)
                rec["xy"].append(xy[i].cpu().numpy())
                rec["z"].append(float(z[i]))
                rec["qpos"].append(qpos[i].cpu().numpy())
                rec["qvel"].append(qvel[i].cpu().numpy())
                rec["tau"].append(tau[i].cpu().numpy())
                rec["contact"].append(contact[i].cpu().numpy())
                rec["foot_xy"].append(foot_xy[i].cpu().numpy())
                rec["foot_z"].append(foot_z[i].cpu().numpy())
                if rec["ee"] is not None:
                    rec["ee"].append(self._ee_row(i))
                done = dones.bool()
                if bool(done.any()):
                    n_done += int(done.sum())
                    for name in term_names:
                        term_counts[name] += int(
                            self.raw.termination_manager.get_term(name)[done].sum()
                        )
            schedule.append((t0, t, tuple(vel), None if body is None else tuple(body)))
            print(f"[report]   seg {len(schedule)}/{len(segments)} vel={tuple(vel)}"
                  f" body={None if body is None else tuple(round(b, 3) for b in body)}"
                  f"（{n_steps} 步 / {time.perf_counter() - t_wall:.1f} s）")
        data = EpisodeData(label=self.label, command=(0.0, 0.0, 0.0))
        data.t = np.array(rec["t"]); data.cmd = np.array(rec["cmd"])
        data.vel_b = np.array(rec["vel_b"]); data.yaw_rate = np.array(rec["yaw"])
        data.height = np.array(rec["h"]); data.pitch = np.array(rec["pitch"])
        data.roll = np.array(rec["roll"]); data.body_cmd = np.array(rec["body_cmd"])
        data.root_xy = np.array(rec["xy"]); data.root_z = np.array(rec["z"])
        data.joint_pos = np.array(rec["qpos"]); data.joint_vel = np.array(rec["qvel"])
        data.torque = np.array(rec["tau"]); data.contact = np.array(rec["contact"], dtype=bool)
        data.foot_xy_b = np.array(rec["foot_xy"]); data.foot_z_w = np.array(rec["foot_z"])
        data.schedule = schedule
        data.term_counts = term_counts
        data.n_done = n_done
        if rec["ee"] is not None:
            ee = np.array(rec["ee"])
            data.ee_cmd_pos_b, data.ee_pos_b, data.ee_ori_err = ee[:, :3], ee[:, 3:6], ee[:, 6]
        data.joint_names_all = list(self.joint_names)
        data.col = {n: i for i, n in enumerate(self.joint_names)}
        data.torque_limit = self.torque_limit
        data.mirror_rms = mirror_rms(data.joint_pos, self.joint_names)
        data.per_env = None
        data.env_terrain = self.env_terrain
        data.env_terrain_level = self.env_terrain_level
        return data

    def _contact_mask(self) -> torch.Tensor:
        if self.contact_ids is None:
            return torch.zeros(self.n_envs, 4, dtype=torch.bool, device=self.device)
        f = self.raw.scene["contact_forces"].data.net_forces_w[:, self.contact_ids, :]
        return f.norm(dim=-1) > self.contact_threshold

    def _ee_row(self, i: int) -> np.ndarray:
        """[cmd_ee_pos_b(3), actual_ee_pos_b(3), ori_err(rad)]，仅用于画臂的跟踪。"""
        import isaaclab.utils.math as math_utils  # noqa: PLC0415

        cmd = self.ee_cmd.pose_command_b[i]
        pos_w = self.robot.data.body_pos_w[i, self.ee_body_idx]
        quat_w = self.robot.data.body_quat_w[i, self.ee_body_idx]
        root_pos = self.robot.data.root_pos_w[i]
        root_quat = self.robot.data.root_quat_w[i]
        pos_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), pos_w - root_pos)
        quat_b = math_utils.quat_mul(math_utils.quat_conjugate(root_quat), quat_w)
        # 两个单位四元数之间的夹角（取绝对值 ⇒ 忽略 q 与 -q 的等价性）
        dot = torch.clamp(torch.abs(torch.dot(quat_b, cmd[3:10])), 0.0, 1.0)
        ori_err = 2.0 * torch.acos(dot)
        return np.concatenate([
            cmd[:3].cpu().numpy(), pos_b.cpu().numpy(), np.array([float(ori_err)], dtype=float)
        ])


def _foot_xy_body(robot, foot_sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """4 个轮在 root 系下的 xy（俯视图用；同时把 z 也返回，便于画离地间隙）。"""
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    pos_w = robot.data.body_pos_w[:, foot_sel, :]
    # 把 root 四元数广播到"每个足端"这一维（quat_apply 要求形状对齐到最后一维 4/3）
    quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, pos_w.shape[1], -1)
    quat_inv = math_utils.quat_conjugate(quat)
    pos_b = math_utils.quat_apply(quat_inv, pos_w - robot.data.root_pos_w.unsqueeze(1))
    return pos_b[:, :, :2], pos_b[:, :, 2]


def _resolve_effort_limits(robot, joint_sel: torch.Tensor) -> np.ndarray | None:
    """取每个关节的力矩上限（N·m）；取不到返回 None。

    注意坑：`robot.data.joint_effort_limits` 在本资产上全是 **1e9**（IsaacLab 的
    "无限制"占位值，不是真实限幅），直接拿它算占用率会永远得到 0.00。
    所以这里**优先**用 actuator 配置里的 `effort_limit`（`assets/deeprobotics.py` 里
    24/36/76.4/21.6/100/10 那些真实值），只有它拿不到时才退到 data 张量。
    """
    sel = joint_sel.detach().cpu().numpy()
    _SANE = 1.0e6  # 大于这个数就当占位值/无限制

    # ① actuator 配置（最可信）
    try:
        order = np.zeros(robot.num_joints, dtype=float)
        for act in robot.actuators.values():
            eff = getattr(act.cfg, "effort_limit", None)
            if eff is None:
                continue
            ids = np.asarray(act.joint_indices, dtype=int).ravel()
            order[ids] = float(np.mean(np.asarray(eff, dtype=float).ravel()))
        v = order[sel]
        ok = (v > 0.0) & (v < _SANE)
        if ok.mean() > 0.5:
            v = np.where(ok, v, np.nan)
            return v
    except Exception:  # noqa: BLE001
        pass

    # ② data 张量（可能全是 1e9 占位）
    for attr in ("joint_effort_limits", "joint_effort_limits_sim"):
        lim = getattr(robot.data, attr, None)
        if lim is None:
            continue
        try:
            v = np.asarray(lim[0, joint_sel].detach().cpu().numpy(), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        ok = (v > 0.0) & (v < _SANE)
        if ok.mean() > 0.5:
            return np.where(ok, v, np.nan)
    return None


# ──────────────────────────── 指标（纯函数）────────────────────────────
def mirror_rms(qpos: np.ndarray, joint_names: list[str]) -> dict[str, list[float]]:
    """4 个镜像对的逐关节类型 RMS（符号约定见 DEF-027）。"""
    out: dict[str, list[float]] = {}
    if qpos is None or np.size(qpos) == 0 or not joint_names:
        return out
    for a, b, signs in MIRROR_PAIRS:
        vals = []
        for jt in JOINT_TYPES:
            if f"{a}_{jt}_joint" not in joint_names or f"{b}_{jt}_joint" not in joint_names:
                return {}
            ia, ib = joint_names.index(f"{a}_{jt}_joint"), joint_names.index(f"{b}_{jt}_joint")
            vals.append(float(np.sqrt(np.mean((qpos[:, ia] - signs[jt] * qpos[:, ib]) ** 2))))
        out[f"{a}~{b}"] = vals
    return out


def lateral_asymmetry(foot_xy_b: np.ndarray) -> tuple[float, float]:
    """左右不对称度（0 = 完全对称）。列序 = fl, fr, hl, hr。"""
    y = foot_xy_b[:, :, 1].mean(axis=0)
    return float(y[0] + y[1]), float(y[2] + y[3])


def stance_width(foot_xy_b: np.ndarray) -> tuple[float, float]:
    y = foot_xy_b[:, :, 1].mean(axis=0)
    return float(y[0] - y[1]), float(y[2] - y[3])


def duty_factor(contact: np.ndarray) -> np.ndarray:
    """每条腿的触地占比（滚动步态 ≈1.0，跳跃/踏步 <1.0）。"""
    return contact.mean(axis=0)


def dominant_freq(sig: np.ndarray, dt: float) -> float:
    """去均值后的主频（Hz）；信号太短/太平时返回 0。"""
    x = np.asarray(sig, dtype=float)
    if x.size < 8:
        return 0.0
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=dt)
    k = int(np.argmax(spec[1:]) + 1) if spec.size > 1 else 0
    return float(freqs[k])


def summarize(ep: EpisodeData, joint_names: list[str], torque_limit: np.ndarray | None) -> dict:
    """一档命令的标量指标（写进 report.md / JSON）。"""
    has_joints = np.size(ep.joint_pos) > 0 and bool(joint_names)
    has_feet = np.size(ep.foot_xy_b) > 0
    has_contact = np.size(ep.contact) > 0
    mir = mirror_rms(ep.joint_pos, joint_names)
    fa, ha = lateral_asymmetry(ep.foot_xy_b) if has_feet else (float("nan"), float("nan"))
    swf, swh = stance_width(ep.foot_xy_b) if has_feet else (float("nan"), float("nan"))
    pw = ep.torque * ep.joint_vel if has_joints else np.zeros(0)
    # ── 尖刺指标（DEF-043 的分析：段首阶跃 vs push 冲击）────────────────────
    # 尖刺定义：|err_xy| > max(均值+3σ, 0.35 m/s)；再按"是否落在每段前 0.3 s"分成两类。
    err_xy = ep.err_xy
    _sp_thr = max(float(err_xy.mean() + 3 * err_xy.std()), 0.35)
    _sp = err_xy > _sp_thr
    _sp_n = int(_sp.sum())
    spike = {
        "spike_threshold": _sp_thr,
        "spike_count": _sp_n,
        "spike_rate": float(_sp.mean()),
        "spike_max_err": float(err_xy.max()),
        "spike_max_speed": float(np.linalg.norm(ep.vel_b, axis=1).max()),
        "spike_segments": 0,
        "spike_recovery_s": float("nan"),
    }
    # 连续段 + 恢复时间（回到阈值以下所需时间）
    j = 0
    recover = []
    while j < _sp.size:
        if _sp[j]:
            k = j
            while k + 1 < _sp.size and _sp[k + 1]:
                k += 1
            spike["spike_segments"] += 1
            m = k + 1
            while m < _sp.size and err_xy[m] > _sp_thr * 0.5:
                m += 1
            recover.append((m - j) * float(ep.t[1] - ep.t[0]) if ep.t.size > 1 else float("nan"))
            j = k + 1
        else:
            j += 1
    if recover:
        spike["spike_recovery_s"] = float(np.mean(recover))
    out = {
        "label": ep.label,
        "command": list(ep.command),
        "steps": int(ep.t.size),
        "duration_s": float(ep.t[-1] - ep.t[0]) if ep.t.size else 0.0,
        "n_done": int(ep.n_done),
        "terminations": dict(ep.term_counts),
        "err_vel_xy_mean": float(ep.err_xy.mean()),
        "err_vel_xy_p95": _pct(ep.err_xy, 95),
        "err_vel_yaw_mean": float(ep.err_yaw.mean()),
        "vel_bias": [float(ep.cmd[:, i].mean() - ep.vel_b[:, i].mean()) for i in range(2)],
        "vel_std": [float(ep.vel_b[:, i].std()) for i in range(2)],
        "yaw_bias": float(ep.cmd[:, 2].mean() - ep.yaw_rate.mean()),
        "height_mean": float(ep.height.mean()),
        "height_std": float(ep.height.std()),
        "pitch_mean_deg": float(np.degrees(ep.pitch.mean())),
        "roll_mean_deg": float(np.degrees(ep.roll.mean())),
        "pitch_std_deg": float(np.degrees(ep.pitch.std())),
        "roll_std_deg": float(np.degrees(ep.roll.std())),
        "joint_pos_std": ([float(ep.joint_pos[:, i].std()) for i in range(len(joint_names))]
                          if has_joints else []),
        "torque_rms": ([float(np.sqrt(np.mean(ep.torque[:, i] ** 2))) for i in range(len(joint_names))]
                       if has_joints else []),
        "torque_absmax": ([float(np.abs(ep.torque[:, i]).max()) for i in range(len(joint_names))]
                          if has_joints else []),
        "power_mean_abs": float(np.abs(pw).mean()) if pw.size else float("nan"),
        "duty_factor": [float(x) for x in duty_factor(ep.contact)] if has_contact else [],
        "wheel_omega_mean": (
            [float(ep.joint_vel[:, joint_names.index(f"{leg}_wheel_joint")].mean()) for leg in LEGS]
            if has_joints else []
        ),
        "step_freq_hz": (
            {leg: dominant_freq(ep.foot_z_w[:, k], 0.02) for k, leg in enumerate(LEGS)}
            if has_feet else {}
        ),
        "mirror_rms": mir,
        "lateral_asymmetry_cm": [fa * 100.0, ha * 100.0],
        "stance_width": [swf, swh],
        "path_length_m": float(np.linalg.norm(np.diff(ep.root_xy, axis=0), axis=1).sum()),
        **spike,
    }
    if torque_limit is not None and has_joints:
        out["torque_limit_usage"] = [
            (
                float(np.abs(ep.torque[:, i]).max() / torque_limit[i])
                if np.isfinite(torque_limit[i]) and torque_limit[i] > 0
                else None
            )
            for i in range(len(joint_names))
        ]
    return out


# ──────────────────────────── 绘图 ────────────────────────────
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _switch_times(ep: EpisodeData) -> list[float]:
    """schedule 的切换时刻（第一段 t0=0 不算）。"""
    return [float(s[0]) for s in (ep.schedule or [])][1:]


def _mark_switches(ax, ep: EpisodeData) -> None:
    for t0 in _switch_times(ep):
        ax.axvline(t0, color="0.75", lw=0.8, zorder=0)


def seg_metrics(ep: EpisodeData) -> list[dict]:
    """把一条滚动按 schedule **分段**算稳态指标。

    每段取后半段当稳态（跳过切换后的瞬态），给 fig02 / fig03 的逐段柱状图用。
    没有 schedule 时退化成"整段就是一段"。
    """
    if ep.cmd.size == 0:
        return []
    if not ep.schedule:
        sl = slice(ep.t.size // 2, ep.t.size)
        row = {
            "t0": 0.0, "t1": float(ep.t[-1]) if ep.t.size else 0.0, "cmd": list(ep.command),
            "body": None, "dur_s": float(ep.t[-1] - ep.t[0]) if ep.t.size else 0.0,
            "err_vx": float(np.abs(ep.cmd[sl, 0] - ep.vel_b[sl, 0]).mean()),
            "err_vy": float(np.abs(ep.cmd[sl, 1] - ep.vel_b[sl, 1]).mean()),
            "err_wz": float(np.abs(ep.cmd[sl, 2] - ep.yaw_rate[sl]).mean()),
            "err_h": float(np.abs(ep.height[sl] - ep.body_cmd[sl, 0]).mean())
            if ep.body_cmd is not None else float("nan"),
            "err_pitch_deg": float(np.degrees(np.abs(ep.pitch[sl] - ep.body_cmd[sl, 1])).mean())
            if ep.body_cmd is not None else float("nan"),
            "err_roll_deg": float(np.degrees(np.abs(ep.roll[sl] - ep.body_cmd[sl, 2])).mean())
            if ep.body_cmd is not None else float("nan"),
        }
        return [row]
    dt = float(ep.t[1] - ep.t[0]) if ep.t.size > 1 else 0.02
    out: list[dict] = []
    for t0, t1, vel, body in ep.schedule:
        i0 = max(int(round(t0 / dt)), 0)
        i1 = min(int(round(t1 / dt)), ep.t.size)
        if i1 - i0 < 4:
            continue
        s0 = i0 + (i1 - i0) // 2
        sl = slice(s0, i1)
        row = {
            "t0": float(t0), "t1": float(t1), "cmd": [float(v) for v in vel],
            "body": None if body is None else [float(b) for b in body],
            "dur_s": float(t1 - t0),
            "err_vx": float(np.abs(ep.cmd[sl, 0] - ep.vel_b[sl, 0]).mean()),
            "err_vy": float(np.abs(ep.cmd[sl, 1] - ep.vel_b[sl, 1]).mean()),
            "err_wz": float(np.abs(ep.cmd[sl, 2] - ep.yaw_rate[sl]).mean()),
            "err_h": float("nan"), "err_pitch_deg": float("nan"), "err_roll_deg": float("nan"),
        }
        if ep.body_cmd is not None and body is not None:
            row["err_h"] = float(np.abs(ep.height[sl] - ep.body_cmd[sl, 0]).mean())
            row["err_pitch_deg"] = float(np.degrees(np.abs(ep.pitch[sl] - ep.body_cmd[sl, 1])).mean())
            row["err_roll_deg"] = float(np.degrees(np.abs(ep.roll[sl] - ep.body_cmd[sl, 2])).mean())
        out.append(row)
    return out


def _dense_heightmap(ax, pts: np.ndarray, traj: np.ndarray | None, name: str, cmap="terrain"):
    """把稀疏的 ray-cast 命中点画成**稠密**高度热力图（tricontourf 插值）+ 轨迹。

    用户的抱怨原话："这么稀疏的点云谁能看懂"（需求 8）⇒ 这里用三角剖分填充，
    再把轨迹画粗（lw=2.2）保证在一片色块里看得见。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        ax.set_axis_off()
        return
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    x, y, z = pts[:, 0] - x0, pts[:, 1] - y0, pts[:, 2]
    art = None
    try:
        art = ax.tricontourf(x, y, z, levels=24, cmap=cmap)
    except Exception:  # noqa: BLE001 - 退化三角剖分时退回散点（不要毁掉整份报告）
        art = ax.scatter(x, y, c=z, s=6, cmap=cmap)
    if traj is not None and np.size(traj):
        tr = np.asarray(traj, dtype=float).reshape(-1, 2)
        if tr.shape[0] > 1:
            # 轨迹可能比扫描格长得多（8 s × 1.5 m/s = 12 m vs 1.6 m 的格子）⇒ 裁到
            # 扫描范围附近再画，否则 `aspect="equal"` 会把热力图压成一个点。
            padx = 0.15 * max(float(x.max() - x.min()), 1e-3)
            pady = 0.15 * max(float(y.max() - y.min()), 1e-3)
            tx, ty = tr[:, 0] - x0, tr[:, 1] - y0
            inside = ((tx >= x.min() - padx) & (tx <= x.max() + padx)
                      & (ty >= y.min() - pady) & (ty <= y.max() + pady))
            ax.plot(np.where(inside, tx, np.nan), np.where(inside, ty, np.nan),
                    color="k", lw=2.2, solid_capstyle="round")
            ax.plot(tr[:1, 0] - x0, tr[:1, 1] - y0, "wo", ms=6, mec="k", mew=1.2,
                    label="start")
            ax.plot(tr[-1:, 0] - x0, tr[-1:, 1] - y0, "k^", ms=7, label="end")
    ax.set_title(f"terrain: {name}", fontsize=9)
    ax.set_xlabel("x - x0 (m)", fontsize=8)
    ax.set_ylabel("y - y0 (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    return art


def _save(fig, out_dir: str, name: str, dpi: int) -> str:
    path = os.path.join(out_dir, f"{name}.png")
    with warnings.catch_warnings():
        # gridspec 版式（fig04）与 tight_layout 不兼容，会刷一堆 UserWarning —— 无伤大雅，静音
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001
            pass
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[report] 图已保存: {path}")
    return path


def fig01_tracking(series, out_dir, dpi, meta):
    """一条**连续切换**轨迹上的速度 / 角速度跟踪（command vs actual + 逐轴误差）。

    用 schedule 那条数据（``series[lab][0]``）：vx / vy / wz 都会走一遍并来回切换，
    所以一张图里既看稳态精度、也看切换时刻（灰竖线）的响应（用户需求 2）。
    """
    labels = list(series.keys())
    names = ("vx", "vy", "wz")
    units = ("m/s", "m/s", "rad/s")
    fig, axes = plt.subplots(3, 2, figsize=(15, 8.4), squeeze=False)
    for m, lab in enumerate(labels):
        ep = series[lab][0]
        c = COLORS[m % len(COLORS)]
        for a, name in enumerate(names):
            y = ep.yaw_rate if a == 2 else ep.vel_b[:, a]
            ax = axes[a][0]
            if m == 0:
                ax.plot(ep.t, ep.cmd[:, a], "k--", lw=1.4, label="command")
            ax.plot(ep.t, y, color=c, lw=1.0, label=lab)
            _mark_switches(ax, ep)
            ax.set_ylabel(f"{name} ({units[a]})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            if a == 0:
                ax.set_title("command vs actual   (grey lines = command switch)")
            axe = axes[a][1]
            axe.plot(ep.t, np.abs(ep.cmd[:, a] - y), color=c, lw=1.0, label=lab)
            _mark_switches(axe, ep)
            axe.set_ylabel(f"|err {name}| ({units[a]})")
            axe.grid(alpha=0.3)
            axe.legend(fontsize=7)
            if a == 0:
                axe.set_title("absolute tracking error")
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle(f"(1) Velocity tracking over a switching schedule -- {meta['task']}"
                 f"  [{meta.get('sched_s', 0):.1f} s]")
    return _save(fig, out_dir, "fig01_tracking_timeseries", dpi)


def fig02_tracking_summary(series, out_dir, dpi, meta):
    """跟踪汇总：**逐段**误差柱状（vx/vy/wz 各一格）+ 指令-实际散点（三个轴）。

    以前只有 vx 一档命令，"只测了 vx" ⇒ 图很单薄（用户需求 3）。现在按 schedule
    的每一段算稳态误差，三个速度轴分别成柱，下面再给三张 cmd-vs-actual 散点。
    """
    labels = list(series.keys())
    sm = {lab: seg_metrics(series[lab][0]) for lab in labels}
    names = ("vx", "vy", "wz")
    n = len(sm[labels[0]])
    x = np.arange(n)
    w = 0.8 / max(len(labels), 1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6), squeeze=False)
    xt = [f"{r['cmd'][0]:g},{r['cmd'][1]:g},{r['cmd'][2]:g}" for r in sm[labels[0]]]
    for a, name in enumerate(names):
        ax = axes[0][a]
        for m, lab in enumerate(labels):
            vals = [r[f"err_{name}"] for r in sm[lab]]
            ax.bar(x + m * w, vals, w, color=COLORS[m % len(COLORS)], label=lab)
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels(xt, fontsize=6, rotation=90)
        ax.set_ylabel(f"mean |{name} cmd - {name}|")
        ax.set_title(f"{name}: steady-state error per command segment")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    for a, name in enumerate(names):
        ax = axes[1][a]
        lo, hi = 0.0, 0.0
        for m, lab in enumerate(labels):
            ep = series[lab][0]
            y = ep.yaw_rate if a == 2 else ep.vel_b[:, a]
            ax.scatter(ep.cmd[:, a], y, s=3, alpha=0.25,
                       color=COLORS[m % len(COLORS)], label=lab)
            lo = min(lo, float(ep.cmd[:, a].min()), float(y.min()))
            hi = max(hi, float(ep.cmd[:, a].max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="ideal")
        ax.set_xlabel(f"{name} command")
        ax.set_ylabel(f"{name} actual")
        ax.set_title(f"{name}: command vs actual")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"(2) Tracking summary -- {meta['task']}")
    return _save(fig, out_dir, "fig02_tracking_summary", dpi)


def fig03_posture(series, out_dir, dpi, meta):
    """机身姿态：height / pitch / roll 各自的跟踪 + **逐段**稳态误差。

    用户需求 4：pitch 与 roll 不挤在一张图里；同一条 10 s+ 轨迹里切换不同位姿。
    左列 = 指令 vs 实际（灰竖线 = 切换），右列 = 每段后半段的稳态误差。
    """
    labels = list(series.keys())
    rows = (("height", "m", 0, "err_h"), ("pitch", "deg", 1, "err_pitch_deg"),
            ("roll", "deg", 2, "err_roll_deg"))
    fig, axes = plt.subplots(3, 2, figsize=(15, 9.2), squeeze=False)
    sm0 = seg_metrics(series[labels[0]][0])
    n = max(len(sm0), 1)
    x = np.arange(n)
    w = 0.8 / max(len(labels), 1)

    def _seg_label(r):
        b = r.get("body")
        if not b:
            return ""
        return f"h={b[0]:.2f}\np={b[1]:+.2f}\nr={b[2]:+.2f}"

    xt = [_seg_label(r) for r in sm0]
    for a, (name, unit, idx, key) in enumerate(rows):
        ax = axes[a][0]
        for m, lab in enumerate(labels):
            ep = series[lab][0]
            c = COLORS[m % len(COLORS)]
            if ep.body_cmd is not None:
                cmd = ep.body_cmd[:, idx]
                cmd = np.degrees(cmd) if idx else cmd
                if m == 0:
                    ax.plot(ep.t, cmd, "k--", lw=1.4, label="command")
            act = ep.height if idx == 0 else (ep.pitch if idx == 1 else ep.roll)
            act = np.degrees(act) if idx else act
            ax.plot(ep.t, act, color=c, lw=1.0, label=lab)
            _mark_switches(ax, ep)
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if a == 0:
            ax.set_title("body pose: command vs actual  (grey = switch)")
        axe = axes[a][1]
        for m, lab in enumerate(labels):
            vals = [r.get(key, float("nan")) for r in seg_metrics(series[lab][0])]
            axe.bar(x + m * w, vals, w, color=COLORS[m % len(COLORS)], label=lab)
        axe.set_xticks(x + w * (len(labels) - 1) / 2)
        axe.set_xticklabels(xt, fontsize=6)
        axe.set_ylabel(f"steady |{name} err| ({unit})")
        axe.set_title("per-segment steady-state error")
        axe.grid(alpha=0.3, axis="y")
        axe.legend(fontsize=7)
        if a == 2:
            axe.set_xlabel("segment command  (h = height cmd, p/r = pitch/roll cmd)")
    for ax in axes[-1]:
        if ax.get_xlabel() == "":
            ax.set_xlabel("t (s)")
    fig.suptitle(f"(3) Body pose tracking over a switching schedule -- {meta['task']}")
    return _save(fig, out_dir, "fig03_posture", dpi)


def fig04_gait(series, out_dir, dpi, meta):
    """步态图：**每一档速度指令一列**，画四足触地条带 + 底部占空比。

    用户需求 5：轮腿在纯 vx 下"轮子转就行、不需要迈步"⇒ 旧版只画第一档
    （静止）当然全程着地、什么都看不出来。现在把 --commands 里每一档（含 vy/wz）
    各画一列，一眼就能分出"滚动"和"迈步/跳跃"。
    """
    labels = list(series.keys())
    eps = series[labels[0]]
    if np.size(eps[0].contact) == 0:
        return None  # 没有接触传感器的任务，直接跳过
    n = len(eps)
    fig = plt.figure(figsize=(max(9.5, 2.7 * n), 10.5))
    gs = fig.add_gridspec(5, n, height_ratios=[1, 1, 1, 1, 1.7], hspace=0.34, wspace=0.16)
    for j, ep in enumerate(eps):
        c = ep.command
        for k, leg in enumerate(LEGS):
            ax = fig.add_subplot(gs[k, j])
            ax.fill_between(ep.t, 0.0, 1.0, where=ep.contact[:, k], step="mid",
                            color=COLORS[j % len(COLORS)], alpha=0.85)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.grid(alpha=0.2, axis="x")
            if k == 0:
                ax.set_title(f"cmd=({c[0]:g},{c[1]:g},{c[2]:g})", fontsize=8)
            if j == 0:
                ax.set_ylabel(leg, rotation=0, ha="right", va="center")
            if k == 3:
                ax.set_xlabel("t (s)", fontsize=8)
    ax = fig.add_subplot(gs[4, :])
    xx = np.arange(len(LEGS))
    ww = 0.8 / max(n, 1)
    for j, ep in enumerate(eps):
        c = ep.command
        ax.bar(xx + j * ww, duty_factor(ep.contact), ww, color=COLORS[j % len(COLORS)],
               label=f"({c[0]:g},{c[1]:g},{c[2]:g})")
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xticks(xx + ww * (n - 1) / 2)
    ax.set_xticklabels(LEGS)
    ax.set_ylabel("duty factor\n(contact fraction)")
    ax.set_ylim(0, 1.15)
    ax.set_title("duty factor per leg  --  ~1.0 = wheels just rolling,  <1.0 = stepping",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=min(n, 5))
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"(4) Gait diagram per velocity command -- {meta['task']}")
    return _save(fig, out_dir, "fig04_gait_diagram", dpi)


def fig05_joints(series, out_dir, dpi, meta):
    """12 个腿关节的位置 / 速度网格（在**同一条切换轨迹**上，所以能看到指令组合的影响）。"""
    labels = list(series.keys())
    if np.size(series[labels[0]][0].joint_pos) == 0:
        return None
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5), squeeze=False)
    for k, leg in enumerate(LEGS):
        for a, jt in enumerate(JOINT_TYPES):
            name = f"{leg}_{jt}_joint"
            ax = axes[a][k]
            for m, lab in enumerate(labels):
                e = series[lab][0]
                col = e.col[name]
                ax.plot(e.t, e.joint_pos[:, col], color=COLORS[m % len(COLORS)], lw=0.9, label=lab)
                ax.plot(e.t, e.joint_vel[:, col], color=COLORS[m % len(COLORS)], lw=0.7, ls=":")
            _mark_switches(ax, series[labels[0]][0])
            ax.set_title(f"{leg} {jt}", fontsize=9)
            ax.grid(alpha=0.25)
            if k == 0 and a == 0:
                ax.legend(fontsize=7)
                ax.set_ylabel("pos (solid) / vel (dotted)")
    for k in range(4):
        axes[2][k].set_xlabel("t (s)")
    fig.suptitle(f"(5) Leg joint trajectories over the switching schedule -- {meta['task']}")
    return _save(fig, out_dir, "fig05_joints", dpi)


def fig06_actuation(series, out_dir, dpi, meta):
    """执行器：逐关节力矩统计 + **力矩时程**（髋/膝/轮），看负载是否平滑。"""
    labels = list(series.keys())
    if np.size(series[labels[0]][0].torque) == 0:
        return None
    names = series[labels[0]][0].joint_names_all
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.6), squeeze=False)
    x = np.arange(len(names))
    w = 0.8 / max(len(labels), 1)
    for m, lab in enumerate(labels):
        e = series[lab][0]
        c = COLORS[m % len(COLORS)]
        rms = np.sqrt((e.torque ** 2).mean(axis=0))
        axes[0][0].bar(x + m * w, rms, w, color=c, label=lab)
        axes[0][1].bar(x + m * w, np.abs(e.torque).max(axis=0), w, color=c, label=lab)
        axes[0][2].bar(x + m * w, np.abs(e.torque * e.joint_vel).mean(axis=0), w, color=c, label=lab)
    for ax, title, ylab in (
        (axes[0][0], "joint torque RMS", "N*m"),
        (axes[0][1], "joint torque peak |tau|max", "N*m"),
        (axes[0][2], "mean joint power |tau*dq|", "W"),
    ):
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    # ── 下行：力矩时程（膝盖 + 轮子），这是"哪段时间负载猛"的直接证据 ──
    ax = axes[1][0]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        for k, leg in enumerate(LEGS):
            col = e.col.get(f"{leg}_knee_joint")
            if col is None:
                continue
            ax.plot(e.t, e.torque[:, col], color=COLORS[m % len(COLORS)], lw=0.8,
                    ls=["-", "--", "-.", ":"][k], alpha=0.9, label=f"{lab} {leg}" if m == 0 else None)
    _mark_switches(ax, series[labels[0]][0])
    ax.set_title("knee joint torque over time")
    ax.set_ylabel("N*m")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1][1]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        for k, leg in enumerate(LEGS):
            col = e.col.get(f"{leg}_wheel_joint")
            if col is None:
                continue
            ax.plot(e.t, e.torque[:, col], color=COLORS[m % len(COLORS)], lw=0.8,
                    ls=["-", "--", "-.", ":"][k], alpha=0.9, label=f"{lab} {leg}" if m == 0 else None)
    _mark_switches(ax, series[labels[0]][0])
    ax.set_title("wheel joint torque over time")
    ax.set_ylabel("N*m")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    ax = axes[1][2]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        rms = np.sqrt((e.torque ** 2).mean(axis=0))
        # 峰值因子 = |tau|max / RMS（不用力矩限幅：本资产在 IsaacLab 里读到的是 1e9 占位值）
        crest = np.abs(e.torque).max(axis=0) / np.where(rms > 1e-6, rms, np.nan)
        ax.bar(x + m * w, crest, w, color=COLORS[m % len(COLORS)], label=lab)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="crest = 3")
    ax.set_title("crest factor |tau|max / RMS")
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.suptitle(f"(6) Actuation over the switching schedule -- {meta['task']}")
    return _save(fig, out_dir, "fig06_actuation", dpi)


def fig07_symmetry(series, out_dir, dpi, meta):
    """左右对称："撇腿"专项 —— 镜像 RMS + 足端不对称度 / 轮距的**时程**+ 俯视图。"""
    labels = list(series.keys())
    if not series[labels[0]][0].mirror_rms:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ep0 = series[labels[0]][0]
    pairs = list(ep0.mirror_rms.keys())
    x = np.arange(len(pairs) * len(JOINT_TYPES))
    w = 0.8 / max(len(labels), 1)
    ticks, ticklabels = [], []
    for pi, pair in enumerate(pairs):
        for a, jt in enumerate(JOINT_TYPES):
            k = pi * len(JOINT_TYPES) + a
            ticks.append(k + w * (len(labels) - 1) / 2)
            ticklabels.append(f"{pair}\n{jt}")
            for m, lab in enumerate(labels):
                v = series[lab][0].mirror_rms[pair][a]
                axes[0][0].bar(k + m * w, v, w, color=COLORS[m % len(COLORS)],
                               label=lab if k == 0 else None)
    axes[0][0].set_xticks(ticks)
    axes[0][0].set_xticklabels(ticklabels, fontsize=7)
    axes[0][0].set_title("mirror RMS (rad) -- smaller = more symmetric")
    axes[0][0].set_ylabel("rad")
    axes[0][0].grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=7)

    # 时程版：每一步的左右足端 y 偏差（cm）——比"每档一个点"能看出指令依赖
    ax = axes[0][1]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        if np.size(e.foot_xy_b) == 0:
            continue
        y = np.asarray(e.foot_xy_b)[:, :, 1]
        ax.plot(e.t, (y[:, 0] + y[:, 1]) * 100, color=COLORS[m % len(COLORS)], lw=1.0,
                label=f"{lab} front L+R")
        ax.plot(e.t, (y[:, 2] + y[:, 3]) * 100, color=COLORS[m % len(COLORS)], lw=0.9, ls="--",
                label=f"{lab} rear L+R")
        _mark_switches(ax, e)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("lateral asymmetry over time (cm),  0 = perfectly symmetric")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    for m, lab in enumerate(labels):
        e = series[lab][0]
        if np.size(e.foot_xy_b) == 0:
            continue
        xy = np.asarray(e.foot_xy_b).mean(axis=0)
        axes[1][0].scatter(xy[:, 0], xy[:, 1], s=60, color=COLORS[m % len(COLORS)], label=lab)
        for k, leg in enumerate(LEGS):
            axes[1][0].annotate(leg, (xy[k, 0], xy[k, 1]), fontsize=8, xytext=(4, 4),
                                textcoords="offset points")
    axes[1][0].axhline(0, color="k", lw=0.6, ls=":")
    axes[1][0].set_xlabel("body x (m)")
    axes[1][0].set_ylabel("body y (m)")
    axes[1][0].set_title("wheel centre top view (mean position, body frame)")
    axes[1][0].grid(alpha=0.3)
    axes[1][0].legend(fontsize=7)
    axes[1][0].axis("equal")

    # 时程版轮距：o = 前轮距、x = 后轮距（"前轮距收窄"是 DEF-038 §6 的观察项）
    ax = axes[1][1]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        if np.size(e.foot_xy_b) == 0:
            continue
        y = np.asarray(e.foot_xy_b)[:, :, 1]
        ax.plot(e.t, (y[:, 0] - y[:, 1]) * 100, color=COLORS[m % len(COLORS)], lw=1.0,
                label=f"{lab} front track")
        ax.plot(e.t, (y[:, 2] - y[:, 3]) * 100, color=COLORS[m % len(COLORS)], lw=0.9, ls="--",
                label=f"{lab} rear track")
        _mark_switches(ax, e)
    ax.set_title("track width over time (cm):  solid = front, dashed = rear")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"(7) Left/right symmetry & foot geometry -- {meta['task']}")
    return _save(fig, out_dir, "fig07_symmetry", dpi)


def fig08_arm(series, out_dir, dpi, meta):
    """机械臂：EE 位置 / 姿态跟踪（多个目标，带均值虚线）+ 臂关节**力矩**。

    用户需求 7：一条 10 s+ 的轨迹里多测几个末端目标；平均误差用虚线 + 文字标出来；
    原来记"臂关节角 (rad)"看不出东西 ⇒ 换成**各臂关节力矩变化**（更贴近执行器负载）。
    """
    labels = [lab for lab in series if series[lab][0].ee_cmd_pos_b is not None]
    if not labels:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9))
    ep0 = series[labels[0]][0]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        c = COLORS[m % len(COLORS)]
        err = np.linalg.norm(e.ee_cmd_pos_b - e.ee_pos_b, axis=1) * 100.0
        ori = np.degrees(e.ee_ori_err)
        axes[0][0].plot(e.t, err, color=c, lw=1.0, label=lab)
        axes[0][0].axhline(float(err.mean()), color=c, lw=1.0, ls="--")
        axes[0][0].text(0.99, 0.02 + 0.06 * m, f"{lab} mean = {err.mean():.1f} cm",
                        color=c, ha="right", va="bottom", transform=axes[0][0].transAxes,
                        fontsize=8)
        axes[0][1].plot(e.t, ori, color=c, lw=1.0, label=lab)
        axes[0][1].axhline(float(ori.mean()), color=c, lw=1.0, ls="--")
        axes[0][1].text(0.99, 0.02 + 0.06 * m, f"{lab} mean = {ori.mean():.1f} deg",
                        color=c, ha="right", va="bottom", transform=axes[0][1].transAxes,
                        fontsize=8)
    # EE 目标切换时刻（目标位置跳变 > 1 mm 的地方）
    if ep0.ee_cmd_pos_b is not None and ep0.t.size > 1:
        jump = np.linalg.norm(np.diff(ep0.ee_cmd_pos_b, axis=0), axis=1) > 1e-3
        for tj in ep0.t[1:][jump]:
            for ax in (axes[0][0], axes[0][1]):
                ax.axvline(float(tj), color="0.75", lw=0.8, zorder=0)
    axes[0][0].set_title("end-effector position tracking error (grey = new target)")
    axes[0][0].set_ylabel("cm")
    axes[0][1].set_title("end-effector orientation error")
    axes[0][1].set_ylabel("deg")
    for ax in (axes[0][0], axes[0][1]):
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for m, lab in enumerate(labels):
        e = series[lab][0]
        c = COLORS[m % len(COLORS)]
        arm_cols = [cc for n, cc in e.col.items() if n.startswith(("arm_joint", "gripper_joint"))]
        for cc in arm_cols:
            axes[1][0].plot(e.t, e.torque[:, cc], color=c, lw=0.8, alpha=0.8,
                            label=f"{lab} {e.joint_names_all[cc]}" if cc == arm_cols[0] else None)
        rms = [float(np.sqrt(np.mean(e.torque[:, cc] ** 2))) for cc in arm_cols]
        axes[1][1].bar(np.arange(len(arm_cols)) + m * (0.8 / max(len(labels), 1)),
                       rms, 0.8 / max(len(labels), 1), color=c, label=lab)
    axes[1][0].set_title("arm joint torque over time (replaces the old joint-angle panel)")
    axes[1][0].set_ylabel("N*m")
    axes[1][0].set_xlabel("t (s)")
    axes[1][0].grid(alpha=0.3)
    axes[1][0].legend(fontsize=6, ncol=2)
    axes[1][1].set_title("arm joint torque RMS")
    arm_names = [n for n in ep0.joint_names_all if n.startswith(("arm_joint", "gripper_joint"))]
    _w = 0.8 / max(len(labels), 1)
    axes[1][1].set_xticks(np.arange(len(arm_names)) + _w * (len(labels) - 1) / 2)
    axes[1][1].set_xticklabels(arm_names, rotation=30, fontsize=7)
    axes[1][1].set_ylabel("N*m")
    axes[1][1].grid(alpha=0.3, axis="y")
    axes[1][1].legend(fontsize=7)
    fig.suptitle(f"(8) Arm / end-effector tracking and torque -- {meta['task']}"
                 f"  [{meta.get('arm_s', 0):.1f} s, {meta.get('arm_targets', 0)} target switches]")
    return _save(fig, out_dir, "fig08_arm_ee", dpi)


def fig09_terrain(series, out_dir, dpi, meta):
    """地形：**每个子地形一张稠密高度热力图 + 该 env 的真实轨迹** + 足端离地间隙。

    用户需求 8：稀疏点云没人看得懂 ⇒ 用 tricontourf 插值成高度图、轨迹画粗；
    "地形高度随时间波动"这张删掉（地形在一次测试里根本不变）；
    并且"每个地形简单测一下" —— 这里每种地形取一个代表 env 各画一格。
    """
    labels = [lab for lab in series if series[lab][0].terrain_maps]
    if not labels:
        labels = [lab for lab in series if series[lab][0].terrain_pts_w is not None]
        if not labels:
            return None
    # 取"走得最远"的那一档命令 ⇒ 轨迹信息量最大（静止档的轨迹就是一个点）
    def _path_len(e) -> float:
        if e.terrain_maps is None or np.size(e.root_xy) == 0:
            return -1.0
        return float(np.linalg.norm(np.diff(np.asarray(e.root_xy, dtype=float), axis=0), axis=1).sum())

    ep0 = max(series[labels[0]], key=_path_len)
    maps = dict(ep0.terrain_maps or {})
    trajs = dict(ep0.terrain_traj or {})
    if not maps and ep0.terrain_pts_w is not None:
        maps = {"env_id": np.asarray(ep0.terrain_pts_w, dtype=float).reshape(-1, 3)}
        trajs = {"env_id": np.asarray(ep0.root_xy, dtype=float)}
    names = sorted(maps.keys())
    n = len(names)
    cols = min(4, n + 1)
    rows = int(np.ceil((n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.9 * rows), squeeze=False)
    for i, nm in enumerate(names):
        ax = axes[i // cols][i % cols]
        art = _dense_heightmap(ax, maps[nm], trajs.get(nm), nm)
        if art is not None and i == 0:
            fig.colorbar(art, ax=ax, label="terrain z (m)", fraction=0.046)
    ax = axes[n // cols][n % cols]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        if np.size(e.foot_z_w) == 0:
            continue
        fz = np.asarray(e.foot_z_w, dtype=float)
        low = np.sort(fz, axis=1)[:, 0]
        clear = fz - float(np.median(low))
        ax.plot(e.t, clear.mean(axis=1) * 100, color=COLORS[m % len(COLORS)], lw=1.0, label=lab)
        _mark_switches(ax, e)
    ax.set_title("mean wheel clearance (cm)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("cm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    for i in range(n + 1, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(f"(9) Terrain: dense height maps per sub-terrain + trajectories -- {meta['task']}")
    return _save(fig, out_dir, "fig09_terrain", dpi)


def fig10_compare(series, out_dir, dpi, meta):
    """两个 checkpoint 的关键指标并排（需要一个以上 label）。"""
    labels = list(series.keys())
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    keys = [
        ("err_vel_xy_mean", "linear velocity error (m/s)", False),
        ("height_std", "base height jitter std (m)", False),
        ("power_mean_abs", "mean |joint power| (W)", False),
        ("duration_s", "recorded duration (s)", False),
    ]
    n = min(len(v) for v in series.values())
    x = np.arange(n)
    w = 0.8 / len(labels)
    for ax, (key, title, _) in zip(axes, keys):
        for m, lab in enumerate(labels):
            vals = [series[lab][i].mirror_rms and _metric_of(series[lab][i], key) for i in range(n)]
            ax.bar(x + m * w, vals, w, color=COLORS[m % len(COLORS)], label=lab)
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([str(e.command) for e in series[labels[0]]], fontsize=6, rotation=20)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    fig2, ax2 = plt.subplots(figsize=(11, 3.6))
    rows = []
    for m, lab in enumerate(labels):
        for i in range(n):
            s = summarize(series[lab][i], series[lab][i].joint_names_all, series[lab][i].torque_limit)
            knee = max(s["mirror_rms"]["hl~hr"]) if s["mirror_rms"] else float("nan")
            rows.append((lab, str(series[lab][i].command), s["err_vel_xy_mean"], s["height_std"],
                         s["lateral_asymmetry_cm"][1], knee))
    ax2.axis("off")
    tbl = ax2.table(
        cellText=[[r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:+.2f}", f"{r[5]:.3f}"] for r in rows],
        colLabels=["policy", "command", "err_vel_xy", "height_std", "rear asym (cm)",
                   "hl~hr knee RMS"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
    path2 = _save(fig2, out_dir, "fig10_compare_table", dpi)
    fig.suptitle(f"(10) A/B comparison (same env / same commands / same seed) -- {meta['task']}")
    return [_save(fig, out_dir, "fig10_compare", dpi), path2]


def _metric_of(ep: EpisodeData, key: str) -> float:
    s = summarize(ep, ep.joint_names_all, ep.torque_limit)
    return float(s[key])


# ───────────────── ⑪ 分地形统计（多地形任务）─────────────────
def terrs_sorted(table: dict) -> list:
    """地形名的稳定排序（按"样本量"降序，方便一眼看出哪类地形测得最多）。"""
    def n_env(t):
        for lab in table[t].values():
            if "n_env" in lab:
                return -int(np.mean(lab["n_env"]))
        return 0

    return sorted(table.keys(), key=n_env)


def per_terrain_table(series: dict) -> dict:
    """把逐 env 指标按"落在哪种地形上"分组求均值 → ``{地形名: {指标: 值}}``。"""
    out: dict = {}
    for lab, eps in series.items():
        for ep in eps:
            if ep.per_env is None or ep.env_terrain is None:
                continue
            names = np.array(ep.env_terrain)
            for tname in sorted(set(names.tolist())):
                sel = names == tname
                if not sel.any():
                    continue
                d = out.setdefault(tname, {}).setdefault(lab, {})
                for key in ("err_xy", "err_yaw", "height_std", "tau_rms_max", "done"):
                    v = np.asarray(ep.per_env[key], dtype=float)[sel]
                    d.setdefault(key, []).append(float(np.mean(v)))
                d.setdefault("duty_min", []).append(float(np.asarray(ep.per_env["duty"])[sel].min()))
                d.setdefault("n_env", []).append(int(sel.sum()))
                cmd_key = f"cmd{tuple(round(c, 2) for c in ep.command)}"
                d.setdefault(cmd_key, []).append(float(np.mean(np.asarray(ep.per_env["err_xy"])[sel])))
    return out


def fig11_per_terrain(series, out_dir, dpi, meta):
    """分地形对比：不同子地形上的误差 / 抖动 / 力矩 / 摔倒 / 占空比。"""
    table = per_terrain_table(series)
    if not table:
        return None
    terrs = list(table.keys())
    labels = list(series.keys())
    cmds = sorted({k for t in table.values() for lab in t.values() for k in lab if k.startswith("cmd")})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
    metrics = [("err_xy", "linear velocity error (m/s)"), ("err_yaw", "yaw rate error (rad/s)"),
               ("height_std", "base height jitter std (m)")]
    x = np.arange(len(terrs))
    w = 0.8 / max(len(cmds), 1)
    for ax, (key, title) in zip(axes[0], metrics):
        for j, ck in enumerate(cmds):
            vals = [np.mean(table[t].get(labels[0], {}).get(ck, [np.nan]))
                    for t in terrs]
            ax.bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
        ax.set_xticks(x + w * (len(cmds) - 1) / 2)
        ax.set_xticklabels(terrs, fontsize=8, rotation=15)
        ax.set_title(f"{title} -- by terrain")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    for j, ck in enumerate(cmds):
        vals = [np.mean(table[t].get(labels[0], {}).get("done", [np.nan]))
                for t in terrs]
        axes[1][0].bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
    axes[1][0].set_xticks(x + w * (len(cmds) - 1) / 2)
    axes[1][0].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][0].set_title("terminations per env (by terrain)")
    axes[1][0].grid(alpha=0.3, axis="y")
    for j, ck in enumerate(cmds):
        vals = [np.mean(table[t].get(labels[0], {}).get("duty_min", [np.nan]))
                for t in terrs]
        axes[1][1].bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
    axes[1][1].set_xticks(x + w * (len(cmds) - 1) / 2)
    axes[1][1].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][1].set_title("worst-leg duty factor (by terrain)")
    axes[1][1].set_ylim(0, 1.05)
    axes[1][1].grid(alpha=0.3, axis="y")
    n_env = [int(np.mean(table[t].get(labels[0], {}).get("n_env", [0]))) for t in terrs]
    axes[1][2].bar(x, n_env, 0.6)
    axes[1][2].set_xticks(x)
    axes[1][2].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][2].set_title("number of envs per terrain (sample size)")
    axes[1][2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"(11) Per-terrain metrics -- {meta['task']}")
    return _save(fig, out_dir, "fig11_per_terrain", dpi)


# ───────────────── ⑫ 指令切换（变换能力）─────────────────
def transition_metrics(ep: EpisodeData, tol_frac: float = 0.25, hold_s: float = 0.30) -> list[dict]:
    """每个切换点之后：峰值误差 / 稳定时间 / 稳态误差（速度与高度各一份）。"""
    if not ep.schedule:
        return []
    dt = float(ep.t[1] - ep.t[0]) if ep.t.size > 1 else 0.02
    err_xy = ep.err_xy
    err_h = np.abs(ep.height - ep.body_cmd[:, 0]) if ep.body_cmd is not None else None
    out = []
    for k, (t0, t1, vel, body) in enumerate(ep.schedule):
        i0, i1 = int(t0 / dt), min(int(t1 / dt), ep.t.size)
        if i1 - i0 < 5:
            continue
        # 稳态：该段最后 40%
        s0 = i0 + int(0.6 * (i1 - i0))
        ss_xy = float(np.mean(err_xy[s0:i1])) if i1 > s0 else float("nan")
        thr = max(ss_xy * (1 + tol_frac), 0.03)
        hold_n = max(int(hold_s / dt), 1)
        settle = None
        for j in range(i0, i1 - hold_n):
            if np.all(err_xy[j:j + hold_n] < thr):
                settle = float(ep.t[j] - t0)
                break
        win = slice(i0, min(i0 + int(0.5 / dt), i1))
        row = {
            "seg": k,
            "t0": round(float(t0), 2),
            "vel_cmd": [round(float(v), 2) for v in vel],
            "body_cmd": [round(float(b), 3) for b in body] if body is not None else None,
            "peak_err_xy_0p5s": float(np.max(err_xy[win])) if err_xy[win].size else float("nan"),
            "steady_err_xy": ss_xy,
            "settle_s": settle,
            "steady_err_height": (
                float(np.mean(err_h[s0:i1])) if err_h is not None and i1 > s0 else float("nan")
            ),
        }
        out.append(row)
    return out


def fig12_switch(series, out_dir, dpi, meta):
    """指令切换测试：速度/姿态指令跳变后的跟踪与稳定过程。"""
    labels = [lab for lab in series if series[lab][0].schedule]
    if not labels:
        return None
    eps = {lab: series[lab][0] for lab in labels}
    ep0 = eps[labels[0]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    for lab in labels:
        ep = eps[lab]
        c = COLORS[labels.index(lab) % len(COLORS)]
        axes[0][0].plot(ep.t, ep.cmd[:, 0], "k--", lw=1.0, label="vx cmd" if lab == labels[0] else None)
        axes[0][0].plot(ep.t, ep.vel_b[:, 0], color=c, lw=1.1, label=f"{lab} vx actual")
        axes[0][1].plot(ep.t, ep.err_xy, color=c, lw=1.0, label=lab)
        if ep.body_cmd is not None:
            axes[1][0].plot(ep.t, ep.body_cmd[:, 0], "k--", lw=1.0,
                            label="height cmd" if lab == labels[0] else None)
            axes[1][0].plot(ep.t, ep.height, color=c, lw=1.1, label=f"{lab} height")
            axes[1][1].plot(ep.t, np.degrees(ep.body_cmd[:, 1]), "k--", lw=1.0, label="pitch cmd")
            axes[1][1].plot(ep.t, np.degrees(ep.pitch), color=c, lw=1.1, label=f"{lab} pitch")
            axes[1][1].plot(ep.t, np.degrees(ep.body_cmd[:, 2]), "k:", lw=1.0, label="roll cmd")
            axes[1][1].plot(ep.t, np.degrees(ep.roll), color=c, lw=1.0, ls=":", label=f"{lab} roll")
    for ax in axes.ravel():
        for k, (t0, t1, _v, _b) in enumerate(ep0.schedule):
            ax.axvline(t0, color="0.6", lw=0.8, ls="-")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    axes[0][0].set_title("vx: command vs actual (vertical lines = switch)")
    axes[0][1].set_ylabel("|v_cmd - v| (m/s)")
    axes[0][1].set_title("linear velocity error")
    axes[1][0].set_title("base height: command vs actual")
    axes[1][1].set_ylabel("deg")
    axes[1][1].set_title("pitch / roll: command vs actual")
    for ax in axes.ravel():
        ax.set_xlabel("t (s)")
    fig.suptitle(f"(12) Command switching -- {meta['task']}"
                 f"  (each segment {meta.get('switch_s')} s)")
    paths = [_save(fig, out_dir, "fig12_switch", dpi)]
    # 数字表
    rows = []
    for lab in labels:
        for r in transition_metrics(eps[lab]):
            rows.append((lab, r["seg"], str(r["vel_cmd"]), str(r["body_cmd"]),
                         f"{r['peak_err_xy_0p5s']:.3f}", f"{r['steady_err_xy']:.3f}",
                         "-" if r["settle_s"] is None else f"{r['settle_s']:.2f}",
                         f"{r['steady_err_height']:.3f}"))
    fig2, ax2 = plt.subplots(figsize=(11, 0.4 + 0.32 * max(len(rows), 3)))
    ax2.axis("off")
    tbl = ax2.table(
        cellText=[list(r) for r in rows],
        colLabels=["policy", "seg", "velocity cmd", "body pose cmd",
                   "peak err first 0.5 s", "steady err", "settle time (s)",
                   "height steady err"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    paths.append(_save(fig2, out_dir, "fig12_switch_table", dpi))
    return paths


# ───────────────── ⑬ push 抗扰扫描（训练口径以外的泛化）─────────────────
def push_metrics(ep: EpisodeData, n_envs: int) -> dict:
    """一条 push 档位滚动的"抗扰"指标：生还率 / 终止构成 / 尖刺 / 最大瞬时误差。"""
    s = summarize(ep, ep.joint_names_all, ep.torque_limit)
    dur = max(float(s["duration_s"]), 1e-6)
    n_env = max(int(n_envs), 1)
    per_min = 60.0 / dur
    return {
        "err_vel_xy_mean": float(s["err_vel_xy_mean"]),
        "term_rate_per_env_min": float(s["n_done"]) / n_env * per_min,
        "terminations": dict(s["terminations"]),
        "spike_per_min": float(s["spike_count"]) / n_env * per_min,
        "spike_recovery_s": float(s["spike_recovery_s"]),
        "spike_max_err": float(s["spike_max_err"]),
        "spike_max_speed": float(s["spike_max_speed"]),
        "duration_s": dur,
    }


def fig13_push_robustness(series, out_dir, dpi, meta):
    """push 抗扰扫描：把"推得更勤 / 推得更狠"做成若干档，看生还率与尖刺如何退化。

    用户需求 1：频率高于训练、力度外推到训练分布之外 ⇒ 直接读泛化边界。
    横轴 = 档位（`力度 x 频率`），1x1 就是训练口径。
    """
    labels = list(series.keys())
    if not labels:
        return None
    n_envs = int(meta.get("num_envs", 64))
    mets = {lab: push_metrics(series[lab][0], n_envs) for lab in labels}
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6), squeeze=False)
    # (0,0) 跟踪误差
    ax = axes[0][0]
    ax.bar(x, [mets[lab]["err_vel_xy_mean"] for lab in labels], 0.6,
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean |v_cmd - v| (m/s)")
    ax.set_title("tracking error under different push levels")
    ax.grid(alpha=0.3, axis="y")
    # (0,1) 终止率（按终止项堆叠）
    ax = axes[0][1]
    groups = sorted({g for lab in labels for g in mets[lab]["terminations"]})
    bottom = np.zeros(len(labels))
    for gi, g in enumerate(groups):
        vals = np.array([float(mets[lab]["terminations"].get(g, 0)) / n_envs
                         * (60.0 / mets[lab]["duration_s"]) for lab in labels])
        ax.bar(x, vals, 0.6, bottom=bottom,
               color=COLORS[gi % len(COLORS)], label=g.split("/")[-1])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("terminations / env / minute")
    ax.set_title("falls (survivability): lower = more robust")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    # (1,0) 尖刺频次 + 恢复时间
    ax = axes[1][0]
    ax.bar(x, [mets[lab]["spike_per_min"] for lab in labels], 0.6,
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("velocity-error spikes / env / minute")
    ax.set_title("spike frequency (threshold = mean + 3 sigma, min 0.35 m/s)")
    ax.grid(alpha=0.3, axis="y")
    ax2 = ax.twinx()
    ax2.plot(x, [mets[lab]["spike_recovery_s"] for lab in labels], "ko--", lw=1.2)
    ax2.set_ylabel("mean recovery time (s)")
    # (1,1) 逐段误差时程（细线）
    ax = axes[1][1]
    for i, lab in enumerate(labels):
        ep = series[lab][0]
        ax.plot(ep.t, ep.err_xy, color=COLORS[i % len(COLORS)], lw=0.8, alpha=0.9, label=lab)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|v_cmd - v| (m/s)")
    ax.set_title("per-step tracking error (spikes = push impacts)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(f"(13) Push robustness sweep -- {meta['task']}"
                 f"  (1x1 = training push settings)")
    paths = [_save(fig, out_dir, "fig13_push_robustness", dpi)]
    rows = [
        (lab, f"{mets[lab]['err_vel_xy_mean']:.4f}",
         f"{mets[lab]['term_rate_per_env_min']:.2f}",
         f"{mets[lab]['spike_per_min']:.1f}",
         f"{mets[lab]['spike_recovery_s']:.2f}",
         f"{mets[lab]['spike_max_err']:.2f}")
        for lab in labels
    ]
    fig2, ax3 = plt.subplots(figsize=(10, 0.5 + 0.34 * max(len(rows), 3)))
    ax3.axis("off")
    tbl = ax3.table(
        cellText=[list(r) for r in rows],
        colLabels=["push level", "err_vel_xy (m/s)", "falls/env/min", "spikes/env/min",
                   "recovery (s)", "max err (m/s)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    paths.append(_save(fig2, out_dir, "fig13_push_table", dpi))
    return paths


# ──────────────────────────── 报告 ────────────────────────────
#: NPZ 里每条 episode 存哪些字段（`groups` 分 cmd / sched / arm / push 四组）
_EP_FIELDS = ("t", "cmd", "vel_b", "yaw_rate", "root_xy", "root_z", "height", "pitch", "roll",
              "joint_pos", "joint_vel", "torque", "contact", "foot_xy_b", "foot_z_w")
_EP_OPT = ("body_cmd", "ee_cmd_pos_b", "ee_pos_b", "ee_ori_err", "scan_min", "scan_max",
           "scan_mean", "env_terrain_level")


def _dump_groups(path: str, groups: dict) -> None:
    """把 ``{组名: {label: [EpisodeData]}}`` 落成一份 npz（键 = ``组名~label|序号|字段``）。"""
    npz: dict[str, np.ndarray] = {}
    for grp, series in groups.items():
        for lab in series:
            for i, ep in enumerate(series[lab]):
                pre = f"{grp}~{lab}|{i}|"
                for field_name in _EP_FIELDS:
                    npz[pre + field_name] = np.asarray(getattr(ep, field_name))
                for opt in _EP_OPT:
                    v = getattr(ep, opt, None)
                    if v is not None:
                        npz[pre + opt] = np.asarray(v)
                if ep.terrain_pts_w is not None:
                    npz[pre + "terrain_pts_w"] = np.asarray(ep.terrain_pts_w)
                for nm, pts in (ep.terrain_maps or {}).items():
                    npz[pre + f"tmap|{nm}"] = np.asarray(pts)
                for nm, tr in (ep.terrain_traj or {}).items():
                    npz[pre + f"ttraj|{nm}"] = np.asarray(tr)
                if ep.per_env is not None:
                    for k, v in ep.per_env.items():
                        npz[pre + f"pe|{k}"] = np.asarray(v)
                if ep.env_terrain is not None:
                    npz[pre + "env_terrain"] = np.asarray(ep.env_terrain, dtype=object).astype("U32")
                npz[pre + "meta"] = np.array([json.dumps({
                    "label": ep.label,
                    "command": list(ep.command),
                    "joint_names_all": ep.joint_names_all,
                    "schedule": [list(s) for s in (ep.schedule or [])],
                    "term_counts": ep.term_counts,
                    "n_done": int(ep.n_done),
                }, ensure_ascii=False)])
                if ep.torque_limit is not None:
                    npz[pre + "torque_limit"] = np.asarray(ep.torque_limit)
    np.savez_compressed(path, **npz)


def write_report(out_dir: str, groups: dict, meta: dict, figures: list[str]) -> dict:
    """落地 report.md / summary.json / data.npz，并返回 summary（给 stdout 用）。"""
    series = groups.get("cmd") or next((g for g in groups.values() if g), {})
    labels = list(series.keys())
    summary = {
        lab: [summarize(ep, ep.joint_names_all, ep.torque_limit) for ep in series[lab]]
        for lab in labels
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

    _dump_groups(os.path.join(out_dir, "data.npz"), groups)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    lines = [
        f"# 低层策略体检报告 —— {meta['task']}",
        "",
        f"* checkpoint A：`{meta['checkpoint_a']}`",
    ]
    if meta.get("checkpoint_b"):
        lines.append(f"* checkpoint B：`{meta['checkpoint_b']}`")
    lines += [
        f"* 环境：`num_envs={meta['num_envs']}` / 每档 `{meta['steps']}` 步（warmup {meta['warmup']}）"
        f" / seed `{meta['seed']}` / push `{'关' if meta['no_push'] else '开'}`",
        f"* 命令：`{meta['commands']}`",
        "",
        "## 1. 关键指标（逐档命令）",
        "",
        "| 策略 | 命令 | err_vel_xy | err_yaw | 高度均值 | 高度std | pitch均值 | 后不对称 | hl~hr 膝RMS | 终止 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lab in labels:
        for s in summary[lab]:
            term = ", ".join(f"{k.split('/')[-1]}={v}" for k, v in s["terminations"].items() if v)
            lines.append(
                f"| {lab} | {tuple(s['command'])} | {s['err_vel_xy_mean']:.4f} | {s['err_vel_yaw_mean']:.4f} "
                f"| {s['height_mean']:.4f} | {s['height_std']:.4f} | {s['pitch_mean_deg']:+.2f}° "
                f"| {s['lateral_asymmetry_cm'][1]:+.2f} cm | {max(s['mirror_rms']['hl~hr']):.3f} "
                f"| {term or '-'} |"
            )

    lines += ["", "## 2. 镜像 RMS（rad，越小越对称）", "",
              "| 策略 | 命令 | fl~hr (hipx/hipy/knee) | fr~hl | fl~fr | hl~hr |",
              "|---|---|---|---|---|---|"]
    for lab in labels:
        for s in summary[lab]:
            if not s["mirror_rms"]:
                continue
            cells = ["/".join(f"{v:.3f}" for v in s["mirror_rms"][p])
                     for p in ("fl~hr", "fr~hl", "fl~fr", "hl~hr")]
            lines.append(f"| {lab} | {tuple(s['command'])} | " + " | ".join(cells) + " |")

    lines += ["", "## 3. 执行器与能耗（每档命令）", "",
              "| 策略 | 命令 | 腿 |tau| RMS 最大 | 轮 |tau| RMS 最大 | 臂 |tau| RMS 最大 "
              "| 峰值因子 最大 | 平均 |功率| | 轨迹长 |",
              "|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        for s in summary[lab]:
            nleg, nwheel, narm = 12, 4, max(len(s["torque_rms"]) - 16, 0)
            t = s["torque_rms"]
            crest = [
                (s["torque_absmax"][i] / t[i] if t[i] > 1e-6 else float("nan"))
                for i in range(len(t))
            ]
            crest_max = np.nanmax(crest) if np.any(np.isfinite(crest)) else float("nan")
            lines.append(
                f"| {lab} | {tuple(s['command'])} | {max(t[:nleg]):.2f} "
                f"| {max(t[nleg:nleg + nwheel]):.2f} "
                f"| {(max(t[nleg + nwheel:]) if narm else float('nan')):.2f} "
                f"| {crest_max:.2f} | {s['power_mean_abs']:.2f} W | {s['path_length_m']:.2f} m |"
            )

    # 后面几节是条件出现的（分地形 / 指令切换 / push 扫描）⇒ 编号用计数器，别写死
    sec_no = 4
    terr = per_terrain_table(series)
    if terr:
        lines += ["", f"## {sec_no}. 分地形（逐 env 指标按『落在哪种地形上』分组）", "",
                  "| 地形 | 命令 | err_vel_xy | err_vel_yaw | 高度std | 力矩RMS最大 | 平均终止次数 | 最差腿触地占比 | env 数 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tname in terrs_sorted(terr):
            for lab in terr[tname]:
                d = terr[tname][lab]
                for ck in sorted(k for k in d if k.startswith("cmd")):
                    lines.append(
                        f"| {tname} | {ck.replace('cmd', '')} | {d[ck][0]:.4f} "
                        f"| {np.mean(d['err_yaw']):.4f} | {np.mean(d['height_std']):.4f} "
                        f"| {np.mean(d['tau_rms_max']):.1f} | {np.mean(d['done']):.2f} "
                        f"| {np.mean(d['duty_min']):.3f} | {int(np.mean(d['n_env']))} |"
                    )
        sec_no += 1

    switch_rows = []
    for lab in (groups.get("sched") or series):
        for ep in (groups.get("sched") or series)[lab]:
            for r in transition_metrics(ep):
                switch_rows.append((lab, r))
    if switch_rows:
        lines += ["", f"## {sec_no}. 指令切换（变换能力：每段切换后的峰值误差/稳定时间/稳态误差）", "",
                  "| 策略 | 段 | 速度指令 | 姿态指令 | 切换后 0.5 s 峰值误差 | 稳态误差 | 稳定时间 (s) | 高度稳态误差 |",
                  "|---|---|---|---|---|---|---|---|"]
        for lab, r in switch_rows:
            st = "-" if r["settle_s"] is None else f"{r['settle_s']:.2f}"
            lines.append(
                f"| {lab} | {r['seg']} | {r['vel_cmd']} | {r['body_cmd']} "
                f"| {r['peak_err_xy_0p5s']:.3f} | {r['steady_err_xy']:.3f} | {st} "
                f"| {r['steady_err_height']:.3f} |"
            )
        sec_no += 1

    push = groups.get("push") or {}
    if push:
        n_env = max(int(meta.get("num_envs", 64)), 1)
        lines += ["", f"## {sec_no}. push 抗扰扫描（1x1 = 训练口径；>1 为外推）", "",
                  "| 档位 | err_vel_xy (m/s) | 终止 / 环境 / 分钟 | 尖刺 / 环境 / 分钟 "
                  "| 平均恢复 (s) | 最大瞬时误差 (m/s) |",
                  "|---|---|---|---|---|---|"]
        for lab in push:
            m = push_metrics(push[lab][0], n_env)
            lines.append(
                f"| {lab} | {m['err_vel_xy_mean']:.4f} | {m['term_rate_per_env_min']:.2f} "
                f"| {m['spike_per_min']:.1f} | {m['spike_recovery_s']:.2f} "
                f"| {m['spike_max_err']:.2f} |"
            )
        sec_no += 1

    lines += ["", f"## {sec_no}. 图（每张一个角度）", ""]
    lines += [f"* `{os.path.basename(p)}`" for p in figures]
    lines += ["", "> 原始数据在 `data.npz`，标量指标在 `summary.json`。", ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary


# ──────────────────────────── 主流程 ────────────────────────────
def load_series_npz(npz_path: str) -> tuple[dict, dict]:
    """把 `data.npz` 读回 `(groups, meta)`（配合 `--from-npz` 做"云端采集 + 本机画图"）。

    键格式 `组名~label|序号|字段`（组名 ∈ cmd / sched / arm / push）；老版本（无 `~`）
    一律当作 `cmd` 组 ⇒ 老 npz 仍然能画。
    """
    z = np.load(npz_path, allow_pickle=True)
    meta_path = os.path.join(os.path.dirname(os.path.abspath(npz_path)), "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    groups: dict[str, dict[str, list[EpisodeData]]] = {}
    seen: dict[tuple[str, str], set] = {}
    for key in z.files:
        parts = key.split("|")
        if len(parts) < 3:
            continue
        head = parts[0]
        grp, lab = head.split("~", 1) if "~" in head else ("cmd", head)
        seen.setdefault((grp, lab), set()).add(int(parts[1]))
    for (grp, lab), idxs in seen.items():
        bucket = groups.setdefault(grp, {}).setdefault(lab, [])
        for i in sorted(idxs):
            # 新格式带组名前缀；老 npz 只有 `label|i|field` ⇒ 自动兼容
            new_pre = f"{grp}~{lab}|{i}|"
            prefix = new_pre if f"{new_pre}meta" in z.files else f"{lab}|{i}|"

            def _k(field: str, _p: str = prefix) -> str:
                return _p + field

            info = {}
            if _k("meta") in z.files:
                info = json.loads(str(z[_k("meta")][0]))
            ep = EpisodeData(
                label=info.get("label", lab),
                command=tuple(info.get("command", (0.0, 0.0, 0.0))),
            )
            for field_name in _EP_FIELDS:
                if _k(field_name) in z.files:
                    setattr(ep, field_name, np.asarray(z[_k(field_name)]))
            for opt in _EP_OPT:
                if _k(opt) in z.files:
                    setattr(ep, opt, np.asarray(z[_k(opt)]))
            if _k("torque_limit") in z.files:
                ep.torque_limit = np.asarray(z[_k("torque_limit")], dtype=float)
            if _k("terrain_pts_w") in z.files:
                ep.terrain_pts_w = np.asarray(z[_k("terrain_pts_w")])
            tm_pre, tt_pre = _k("tmap|"), _k("ttraj|")
            tm = {key[len(tm_pre):]: np.asarray(z[key]) for key in z.files if key.startswith(tm_pre)}
            tt = {key[len(tt_pre):]: np.asarray(z[key]) for key in z.files if key.startswith(tt_pre)}
            ep.terrain_maps = tm or None
            ep.terrain_traj = tt or None
            if _k("env_terrain") in z.files:
                ep.env_terrain = [str(x) for x in np.asarray(z[_k("env_terrain")])]
            pe = {}
            pe_pre = _k("pe|")
            for key in z.files:
                if key.startswith(pe_pre):
                    pe[key[len(pe_pre):]] = np.asarray(z[key])
            ep.per_env = pe or None
            ep.joint_names_all = list(info.get("joint_names_all", []))
            if not ep.joint_names_all and ep.joint_pos.ndim == 2 and ep.joint_pos.shape[1] >= 16:
                # 老版本 npz 没存列名 ⇒ 按本脚本的固定列序重建（12 腿 + 4 轮 [+ 臂/夹爪]）
                names = [f"{leg}_{jt}_joint" for leg in LEGS for jt in JOINT_TYPES]
                names += [f"{leg}_wheel_joint" for leg in LEGS]
                extra = ep.joint_pos.shape[1] - len(names)
                names += [f"extra_joint{j}" for j in range(extra)]
                ep.joint_names_all = names
            ep.col = {n: j for j, n in enumerate(ep.joint_names_all)}
            ep.schedule = [tuple(x) for x in info.get("schedule", [])] or None
            ep.term_counts = info.get("term_counts", {})
            ep.n_done = int(info.get("n_done", 0))
            if ep.joint_pos.size and ep.joint_names_all:
                ep.mirror_rms = mirror_rms(ep.joint_pos, ep.joint_names_all)
            bucket.append(ep)
    total = sum(len(v) for g in groups.values() for v in g.values())
    print(f"[report] 从 {npz_path} 读回 {total} 段数据"
          f"（groups={ {g: list(v) for g, v in groups.items()} }）")
    return groups, meta


def render_all(groups: dict, meta: dict, out_dir: str, dpi: int) -> dict:
    """把 ``groups``（cmd / sched / arm / push 四组数据）渲染成整套图 + 报告。"""
    os.makedirs(out_dir, exist_ok=True)
    series = groups.get("cmd") or {}
    sched = groups.get("sched") or series
    arm = groups.get("arm") or sched
    push = groups.get("push") or {}
    figures: list[str] = []
    plan = [
        (fig01_tracking, sched), (fig02_tracking_summary, sched), (fig03_posture, sched),
        (fig04_gait, series), (fig05_joints, sched), (fig06_actuation, sched),
        (fig07_symmetry, sched), (fig08_arm, arm), (fig09_terrain, series),
        (fig10_compare, series), (fig11_per_terrain, series), (fig12_switch, sched),
        (fig13_push_robustness, push),
    ]
    for fn, data in plan:
        if not data:
            continue
        if fn is fig10_compare and len(data) < 2:
            continue
        try:
            out = fn(data, out_dir, dpi, meta)
        except Exception as exc:  # noqa: BLE001 - 一张图失败不该毁掉整份报告
            print(f"[report][WARN] {fn.__name__} 画图失败：{type(exc).__name__}: {exc}")
            continue
        if out is None:
            continue
        figures += out if isinstance(out, list) else [out]
    summary = write_report(out_dir, groups, meta, figures)
    print("\n[report] === 汇总（逐档命令）===")
    print(f"{'策略':<28}{'命令':<18}{'err_xy':>9}{'err_yaw':>9}{'高度std':>9}{'后不对称cm':>12}{'膝RMS(hl~hr)':>14}")
    for lab in summary:
        for s in summary[lab]:
            knee = max(s["mirror_rms"]["hl~hr"]) if s["mirror_rms"] else float("nan")
            print(f"{lab[:27]:<28}{str(tuple(s['command'])):<18}"
                  f"{s['err_vel_xy_mean']:>9.4f}{s['err_vel_yaw_mean']:>9.4f}{s['height_std']:>9.4f}"
                  f"{s['lateral_asymmetry_cm'][1]:>12.2f}{knee:>14.3f}")
    print(f"\n[report] 输出目录：{os.path.abspath(out_dir)}"
          f"（report.md / summary.json / data.npz / {len(figures)} 张 PNG）")
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # ── 只画图模式：从 data.npz 读回数据渲染（不建环境、不装 checkpoint）──
    if args_cli.from_npz:
        groups, meta = load_series_npz(args_cli.from_npz)
        render_all(groups, meta, args_cli.out_dir, args_cli.dpi)
        return
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 固定命令 + 去噪（与 eval_fixed_command / probe_gait_symmetry 同口径）
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.no_push:
        env_cfg.events.randomize_push_robot = None
        env_cfg.events.randomize_apply_external_force_torque = None

    # 地形：小环境数下把网格缩小到 5×5（生成快），大环境数保持训练网格（DEF-040 §4 的坑）
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.max_init_terrain_level = None
        # 本仓库的 WBC 配置把 height_scanner 关了（`RoughEnvWBCConfig.__post_init__`）。
        # 诊断报告需要它来画地形点云 ⇒ 临时加一份**只用于诊断**的（不往 observations 里塞东西，
        # 所以不影响策略输入；参数与 `velocity_env_cfg.EventCfg` 里原来那份一致）。
        if not args_cli.no_height_scan and getattr(env_cfg.scene, "height_scanner", None) is None:
            from isaaclab.sensors import RayCasterCfg, patterns  # noqa: PLC0415

            base = getattr(env_cfg, "base_link_name", "base_link")
            env_cfg.scene.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + base,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            print(f"[report] 临时加诊断用 height_scanner（prim={base}）⇒ 才能画 fig09_terrain")
        _shrink = args_cli.terrain_grid == "5x5" or (
            args_cli.terrain_grid == "auto" and args_cli.num_envs <= 16
        )
        if _shrink:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False
            print("[report] 地形网格 → 5×5（注意：5 列会丢掉部分子地形，"
                  "分地形统计请用 --terrain-grid keep）")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped
    commands = parse_commands(args_cli.commands)
    term_names = list(unwrapped.termination_manager.active_terms)
    os.makedirs(args_cli.out_dir, exist_ok=True)

    ckpt_a = os.path.abspath(args_cli.checkpoint)
    label_a = args_cli.label or os.path.basename(os.path.dirname(ckpt_a))
    jobs = [(label_a, ckpt_a)]
    if args_cli.compare:
        ckpt_b = os.path.abspath(args_cli.compare)
        jobs.append((args_cli.label_b or os.path.basename(os.path.dirname(ckpt_b)), ckpt_b))

    print(f"[report] task={task_name}  num_envs={unwrapped.num_envs}  dt={unwrapped.step_dt:.4f}s")
    print(f"[report] 命令 {commands}；每档 {args_cli.steps} 步（warmup {args_cli.warmup}）")
    print(f"[report] 终止项：{term_names}")

    harness = Harness(
        env, unwrapped, label_a, args_cli.env_id,
        terrain_gen=env_cfg.scene.terrain.terrain_generator,
    )
    # ── 机身姿态命令区间（有 body_pose 才有）──────────────────────────────
    bp = getattr(env_cfg.commands, "body_pose", None)
    if bp is not None:
        hr = tuple(getattr(bp, "height_range", (0.513, 0.513)))
        pr = tuple(getattr(bp, "pitch_range", (0.0, 0.0)))
        rr = tuple(getattr(bp, "roll_range", (0.0, 0.0)))
        bp.debug_vis = False
    else:
        hr = pr = rr = (0.0, 0.0)

    def _body_of(key: str) -> tuple[float, float, float] | None:
        if bp is None:
            return None
        h_mid, h_lo, h_hi = 0.5 * (hr[0] + hr[1]), hr[0], hr[1]
        # 姿态段（pitch/roll）用**标称站姿**高度（0.513 = 常规站立，落在 height_range 内），
        # 不用 h_mid（=(0.33+0.55)/2=0.44，那是"蹲着"）——否则俯仰/侧倾会顺手测成蹲姿。
        h_nom = min(0.513, hr[1])
        table = {
            "mid": (h_mid, 0.0, 0.0),
            "h_mid": (h_mid, 0.0, 0.0),
            "h_lo": (h_lo, 0.0, 0.0),
            "h_hi": (h_hi, 0.0, 0.0),
            "pitch_hi": (h_nom, pr[1], 0.0),
            "pitch_lo": (h_nom, pr[0], 0.0),
            "roll_hi": (h_nom, 0.0, rr[1]),
            "roll_lo": (h_nom, 0.0, rr[0]),
        }
        return table.get(key, table["mid"])

    # 速度段的机身姿态**不写**（保持重置时采样到的"通常直立"值，≈0.513 m）——
    # 写 h_mid=0.44 会把整段测成"蹲着跑"，那不是训练时的常规工况。
    neutral_body = None
    # 臂 / push 两段为了各档位可比，写一个**固定**的标称站姿（0.513 = 常规站立高度）
    nominal_body = ((min(0.513, hr[1]), 0.0, 0.0) if bp is not None else None)

    # ── schedule（一条连续轨迹里切换速度 + 机身姿态）─────────────────────
    segments = None
    sched_s = 0.0
    if args_cli.schedule == "full":
        seg_s = max(float(args_cli.switch) if float(args_cli.switch) > 0 else float(args_cli.seg_s),
                    0.2)
        segments = [(seg_s, c, neutral_body) for c in DEFAULT_SCHEDULE]
        if bp is not None:
            segments += [(seg_s, v, _body_of(k)) for v, k in DEFAULT_POSTURE_SCHEDULE]
            print("[report] schedule 含姿态段（body_pose 重采样已关，脚本手写）")
        else:
            print("[report] 该任务没有 body_pose 命令 ⇒ schedule 只切速度")
        sched_s = len(segments) * seg_s
        print(f"[report] schedule：{len(segments)} 段 × {seg_s}s = {sched_s:.1f}s"
              f"（vx/vy/wz + height/pitch/roll）")
    if bp is not None:
        # 无论哪种模式都关掉 body_pose 自己的重采样：命令要么脚本手写、要么保持重置采样值
        harness.disable_body_resample()

    # ── fig08 专用的臂测试：更长、切多个末端目标 ───────────────────────
    arm_s = 0.0
    push_sched = None
    push_grid: list[tuple[float, float]] = []
    if args_cli.push_sweep:
        for chunk in args_cli.push_sweep.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            mv, fv = (chunk.split(",") + ["1"])[:2]
            push_grid.append((float(mv), float(fv)))
    series: dict[str, dict[str, list[EpisodeData]]] = {}
    for label, ckpt in jobs:
        harness.label = label
        runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=unwrapped.device)
        print(f"[report] 滚动 `{label}` ← {ckpt}")
        series.setdefault("cmd", {})[label] = harness.collect(
            commands, args_cli.steps, args_cli.warmup, policy, term_names
        )
        if segments is not None:
            series.setdefault("sched", {})[label] = [
                harness.collect_schedule(segments, policy, term_names)
            ]
        # 臂：拉长 + 一条轨迹里切多个末端目标（用户需求 7）
        ee_cfg = getattr(env_cfg.commands, "ee_pose", None)
        if ee_cfg is not None and int(args_cli.arm_targets) > 0:
            arm_s = min(3.0 * (int(args_cli.arm_targets) + 1), 30.0)
            if hasattr(ee_cfg, "resampling_time_range"):
                ee_cfg.resampling_time_range = (arm_s / (args_cli.arm_targets + 1),) * 2
            series.setdefault("arm", {})[label] = [
                harness.collect_schedule([(arm_s, (0.0, 0.0, 0.0), nominal_body)], policy, term_names)
            ]
        # push 抗扰扫描（频率高于训练 + 力度外推）
        if push_grid and harness.push_term is not None:
            resample_iv = getattr(ee_cfg, "resampling_time_range", (5.0, 5.0)) if ee_cfg else None
            if ee_cfg is not None:
                ee_cfg.resampling_time_range = (4.0, 4.0)
            push_sched = [(4.0, c, nominal_body) for c in
                          ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.6))]
            for mv, fv in push_grid:
                harness.set_push(mv, fv)
                plabel = f"{label} push {mv:g}x{fv:g}"
                print(f"[report]   push 档位 {plabel}")
                series.setdefault("push", {})[plabel] = [
                    harness.collect_schedule(push_sched, policy, term_names)
                ]
            harness.set_push(1.0, 1.0)
            if ee_cfg is not None and resample_iv is not None:
                ee_cfg.resampling_time_range = resample_iv

    groups = series

    meta = {
        "task": task_name,
        "checkpoint_a": ckpt_a,
        "checkpoint_b": os.path.abspath(args_cli.compare) if args_cli.compare else None,
        "num_envs": int(unwrapped.num_envs),
        "steps": int(args_cli.steps),
        "warmup": int(args_cli.warmup),
        "seed": int(args_cli.seed),
        "commands": args_cli.commands,
        "no_push": bool(args_cli.no_push),
        "env_id": int(args_cli.env_id),
        "switch_s": float(args_cli.seg_s),
        "seg_s": float(args_cli.seg_s),
        "sched_s": float(sched_s),
        "arm_s": float(arm_s),
        "arm_targets": int(args_cli.arm_targets),
        "push_grid": [list(p) for p in push_grid],
        "terrain_grid": args_cli.terrain_grid,
    }

    render_all(groups, meta, args_cli.out_dir, args_cli.dpi)


if __name__ == "__main__":
    main()
    # Isaac 的 `simulation_app.close()` 在本机（Windows + A6000/A4000）会偶发挂住，
    # 报告已经落盘 ⇒ 直接退出，别让命令行卡死（与 probe_root_height_termination.py 同一处理）。
    sys.stdout.flush()
    os._exit(0)





