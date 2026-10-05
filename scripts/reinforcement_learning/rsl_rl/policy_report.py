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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# 画图依赖放在 AppLauncher 之前 import：这样才能支持 `--from-npz` 的"只画图"模式
# （云端只负责采集，图回本机画 —— 云端通常没有中文字体）。
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
              "WenQuanYi Micro Hei", "DejaVu Sans"]
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
    matplotlib.rcParams["font.sans-serif"] = [_fam] + list(matplotlib.rcParams["font.sans-serif"])
    print(f"[report] 使用中文字体：{_fam} ← {_font_path}")

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

DEFAULT_COMMANDS = "0,0,0;0.5,0,0;1.0,0,0"


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
    p.add_argument("--commands", type=str, default=DEFAULT_COMMANDS, help='分号分隔的 "vx,vy,wz"')
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
        help=">0 时进入「指令切换」模式：每段这么多秒，按 --commands 切速度、再切机身高度/俯仰/侧倾"
        "（用来考察指令间的变换能力；总时长 ≈ 段数 × 该值）",
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

# 图里要用中文标签：DejaVu Sans 没有 CJK 字形（会画成方块/掉字），
# 换成系统里的中文字体（Windows 是 Microsoft YaHei / SimHei，Linux 上退化成 Noto/文泉驿）。
_CJK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
              "WenQuanYi Micro Hei", "DejaVu Sans"]
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
            self.ee_body_idx = find_one(self.robot, "body", "gripper_base")
        except Exception:  # noqa: BLE001
            self.ee_cmd = None

        # 高度/足端 cfgs（与 body_pose 奖励同一口径）
        from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

        self.asset_cfg = SceneEntityCfg("robot")
        self.feet_cfg = SceneEntityCfg("robot", body_names=".*wheel")
        self.feet_cfg.resolve(self.raw.scene)

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
        for cmd in commands:
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
                    z = hits[:, 2]
                    rec["scan"].append(np.array([float(z.min()), float(z.max()), float(z.mean())]))

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
        schedule: list = []
        term_counts = {n: 0 for n in term_names}
        t = 0.0
        obs = self._force_reset(segments[0][1], obs)
        for dur_s, vel, body in segments:
            n_steps = int(round(dur_s / self.dt))
            t0 = t
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
                done = dones.bool()
                if bool(done.any()):
                    for name in term_names:
                        term_counts[name] += int(
                            self.raw.termination_manager.get_term(name)[done].sum()
                        )
            schedule.append((t0, t, tuple(vel), None if body is None else tuple(body)))
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
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _save(fig, out_dir: str, name: str, dpi: int) -> str:
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[report] 图已保存: {path}")
    return path


def _iter_pairs(series: dict[str, list[EpisodeData]]):
    """按命令档逐个 index 对齐多个 label。"""
    labels = list(series.keys())
    n = min(len(v) for v in series.values())
    for i in range(n):
        yield labels, {lab: series[lab][i] for lab in labels}


def fig01_tracking(series, out_dir, dpi, meta):
    """速度/角速度：指令 vs 实际（逐档命令）。"""
    names = ["vx", "vy", "wz"]
    n = min(len(v) for v in series.values())
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 7.2), squeeze=False)
    for j, (labels, eps) in enumerate(_iter_pairs(series)):
        first = eps[labels[0]]
        for a, key in enumerate(("vel_b", "vel_b", "yaw_rate")):
            ax = axes[a][j]
            if a < 2:
                ax.plot(first.t, first.cmd[:, a], "k--", lw=1.2, label="cmd")
            else:
                ax.plot(first.t, first.cmd[:, 2], "k--", lw=1.2, label="cmd")
            for k, lab in enumerate(labels):
                ep = eps[lab]
                y = ep.vel_b[:, a] if a < 2 else ep.yaw_rate
                ax.plot(ep.t, y, color=COLORS[k % len(COLORS)], lw=1.0, label=lab)
            ax.set_title(f"{names[a]}  cmd={first.command}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(names[a])
            if a == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"① 速度跟踪（{meta['task']}）")
    return _save(fig, out_dir, "fig01_tracking_timeseries", dpi)


def fig02_tracking_summary(series, out_dir, dpi, meta):
    """逐档误差柱状 + 指令-实际散点（带 y=x）+ 终止构成。"""
    labels = list(series.keys())
    n = len(series[labels[0]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(n)
    w = 0.8 / len(labels)
    for k, lab in enumerate(labels):
        errs = [float(series[lab][i].err_xy.mean()) for i in range(n)]
        axes[0].bar(x + k * w, errs, w, color=COLORS[k % len(COLORS)], label=lab)
    axes[0].set_xticks(x + w * (len(labels) - 1) / 2)
    axes[0].set_xticklabels([f"{c}" for c in [e.command for e in series[labels[0]]]], fontsize=7)
    axes[0].set_ylabel("mean |v_cmd - v_actual| (m/s)")
    axes[0].set_title("逐档命令的线速度误差")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")

    for k, lab in enumerate(labels):
        for i in range(n):
            ep = series[lab][i]
            axes[1].scatter(ep.cmd[:, 0], ep.vel_b[:, 0], s=2, alpha=0.25,
                            color=COLORS[k % len(COLORS)], label=lab if i == 0 else None)
    lo = min(min(ep.cmd[:, 0].min(), ep.vel_b[:, 0].min()) for lab in labels for ep in series[lab])
    hi = max(max(ep.cmd[:, 0].max(), ep.vel_b[:, 0].max()) for lab in labels for ep in series[lab])
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_xlabel("vx cmd"); axes[1].set_ylabel("vx actual")
    axes[1].set_title("指令-实际散点（虚线 = 理想）")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    groups = sorted({g for lab in labels for ep in series[lab] for g in ep.term_counts})
    for k, lab in enumerate(labels):
        vals = [sum(ep.term_counts.get(g, 0) for ep in series[lab]) for g in groups]
        axes[2].bar(np.arange(len(groups)) + k * w, vals, w, color=COLORS[k % len(COLORS)], label=lab)
    axes[2].set_xticks(np.arange(len(groups)) + w * (len(labels) - 1) / 2)
    axes[2].set_xticklabels([g.replace("Episode_Termination/", "") for g in groups],
                            fontsize=7, rotation=20)
    axes[2].set_title("终止次数（全部命令档合计）")
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"② 跟踪汇总（{meta['task']}）")
    return _save(fig, out_dir, "fig02_tracking_summary", dpi)


def fig03_posture(series, out_dir, dpi, meta):
    """机身高度（相对足端）/俯仰/侧倾 + 高度误差直方图。"""
    n = min(len(v) for v in series.values())
    fig, axes = plt.subplots(2, n + 1, figsize=(4.2 * (n + 1), 7.0), squeeze=False)
    labels = list(series.keys())
    for j, (_, eps) in enumerate(_iter_pairs(series)):
        first = eps[labels[0]]
        if first.body_cmd is not None:
            axes[0][j].plot(first.t, first.body_cmd[:, 0], "k--", lw=1.2, label="height cmd")
            axes[1][j].plot(first.t, np.degrees(first.body_cmd[:, 1]), "k--", lw=1.0, label="pitch cmd")
            axes[1][j].plot(first.t, np.degrees(first.body_cmd[:, 2]), "k:", lw=1.0, label="roll cmd")
        else:
            axes[0][j].plot(first.t, first.height.mean() * np.ones_like(first.t), "k--", lw=1.0,
                            label="mean actual")
        for k, lab in enumerate(labels):
            ep = eps[lab]
            axes[0][j].plot(ep.t, ep.height, color=COLORS[k % len(COLORS)], lw=1.0, label=lab)
            axes[1][j].plot(ep.t, np.degrees(ep.pitch), color=COLORS[k % len(COLORS)],
                            lw=1.0, label=f"{lab} pitch")
            axes[1][j].plot(ep.t, np.degrees(ep.roll), color=COLORS[k % len(COLORS)],
                            lw=0.8, ls=":", label=f"{lab} roll")
        axes[0][j].set_title(f"机身高度 cmd={first.command}")
        axes[1][j].set_title("俯仰(实线)/侧倾(点线) [deg]")
        for a in (0, 1):
            axes[a][j].grid(alpha=0.3)
            if j == 0:
                axes[a][j].legend(fontsize=6)
    # root_z 的物理含义容易混：它是**世界系**机身高度，会被地形起伏/抬腿带着上下动；
    # 而上面第一列画的"机身高度"是**相对足端**的量（= 机身 − 足端，身体姿态控制的那个量）。
    # 用散点把两者的关系画出来最直观：纵轴 root_z、横轴"相对足端高度"。
    ax = axes[0][n]
    for k, lab in enumerate(labels):
        for ep in series[lab]:
            step = max(ep.t.size // 400, 1)
            ax.scatter(ep.height[::step], ep.root_z[::step], s=3, alpha=0.35,
                       color=COLORS[k % len(COLORS)], label=lab if ep is series[lab][0] else None)
    ax.set_xlabel("机身高度（相对足端, m）")
    ax.set_ylabel("root_z（世界系, m）")
    ax.set_title("root_z vs 相对足端高度\n（两者相差 = 足端离地；地形起伏只改 root_z）", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.suptitle(f"③ 姿态与高度（{meta['task']}）")
    return _save(fig, out_dir, "fig03_posture", dpi)


def fig04_gait(series, out_dir, dpi, meta):
    """步态图：四足触地时序 + 占空比 + 踏步主频 + 轮速。"""
    labels = list(series.keys())
    if np.size(series[labels[0]][0].contact) == 0:
        return None  # 切换模式等没记录接触数据的情形，直接跳过
    fig, axes = plt.subplots(4, 2, figsize=(13, 9), squeeze=False)
    ep = series[labels[0]][0]
    for k, leg in enumerate(LEGS):
        ax = axes[k][0]
        for m, lab in enumerate(labels):
            e = series[lab][0]
            ax.fill_between(e.t, m, m + 0.8, where=e.contact[:, k], step="mid",
                            color=COLORS[m % len(COLORS)], alpha=0.55, label=lab if k == 0 else None)
            ax.plot(e.t, e.foot_z_w[:, k] - min(0.0, e.foot_z_w[:, k].min()),
                    color=COLORS[m % len(COLORS)], lw=0.8, ls=":")
        ax.set_ylabel(leg, rotation=0, ha="right")
        ax.set_yticks([])
        ax.grid(alpha=0.25, axis="x")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title("触地条带（点线 = 轮心离地高度，已平移到 0 起）")
        if k == 3:
            ax.set_xlabel("t (s)")
    ax = axes[0][1]
    duty = np.array([duty_factor(series[lab][0].contact) for lab in labels])  # (L,4)
    xx = np.arange(4)
    w = 0.8 / max(len(labels), 1)
    for m, lab in enumerate(labels):
        ax.bar(xx + m * w, duty[m], w, color=COLORS[m % len(COLORS)], label=lab)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xx + w * (len(labels) - 1) / 2); ax.set_xticklabels(LEGS)
    ax.set_ylim(0, 1.15); ax.set_ylabel("触地占比")
    ax.set_title("占空比（轮足滚动 ≈1.0；踏步/跳跃 <1.0）")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        f = [dominant_freq(e.foot_z_w[:, k], e.t[1] - e.t[0]) for k in range(4)]
        ax.bar(np.arange(4) + m * w, f, w, color=COLORS[m % len(COLORS)], label=lab)
    ax.set_xticks(np.arange(4) + w * (len(labels) - 1) / 2); ax.set_xticklabels(LEGS)
    ax.set_ylabel("主频 (Hz)"); ax.set_title("轮心垂直运动主频（≈0 = 纯滚动）")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2][1]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        for k, leg in enumerate(LEGS):
            ax.plot(e.t, e.joint_vel[:, 12 + k], color=COLORS[m % len(COLORS)],
                    lw=0.9, alpha=0.9, ls=["-", "--", "-.", ":"][k],
                    label=f"{lab} {leg}" if m == 0 else None)
    ax.set_ylabel("轮转速 (rad/s)"); ax.set_xlabel("t (s)")
    ax.set_title("四个轮的转速（分叉 = 打滑/拖拽）")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    axes[3][1].axis("off")
    fig.suptitle(f"④ 步态（{meta['task']} / cmd={ep.command}）")
    return _save(fig, out_dir, "fig04_gait_diagram", dpi)


def fig05_joints(series, out_dir, dpi, meta):
    """12 个腿关节的位置/速度（网格）+ 4 个轮转速。"""
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
            ax.set_title(f"{leg} {jt}", fontsize=9)
            ax.grid(alpha=0.25)
            if k == 0 and a == 0:
                ax.legend(fontsize=7)
                ax.set_ylabel("pos(实线)/vel(点线)")
    fig.suptitle(f"⑤ 腿关节轨迹（{meta['task']} / cmd={series[labels[0]][0].command}）")
    return _save(fig, out_dir, "fig05_joints", dpi)


def fig06_actuation(series, out_dir, dpi, meta):
    """力矩：逐关节 RMS / 峰值占限幅比例 / 关节功率。"""
    labels = list(series.keys())
    if np.size(series[labels[0]][0].torque) == 0:
        return None
    names = series[labels[0]][0].joint_names_all
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    x = np.arange(len(names))
    w = 0.8 / max(len(labels), 1)
    for m, lab in enumerate(labels):
        e = series[lab][0]
        rms = np.sqrt((e.torque ** 2).mean(axis=0))
        axes[0][0].bar(x + m * w, rms, w, color=COLORS[m % len(COLORS)], label=lab)
        axes[1][0].bar(x + m * w, np.abs(e.torque).max(axis=0), w, color=COLORS[m % len(COLORS)])
        axes[0][1].bar(x + m * w, np.abs(e.torque * e.joint_vel).mean(axis=0), w,
                       color=COLORS[m % len(COLORS)], label=lab)
    for ax, title, ylab in (
        (axes[0][0], "力矩 RMS (N·m)", "N·m"),
        (axes[1][0], "力矩峰值 |τ|max (N·m)", "N·m"),
        (axes[0][1], "关节功率均值 |tau*dq| (W)", "W"),
    ):
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=7)
    # 峰值因子 = |tau|max / RMS（不用力矩限幅：本资产在 IsaacLab 里读到的是 1e9 占位值）
    for m, lab in enumerate(labels):
        e = series[lab][0]
        rms = np.sqrt((e.torque ** 2).mean(axis=0))
        crest = np.abs(e.torque).max(axis=0) / np.where(rms > 1e-6, rms, np.nan)
        axes[1][1].bar(x + m * w, crest, w, color=COLORS[m % len(COLORS)], label=lab)
    axes[1][1].axhline(3.0, color="k", ls="--", lw=1)
    axes[1][1].set_title("峰值因子 |tau|max / RMS（虚线 = 3，越尖的负载越不平滑）")
    axes[1][1].set_xticks(x + w * (len(labels) - 1) / 2)
    axes[1][1].set_xticklabels(names, rotation=90, fontsize=6)
    axes[1][1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"⑥ 执行器（{meta['task']} / cmd={series[labels[0]][0].command}）")
    return _save(fig, out_dir, "fig06_actuation", dpi)


def fig07_symmetry(series, out_dir, dpi, meta):
    """左右镜像：散点 + RMS 柱状 + 足端俯视图（"撇腿"角度）。"""
    labels = list(series.keys())
    if not series[labels[0]][0].mirror_rms:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ep = series[labels[0]][0]
    pairs = list(ep.mirror_rms.keys())
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
    axes[0][0].set_xticks(ticks); axes[0][0].set_xticklabels(ticklabels, fontsize=7)
    axes[0][0].set_title("镜像 RMS（越小越对称）")
    axes[0][0].set_ylabel("rad"); axes[0][0].grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=7)

    for m, lab in enumerate(labels):
        fa, ha = np.array([lateral_asymmetry(series[lab][i].foot_xy_b)
                           for i in range(len(series[lab]))]).T
        axes[0][1].scatter(np.arange(len(fa)), fa * 100, color=COLORS[m % len(COLORS)], label=f"{lab} 前")
        axes[0][1].scatter(np.arange(len(fa)), ha * 100, color=COLORS[m % len(COLORS)],
                           marker="x", label=f"{lab} 后")
    axes[0][1].axhline(0, color="k", lw=0.8)
    axes[0][1].set_title("左右不对称度 (cm) = 0 完全对称")
    axes[0][1].set_xlabel("命令档编号"); axes[0][1].grid(alpha=0.3); axes[0][1].legend(fontsize=7)

    for m, lab in enumerate(labels):
        e = series[lab][0]
        xy = e.foot_xy_b.mean(axis=0)
        axes[1][0].scatter(xy[:, 0], xy[:, 1], s=60, color=COLORS[m % len(COLORS)], label=lab)
        for k, leg in enumerate(LEGS):
            axes[1][0].annotate(leg, (xy[k, 0], xy[k, 1]), fontsize=8, xytext=(4, 4),
                                textcoords="offset points")
    axes[1][0].axhline(0, color="k", lw=0.6, ls=":")
    axes[1][0].set_xlabel("body x (m)"); axes[1][0].set_ylabel("body y (m)")
    axes[1][0].set_title("足端俯视图（均值位置）"); axes[1][0].grid(alpha=0.3)
    axes[1][0].legend(fontsize=7); axes[1][0].axis("equal")

    for m, lab in enumerate(labels):
        for i in range(len(series[lab])):
            sw = stance_width(series[lab][i].foot_xy_b)
            axes[1][1].scatter(i, sw[0] * 100, color=COLORS[m % len(COLORS)], marker="o")
            axes[1][1].scatter(i, sw[1] * 100, color=COLORS[m % len(COLORS)], marker="x")
    axes[1][1].set_title("前后轮距 (cm)：o=前 x=后")
    axes[1][1].set_xlabel("命令档编号"); axes[1][1].grid(alpha=0.3)
    fig.suptitle(f"⑦ 对称性（{meta['task']}）")
    return _save(fig, out_dir, "fig07_symmetry", dpi)


def fig08_arm(series, out_dir, dpi, meta):
    """机械臂：EE 位置/姿态跟踪误差 + 臂关节位置速度（没有臂就跳过）。"""
    labels = [lab for lab in series if series[lab][0].ee_cmd_pos_b is not None]
    if not labels:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ep = series[labels[0]][0]
    for m, lab in enumerate(labels):
        e = series[lab][0]
        err = np.linalg.norm(e.ee_cmd_pos_b - e.ee_pos_b, axis=1)
        axes[0].plot(e.t, err * 100, color=COLORS[m % len(COLORS)], lw=1.0, label=lab)
        axes[1].plot(e.t, np.degrees(e.ee_ori_err), color=COLORS[m % len(COLORS)], lw=1.0, label=lab)
        arm_cols = [c for n, c in e.col.items() if n.startswith(("arm_joint", "gripper_joint"))]
        for c in arm_cols:
            axes[2].plot(e.t, e.joint_pos[:, c], color=COLORS[m % len(COLORS)], lw=0.8,
                         alpha=0.75, label=f"{lab} {e.joint_names_all[c]}" if c == arm_cols[0] else None)
    axes[0].set_title("EE 位置跟踪误差 (cm)"); axes[0].set_xlabel("t (s)")
    axes[1].set_title("EE 姿态误差 (deg)"); axes[1].set_xlabel("t (s)")
    axes[2].set_title("臂/夹爪关节角 (rad)"); axes[2].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.suptitle(f"⑧ 机械臂（{meta['task']} / cmd={ep.command}）")
    return _save(fig, out_dir, "fig08_arm_ee", dpi)


def fig09_terrain(series, out_dir, dpi, meta):
    """地形：扫描点云俯视图 + 足端离地间隙 + 地形高度/粗糙度随时间。"""
    labels = [lab for lab in series if series[lab][0].terrain_pts_w is not None]
    if not labels:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for m, lab in enumerate(labels):
        e = series[lab][0]
        pts = e.terrain_pts_w
        base = np.asarray(e.root_xy[0], dtype=float)  # (2,)
        if pts is not None and np.size(pts):
            pts = np.asarray(pts, dtype=float).reshape(-1, 3)
            if m == 0:
                sc = axes[0].scatter(pts[:, 0] - base[0], pts[:, 1] - base[1], c=pts[:, 2],
                                     s=4, cmap="terrain")
                fig.colorbar(sc, ax=axes[0], label="terrain z (m)")
        axes[0].scatter(e.root_xy[:, 0] - base[0], e.root_xy[:, 1] - base[1],
                        color=COLORS[m % len(COLORS)], s=8, label=f"{lab} 轨迹")
        if e.scan_mean is not None:
            axes[1].plot(e.t, e.scan_min - e.scan_min.mean(), color=COLORS[m % len(COLORS)],
                         lw=0.9, label=f"{lab} scan min")
            axes[1].plot(e.t, e.scan_max - e.scan_min.mean(), color=COLORS[m % len(COLORS)],
                         lw=0.9, ls="--", label=f"{lab} scan max")
        # 离地间隙：以"每一步最低的那个足端高度"的中位数作为地面参考
        low = np.sort(np.asarray(e.foot_z_w, dtype=float), axis=1)[:, 0]
        clear = np.asarray(e.foot_z_w, dtype=float) - float(np.median(low))
        axes[2].plot(e.t, clear.mean(axis=1) * 100, color=COLORS[m % len(COLORS)], lw=1.0, label=lab)
    axes[0].set_title("地形扫描点云（俯视图，颜色 = 高度）")
    axes[0].set_xlabel("x - x0 (m)"); axes[0].set_ylabel("y - y0 (m)")
    axes[0].axis("equal"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title("地形高度随时间的波动（已去均值）")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("z (m)"); axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    axes[2].set_title("足端平均离地高度 (cm)")
    axes[2].set_xlabel("t (s)"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    fig.suptitle(f"⑨ 地形（{meta['task']}）")
    return _save(fig, out_dir, "fig09_terrain", dpi)


def fig10_compare(series, out_dir, dpi, meta):
    """两个 checkpoint 的关键指标并排（需要一个以上 label）。"""
    labels = list(series.keys())
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    keys = [
        ("err_vel_xy_mean", "线速度误差 (m/s)", False),
        ("height_std", "机身高度抖动 std (m)", False),
        ("power_mean_abs", "平均 |关节功率| (W)", False),
        ("duration_s", "记录时长 (s)", False),
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
        colLabels=["策略", "命令", "err_vel_xy", "height_std", "后不对称(cm)", "hl~hr 膝RMS"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
    path2 = _save(fig2, out_dir, "fig10_compare_table", dpi)
    fig.suptitle("⑩ A/B 对比（同 env / 同命令 / 同 seed）")
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
    metrics = [("err_xy", "线速度误差 (m/s)"), ("err_yaw", "角速度误差 (rad/s)"),
               ("height_std", "机身高度抖动 std (m)")]
    x = np.arange(len(terrs))
    w = 0.8 / max(len(cmds), 1)
    for ax, (key, title) in zip(axes[0], metrics):
        for j, ck in enumerate(cmds):
            vals = [np.mean(table[t].get(labels[0], {}).get(ck, [np.nan]))
                    for t in terrs]
            ax.bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
        ax.set_xticks(x + w * (len(cmds) - 1) / 2)
        ax.set_xticklabels(terrs, fontsize=8, rotation=15)
        ax.set_title(f"{title}（按地形）")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    for j, ck in enumerate(cmds):
        vals = [np.mean(table[t].get(labels[0], {}).get("done", [np.nan]))
                for t in terrs]
        axes[1][0].bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
    axes[1][0].set_xticks(x + w * (len(cmds) - 1) / 2)
    axes[1][0].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][0].set_title("每 env 平均终止次数（按地形）")
    axes[1][0].grid(alpha=0.3, axis="y")
    for j, ck in enumerate(cmds):
        vals = [np.mean(table[t].get(labels[0], {}).get("duty_min", [np.nan]))
                for t in terrs]
        axes[1][1].bar(x + j * w, vals, w, label=ck.replace("cmd", "cmd="))
    axes[1][1].set_xticks(x + w * (len(cmds) - 1) / 2)
    axes[1][1].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][1].set_title("最差那条腿的触地占比（按地形）")
    axes[1][1].set_ylim(0, 1.05)
    axes[1][1].grid(alpha=0.3, axis="y")
    n_env = [int(np.mean(table[t].get(labels[0], {}).get("n_env", [0]))) for t in terrs]
    axes[1][2].bar(x, n_env, 0.6)
    axes[1][2].set_xticks(x)
    axes[1][2].set_xticklabels(terrs, fontsize=8, rotation=15)
    axes[1][2].set_title("每种地形分到多少个 env（样本量）")
    axes[1][2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"⑪ 分地形（{meta['task']}）")
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
    axes[0][0].set_title("vx：指令 vs 实际（竖线 = 切换时刻）")
    axes[0][1].set_ylabel("|v_cmd - v| (m/s)"); axes[0][1].set_title("线速度误差")
    axes[1][0].set_title("机身高度：指令 vs 实际")
    axes[1][1].set_ylabel("deg")
    axes[1][1].set_title("俯仰/侧倾：指令 vs 实际")
    for ax in axes.ravel():
        ax.set_xlabel("t (s)")
    fig.suptitle(f"⑫ 指令切换（{meta['task']}，每段 {meta.get('switch_s')} s）")
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
        colLabels=["策略", "段", "速度指令", "姿态指令", "切换后 0.5s 峰值误差",
                   "稳态误差", "稳定时间(s)", "高度稳态误差"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    paths.append(_save(fig2, out_dir, "fig12_switch_table", dpi))
    return paths


# ──────────────────────────── 报告 ────────────────────────────
def write_report(out_dir: str, series: dict, meta: dict, figures: list[str]) -> dict:
    """落地 report.md / summary.json / data.npz，并返回 summary（给 stdout 用）。"""
    labels = list(series.keys())
    summary = {
        lab: [summarize(ep, ep.joint_names_all, ep.torque_limit) for ep in series[lab]]
        for lab in labels
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

    npz: dict[str, np.ndarray] = {}
    for lab in labels:
        for i, ep in enumerate(series[lab]):
            for field_name in ("t", "cmd", "vel_b", "yaw_rate", "root_xy", "root_z", "height",
                               "pitch", "roll", "joint_pos", "joint_vel", "torque", "contact",
                               "foot_xy_b", "foot_z_w"):
                npz[f"{lab}|{i}|{field_name}"] = np.asarray(getattr(ep, field_name))
            for opt in ("body_cmd", "ee_cmd_pos_b", "ee_pos_b", "ee_ori_err",
                        "scan_min", "scan_max", "scan_mean", "env_terrain_level"):
                v = getattr(ep, opt, None)
                if v is not None:
                    npz[f"{lab}|{i}|{opt}"] = np.asarray(v)
            if ep.terrain_pts_w is not None:
                npz[f"{lab}|{i}|terrain_pts_w"] = np.asarray(ep.terrain_pts_w)
            if ep.per_env is not None:
                for k, v in ep.per_env.items():
                    npz[f"{lab}|{i}|pe|{k}"] = np.asarray(v)
            if ep.env_terrain is not None:
                npz[f"{lab}|{i}|env_terrain"] = np.asarray(ep.env_terrain, dtype=object).astype("U32")
            npz[f"{lab}|{i}|meta"] = np.array([
                json.dumps({
                    "label": ep.label,
                    "command": list(ep.command),
                    "joint_names_all": ep.joint_names_all,
                    "schedule": [list(s) for s in (ep.schedule or [])],
                    "term_counts": ep.term_counts,
                    "n_done": int(ep.n_done),
                }, ensure_ascii=False)
            ])
            if ep.torque_limit is not None:
                npz[f"{lab}|{i}|torque_limit"] = np.asarray(ep.torque_limit)
    np.savez_compressed(os.path.join(out_dir, "data.npz"), **npz)
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

    terr = per_terrain_table(series)
    if terr:
        lines += ["", "## 4. 分地形（逐 env 指标按『落在哪种地形上』分组）", "",
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

    switch_rows = []
    for lab in labels:
        for ep in series[lab]:
            for r in transition_metrics(ep):
                switch_rows.append((lab, r))
    if switch_rows:
        lines += ["", "## 5. 指令切换（变换能力：每段切换后的峰值误差/稳定时间/稳态误差）", "",
                  "| 策略 | 段 | 速度指令 | 姿态指令 | 切换后 0.5 s 峰值误差 | 稳态误差 | 稳定时间 (s) | 高度稳态误差 |",
                  "|---|---|---|---|---|---|---|---|"]
        for lab, r in switch_rows:
            st = "-" if r["settle_s"] is None else f"{r['settle_s']:.2f}"
            lines.append(
                f"| {lab} | {r['seg']} | {r['vel_cmd']} | {r['body_cmd']} "
                f"| {r['peak_err_xy_0p5s']:.3f} | {r['steady_err_xy']:.3f} | {st} "
                f"| {r['steady_err_height']:.3f} |"
            )

    lines += ["", "## 6. 图（每张一个角度）", ""]
    lines += [f"* `{os.path.basename(p)}`" for p in figures]
    lines += ["", "> 原始数据在 `data.npz`，标量指标在 `summary.json`。", ""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary


# ──────────────────────────── 主流程 ────────────────────────────
def load_series_npz(npz_path: str) -> tuple[dict, dict]:
    """把 `data.npz` 读回 `(series, meta)`（配合 `--from-npz` 做"云端采集 + 本机画图"）。"""
    z = np.load(npz_path, allow_pickle=True)
    meta_path = os.path.join(os.path.dirname(os.path.abspath(npz_path)), "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    series: dict[str, list[EpisodeData]] = {}
    seen: dict[str, set] = {}
    for key in z.files:
        parts = key.split("|")
        if len(parts) < 3:
            continue
        lab, idx, field = parts[0], int(parts[1]), "|".join(parts[2:])
        seen.setdefault(lab, set()).add(idx)
    for lab, idxs in seen.items():
        series[lab] = []
        for i in sorted(idxs):
            info = {}
            if f"{lab}|{i}|meta" in z.files:
                info = json.loads(str(z[f"{lab}|{i}|meta"][0]))
            ep = EpisodeData(
                label=info.get("label", lab),
                command=tuple(info.get("command", (0.0, 0.0, 0.0))),
            )
            for field_name in ("t", "cmd", "vel_b", "yaw_rate", "root_xy", "root_z", "height",
                               "pitch", "roll", "joint_pos", "joint_vel", "torque", "contact",
                               "foot_xy_b", "foot_z_w"):
                if f"{lab}|{i}|{field_name}" in z.files:
                    setattr(ep, field_name, np.asarray(z[f"{lab}|{i}|{field_name}"]))
            for opt in ("body_cmd", "ee_cmd_pos_b", "ee_pos_b", "ee_ori_err",
                        "scan_min", "scan_max", "scan_mean", "env_terrain_level"):
                if f"{lab}|{i}|{opt}" in z.files:
                    setattr(ep, opt, np.asarray(z[f"{lab}|{i}|{opt}"]))
            if f"{lab}|{i}|torque_limit" in z.files:
                ep.torque_limit = np.asarray(z[f"{lab}|{i}|torque_limit"], dtype=float)
            if f"{lab}|{i}|terrain_pts_w" in z.files:
                ep.terrain_pts_w = np.asarray(z[f"{lab}|{i}|terrain_pts_w"])
            if f"{lab}|{i}|env_terrain" in z.files:
                ep.env_terrain = [str(x) for x in np.asarray(z[f"{lab}|{i}|env_terrain"])]
            pe = {}
            for k in z.files:
                pref = f"{lab}|{i}|pe|"
                if k.startswith(pref):
                    pe[k[len(pref):]] = np.asarray(z[k])
            ep.per_env = pe or None
            ep.joint_names_all = list(info.get("joint_names_all", []))
            if not ep.joint_names_all and ep.joint_pos.ndim == 2 and ep.joint_pos.shape[1] >= 16:
                # 老版本 npz 没存列名 ⇒ 按本脚本的固定列序重建（12 腿 + 4 轮 [+ 臂/夹爪]）
                names = [f"{leg}_{jt}_joint" for leg in LEGS for jt in JOINT_TYPES]
                names += [f"{leg}_wheel_joint" for leg in LEGS]
                extra = ep.joint_pos.shape[1] - len(names)
                names += [f"extra_joint{i}" for i in range(extra)]
                ep.joint_names_all = names
            ep.col = {n: j for j, n in enumerate(ep.joint_names_all)}
            ep.schedule = [tuple(x) for x in info.get("schedule", [])] or None
            ep.term_counts = info.get("term_counts", {})
            ep.n_done = int(info.get("n_done", 0))
            if ep.joint_pos.size and ep.joint_names_all:
                ep.mirror_rms = mirror_rms(ep.joint_pos, ep.joint_names_all)
            series[lab].append(ep)
    print(f"[report] 从 {npz_path} 读回 {sum(len(v) for v in series.values())} 段数据"
          f"（labels={list(series)}）")
    return series, meta


def render_all(series: dict, meta: dict, out_dir: str, dpi: int) -> dict:
    """把 series 渲染成"每角度一张图 + report.md + summary.json + data.npz"。"""
    os.makedirs(out_dir, exist_ok=True)
    figures: list[str] = []
    for fn in (fig01_tracking, fig02_tracking_summary, fig03_posture, fig04_gait, fig05_joints,
               fig06_actuation, fig07_symmetry, fig08_arm, fig09_terrain, fig10_compare,
               fig11_per_terrain, fig12_switch):
        if fn is fig10_compare and len(series) < 2:
            continue
        try:
            out = fn(series, out_dir, dpi, meta)
        except Exception as exc:  # noqa: BLE001 - 一张图失败不该毁掉整份报告
            print(f"[report][WARN] {fn.__name__} 画图失败：{type(exc).__name__}: {exc}")
            continue
        if out is None:
            continue
        figures += out if isinstance(out, list) else [out]
    summary = write_report(out_dir, series, meta, figures)
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
        series, meta = load_series_npz(args_cli.from_npz)
        render_all(series, meta, args_cli.out_dir, args_cli.dpi)
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
    series: dict[str, list[EpisodeData]] = {}
    segments = None
    if args_cli.switch > 0:
        seg_s = float(args_cli.switch)
        bp = getattr(env_cfg.commands, "body_pose", None)
        hr = tuple(getattr(bp, "height_range", (0.513, 0.513)))
        pr = tuple(getattr(bp, "pitch_range", (0.0, 0.0)))
        rr = tuple(getattr(bp, "roll_range", (0.0, 0.0)))
        h_mid, h_lo, h_hi = 0.5 * (hr[0] + hr[1]), hr[0], hr[1]
        segments = [(seg_s, c, (h_mid, 0.0, 0.0)) for c in commands]
        segments += [
            (seg_s, (0.0, 0.0, 0.0), (h_hi, 0.0, 0.0)),   # 抬升机身
            (seg_s, (0.0, 0.0, 0.0), (h_lo, 0.0, 0.0)),   # 压低机身
            (seg_s, (0.0, 0.0, 0.0), (h_mid, pr[1], 0.0)),  # 抬头
            (seg_s, (0.0, 0.0, 0.0), (h_mid, 0.0, rr[1])),  # 侧倾
        ]
        harness.disable_body_resample()
        print(f"[report] 指令切换模式：{len(segments)} 段 × {seg_s}s "
              f"= {len(segments) * seg_s:.1f}s（速度 {len(commands)} 段 + 姿态 4 段）")
    for label, ckpt in jobs:
        harness.label = label
        runner = OnPolicyRunnerHis(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=unwrapped.device)
        print(f"[report] 滚动 `{label}` ← {ckpt}")
        if segments is not None:
            series[label] = [harness.collect_schedule(segments, policy, term_names)]
        else:
            series[label] = harness.collect(
                commands, args_cli.steps, args_cli.warmup, policy, term_names
            )

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
        "switch_s": float(args_cli.switch),
        "terrain_grid": args_cli.terrain_grid,
    }

    render_all(series, meta, args_cli.out_dir, args_cli.dpi)


if __name__ == "__main__":
    main()
    # Isaac 的 `simulation_app.close()` 在本机（Windows + A6000/A4000）会偶发挂住，
    # 报告已经落盘 ⇒ 直接退出，别让命令行卡死（与 probe_root_height_termination.py 同一处理）。
    sys.stdout.flush()
    os._exit(0)





