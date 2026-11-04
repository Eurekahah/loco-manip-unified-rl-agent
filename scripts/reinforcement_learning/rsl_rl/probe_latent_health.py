# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""特权/历史编码器"健康度"探针：只靠 checkpoint，不用起 Isaac。

为什么需要它
    `Loss/constraint` / `Loss/latent_distance` 这类量**可以靠"两边一起变常数"骗到 ~0**：
    实测那条 20k run 在 λ 一开（it≈3000）时两条 loss 一步掉 80~97%，但那是特权编码器
    （教师）被拉向学生、**输入敏感度掉到原来的 4%** 的结果
    （见 docs/history_encoder_loss_cliff_zh.md）。所以判"这个模块到底有没有在编码环境信息"，
    必须看与损失无关的量：

    * **输入敏感度** ``||dz_priv/dx||_F``：与输入分布无关。好的教师应当**不随时间塌到 ~0**；
    * **跨环境分散度**：按域随机化范围采样一批特权向量，看 z_priv 的逐维 std 与有效秩；
    * 顺带打印 ``mean|z_priv|``，便于识别"输出被整体压平"。

用法::

    python scripts/reinforcement_learning/rsl_rl/probe_latent_health.py \
        --run logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k

    python .../probe_latent_health.py --run <run> --iters 3000 3500 5000 19999 --batch 2048

不启动 Isaac Sim（纯 torch）。特权向量的采样区间取自 `velocity_env_cfg.py: EventCfg`
（负载/惯量/PD 增益/轮子摩擦），布局与 `mdp/observations.py` 的特权项顺序一致。
**这是代表性采样、不是真实 batch**：跨 checkpoint 对比有效，绝对值只作量级参考。
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

#: 特权向量布局假设（与 WBCObservationsCfg.PrivilegedCfg 的项顺序一致）：
#:   base_extra_payload(1, add in [-1, 3] kg)
#:   end_effector_payload(1, scale in [1, 3])
#:   inertia_scale(?, scale in [0.85, 1.15])
#:   gain_scale(?, scale in [0.85, 1.15])
#:   material_properties(4 轮 x 3: 静摩擦[0.3,1.0] / 动摩擦[0.3,0.8] / 恢复[0,0.4])
LAYOUT = [
    ("base_extra_payload", 1, -1.0, 3.0),
    ("end_effector_payload", 1, 1.0, 3.0),
    ("inertia_scale", None, 0.85, 1.15),
    ("gain_scale", None, 0.85, 1.15),
]
MATERIAL_WHEELS, MATERIAL_SLOTS = 4, 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="特权编码器健康度探针（敏感度 / 分散度 / 有效秩）")
    p.add_argument("--run", required=True, help="run 目录（含 model_*.pt）")
    p.add_argument("--iters", nargs="*", type=int, default=None, help="要测的迭代号；不给就等间隔取 9 个")
    p.add_argument("--batch", type=int, default=1024, help="采样条数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jac-batch", type=int, default=16, help="算雅可比用的样本数（反传开销小）")
    return p.parse_args()


def list_checkpoints(run: str, iters: list[int] | None) -> list[tuple[int, str]]:
    found: dict[int, str] = {}
    for path in glob.glob(os.path.join(run, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    if not found:
        raise SystemExit(f"{run} 下没有 model_*.pt")
    if iters:
        missing = [i for i in iters if i not in found]
        if missing:
            print(f"[warn] 缺这些迭代的 checkpoint：{missing}")
        return [(i, found[i]) for i in sorted(iters) if i in found]
    keys = sorted(found)
    grid = [keys[round(k * (len(keys) - 1) / 8)] for k in range(9)]
    return [(i, found[i]) for i in dict.fromkeys(grid)]


def make_privileged_encoder(sd: dict) -> torch.nn.Module:
    """按 state_dict 形状重建特权编码器的 MLP（纯 Linear+ELU，不依赖仓库代码版本）。"""
    layers, i = [], 0
    while f"privileged_encoder.encoder.{i}.weight" in sd:
        w = sd[f"privileged_encoder.encoder.{i}.weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(sd[f"privileged_encoder.encoder.{i}.bias"])
        layers += [lin, torch.nn.ELU()]
        i += 2
    return torch.nn.Sequential(*layers)  # 保留末尾 ELU，与网络定义一致


def sample_privileged(dim: int, batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    fixed = sum(n for _, n, *_ in LAYOUT if n is not None) + MATERIAL_WHEELS * MATERIAL_SLOTS
    rest = dim - fixed
    assert rest >= 0, f"特权维度 {dim} 小于布局假设 {fixed}，请检查 LAYOUT"
    second = rest // 2
    widths = [1, 1, second, rest - second]
    cols = []
    for (_, _, lo, hi), width in zip(LAYOUT, widths):
        cols.append(lo + (hi - lo) * torch.rand(batch, width, generator=g))
    for lo, hi in ((0.3, 1.0), (0.3, 0.8), (0.0, 0.4)):
        cols.append(lo + (hi - lo) * torch.rand(batch, MATERIAL_WHEELS, generator=g))
    return torch.cat(cols, dim=1)


def health(enc: torch.nn.Module, x: torch.Tensor, jac_batch: int) -> dict:
    with torch.no_grad():
        z = enc(x)
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    eff_rank = float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))
    xj = x[:jac_batch].clone().requires_grad_(True)
    zj = enc(xj)
    acc = torch.zeros(xj.shape[0])
    for k in range(zj.shape[1]):
        g = torch.autograd.grad(zj[:, k].sum(), xj, retain_graph=True)[0]
        acc += g.pow(2).sum(dim=1)
    return {
        "spread": float(zc.std(0).mean()),
        "eff_rank": eff_rank,
        "mean_norm": float(z.norm(dim=1).mean()),
        "jac": float(acc.sqrt().mean()),
    }


def main() -> int:
    args = parse_args()
    ckpts = list_checkpoints(args.run, args.iters)
    first = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)["model_state_dict"]
    dim = first["privileged_encoder.encoder.0.weight"].shape[1]
    x = sample_privileged(dim, args.batch, args.seed)
    print(f"# {os.path.basename(os.path.abspath(args.run))}")
    print(f"  特权维度={dim}  batch={args.batch}  jac_batch={args.jac_batch}  layout={[n for n, *_ in LAYOUT]}")
    print("  spread=跨环境逐维 std 均值；eff_rank=协方差参与比；jac=||dz/dx||_F（与输入分布无关）")
    print(f"  {'iter':>7} {'spread':>10} {'eff_rank':>9} {'mean|z|':>9} {'jac':>9} {'jac/首行':>9}")
    base = None
    for it, path in ckpts:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        enc = make_privileged_encoder(sd)
        h = health(enc, x, args.jac_batch)
        if base is None:
            base = h["jac"]
        ratio = h["jac"] / base if base else float("nan")
        print(f"  {it:>7} {h['spread']:>10.4f} {h['eff_rank']:>9.2f} {h['mean_norm']:>9.3f} "
              f"{h['jac']:>9.4f} {ratio:>8.2f}x")
    print()
    print("判据：修好之后 jac 应当**保持同一量级**（不塌）；若掉到首行的几个百分点，")
    print("      说明特权编码器退化成近似常数函数，那条 loss 的下降是假信号。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
