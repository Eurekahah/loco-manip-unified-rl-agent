# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
#
"""把低层 checkpoint 导出成「高层 replay 可直接加载」的 TorchScript 部署策略。

为什么需要它
------------
``play.py`` 走的是 ``export_policy_as_jit(policy_nn.actor)``：**只导出 actor**。
对于 ``ActorCriticHistory``（ROA / RMA 风格、带 history encoder）的策略，
actor 的输入是 ``policy_obs + latent``，而 latent 由**另一个模块**
``history_encoder`` 从 10 步本体感受历史里算出来。只导出 actor 会得到一个
「要 108 维输入、但那 32 维没人给」的模型，高层 replay 无法使用。

本脚本导出的是**部署态**包装（等价于 ``ActorCriticHistory.act_inference``）：

    forward(policy_obs, history_flat) -> action
        latent = history_encoder(history_flat)     # history_length 步展平
        return actor(cat([policy_obs, latent], -1))

同时写 ``policy_layout.json`` 描述需要的输入维度，让 replay 侧显式校验而不是猜。

用法
----
    python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
        --run logs/rsl_rl/history_adaptation/2026-09-18_19-33-47 \
        --checkpoint model_7500.pt

不传 ``--checkpoint`` 时取该 run 里迭代号最大的一个。**不需要启动 Isaac Sim**（纯 torch）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.nn as nn
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RSL_RL_ROOT = os.path.join(REPO_ROOT, "rsl_rl")
if RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, RSL_RL_ROOT)

from rsl_rl.networks.history_encoder import HistoryEncoder  # noqa: E402
from rsl_rl.networks.mlp import MLP  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _load_agent_cfg(run_dir: str) -> dict:
    """读取 ``params/agent.yaml`` 的 ``policy`` 段。

    Isaac Lab 的 ``dump_yaml`` 会把 tuple 写成 ``!!python/tuple``，SafeLoader 默认不认，
    所以这里只给这一个标签注册构造器（不用 ``unsafe_load``）。
    """
    path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.exists(path):
        return {}

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
    )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_Loader) or {}
    return cfg.get("policy", {}) or {}


def _linear_chain(sd: dict, prefix: str) -> list[tuple[int, int]]:
    """返回 ``[(out_dim, in_dim), ...]``，按层号排序。"""
    found = []
    for key, tensor in sd.items():
        m = re.fullmatch(re.escape(prefix) + r"(\d+)\.weight", key)
        if m:
            found.append((int(m.group(1)), (int(tensor.shape[0]), int(tensor.shape[1]))))
    found.sort(key=lambda x: x[0])
    if not found:
        raise KeyError(f"state_dict 里找不到任何 {prefix}<i>.weight")
    return [shape for _, shape in found]


def _build_mlp(sd: dict, prefix: str, activation: str) -> MLP:
    chain = _linear_chain(sd, prefix)
    mlp = MLP(chain[0][1], chain[-1][0], [out for out, _ in chain[:-1]], activation)
    mlp.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    return mlp


def _conv_layer_indices(sd: dict) -> list[int]:
    idx = {
        int(m.group(1))
        for k in sd
        if (m := re.fullmatch(r"history_encoder\.conv\.(\d+)\.weight", k))
    }
    return sorted(idx)


# ---------------------------------------------------------------------------
# 部署态包装
# ---------------------------------------------------------------------------


class DeployHistoryPolicy(nn.Module):
    """``act_inference`` 的等价实现（不含 privileged 分支，部署时也不需要它）。

    输入:
        policy_obs    : (B, policy_obs_dim)
        history_flat  : (B, history_length * history_single_step_dim)
    输出:
        action        : (B, action_dim)
    """

    def __init__(self, actor: nn.Module, history_encoder: nn.Module):
        super().__init__()
        self.actor = actor
        self.history_encoder = history_encoder

    def forward(self, policy_obs: torch.Tensor, history_flat: torch.Tensor) -> torch.Tensor:
        latent = self.history_encoder(history_flat)
        return self.actor(torch.cat([policy_obs, latent], dim=-1))


class DeployActorPolicy(nn.Module):
    """普通 ``ActorCritic`` 的部署态包装。

    刻意保持与 ``play.py`` 的 ``_TorchPolicyExporter`` 相同的结构
    （``actor`` + ``normalizer``），这样"读 ``policy.actor`` 取维度"之类的
    既有代码不用区分是哪种导出。
    """

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor
        self.normalizer = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def build_export_module(sd: dict, policy_cfg: dict):
    """按 state_dict + agent.yaml 重建导出模块，并给出 layout 描述。"""
    activation = policy_cfg.get("activation", "elu")
    actor = _build_mlp(sd, "actor.", activation)

    if not any(k.startswith("history_encoder.") for k in sd):
        module = DeployActorPolicy(actor).eval()
        chain = _linear_chain(sd, "actor.")
        return module, {
            "kind": "actor",
            "policy_obs_dim": chain[0][1],
            "action_dim": chain[-1][0],
            "activation": activation,
        }

    conv_idx = _conv_layer_indices(sd)
    num_single_step_obs = int(sd["history_encoder.conv.0.weight"].shape[1])
    latent_dim = int(policy_cfg.get("latent_dim", sd["history_encoder.head.0.weight"].shape[0] // 2))
    history_length = policy_cfg.get("history_length")
    if history_length is None:
        raise ValueError("agent.yaml 里没有 policy.history_length，请用 --history_length 指定。")
    history_length = int(history_length)

    # conv 结构：优先用 agent.yaml，缺失时从 state_dict 推断（stride 不在 state_dict 里）
    hidden_channels = policy_cfg.get("history_encoder_hidden_channels") or [
        int(sd[f"history_encoder.conv.{i}.weight"].shape[0]) for i in conv_idx
    ]
    kernel_sizes = policy_cfg.get("history_encoder_kernel_sizes") or [
        int(sd[f"history_encoder.conv.{i}.weight"].shape[2]) for i in conv_idx
    ]
    strides = policy_cfg.get("history_encoder_strides") or [1] * len(conv_idx)
    print(
        f"[export] history_encoder: single_step={num_single_step_obs} "
        f"len={history_length} latent={latent_dim} "
        f"channels={tuple(hidden_channels)} kernels={tuple(kernel_sizes)} strides={tuple(strides)}"
    )
    history_encoder = HistoryEncoder(
        num_single_step_obs=num_single_step_obs,
        history_length=history_length,
        latent_dim=latent_dim,
        hidden_channels=tuple(hidden_channels),
        kernel_sizes=tuple(kernel_sizes),
        strides=tuple(strides),
        activation=activation,
    )
    # load_state_dict 会在 shape 不符时报错 —— 顺带校验 conv 结构推断是否正确
    history_encoder.load_state_dict(
        {k[len("history_encoder."):]: v for k, v in sd.items() if k.startswith("history_encoder.")}
    )

    chain = _linear_chain(sd, "actor.")
    module = DeployHistoryPolicy(actor, history_encoder).eval()
    return module, {
        "kind": "history",
        "policy_obs_dim": chain[0][1] - latent_dim,
        "history_single_step_dim": num_single_step_obs,
        "history_length": history_length,
        "latent_dim": latent_dim,
        "action_dim": chain[-1][0],
        "activation": activation,
    }


def _latest_checkpoint(run_dir: str) -> str:
    found = []
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"model_(\d+)\.pt", name)
        if m:
            found.append((int(m.group(1)), name))
    if not found:
        raise FileNotFoundError(f"{run_dir} 里没有 model_<iter>.pt")
    return max(found)[1]


# ---------------------------------------------------------------------------


def _cross_check_with_rsl_rl(sd, policy_cfg, layout, scripted, activation) -> None:
    """可选自检：与 rsl_rl 的 ``ActorCriticHistory.act_inference`` 比数值。"""
    if layout["kind"] != "history":
        return
    try:
        from rsl_rl.modules.actor_critic_history import ActorCriticHistory
    except ImportError as e:
        print(f"[export] 跳过 act_inference 对比（{e}）")
        return

    conv_idx = _conv_layer_indices(sd)
    critic_chain = _linear_chain(sd, "critic.")
    priv_chain = _linear_chain(sd, "privileged_encoder.encoder.")
    obs_groups = {
        "policy": ["policy"], "critic": ["critic"],
        "history": ["history"], "privileged": ["privileged"],
    }
    batch = 4
    fake_obs = {
        "policy": torch.zeros(batch, layout["policy_obs_dim"]),
        "critic": torch.zeros(batch, critic_chain[0][1]),
        "history": torch.zeros(
            batch, layout["history_length"] * layout["history_single_step_dim"]
        ),
        "privileged": torch.zeros(batch, priv_chain[0][1]),
    }
    policy = ActorCriticHistory(
        obs=fake_obs,
        obs_groups=obs_groups,
        num_actions=layout["action_dim"],
        noise_std_type=policy_cfg.get("noise_std_type", "log"),
        latent_dim=layout["latent_dim"],
        history_length=layout["history_length"],
        history_encoder_hidden_channels=tuple(
            policy_cfg.get("history_encoder_hidden_channels")
            or [int(sd[f"history_encoder.conv.{i}.weight"].shape[0]) for i in conv_idx]
        ),
        history_encoder_kernel_sizes=tuple(
            policy_cfg.get("history_encoder_kernel_sizes")
            or [int(sd[f"history_encoder.conv.{i}.weight"].shape[2]) for i in conv_idx]
        ),
        history_encoder_strides=tuple(
            policy_cfg.get("history_encoder_strides") or [1] * len(conv_idx)
        ),
        privileged_encoder_hidden_dims=tuple(
            policy_cfg.get("privileged_encoder_hidden_dims")
            or [out for out, _ in priv_chain[:-1]]
        ),
        actor_hidden_dims=[out for out, _ in _linear_chain(sd, "actor.")[:-1]],
        critic_hidden_dims=[out for out, _ in critic_chain[:-1]],
        activation=activation,
    )
    # 注意：本仓库的 ActorCriticHistory.load_state_dict 被覆写成返回 bool，不能解包
    policy.load_state_dict(sd, strict=True)
    print("[export] 已把 checkpoint 全部权重 strict=True 载入参考 ActorCriticHistory")
    torch.manual_seed(0)
    po = torch.randn(batch, layout["policy_obs_dim"])
    hf = torch.randn(batch, layout["history_length"] * layout["history_single_step_dim"])
    with torch.no_grad():
        y_ref = policy.act_inference({**fake_obs, "policy": po, "history": hf})
        y_ours = scripted(po, hf)
    err = float((y_ref - y_ours).abs().max())
    print(f"[export] 与 rsl_rl act_inference 的最大误差: {err:.3e}")
    if err > 1e-5:
        raise RuntimeError("导出的部署策略与 rsl_rl ActorCriticHistory.act_inference 不一致")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run 目录 logs/rsl_rl/<exp>/<timestamp>")
    parser.add_argument("--checkpoint", default=None, help="模型文件名；默认取迭代号最大的")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="默认 <run>/exported_deploy"
        "（**不要**用 play.py 的 <run>/exported：那里是 actor-only，缺 history encoder）",
    )
    args = parser.parse_args()

    run_dir = args.run
    ckpt_name = args.checkpoint or _latest_checkpoint(run_dir)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    out_dir = args.out_dir or os.path.join(run_dir, "exported_deploy")
    os.makedirs(out_dir, exist_ok=True)

    # 部署态导出必须落在 exported_deploy/：play.py 的 exported/policy.pt 是 actor-only
    # （只有 actor，history 策略缺 encoder），拿去 sim2sim 会静默算错。
    actor_only_dir = os.path.join(run_dir, "exported")
    if os.path.basename(os.path.normpath(out_dir)) == "exported":
        print(
            "[export] ⚠️  --out_dir 指向 <run>/exported，那是 play.py 的 actor-only 导出目录。"
            "默认的 <run>/exported_deploy 才是部署态（含 history encoder）。"
        )
    elif os.path.exists(os.path.join(actor_only_dir, "policy.pt")):
        print(
            f"[export] 提示：{actor_only_dir} 下已有一份 play.py 导出的 actor-only policy.pt，"
            "本次写出的是它的部署态版本（本目录），两者不要混用。"
        )

    policy_cfg = _load_agent_cfg(run_dir)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["model_state_dict"]

    module, layout = build_export_module(sd, policy_cfg)
    layout["source_checkpoint"] = ckpt_name
    layout["source_iteration"] = payload.get("iter")
    layout["source_run"] = os.path.relpath(run_dir).replace("\\", "/")
    if policy_cfg.get("class_name"):
        layout["source_policy_class"] = policy_cfg["class_name"]

    scripted = torch.jit.script(module)
    policy_path = os.path.join(out_dir, "policy.pt")
    scripted.save(policy_path)
    layout_path = os.path.join(out_dir, "policy_layout.json")
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    print(f"[export] checkpoint  : {ckpt_path} (iter={layout['source_iteration']})")
    print(f"[export] kind        : {layout['kind']}")
    print(f"[export] 输入维度    : policy_obs={layout['policy_obs_dim']}"
          + (f", history={layout['history_length']} x {layout['history_single_step_dim']}"
             if layout["kind"] == "history" else ""))
    print(f"[export] 输出维度    : action={layout['action_dim']}")
    print(f"[export] 写出        : {policy_path}")
    print(f"[export] 写出        : {layout_path}")

    torch.manual_seed(0)
    if layout["kind"] == "history":
        po = torch.randn(8, layout["policy_obs_dim"])
        hf = torch.randn(8, layout["history_length"] * layout["history_single_step_dim"])
        with torch.no_grad():
            err = float((module(po, hf) - scripted(po, hf)).abs().max())
    else:
        po = torch.randn(8, layout["policy_obs_dim"])
        with torch.no_grad():
            err = float((module(po) - scripted(po)).abs().max())
    print(f"[export] scripted vs eager 最大误差: {err:.3e}")
    if err > 1e-5:
        raise RuntimeError("TorchScript 导出与 eager 前向不一致")

    _cross_check_with_rsl_rl(sd, policy_cfg, layout, scripted, layout["activation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
