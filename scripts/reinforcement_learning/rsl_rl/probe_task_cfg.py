"""probe：**只解析任务配置**，不创建 env —— 用于本机跑不了 terrain generator 的任务（DEF-031）。

验证 4 件事（静态、秒级）：
  1. env cfg 类名 + 地形生成器是否存在、是否开课程；
  2. 各观测组的 term 函数名（确认 privileged=privileged_env_params_v4、history=正确的那一个）；
  3. agent cfg 类名 + 实验名 + 教师锚/探索口径超参；
  4. 打印 play 任务的关键命令/权重，确认 PLAY 覆盖生效。

用法（必须 --headless）::

    python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py \
        --task Rough-Slopes-History-Anchor-Deeprobotics-M20-v0 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="仅解析 task cfg（不建 env）")
parser.add_argument("--task", required=True, help="任务 id")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rl_training.tasks  # noqa: F401,E402


def _fn_name(f) -> str:
    if isinstance(f, str):
        return f
    return getattr(f, "__name__", type(f).__name__)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    print("=" * 78)
    print(f"env_cfg      : {type(env_cfg).__name__}")
    tg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    print(f"terrain_gen  : {type(tg).__name__ if tg is not None else None}"
          f"  curriculum={getattr(tg, 'curriculum', None)}  size={getattr(tg, 'size', None)}")
    for gname in ("policy", "critic", "history", "privileged"):
        grp = getattr(env_cfg.observations, gname, None)
        if grp is None:
            print(f"obs[{gname:<10}]: None")
            continue
        terms = []
        for n, t in vars(grp).items():
            if isinstance(t, ObsTerm):
                terms.append(f"{n}={_fn_name(t.func)}" + (f"[h={t.history_length}]" if t.history_length else ""))
        print(f"obs[{gname:<10}]: {', '.join(terms)}")
    print(f"agent_cfg    : {type(agent_cfg).__name__}  experiment={agent_cfg.experiment_name}")
    camera_names = [
        name for name, value in vars(env_cfg.scene).items() if isinstance(value, TiledCameraCfg)
    ]
    recorder_names = [
        name
        for name, value in vars(env_cfg.recorders).items()
        if name not in {
            "dataset_file_handler_class_type",
            "dataset_filename",
            "dataset_export_dir_path",
            "dataset_export_mode",
            "export_in_record_pre_reset",
            "export_in_close",
        }
        and value is not None
    ]
    print(f"cameras      : {camera_names}")
    print(f"recorders    : {recorder_names}")
    p, a = agent_cfg.policy, agent_cfg.algorithm
    print(f"  teacher_recon_hidden_dims={getattr(p, 'teacher_recon_hidden_dims', None)}"
          f"  teacher_recon_coef={getattr(a, 'teacher_recon_coef', None)}")
    print(f"  entropy_coef={a.entropy_coef}  max_noise_std={getattr(p, 'max_noise_std', None)}"
          f"  history_length={getattr(p, 'history_length', None)}")
    print(f"  obs_groups={agent_cfg.obs_groups}")
    print("=" * 78)


if __name__ == "__main__":
    main()
    os._exit(0)
