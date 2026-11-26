from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def compute_base_height_rel_to_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    wheel_radius: float = 0.09,
) -> torch.Tensor:
    """base 相对足端底部(平均)的高度，单位 m。

    ``wheel_radius`` 是该机型的轮半径（轮心离地高度）。默认 0.09 保持 M20 行为；
    轮式机器人必须按资产实测值传入（AS2W-PiperX 实测 0.0774 m）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_h_w = asset.data.body_pos_w[:, feet_cfg.body_ids, 2]  # (N, num_feet)
    base_h_w = asset.data.root_pos_w[:, 2]                     # (N,)
    return base_h_w - feet_h_w.mean(dim=1) + float(wheel_radius)  # (N,)
