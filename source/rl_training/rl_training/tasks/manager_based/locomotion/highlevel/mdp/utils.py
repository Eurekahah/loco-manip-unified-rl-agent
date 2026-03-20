# mdp/observations.py
from __future__ import annotations
import torch
import torchvision.transforms.functional as TF



import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# Utility helpers (not exposed as mdp terms directly)
# =============================================================================

def get_asset(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> RigidObject | Articulation:
    """Resolve a SceneEntityCfg to the actual scene asset."""
    cfg.resolve(env.scene)
    return env.scene[cfg.name]


def robot_root_pos_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame root position of the robot, shape (N, 3)."""
    robot: Articulation = get_asset(env, robot_cfg)
    return robot.data.root_pos_w  # (N, 3)


def robot_root_quat_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame root quaternion (w, x, y, z) of the robot, shape (N, 4)."""
    robot: Articulation = get_asset(env, robot_cfg)
    return robot.data.root_quat_w  # (N, 4)


def object_root_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame root position of any rigid object or articulation, shape (N, 3)."""
    asset = get_asset(env, object_cfg)
    return asset.data.root_pos_w  # (N, 3)