import torch
import numpy as np
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_discontinuous(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_ranges: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    在 default 根状态基础上，叠加不连续范围内的随机偏移来重置物体位置和姿态。

    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID
        pose_ranges: 偏移范围列表，每个元素是一个字典，包含：
            {
                "x": (min, max),      # 位置偏移，单位 m
                "y": (min, max),
                "z": (min, max),
                "roll": (min, max),   # 姿态偏移，单位 rad
                "pitch": (min, max),
                "yaw": (min, max)
            }
        asset_cfg: 物体配置
    """
    if len(env_ids) == 0:
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # 为每个环境独立随机选择一个 pose_range
    range_indices = np.random.randint(0, len(pose_ranges), size=num_envs)

    # 采样各轴偏移量，shape: (num_envs,)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    delta = {}
    for key in keys:
        values = np.zeros(num_envs, dtype=np.float32)
        for i, ridx in enumerate(range_indices):
            chosen = pose_ranges[ridx]
            if key in chosen:
                lo, hi = chosen[key]
                values[i] = np.random.uniform(lo, hi)
            # else 保持 0.0，即无偏移
        delta[key] = torch.tensor(values, device=env.device)

    # 取出对应 env_ids 的默认根状态
    # default_root_state shape: (num_envs_total, 13) [pos(3), rot(4), vel_lin(3), vel_ang(3)]
    root_state = asset.data.default_root_state[env_ids].clone()

    # --- 位置偏移：直接相加 ---
    root_state[:, 0] += delta["x"]
    root_state[:, 1] += delta["y"]
    root_state[:, 2] += delta["z"]

    # --- 姿态偏移：先提取 default 欧拉角，再叠加偏移，最后转回四元数 ---
    default_quat = root_state[:, 3:7]  # (w, x, y, z)
    default_roll, default_pitch, default_yaw = euler_xyz_from_quat(default_quat)

    new_roll  = default_roll  + delta["roll"]
    new_pitch = default_pitch + delta["pitch"]
    new_yaw   = default_yaw   + delta["yaw"]

    new_quat = quat_from_euler_xyz(new_roll, new_pitch, new_yaw)  # shape: (num_envs, 4)
    root_state[:, 3:7] = new_quat

    # 速度部分 (index 7~12) 保留 default 值，不额外修改

    # 写入仿真
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)