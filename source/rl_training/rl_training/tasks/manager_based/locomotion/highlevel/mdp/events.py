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
    支持并行多环境（已正确叠加各 env 的世界坐标原点）。

    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID
        pose_ranges: 偏移范围列表，每个元素是一个字典，包含：
            {
                "x": (min, max),      # 相对于 default_root_state 的位置偏移，单位 m
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
    num_ranges = len(pose_ranges)

    # ------------------------------------------------------------------ #
    # 1. 向量化随机选择每个环境使用哪个 pose_range
    # ------------------------------------------------------------------ #
    range_indices = torch.randint(0, num_ranges, (num_envs,), device=env.device)

    # ------------------------------------------------------------------ #
    # 2. 向量化采样各轴偏移量
    #    将所有 pose_range 的上下界预先构建为 tensor，再用 gather 索引
    # ------------------------------------------------------------------ #
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]

    # lo_table / hi_table: shape (num_ranges, 6)
    lo_table = torch.zeros(num_ranges, 6, device=env.device)
    hi_table = torch.zeros(num_ranges, 6, device=env.device)
    for r_idx, r in enumerate(pose_ranges):
        for k_idx, key in enumerate(keys):
            if key in r:
                lo_table[r_idx, k_idx] = r[key][0]
                hi_table[r_idx, k_idx] = r[key][1]
            # else 保持 0.0，即无偏移

    # 按 range_indices 取出每个 env 对应的上下界，shape: (num_envs, 6)
    lo = lo_table[range_indices]  # (num_envs, 6)
    hi = hi_table[range_indices]  # (num_envs, 6)

    # 均匀采样，shape: (num_envs, 6)
    rand = torch.rand(num_envs, 6, device=env.device)
    delta = lo + rand * (hi - lo)   # delta[:, 0~2] = xyz, delta[:, 3~5] = rpy

    # ------------------------------------------------------------------ #
    # 3. 取出对应 env_ids 的默认根状态（局部坐标系）
    #    default_root_state shape: (num_envs_total, 13)
    #    [pos(3), rot(4, wxyz), vel_lin(3), vel_ang(3)]
    # ------------------------------------------------------------------ #
    root_state = asset.data.default_root_state[env_ids].clone()

    # ------------------------------------------------------------------ #
    # 4. 位置偏移：default 局部坐标 + 随机偏移 + env 世界原点
    #    ★ 关键修复：必须加上 env.scene.env_origins[env_ids]
    #      才能把局部坐标转换到世界坐标系
    # ------------------------------------------------------------------ #
    root_state[:, 0] += delta[:, 0]                          # x
    root_state[:, 1] += delta[:, 1]                          # y
    root_state[:, 2] += delta[:, 2]                          # z
    root_state[:, :3] += env.scene.env_origins[env_ids]      # 叠加世界原点

    # ------------------------------------------------------------------ #
    # 5. 姿态偏移：提取 default 欧拉角 → 叠加偏移 → 转回四元数
    #    四元数约定：(w, x, y, z)，与 IsaacLab 一致
    # ------------------------------------------------------------------ #
    default_quat = root_state[:, 3:7]   # (num_envs, 4), wxyz
    default_roll, default_pitch, default_yaw = euler_xyz_from_quat(default_quat)

    new_roll  = default_roll  + delta[:, 3]
    new_pitch = default_pitch + delta[:, 4]
    new_yaw   = default_yaw   + delta[:, 5]

    root_state[:, 3:7] = quat_from_euler_xyz(new_roll, new_pitch, new_yaw)

    # ------------------------------------------------------------------ #
    # 6. 速度部分（index 7~12）保留 default 值，写入仿真
    # ------------------------------------------------------------------ #
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)