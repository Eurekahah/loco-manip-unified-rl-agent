# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .arm_rewards import * # noqa: F401, F403
from .actions import *  # noqa: F401, F403
from .terrains import *  # noqa: F401, F403

# --------------------------------------------------------------------------- #
# 关于"星号导入造成的同名遮蔽"（原 known_issues 工程债，2026-09-30 复核）
# --------------------------------------------------------------------------- #
# 本模块的星号导入顺序决定 `mdp.<name>` 的最终解析结果，所以**下面这些名字是故意覆盖
# 官方实现的**（本仓库的 cfg 全部按"用本仓库版本"写的；复核命令：把本目录与
# `isaaclab/envs/mdp/*.py` 的顶层定义名做集合交集）：
#
#   rewards：track_lin_vel_xy_exp / track_ang_vel_z_exp / base_height_l2 / lin_vel_z_l2 /
#            ang_vel_xy_l2 / flat_orientation_l2 / undesired_contacts
#   events ：（无）—— `randomize_rigid_body_inertia` / `randomize_com_positions` 是**本仓库
#            新增**的，IsaacLab 5.1 官方 events.py 里没有这两个名字（官方叫
#            `randomize_rigid_body_com`），所以不构成遮蔽。
#
# 唯一**意外**的同名冲突是地形：本目录 `terrains.py` 里那份"原始混合地形"原先也叫
# `ROUGH_TERRAINS_CFG`，与 `isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`（官方，
# 被 `velocity_env_cfg.py` 显式 import 后用于 terrain_generator）同名 ⇒ 已改名为
# `MIXED_TERRAINS_CFG`（`TERRAIN_CFGS["mixed"]` 不变）。详见 DEFECT_LOG_zh.md DEF-037。
