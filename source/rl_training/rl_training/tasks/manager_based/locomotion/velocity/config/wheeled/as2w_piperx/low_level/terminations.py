# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层终止组（交接单 §9.7）。

``bad_orientation_2`` 定义在 ``velocity/mdp/events.py``（不是 terminations），通过
``mdp`` 命名空间导出。``minimum_height`` 取 Phase 0 实测最低稳定高度（0.332 m）减余量。
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.as2w_piperx.profile import (
    AS2W_PIPERX_PROFILE,
)


_PROFILE = AS2W_PIPERX_PROFILE


@configclass
class TerminationsCfg:
    """超时 / 出界 / 倾角 / 过低。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_buffer": 3.0,
        },
    )

    bad_orientation_2 = DoneTerm(
        func=mdp.bad_orientation_2,
        params={"limit_angle": 0.8},
    )

    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": _PROFILE.termination_height,
        },
    )