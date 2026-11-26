# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 平地场景（train / play / stand 共用）。

* 平地：``terrain_type="plane"``，``collision_group=-1``；
* 机器人：``AS2W_PIPERX_CFG``（资产符号已硬改名）；
* 接触传感器：覆盖 ``Robot/.*``，``update_period=0.005``；
* **没有** height_scanner：平地 ``slope_gate_rad=0`` ⇒ 扫描器完全用不到（rough 阶段再加）；
* 灯光：dome + sun，沿用 stand cfg。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from rl_training.assets.deeprobotics import AS2W_PIPERX_CFG


@configclass
class AS2WPiperXSceneCfg(InteractiveSceneCfg):
    """平地 + AS2W-PiperX + 灯光 + 接触传感器。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = AS2W_PIPERX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0)),
    )
    sun = AssetBaseCfg(
        prim_path="/World/Light/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.9), angle=0.6),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.653, 0.271, 0.271, 0.653)),
    )