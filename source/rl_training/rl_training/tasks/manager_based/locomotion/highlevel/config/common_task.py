# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Shared event and camera helpers for generated high-level task scenes."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import (
    EventCfg as HighLevelEventCfg,
)


SIMPLE_ROOM_HIDDEN_PRIMS = ("/World/WBCRoom/table_low_327",)


def configure_quality_rendering(env_cfg) -> None:
    """Enable the RTX features used by the benchmark reference captures."""
    env_cfg.sim.render.rendering_mode = "quality"
    env_cfg.sim.render.enable_direct_lighting = True
    env_cfg.sim.render.enable_global_illumination = True
    env_cfg.sim.render.enable_reflections = True
    env_cfg.sim.render.enable_shadows = True
    env_cfg.sim.render.enable_dl_denoiser = True
    env_cfg.sim.render.antialiasing_mode = "DLAA"
    env_cfg.sim.render.samples_per_pixel = 4


def hide_simple_room_furniture(env, env_ids=None) -> None:
    """Remove Simple_Room's central display table from the benchmark stage."""
    del env, env_ids
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim_path in SIMPLE_ROOM_HIDDEN_PRIMS:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.SetActive(False)


@configclass
class SimpleTaskEventCfg(HighLevelEventCfg):
    """Reset only the robot; task scenes keep their authored object layout."""

    reset_object_position_discontinuous = None

    hide_simple_room_furniture = EventTerm(
        func=hide_simple_room_furniture,
        mode="startup",
    )

    reset_base = EventTerm(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.03, 0.03)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints_default = EventTerm(
        func=vel_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
