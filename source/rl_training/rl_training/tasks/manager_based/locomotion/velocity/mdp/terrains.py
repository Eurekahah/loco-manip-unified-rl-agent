# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for custom terrains."""
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# NOTE: this config enumerates (almost) every sub-terrain type that Isaac Lab ships with,
# purely so you can visually inspect what each one looks like. Proportions are kept roughly
# equal (they don't need to sum to exactly 1.0 - they're normalized internally), so each
# terrain gets about the same number of rows/cols in the generated grid.
# Increase num_rows/num_cols if you want more repeats of each type to see height-range variation.

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=48,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ---- flat ----
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
        # ---- stairs ----
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ---- grid / random boxes ----
        "boxes_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        # ---- height-field based ----
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # ---- rails ----
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.05,
            rail_thickness_range=(0.05, 0.2),
            rail_height_range=(0.05, 0.3),
            platform_width=2.0,
        ),
        # ---- pit ----
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05,
            pit_depth_range=(0.1, 0.5),
            platform_width=2.0,
            double_pit=False,
        ),
        # ---- box (pyramid-like) ----
        "box": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.05,
            box_height_range=(0.1, 0.5),
            platform_width=2.0,
            double_box=False,
        ),
        # ---- gap ----
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
        ),
        # ---- floating ring ----
        "floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.05,
            ring_width_range=(0.3, 0.8),
            ring_height_range=(0.1, 0.4),
            ring_thickness=0.2,
            platform_width=2.0,
        ),
        # ---- star ----
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.05,
            num_bars=8,
            bar_width_range=(0.3, 0.6),
            bar_height_range=(0.1, 0.4),
            platform_width=2.0,
        ),
        # ---- repeated objects ----
        "repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=10, height=0.1, radius=0.3
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=20, height=0.3, radius=0.4
            ),
            platform_width=2.0,
        ),
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=10, height=0.1, size=(0.3, 0.3)
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=20, height=0.3, size=(0.4, 0.4)
            ),
            platform_width=2.0,
        ),
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=10, height=0.1, radius=0.2
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20, height=0.3, radius=0.3
            ),
            platform_width=2.0,
        ),
    },
)
"""All-terrains configuration (for visually inspecting every terrain type Isaac Lab provides)."""