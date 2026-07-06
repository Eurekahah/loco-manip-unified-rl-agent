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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
拆分后的地形配置。

原始的 ROUGH_TERRAINS_CFG 把多种地形按比例混在同一个 TerrainGeneratorCfg 里，
无法单独测试机器人在某一种地形上的表现。这里把每种地形拆成独立的
TerrainGeneratorCfg（proportion=1.0），并额外加了一个纯平地配置，
方便逐个地形做专项测试。

使用方式：
    from <your_pkg>.terrains_split_cfg import TERRAIN_CFGS
    cfg = TERRAIN_CFGS["stairs"]   # 或 "flat" / "boxes" / ...
"""

# --------------------------------------------------------------------------- #
#  公共参数（与原始 ROUGH_TERRAINS_CFG 保持一致）
# --------------------------------------------------------------------------- #
_COMMON_KW = dict(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
)

# --------------------------------------------------------------------------- #
#  1. 正楼梯 (pyramid_stairs)
# --------------------------------------------------------------------------- #
STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""正向金字塔楼梯地形（上台阶）。"""

# --------------------------------------------------------------------------- #
#  2. 倒楼梯 (pyramid_stairs_inv)
# --------------------------------------------------------------------------- #
STAIRS_INV_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""倒金字塔楼梯地形（下台阶/坑）。"""

# --------------------------------------------------------------------------- #
#  3. 方块地形 (boxes)
# --------------------------------------------------------------------------- #
BOXES_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
    },
)
"""随机方块/网格地形。"""

# --------------------------------------------------------------------------- #
#  4. 随机粗糙地形 (random_rough)
# --------------------------------------------------------------------------- #
RANDOM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)
"""随机粗糙地形（高度场噪声）。"""

# --------------------------------------------------------------------------- #
#  5. 金字塔斜坡 (hf_pyramid_slope)
# --------------------------------------------------------------------------- #
SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""上坡（金字塔斜坡）地形。"""

# --------------------------------------------------------------------------- #
#  6. 倒金字塔斜坡 (hf_pyramid_slope_inv)
# --------------------------------------------------------------------------- #
SLOPE_INV_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""下坡（倒金字塔斜坡）地形。"""

# --------------------------------------------------------------------------- #
#  7. 平地（新增，用于基线测试）
# --------------------------------------------------------------------------- #
FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)
"""纯平地地形，用作基线（无地形干扰）对照测试。"""

# --------------------------------------------------------------------------- #
#  8. 原始混合地形（保留，便于对比/回归测试）
# --------------------------------------------------------------------------- #
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""原始的混合地形配置（保留，未做改动）。"""

# --------------------------------------------------------------------------- #
#  9. 无楼梯混合地形
# --------------------------------------------------------------------------- #
NONE_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON_KW,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
    },
)
# --------------------------------------------------------------------------- #
#  方便测试脚本按名字索引
# --------------------------------------------------------------------------- #
TERRAIN_CFGS = {
    "flat": FLAT_TERRAIN_CFG,
    "stairs": STAIRS_TERRAIN_CFG,
    "stairs_inv": STAIRS_INV_TERRAIN_CFG,
    "boxes": BOXES_TERRAIN_CFG,
    "random_rough": RANDOM_ROUGH_TERRAIN_CFG,
    "slope": SLOPE_TERRAIN_CFG,
    "slope_inv": SLOPE_INV_TERRAIN_CFG,
    "mixed": ROUGH_TERRAINS_CFG,
}
"""名称 -> TerrainGeneratorCfg 的映射，供测试脚本 --terrain 参数使用。"""