from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


TOOLS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from fisheye_equirect import (  # noqa: E402
    build_dual_fisheye_equirect_maps,
    build_four_ideal_fisheye_equirect_maps,
    remap_dual_fisheye_to_equirect,
    remap_four_fisheye_to_equirect,
)
from fisheye_gpu import GpuFisheyePanorama  # noqa: E402


R_BASE_CAM0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _synthetic_images(count: int, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            np.stack(
                [
                    (xx * (index + 1)) % 256,
                    (yy * (index + 2)) % 256,
                    ((xx + yy + index * 31) * 2) % 256,
                ],
                axis=-1,
            )
            for index in range(count)
        ],
        axis=0,
    ).astype(np.uint8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dual_gpu_stitch_matches_cpu() -> None:
    width, height = 160, 100
    panorama_width, panorama_height = 192, 96
    maps = build_dual_fisheye_equirect_maps(
        fisheye_width=width,
        fisheye_height=height,
        fov_deg=200.0,
        equirect_width=panorama_width,
        equirect_height=panorama_height,
        lens_offset=0.02,
    )
    images = _synthetic_images(2, height, width)
    cpu = remap_dual_fisheye_to_equirect(images[0], images[1], maps)
    gpu = GpuFisheyePanorama.from_dual_maps(
        maps,
        source_width=width,
        source_height=height,
        panorama_width=panorama_width,
        panorama_height=panorama_height,
    ).stitch(torch.from_numpy(images).cuda())
    gpu = gpu.cpu().numpy()

    difference = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    assert gpu.shape == cpu.shape
    assert float(difference.mean()) < 1.0
    assert int(difference.max()) <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_four_gpu_stitch_matches_cpu() -> None:
    width, height = 160, 100
    panorama_width, panorama_height = 192, 96
    rotations = []
    for index in range(4):
        yaw = np.deg2rad(90.0 * index)
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_z = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotations.append(R_BASE_CAM0.T @ yaw_z @ R_BASE_CAM0)
    maps = build_four_ideal_fisheye_equirect_maps(
        rotations=rotations,
        translations=[np.zeros(3) for _ in range(4)],
        fov_deg=200.0,
        render_width=width,
        render_height=height,
        equirect_width=panorama_width,
        equirect_height=panorama_height,
    )
    images = _synthetic_images(4, height, width)
    cpu = remap_four_fisheye_to_equirect(list(images), maps[0], maps[1])
    gpu = GpuFisheyePanorama.from_four_maps(
        maps,
        source_width=width,
        source_height=height,
        panorama_width=panorama_width,
        panorama_height=panorama_height,
    ).stitch(torch.from_numpy(images).cuda())
    gpu = gpu.cpu().numpy()

    difference = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    assert gpu.shape == cpu.shape
    assert float(difference.mean()) < 1.0
    assert float(np.quantile(difference, 0.999)) <= 12.0
