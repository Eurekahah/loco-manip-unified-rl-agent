"""GPU fisheye-to-equirectangular projection for teleoperation previews."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


def _float_map_xy(map_pair: tuple[np.ndarray, np.ndarray | None]) -> np.ndarray:
    """Convert an OpenCV fixed-point remap pair back to float ``(x, y)`` pixels."""
    map_x, map_y = cv2.convertMaps(map_pair[0], map_pair[1], cv2.CV_32FC2)
    return np.asarray(map_x, dtype=np.float32)


def normalized_grid(
    map_pair: tuple[np.ndarray, np.ndarray | None],
    *,
    source_width: int,
    source_height: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a ``grid_sample`` grid and validity mask for one fisheye remap.

    The returned grid has shape ``(1, H, W, 2)`` and uses the normalized
    ``x, y`` convention required by :func:`torch.nn.functional.grid_sample`.
    """
    if source_width <= 1 or source_height <= 1:
        raise ValueError("grid_sample requires source dimensions greater than one")

    pixel_xy = _float_map_xy(map_pair)
    valid = (
        (pixel_xy[..., 0] >= 0.0)
        & (pixel_xy[..., 0] <= source_width - 1)
        & (pixel_xy[..., 1] >= 0.0)
        & (pixel_xy[..., 1] <= source_height - 1)
    )

    grid = np.empty_like(pixel_xy, dtype=np.float32)
    grid[..., 0] = 2.0 * pixel_xy[..., 0] / max(source_width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * pixel_xy[..., 1] / max(source_height - 1, 1) - 1.0
    grid[~valid] = 0.0
    grid_tensor = torch.from_numpy(grid).unsqueeze(0).to(device=device)
    valid_tensor = torch.from_numpy(valid).unsqueeze(0).to(device=device)
    return grid_tensor, valid_tensor


class GpuFisheyePanorama:
    """Batch GPU projection and weighted blending for a dual or four rig."""

    def __init__(
        self,
        map_pairs: Sequence[tuple[np.ndarray, np.ndarray | None]],
        weights: np.ndarray,
        *,
        source_width: int,
        source_height: int,
        panorama_width: int,
        panorama_height: int,
        device: torch.device | str | None = None,
    ) -> None:
        if len(map_pairs) not in {2, 4}:
            raise ValueError("expected two or four fisheye map pairs")
        if weights.shape != (len(map_pairs), panorama_height, panorama_width):
            raise ValueError(
                f"expected weights {(len(map_pairs), panorama_height, panorama_width)}, "
                f"got {weights.shape}"
            )

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.source_width = int(source_width)
        self.source_height = int(source_height)
        self.panorama_width = int(panorama_width)
        self.panorama_height = int(panorama_height)
        self.num_cameras = len(map_pairs)

        grids = []
        masks = []
        for map_pair in map_pairs:
            grid, valid = normalized_grid(
                map_pair,
                source_width=self.source_width,
                source_height=self.source_height,
                device=self.device,
            )
            grids.append(grid)
            masks.append(valid)

        self.grid = torch.cat(grids, dim=0).contiguous()
        valid_mask = torch.cat(masks, dim=0)
        weight_tensor = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(self.device)
        self.weights = (weight_tensor * valid_mask.to(weight_tensor.dtype)).unsqueeze(1).contiguous()

    @classmethod
    def from_dual_maps(
        cls,
        maps: tuple[
            tuple[np.ndarray, np.ndarray | None],
            tuple[np.ndarray, np.ndarray | None],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        *,
        source_width: int,
        source_height: int,
        panorama_width: int,
        panorama_height: int,
        device: torch.device | str | None = None,
    ) -> GpuFisheyePanorama:
        return cls(
            [maps[0], maps[1]],
            np.stack([maps[2], maps[3]], axis=0),
            source_width=source_width,
            source_height=source_height,
            panorama_width=panorama_width,
            panorama_height=panorama_height,
            device=device,
        )

    @classmethod
    def from_four_maps(
        cls,
        maps: tuple[list[tuple[np.ndarray, np.ndarray | None]], np.ndarray, np.ndarray],
        *,
        source_width: int,
        source_height: int,
        panorama_width: int,
        panorama_height: int,
        device: torch.device | str | None = None,
    ) -> GpuFisheyePanorama:
        return cls(
            maps[0],
            maps[1],
            source_width=source_width,
            source_height=source_height,
            panorama_width=panorama_width,
            panorama_height=panorama_height,
            device=device,
        )

    def stitch(self, images: torch.Tensor) -> torch.Tensor:
        """Stitch images shaped ``(N, H, W, 3)`` and return ``(Hp, Wp, 3)`` uint8."""
        if images.ndim != 4 or images.shape[0] != self.num_cameras or images.shape[-1] != 3:
            raise ValueError(f"expected {(self.num_cameras, 'H', 'W', 3)} images, got {tuple(images.shape)}")
        if tuple(images.shape[1:3]) != (self.source_height, self.source_width):
            raise ValueError(
                f"expected source {(self.source_height, self.source_width)}, got {tuple(images.shape[1:3])}"
            )

        images = images.to(device=self.device)
        if images.is_floating_point():
            images = images * 255.0
        images = images.to(dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        sampled = functional.grid_sample(
            images,
            self.grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        stitched = (sampled * self.weights).sum(dim=0)
        return stitched.clamp_(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).contiguous()

    def stitch_to_cpu(self, images: torch.Tensor) -> np.ndarray:
        """Run GPU stitching and copy only the final panorama back to CPU."""
        return self.stitch(images).cpu().numpy()
