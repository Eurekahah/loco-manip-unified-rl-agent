"""Reusable dual/four fisheye rigs for teleoperation and HDF5 collection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from scipy.spatial.transform import Rotation

from fisheye_equirect import (
    build_dual_fisheye_equirect_maps,
    build_four_ideal_fisheye_equirect_maps,
    dual_camera_rotations,
    remap_dual_fisheye_to_equirect,
    remap_four_fisheye_to_equirect,
)


R_BASE_CAM0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FisheyeRigSpec:
    """Scene-level description of an optional fisheye rig."""

    mode: str
    scene_keys: tuple[str, ...]
    width: int
    height: int
    fov_deg: float
    lens_offset: float = 0.0
    with_depth: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


def _rotation_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _fisheye_spawn(width: int, height: int, fov_deg: float) -> sim_utils.FisheyeCameraCfg:
    return sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        clipping_range=(0.03, 1000.0),
        focal_length=24.0,
        focus_distance=2.0,
        f_stop=0.0,
        horizontal_aperture=20.955,
        vertical_aperture=20.955,
        fisheye_nominal_width=float(width),
        fisheye_nominal_height=float(height),
        fisheye_optical_centre_x=width * 0.5,
        fisheye_optical_centre_y=height * 0.5,
        fisheye_max_fov=float(fov_deg),
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=math.radians(fov_deg * 0.5) / (min(width, height) * 0.5),
        fisheye_polynomial_c=0.0,
        fisheye_polynomial_d=0.0,
        fisheye_polynomial_e=0.0,
        fisheye_polynomial_f=0.0,
        lock_camera=True,
    )


def _make_dual_camera(
    *,
    prim_path: str,
    pos: np.ndarray,
    rotation: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    update_period: float,
    with_depth: bool,
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
        spawn=_fisheye_spawn(width, height, fov_deg),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in pos),
            rot=_rotation_quat_wxyz(rotation),
            convention="opengl",
        ),
    )


def _make_optical_camera(
    *,
    prim_path: str,
    pos: np.ndarray,
    optical_to_base: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    update_period: float,
    with_depth: bool,
) -> CameraCfg:
    optical_to_opengl = np.diag([1.0, -1.0, -1.0])
    local_rotation = optical_to_base @ optical_to_opengl
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb", "depth"] if with_depth else ["rgb"],
        debug_vis=False,
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
        spawn=_fisheye_spawn(width, height, fov_deg),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in pos),
            rot=_rotation_quat_wxyz(local_rotation),
            convention="opengl",
        ),
    )


def _horizontal_ring_rotations() -> list[np.ndarray]:
    rotations = []
    for index in range(4):
        yaw = math.radians(90.0 * index)
        c, s = math.cos(yaw), math.sin(yaw)
        yaw_z = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotations.append(yaw_z @ R_BASE_CAM0)
    return rotations


def add_fisheye_rig(
    env_cfg,
    *,
    mode: str,
    width: int,
    height: int,
    update_period: float,
    fov_deg: float,
    mount_pos: tuple[float, float, float] | list[float],
    lens_offset: float,
    with_depth: bool = False,
) -> FisheyeRigSpec:
    """Attach the selected fisheye rig to an Isaac Lab scene config."""
    mount = np.asarray(mount_pos, dtype=np.float64)
    if mode == "none":
        return FisheyeRigSpec("none", (), width, height, fov_deg, lens_offset, with_depth)
    if mode not in {"dual", "four"}:
        raise ValueError(f"unsupported fisheye rig mode: {mode}")

    if mode == "dual":
        front_rotation, back_rotation = dual_camera_rotations(0.0)
        front_pos = mount + np.array([lens_offset, 0.0, 0.0], dtype=np.float64)
        back_pos = mount - np.array([lens_offset, 0.0, 0.0], dtype=np.float64)
        env_cfg.scene.fisheye_front = _make_dual_camera(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/M20PanoFront",
            pos=front_pos,
            rotation=front_rotation,
            width=width,
            height=height,
            fov_deg=fov_deg,
            update_period=update_period,
            with_depth=with_depth,
        )
        env_cfg.scene.fisheye_back = _make_dual_camera(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/M20PanoBack",
            pos=back_pos,
            rotation=back_rotation,
            width=width,
            height=height,
            fov_deg=fov_deg,
            update_period=update_period,
            with_depth=with_depth,
        )
        return FisheyeRigSpec(
            mode="dual",
            scene_keys=("fisheye_front", "fisheye_back"),
            width=width,
            height=height,
            fov_deg=fov_deg,
            lens_offset=lens_offset,
            with_depth=with_depth,
        )

    scene_keys = []
    for index, rotation in enumerate(_horizontal_ring_rotations()):
        scene_key = f"fisheye_cam{index}"
        scene_keys.append(scene_key)
        setattr(
            env_cfg.scene,
            scene_key,
            _make_optical_camera(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link/M20FourCam{index}",
                pos=mount,
                optical_to_base=rotation,
                width=width,
                height=height,
                fov_deg=fov_deg,
                update_period=update_period,
                with_depth=with_depth,
            ),
        )
    return FisheyeRigSpec(
        mode="four",
        scene_keys=tuple(scene_keys),
        width=width,
        height=height,
        fov_deg=fov_deg,
        lens_offset=lens_offset,
        with_depth=with_depth,
    )


class FisheyePanorama:
    """Precomputed equirectangular projection and blending for one rig mode."""

    def __init__(
        self,
        spec: FisheyeRigSpec,
        *,
        equirect_width: int,
        equirect_height: int,
        stitch_distance: float = 4.0,
    ) -> None:
        if spec.mode == "dual":
            self._maps = build_dual_fisheye_equirect_maps(
                fisheye_width=spec.width,
                fisheye_height=spec.height,
                fov_deg=spec.fov_deg,
                equirect_width=equirect_width,
                equirect_height=equirect_height,
                pitch_deg=0.0,
                lens_offset=spec.lens_offset,
                stitch_distance=stitch_distance,
                blend_deg=4.0,
            )
        elif spec.mode == "four":
            rotations_base = _horizontal_ring_rotations()
            rotations_cam0 = [R_BASE_CAM0.T @ rotation for rotation in rotations_base]
            translations_cam0 = [np.zeros(3, dtype=np.float64) for _ in range(4)]
            self._maps = build_four_ideal_fisheye_equirect_maps(
                rotations=rotations_cam0,
                translations=translations_cam0,
                fov_deg=spec.fov_deg,
                render_width=spec.width,
                render_height=spec.height,
                equirect_width=equirect_width,
                equirect_height=equirect_height,
                stitch_distance=stitch_distance,
                blend_sigma_deg=24.0,
            )
        else:
            raise ValueError("fisheye panorama requires a dual or four rig")
        self.spec = spec
        self.covered = self._maps[-1]

    def stitch(self, images: list[np.ndarray]) -> np.ndarray:
        if self.spec.mode == "dual":
            return remap_dual_fisheye_to_equirect(images[0], images[1], self._maps)
        return remap_four_fisheye_to_equirect(images, self._maps[0], self._maps[1])


def _as_rgb(data: object) -> np.ndarray:
    rgb = np.asarray(data)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"unexpected RGB shape: {rgb.shape}")
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def read_fisheye_images(env, spec: FisheyeRigSpec) -> list[np.ndarray]:
    images = []
    for scene_key in spec.scene_keys:
        sensor = env.scene.sensors[scene_key]
        rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
        images.append(_as_rgb(rgb))
    return images


class FisheyePanoramaWindow:
    """Separate omni.ui window showing the stitched equirectangular image."""

    def __init__(self, spec: FisheyeRigSpec, width: int, height: int) -> None:
        import omni.ui as ui

        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            f"M20 Fisheye Panorama - {spec.mode}",
            width=width + 24,
            height=height + 52,
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"{spec.mode} -> {width}x{height} equirectangular")
                ui.ImageWithProvider(self._provider, width=width, height=height)

    def update(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        self._provider.set_bytes_data(rgba.reshape(-1).tolist(), [width, height])

    def close(self) -> None:
        self._window.visible = False


class TeleopFisheyeDisplay:
    """Optional panorama renderer owned by a teleoperation script."""

    def __init__(
        self,
        spec: FisheyeRigSpec,
        *,
        show_panorama: bool,
        panorama_width: int,
        panorama_height: int,
    ) -> None:
        self.spec = spec
        self.panorama = None
        self.window = None
        if show_panorama and spec.enabled:
            self.panorama = FisheyePanorama(
                spec,
                equirect_width=panorama_width,
                equirect_height=panorama_height,
            )
            self.window = FisheyePanoramaWindow(spec, panorama_width, panorama_height)

    def update(self, env) -> None:
        if self.window is None or self.panorama is None:
            return
        images = read_fisheye_images(env, self.spec)
        self.window.update(self.panorama.stitch(images))

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None
