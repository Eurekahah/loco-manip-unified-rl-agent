"""Reusable dual/four fisheye rigs for teleoperation and HDF5 collection."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

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
from fisheye_gpu import GpuFisheyePanorama


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
    """Precomputed equirectangular projection with GPU and CPU backends."""

    def __init__(
        self,
        spec: FisheyeRigSpec,
        *,
        equirect_width: int,
        equirect_height: int,
        stitch_distance: float = 4.0,
        backend: str = "auto",
        device: torch.device | str | None = None,
    ) -> None:
        if backend not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"unsupported panorama backend: {backend}")
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
        self.backend = backend
        self.gpu = None
        if backend != "cpu" and torch.cuda.is_available():
            if spec.mode == "dual":
                self.gpu = GpuFisheyePanorama.from_dual_maps(
                    self._maps,
                    source_width=spec.width,
                    source_height=spec.height,
                    panorama_width=equirect_width,
                    panorama_height=equirect_height,
                    device=device,
                )
            else:
                self.gpu = GpuFisheyePanorama.from_four_maps(
                    self._maps,
                    source_width=spec.width,
                    source_height=spec.height,
                    panorama_width=equirect_width,
                    panorama_height=equirect_height,
                    device=device,
                )
            self.backend = "gpu"
        elif backend == "gpu":
            raise RuntimeError("GPU panorama requested but CUDA is not available")
        else:
            self.backend = "cpu"

    def stitch(self, images: list[np.ndarray]) -> np.ndarray:
        """Stitch NumPy RGB images on CPU."""
        if self.spec.mode == "dual":
            return remap_dual_fisheye_to_equirect(images[0], images[1], self._maps)
        return remap_four_fisheye_to_equirect(images, self._maps[0], self._maps[1])

    def stitch_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Stitch ``(N, H, W, 3)`` images on GPU without CPU readback."""
        if self.gpu is None:
            raise RuntimeError("GPU panorama backend is not active")
        return self.gpu.stitch(images)


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


def _as_rgb_tensor(data: torch.Tensor) -> torch.Tensor:
    rgb = data
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"unexpected RGB tensor shape: {tuple(rgb.shape)}")
    return rgb.contiguous()


def read_fisheye_images(env, spec: FisheyeRigSpec) -> list[np.ndarray]:
    images = []
    for scene_key in spec.scene_keys:
        sensor = env.scene.sensors[scene_key]
        rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
        images.append(_as_rgb(rgb))
    return images


def read_fisheye_tensors(
    env,
    spec: FisheyeRigSpec,
    *,
    env_id: int = 0,
) -> torch.Tensor:
    """Return all fisheye RGB frames as one CUDA tensor shaped ``(N, H, W, 3)``."""
    images = []
    for scene_key in spec.scene_keys:
        sensor = env.scene.sensors[scene_key]
        rgb = sensor.data.output["rgb"][env_id]
        images.append(_as_rgb_tensor(rgb))
    return torch.stack(images, dim=0)


def _sensor_frame(env, scene_key: str, env_id: int = 0) -> int:
    frame = env.scene.sensors[scene_key].frame
    if isinstance(frame, torch.Tensor):
        return int(frame[env_id].item())
    return int(frame)


class CameraUiWindow:
    """Throttled image window backed directly by an existing Isaac camera."""

    def __init__(
        self,
        title: str,
        *,
        width: int,
        height: int,
        position: tuple[int, int],
    ) -> None:
        import omni.ui as ui

        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            title,
            width=width + 20,
            height=height + 44,
            position_x=int(position[0]),
            position_y=int(position[1]),
        )
        with self._window.frame:
            ui.ImageWithProvider(self._provider, width=width, height=height)

    def update(self, rgb: np.ndarray) -> None:
        rgb = _as_rgb(rgb)
        height, width = rgb.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        self._provider.set_data_array(np.ascontiguousarray(rgba), [width, height])

    def close(self) -> None:
        self._window.visible = False


class FisheyePanoramaWindow:
    """Separate omni.ui window showing the stitched equirectangular image."""

    def __init__(
        self,
        spec: FisheyeRigSpec,
        width: int,
        height: int,
        *,
        position: tuple[int, int] = (980, 80),
    ) -> None:
        import omni.ui as ui

        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(
            f"M20 Fisheye Panorama - {spec.mode}",
            width=width + 24,
            height=height + 52,
            position_x=int(position[0]),
            position_y=int(position[1]),
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"{spec.mode} -> {width}x{height} equirectangular")
                ui.ImageWithProvider(self._provider, width=width, height=height)

    def update(self, rgb: np.ndarray) -> None:
        rgb = _as_rgb(rgb)
        height, width = rgb.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        self._provider.set_data_array(np.ascontiguousarray(rgba), [width, height])

    def close(self) -> None:
        self._window.visible = False


class TeleopFisheyeDisplay:
    """Throttled wrist/panorama UI windows and optional panorama recording source."""

    def __init__(
        self,
        spec: FisheyeRigSpec,
        *,
        show_panorama: bool,
        panorama_width: int,
        panorama_height: int,
        show_wrist_camera: bool = False,
        wrist_scene_key: str = "wrist_camera",
        wrist_display_hz: float = 20.0,
        panorama_display_hz: float = 12.0,
        preview_hz: float = 15.0,
        panorama_backend: str = "auto",
        wrist_window_width: int = 320,
        wrist_window_height: int = 320,
        wrist_window_pos: tuple[int, int] = (640, 80),
        panorama_window_pos: tuple[int, int] = (980, 80),
        record_panorama: bool = False,
        profiler=None,
        ui_snapshot_dir: str | None = None,
    ) -> None:
        self.spec = spec
        self.panorama = None
        self.panorama_window = None
        self.wrist_window = None
        self.wrist_scene_key = wrist_scene_key
        self.wrist_display_hz = float(wrist_display_hz)
        self.panorama_display_hz = float(panorama_display_hz)
        self._profiler = profiler
        self._last_wrist_update = -float("inf")
        self._last_panorama_update = -float("inf")
        self._last_wrist_frame = -1
        self._last_panorama_frame = -1
        self._last_wrist_rgb: np.ndarray | None = None
        self._last_panorama_rgb: np.ndarray | None = None
        self._wrist_updates = 0
        self._panorama_updates = 0
        self.ui_snapshot_dir = ui_snapshot_dir
        self._panorama_cache: tuple[tuple[int, ...], np.ndarray] | None = None

        if (show_panorama or record_panorama) and spec.enabled:
            self.panorama = FisheyePanorama(
                spec,
                equirect_width=panorama_width,
                equirect_height=panorama_height,
                backend=panorama_backend,
            )
        if show_panorama and self.panorama is not None:
            self.panorama_window = FisheyePanoramaWindow(
                spec,
                panorama_width,
                panorama_height,
                position=panorama_window_pos,
            )
        if show_wrist_camera:
            self.wrist_window = CameraUiWindow(
                "M20 Wrist Camera",
                width=wrist_window_width,
                height=wrist_window_height,
                position=wrist_window_pos,
            )

    def update(self, env) -> None:
        now = time.perf_counter()
        if self.wrist_window is not None:
            frame = _sensor_frame(env, self.wrist_scene_key)
            due = now - self._last_wrist_update >= 1.0 / self.wrist_display_hz
            if due and frame != self._last_wrist_frame:
                if self._profiler is None:
                    self._update_wrist(env)
                else:
                    with self._profiler.measure("wrist_ui_ms"):
                        self._update_wrist(env)
                self._last_wrist_update = now
                self._last_wrist_frame = frame

        if self.panorama_window is not None and self.panorama is not None:
            frames = tuple(_sensor_frame(env, key) for key in self.spec.scene_keys)
            due = now - self._last_panorama_update >= 1.0 / self.panorama_display_hz
            if due and frames != self._last_panorama_frame:
                if self._profiler is None:
                    self._update_panorama_window(env)
                else:
                    with self._profiler.measure("panorama_ui_ms"):
                        self._update_panorama_window(env)
                self._last_panorama_update = now
                self._last_panorama_frame = frames

    def _update_wrist(self, env) -> None:
        sensor = env.scene.sensors[self.wrist_scene_key]
        rgb = sensor.data.output["rgb"][0]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = _as_rgb(rgb)
        self.wrist_window.update(rgb)
        self._last_wrist_rgb = rgb.copy()
        self._wrist_updates += 1

    def _update_panorama_window(self, env) -> None:
        if self.panorama.gpu is not None:
            if self._profiler is None:
                images = read_fisheye_tensors(env, self.spec)
                stitched = self.panorama.stitch_tensor(images)
                panorama = stitched.cpu().numpy()
            else:
                with self._profiler.measure("camera_capture_ms"):
                    images = read_fisheye_tensors(env, self.spec)
                with self._profiler.measure("gpu_stitch_ms"):
                    stitched = self.panorama.stitch_tensor(images)
                with self._profiler.measure("gpu_to_cpu_ms"):
                    panorama = stitched.cpu().numpy()
        else:
            if self._profiler is not None:
                with self._profiler.measure("camera_capture_ms"):
                    images_cpu = read_fisheye_images(env, self.spec)
            else:
                images_cpu = read_fisheye_images(env, self.spec)
            if self._profiler is not None:
                with self._profiler.measure("cpu_stitch_ms"):
                    panorama = self.panorama.stitch(images_cpu)
            else:
                panorama = self.panorama.stitch(images_cpu)
        self.panorama_window.update(panorama)
        self._last_panorama_rgb = np.ascontiguousarray(panorama).copy()
        self._panorama_updates += 1

    def capture_panorama(self, env) -> np.ndarray:
        """Return the current stitched panorama for optional HDF5 recording."""
        if self.panorama is None:
            raise RuntimeError("panorama capture is not configured")
        frames = tuple(_sensor_frame(env, key) for key in self.spec.scene_keys)
        if self._panorama_cache is not None and self._panorama_cache[0] == frames:
            return self._panorama_cache[1]
        if self.panorama.gpu is not None:
            images = read_fisheye_tensors(env, self.spec)
            panorama = self.panorama.stitch_tensor(images).cpu().numpy()
        else:
            panorama = self.panorama.stitch(read_fisheye_images(env, self.spec))
        panorama = np.ascontiguousarray(panorama, dtype=np.uint8)
        self._panorama_cache = (frames, panorama)
        return panorama

    def close(self) -> None:
        if self.panorama_window is not None:
            self.panorama_window.close()
            self.panorama_window = None
        if self.wrist_window is not None:
            self.wrist_window.close()
            self.wrist_window = None

    def status(self) -> dict:
        return {
            "wrist_window_visible": bool(
                self.wrist_window is not None and self.wrist_window._window.visible
            ),
            "panorama_window_visible": bool(
                self.panorama_window is not None and self.panorama_window._window.visible
            ),
            "wrist_updates": self._wrist_updates,
            "panorama_updates": self._panorama_updates,
            "wrist_shape": None
            if self._last_wrist_rgb is None
            else list(self._last_wrist_rgb.shape),
            "panorama_shape": None
            if self._last_panorama_rgb is None
            else list(self._last_panorama_rgb.shape),
            "panorama_backend": None if self.panorama is None else self.panorama.backend,
        }

    def save_ui_snapshots(self) -> dict:
        """Save exactly the last images supplied to the omni.ui providers."""
        status = self.status()
        if not self.ui_snapshot_dir:
            return status
        import cv2

        output_dir = os.path.abspath(os.path.expanduser(self.ui_snapshot_dir))
        os.makedirs(output_dir, exist_ok=True)
        if self._last_wrist_rgb is not None:
            wrist_path = os.path.join(output_dir, "wrist_ui.png")
            cv2.imwrite(wrist_path, cv2.cvtColor(self._last_wrist_rgb, cv2.COLOR_RGB2BGR))
            status["wrist_snapshot"] = wrist_path
        if self._last_panorama_rgb is not None:
            panorama_path = os.path.join(output_dir, "panorama_ui.png")
            cv2.imwrite(
                panorama_path,
                cv2.cvtColor(self._last_panorama_rgb, cv2.COLOR_RGB2BGR),
            )
            status["panorama_snapshot"] = panorama_path
        main_path = os.path.join(output_dir, "main_viewport.png")
        try:
            if capture_active_viewport(main_path):
                status["main_viewport_snapshot"] = main_path
        except Exception as exc:
            status["main_viewport_capture_error"] = str(exc)
        with open(os.path.join(output_dir, "ui_status.json"), "w", encoding="utf-8") as handle:
            json.dump(status, handle, ensure_ascii=False, indent=2)
        return status


def capture_active_viewport(path: str) -> bool:
    """Capture the active desktop viewport while pumping a few Kit updates."""
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    capture = capture_viewport_to_file(viewport, file_path=path)
    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return True
        if capture is not None and getattr(capture, "is_done", False):
            break
    return os.path.isfile(path) and os.path.getsize(path) > 0
