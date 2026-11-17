"""Utilities for converting fisheye and omnidirectional images to equirectangular."""

from __future__ import annotations

import math

import cv2
import numpy as np


def camera_rotation_from_pitch(pitch_deg: float) -> np.ndarray:
    """Return the camera local-to-base rotation for a forward-looking camera.

    The base frame is assumed to use +X forward, +Y left, +Z up.  The USD
    camera uses -Z forward and +Y up.  A positive pitch rotates the optical
    axis downward.
    """
    pitch = math.radians(pitch_deg)
    c, s = math.cos(pitch), math.sin(pitch)

    # Local camera axes expressed in the base frame:
    # +X_camera = -Y_base, +Y_camera = +Z_base, +Z_camera = -X_base.
    forward_camera = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pitch_about_y = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return pitch_about_y @ forward_camera


def dual_camera_rotations(pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return front/back camera local-to-rig rotations for an Insta360-style rig.

    Both cameras stay upright.  The back camera is rotated 180 degrees about
    the rig +Z axis and then pitched downward.
    """
    pitch = math.radians(pitch_deg)
    c, s = math.cos(pitch), math.sin(pitch)
    ry_front = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    ry_back = np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )
    forward_camera = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    yaw_180 = np.diag([-1.0, -1.0, 1.0])
    return ry_front @ forward_camera, ry_back @ yaw_180 @ forward_camera


def build_equirect_maps(
    *,
    fisheye_width: int,
    fisheye_height: int,
    fov_deg: float,
    equirect_width: int,
    equirect_height: int,
    pitch_deg: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Build OpenCV remap tables from an equirectangular canvas to a fisheye image.

    The panorama convention is centered on the base +X axis:

    * longitude increases to the right of the image,
    * latitude increases toward the top of the image,
    * the fisheye is an ideal equidistant model with its image circle inscribed
      in the rendered frame.

    Returns ``(cv_maps, valid_mask)``.  Invalid directions are mapped to -1 so
    ``cv2.remap`` fills them with the border value.
    """
    if fisheye_width <= 0 or fisheye_height <= 0:
        raise ValueError("fisheye resolution must be positive")
    if equirect_width <= 0 or equirect_height <= 0:
        raise ValueError("equirectangular resolution must be positive")
    if not 0.0 < fov_deg <= 360.0:
        raise ValueError("fov_deg must be in (0, 360]")

    # Pixel centers, so the panorama center is exactly the forward direction.
    u = (np.arange(equirect_width, dtype=np.float64) + 0.5) / equirect_width
    v = (np.arange(equirect_height, dtype=np.float64) + 0.5) / equirect_height
    lon = (u - 0.5) * (2.0 * math.pi)
    lat = (0.5 - v) * math.pi
    lon, lat = np.meshgrid(lon, lat)

    cos_lat = np.cos(lat)
    # Positive longitude is to the right, which is base -Y.
    bearings = np.stack(
        [
            cos_lat * np.cos(lon),
            -cos_lat * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )

    rotation = camera_rotation_from_pitch(pitch_deg)
    camera_dirs = bearings @ rotation
    camera_dirs /= np.maximum(np.linalg.norm(camera_dirs, axis=-1, keepdims=True), 1e-12)

    # USD camera looks along -Z_camera.  Image +X is camera +X and image +Y
    # points down, i.e. camera -Y.
    cos_theta = np.clip(-camera_dirs[..., 2], -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(-camera_dirs[..., 1], camera_dirs[..., 0])

    half_fov = math.radians(fov_deg) * 0.5
    radius = theta / half_fov * (min(fisheye_width, fisheye_height) * 0.5)
    cx = (fisheye_width - 1) * 0.5
    cy = (fisheye_height - 1) * 0.5

    map_x = (cx + radius * np.cos(phi)).astype(np.float32)
    map_y = (cy + radius * np.sin(phi)).astype(np.float32)
    valid = theta <= half_fov
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), valid


def remap_fisheye_to_equirect(
    fisheye_rgb: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Sample a fisheye RGB image into an equirectangular RGB image."""
    if fisheye_rgb.ndim != 3 or fisheye_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {fisheye_rgb.shape}")
    return cv2.remap(
        fisheye_rgb,
        maps[0],
        maps[1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _project_fisheye(
    bearings: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    stitch_distance: float,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Project equirectangular bearings into one fisheye lens."""
    points = bearings * stitch_distance - translation
    camera_dirs = points @ rotation
    camera_dirs /= np.maximum(np.linalg.norm(camera_dirs, axis=-1, keepdims=True), 1e-12)

    cos_theta = np.clip(-camera_dirs[..., 2], -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(-camera_dirs[..., 1], camera_dirs[..., 0])

    half_fov = math.radians(fov_deg) * 0.5
    radius = theta / half_fov * (min(width, height) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    map_x = (cx + radius * np.cos(phi)).astype(np.float32)
    map_y = (cy + radius * np.sin(phi)).astype(np.float32)
    valid = theta <= half_fov
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), theta, valid


def build_dual_fisheye_equirect_maps(
    *,
    fisheye_width: int,
    fisheye_height: int,
    fov_deg: float,
    equirect_width: int,
    equirect_height: int,
    pitch_deg: float = 0.0,
    lens_offset: float = 0.02,
    stitch_distance: float = 4.0,
    blend_deg: float = 4.0,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build remap tables for a front/back 200-degree fisheye pair.

    The two lenses share one rigid mount and are separated along the rig X
    axis by ``2 * lens_offset``.  The panorama is centered on rig +X and uses
    the standard equirectangular longitude/latitude layout.
    """
    if fisheye_width <= 0 or fisheye_height <= 0:
        raise ValueError("fisheye resolution must be positive")
    if equirect_width <= 0 or equirect_height <= 0:
        raise ValueError("equirectangular resolution must be positive")
    if not 0.0 < fov_deg <= 360.0:
        raise ValueError("fov_deg must be in (0, 360]")
    if lens_offset < 0.0:
        raise ValueError("lens_offset must be non-negative")
    if stitch_distance <= 0.0:
        raise ValueError("stitch_distance must be positive")

    u = (np.arange(equirect_width, dtype=np.float64) + 0.5) / equirect_width
    v = (np.arange(equirect_height, dtype=np.float64) + 0.5) / equirect_height
    lon = (u - 0.5) * (2.0 * math.pi)
    lat = (0.5 - v) * math.pi
    lon, lat = np.meshgrid(lon, lat)

    cos_lat = np.cos(lat)
    bearings = np.stack(
        [
            cos_lat * np.cos(lon),
            -cos_lat * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )

    rotation_front, rotation_back = dual_camera_rotations(pitch_deg)
    translation_front = np.array([lens_offset, 0.0, 0.0], dtype=np.float64)
    translation_back = np.array([-lens_offset, 0.0, 0.0], dtype=np.float64)
    maps_front, theta_front, valid_front = _project_fisheye(
        bearings,
        rotation=rotation_front,
        translation=translation_front,
        width=fisheye_width,
        height=fisheye_height,
        fov_deg=fov_deg,
        stitch_distance=stitch_distance,
    )
    maps_back, theta_back, valid_back = _project_fisheye(
        bearings,
        rotation=rotation_back,
        translation=translation_back,
        width=fisheye_width,
        height=fisheye_height,
        fov_deg=fov_deg,
        stitch_distance=stitch_distance,
    )

    blend = math.radians(max(blend_deg, 1e-6))
    half_pi = math.pi * 0.5
    weight_front = np.clip((half_pi + blend * 0.5 - theta_front) / blend, 0.0, 1.0)
    weight_back = np.clip((half_pi + blend * 0.5 - theta_back) / blend, 0.0, 1.0)
    weight_front = np.where(valid_front, weight_front, 0.0)
    weight_back = np.where(valid_back, weight_back, 0.0)
    weight_sum = weight_front + weight_back
    covered = weight_sum > 0.0
    weight_front = np.where(covered, weight_front / np.maximum(weight_sum, 1e-12), 0.0)
    weight_back = np.where(covered, weight_back / np.maximum(weight_sum, 1e-12), 0.0)

    return (
        maps_front,
        maps_back,
        weight_front.astype(np.float32),
        weight_back.astype(np.float32),
        covered,
    )


def remap_dual_fisheye_to_equirect(
    front_rgb: np.ndarray,
    back_rgb: np.ndarray,
    maps: tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Stitch front/back fisheye RGB frames into one equirectangular RGB frame."""
    if front_rgb.shape != back_rgb.shape:
        raise ValueError(f"front/back shapes differ: {front_rgb.shape} vs {back_rgb.shape}")
    if front_rgb.ndim != 3 or front_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB images, got {front_rgb.shape}")

    maps_front, maps_back, weight_front, weight_back, _ = maps
    front_eq = cv2.remap(
        front_rgb,
        maps_front[0],
        maps_front[1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)
    back_eq = cv2.remap(
        back_rgb,
        maps_back[0],
        maps_back[1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)
    stitched = front_eq * weight_front[..., None] + back_eq * weight_back[..., None]
    return np.clip(stitched, 0.0, 255.0).astype(np.uint8)


def camera_chain_poses(
    transforms: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Accumulate ``T_cn_cnm1`` transforms into poses in camera-0 coordinates.

    In the Kalibr convention, ``transforms[n]`` maps camera ``n-1``
    coordinates into camera ``n`` coordinates.  The inverse is therefore used
    when accumulating camera-n poses in camera-0 coordinates.
    The returned rotation ``R_n`` maps camera-n coordinates to camera-0
    coordinates and the translation is camera n's origin in camera-0.
    """
    rotations = [np.eye(3, dtype=np.float64)]
    translations = [np.zeros(3, dtype=np.float64)]
    t_prev = np.eye(4, dtype=np.float64)
    for transform in transforms:
        t_prev = t_prev @ np.linalg.inv(np.asarray(transform, dtype=np.float64))
        rotations.append(t_prev[:3, :3].copy())
        translations.append(t_prev[:3, 3].copy())
    return rotations, translations


def _project_opencv_radtan(
    points_camera: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points using OpenCV pinhole + radtan."""
    z = points_camera[..., 2]
    valid_z = z > 1e-6
    safe_z = np.where(valid_z, z, 1.0)
    x = points_camera[..., 0] / safe_z
    y = points_camera[..., 1] / safe_z
    r2 = x * x + y * y

    k1, k2, p1, p2, k3 = [float(v) for v in distortion[:5]]
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    x_pixels = fx * x_distorted + cx
    y_pixels = fy * y_distorted + cy
    valid = (
        valid_z
        & np.isfinite(x_pixels)
        & np.isfinite(y_pixels)
        & (x_pixels >= 0.0)
        & (x_pixels < width)
        & (y_pixels >= 0.0)
        & (y_pixels < height)
    )
    map_x = np.clip(x_pixels, -1.0, width + 1.0).astype(np.float32)
    map_y = np.clip(y_pixels, -1.0, height + 1.0).astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), map_x, valid


def _project_omni_radtan(
    points_camera: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    xi: float,
    distortion: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points with the Kalibr ``omni-radtan`` model.

    The radial-tangential coefficients are ordered ``[k1, k2, p1, p2]``.  The
    separate ``xi`` parameter controls the unified spherical projection.
    """
    norms = np.linalg.norm(points_camera, axis=-1, keepdims=True)
    valid_norm = norms[..., 0] > 1e-9
    unit = points_camera / np.maximum(norms, 1e-12)
    denominator = unit[..., 2] + float(xi)
    valid_denominator = denominator > 1e-6
    safe_denominator = np.where(valid_denominator, denominator, 1.0)
    xu = unit[..., 0] / safe_denominator
    yu = unit[..., 1] / safe_denominator
    r2 = xu * xu + yu * yu
    r4 = r2 * r2

    k1, k2, p1, p2 = [float(v) for v in distortion[:4]]
    radial = 1.0 + k1 * r2 + k2 * r4
    x_distorted = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
    y_distorted = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
    x_pixels = fx * x_distorted + cx
    y_pixels = fy * y_distorted + cy
    valid = (
        valid_norm
        & valid_denominator
        & np.isfinite(x_pixels)
        & np.isfinite(y_pixels)
        & (x_pixels >= 0.0)
        & (x_pixels < width)
        & (y_pixels >= 0.0)
        & (y_pixels < height)
    )
    map_x = np.clip(x_pixels, -1.0, width + 1.0).astype(np.float32)
    map_y = np.clip(y_pixels, -1.0, height + 1.0).astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), map_x, valid


def fit_ftheta_polynomial_from_omni(
    *,
    intrinsic: tuple[float, float, float, float],
    distortion: np.ndarray,
    calibration_width: int,
    calibration_height: int,
    degree: int = 4,
    max_theta_deg: float = 134.9,
) -> tuple[np.ndarray, float, float, float]:
    """Fit Isaac's F-theta lens model to the calibrated omni-radtan camera.

    Isaac Sim 5.1 does not expose a native unified omnidirectional camera
    schema.  The F-theta model is used only for native RTX rendering.  Panorama
    stitching still uses the exact omni-radtan calibration.

    Returns:
        ``(coefficients, max_fov_deg, max_radius_px, rms_error_deg)`` where
        coefficients are ordered ``[k0, k1, ..., kN]`` for
        ``theta = k0 + k1*r + ... + kN*r**N``.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    if not 0.0 < max_theta_deg < 135.0:
        raise ValueError("max_theta_deg must be in (0, 135)")

    fx, _, cx, cy = intrinsic
    xi = float(distortion[0])
    radial_distortion = np.asarray(distortion[1:5], dtype=np.float64)
    corner_radius = max(
        math.hypot(x, y)
        for x in (cx, calibration_width - 1 - cx)
        for y in (cy, calibration_height - 1 - cy)
    )

    theta_scan = np.linspace(0.0, math.radians(max_theta_deg), 20001, dtype=np.float64)
    denominator = np.cos(theta_scan) + xi
    safe_denominator = np.maximum(denominator, 1e-9)
    xu = np.sin(theta_scan) / safe_denominator
    radial = 1.0 + radial_distortion[0] * xu * xu + radial_distortion[1] * xu**4
    radius_scan = fx * xu * radial
    valid_scan = (denominator > 1e-6) & np.isfinite(radius_scan) & (radius_scan >= 0.0)
    if np.count_nonzero(valid_scan) < degree + 2:
        raise RuntimeError("not enough valid samples to fit the F-theta model")

    valid_radius = radius_scan[valid_scan]
    valid_theta = theta_scan[valid_scan]
    max_radius = min(corner_radius, float(valid_radius.max()))
    theta_max = float(np.interp(max_radius, valid_radius, valid_theta))

    sample_theta = np.linspace(0.0, theta_max, 2001, dtype=np.float64)
    sample_denominator = np.cos(sample_theta) + xi
    sample_xu = np.sin(sample_theta) / np.maximum(sample_denominator, 1e-9)
    sample_radial = (
        1.0 + radial_distortion[0] * sample_xu * sample_xu + radial_distortion[1] * sample_xu**4
    )
    sample_radius = fx * sample_xu * sample_radial
    polynomial_descending = np.polyfit(sample_radius, sample_theta, degree)
    fitted_theta = np.polyval(polynomial_descending, sample_radius)
    rms_error_deg = float(np.rad2deg(np.sqrt(np.mean((fitted_theta - sample_theta) ** 2))))
    return (
        polynomial_descending[::-1].astype(np.float64),
        float(np.rad2deg(2.0 * theta_max)),
        float(max_radius),
        rms_error_deg,
    )


def build_four_fisheye_equirect_maps(
    *,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    intrinsics: list[tuple[float, float, float, float]],
    distortions: list[np.ndarray],
    calibration_width: int,
    calibration_height: int,
    render_width: int,
    render_height: int,
    equirect_width: int,
    equirect_height: int,
    stitch_distance: float = 4.0,
    blend_sigma_deg: float = 24.0,
    xi: list[float] | np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Build a four-camera equirectangular stitch.

    The panorama is expressed in camera-0 coordinates.  Camera transforms are
    the chain-accumulated poses returned by :func:`camera_chain_poses`.  Each
    distortion vector must contain ``[xi, k1, k2, p1, p2]`` when ``xi`` is not
    provided separately.
    """
    if not (len(rotations) == len(translations) == len(intrinsics) == len(distortions) == 4):
        raise ValueError("expected exactly four cameras")
    if stitch_distance <= 0.0:
        raise ValueError("stitch_distance must be positive")
    if xi is None:
        if any(len(distortion) < 5 for distortion in distortions):
            raise ValueError(
                "four-camera omni-radtan distortion needs [xi, k1, k2, p1, p2]"
            )
        xi_values = [float(distortion[0]) for distortion in distortions]
        radial_distortions = [
            np.asarray(distortion[1:5], dtype=np.float64) for distortion in distortions
        ]
    else:
        if len(xi) != 4:
            raise ValueError("expected four xi values")
        xi_values = [float(value) for value in xi]
        radial_distortions = [
            np.asarray(distortion[:4], dtype=np.float64) for distortion in distortions
        ]

    u = (np.arange(equirect_width, dtype=np.float64) + 0.5) / equirect_width
    v = (np.arange(equirect_height, dtype=np.float64) + 0.5) / equirect_height
    lon = (u - 0.5) * (2.0 * math.pi)
    lat = (0.5 - v) * math.pi
    lon, lat = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat)
    bearings = np.stack(
        [
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )
    # camera-0 uses +X right, +Y down, +Z forward.  The equirect convention
    # above is centered on +X in the rig frame, so rotate it into the
    # camera-0 optical frame.
    rig_to_camera0 = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    bearings_camera0 = bearings @ rig_to_camera0.T

    scale_x = render_width / float(calibration_width)
    scale_y = render_height / float(calibration_height)
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    weights = []
    sigma = math.radians(max(blend_sigma_deg, 1e-3))
    points_at_distance = bearings_camera0 * stitch_distance
    for rotation, translation, intrinsic, distortion, xi_value in zip(
        rotations, translations, intrinsics, radial_distortions, xi_values
    ):
        points_camera = (points_at_distance - translation) @ rotation
        fx, fy, cx, cy = intrinsic
        maps_camera, map_x, valid = _project_omni_radtan(
            points_camera,
            fx=fx * scale_x,
            fy=fy * scale_y,
            cx=cx * scale_x,
            cy=cy * scale_y,
            xi=xi_value,
            distortion=distortion,
            width=render_width,
            height=render_height,
        )
        axis = rotation[:, 2]
        theta = np.arccos(np.clip(bearings_camera0 @ axis, -1.0, 1.0))
        weight = np.exp(-0.5 * (theta / sigma) ** 2)
        weight = np.where(valid, weight, 0.0)
        maps.append(maps_camera)
        weights.append(weight.astype(np.float32))

    weight_sum = np.sum(weights, axis=0)
    covered = weight_sum > 0.0
    normalized = [
        np.where(covered, weight / np.maximum(weight_sum, 1e-12), 0.0).astype(np.float32)
        for weight in weights
    ]
    return maps, np.stack(normalized, axis=0), covered


def build_four_ideal_fisheye_equirect_maps(
    *,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    fov_deg: float,
    render_width: int,
    render_height: int,
    equirect_width: int,
    equirect_height: int,
    stitch_distance: float = 4.0,
    blend_sigma_deg: float = 24.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Build maps for four identical ideal equidistant fisheye cameras.

    Camera positions and orientations stay unchanged, while all four cameras
    share the same circular equidistant projection.
    """
    if not (len(rotations) == len(translations) == 4):
        raise ValueError("expected exactly four cameras")
    if not 0.0 < fov_deg <= 220.0:
        raise ValueError("fov_deg must be in (0, 220]")
    if stitch_distance <= 0.0:
        raise ValueError("stitch_distance must be positive")

    u = (np.arange(equirect_width, dtype=np.float64) + 0.5) / equirect_width
    v = (np.arange(equirect_height, dtype=np.float64) + 0.5) / equirect_height
    lon = (u - 0.5) * (2.0 * math.pi)
    lat = (0.5 - v) * math.pi
    lon, lat = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat)
    bearings = np.stack(
        [
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )
    rig_to_camera0 = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    bearings_camera0 = bearings @ rig_to_camera0.T
    points_at_distance = bearings_camera0 * stitch_distance

    half_fov = math.radians(fov_deg) * 0.5
    radius_scale = min(render_width, render_height) * 0.5 / half_fov
    cx = render_width * 0.5
    cy = render_height * 0.5
    sigma = math.radians(max(blend_sigma_deg, 1e-3))
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    weights = []
    for rotation, translation in zip(rotations, translations):
        points_camera = (points_at_distance - translation) @ rotation
        directions = points_camera / np.maximum(
            np.linalg.norm(points_camera, axis=-1, keepdims=True), 1e-12
        )
        theta = np.arccos(np.clip(directions[..., 2], -1.0, 1.0))
        phi = np.arctan2(directions[..., 1], directions[..., 0])
        radius = theta * radius_scale
        map_x = (cx + radius * np.cos(phi)).astype(np.float32)
        map_y = (cy + radius * np.sin(phi)).astype(np.float32)
        valid = (
            (theta <= half_fov)
            & (map_x >= 0.0)
            & (map_x < render_width)
            & (map_y >= 0.0)
            & (map_y < render_height)
        )
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0
        maps.append(cv2.convertMaps(map_x, map_y, cv2.CV_16SC2))

        axis = rotation[:, 2]
        axis_angle = np.arccos(np.clip(bearings_camera0 @ axis, -1.0, 1.0))
        weight = np.exp(-0.5 * (axis_angle / sigma) ** 2)
        weights.append(np.where(valid, weight, 0.0).astype(np.float32))

    weight_sum = np.sum(weights, axis=0)
    covered = weight_sum > 0.0
    normalized = [
        np.where(covered, weight / np.maximum(weight_sum, 1e-12), 0.0).astype(np.float32)
        for weight in weights
    ]
    return maps, np.stack(normalized, axis=0), covered


def remap_four_fisheye_to_equirect(
    images: list[np.ndarray],
    maps: list[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Stitch four fisheye images into one equirectangular RGB image."""
    if len(images) != 4 or len(maps) != 4:
        raise ValueError("expected four images and four map pairs")
    image_height, image_width = images[0].shape[:2]
    equirect_height, equirect_width = weights.shape[-2:]
    accumulator = np.zeros((equirect_height, equirect_width, 3), dtype=np.float32)
    for image, map_pair, weight in zip(images, maps, weights):
        if image.shape[:2] != (image_height, image_width):
            raise ValueError("all fisheye images must have the same shape")
        if weight.shape != (equirect_height, equirect_width):
            raise ValueError("all panorama weights must have the same shape")
        remapped = cv2.remap(
            image,
            map_pair[0],
            map_pair[1],
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ).astype(np.float32)
        accumulator += remapped * weight[..., None]
    return np.clip(accumulator, 0.0, 255.0).astype(np.uint8)
