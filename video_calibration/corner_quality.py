from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class CornerQualityConfig:
    """Configuration for local single-corner quality scoring."""

    yellow_h_low: int = 15
    yellow_h_high: int = 45
    yellow_s_low: int = 55
    yellow_v_low: int = 45

    probe_radius_px: int = 3
    min_probe_distance_px: float = 4.0
    max_probe_distance_px: float = 12.0
    probe_distance_height_fraction: float = 0.22

    min_patch_radius_px: int = 14
    max_patch_radius_px: int = 42
    patch_radius_height_fraction: float = 0.80

    min_edge_probe_length_px: float = 12.0
    max_edge_probe_length_px: float = 40.0
    edge_probe_length_height_fraction: float = 0.85
    edge_band_half_width_px: int = 2

    yellow_boundary_weight: float = 0.45
    edge_gradient_weight: float = 0.30
    cornerness_weight: float = 0.20
    local_geometry_weight: float = 0.05


@dataclass(frozen=True)
class CornerQualityResult:
    corner_index: int
    corner_name: str
    score: float
    yellow_boundary_score: float
    yellow_inside_score: float
    yellow_outside_score: float
    edge_gradient_score: float
    adjacent_edge_1_gradient_score: float
    adjacent_edge_2_gradient_score: float
    cornerness_score: float
    local_geometry_score: float
    patch_radius_px: int
    probe_distance_px: float


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(2, dtype=np.float64)
    return vector / norm


def _plate_height(corners: np.ndarray) -> float:
    right = float(np.linalg.norm(corners[2] - corners[1]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    return 0.5 * (right + left)


def _yellow_mask(image_bgr: np.ndarray, config: CornerQualityConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [config.yellow_h_low, config.yellow_s_low, config.yellow_v_low],
        dtype=np.uint8,
    )
    upper = np.array([config.yellow_h_high, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper).astype(np.float32) / 255.0


def _disk_mean(image: np.ndarray, center: np.ndarray, radius: int) -> float:
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    radius = max(1, int(radius))

    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image.shape[1], x + radius + 1)
    y1 = min(image.shape[0], y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0

    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2 <= radius**2
    if not np.any(disk):
        return 0.0
    return float(np.mean(patch[disk]))


def _patch_bounds(shape, center: np.ndarray, radius: int) -> tuple[int, int, int, int]:
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(int(shape[1]), x + radius + 1),
        min(int(shape[0]), y + radius + 1),
    )


def _sample_line_band_mean(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    half_width_px: int,
    samples: int = 30,
) -> float:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    direction_unit = _unit(direction)
    if np.linalg.norm(direction_unit) <= 1e-12:
        return 0.0

    normal = np.array([-direction_unit[1], direction_unit[0]], dtype=np.float64)
    values: list[float] = []

    for alpha in np.linspace(0.0, 1.0, samples):
        base = start + alpha * direction
        for offset in range(-half_width_px, half_width_px + 1):
            point = base + float(offset) * normal
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                values.append(float(image[y, x]))

    return float(np.mean(values)) if values else 0.0


def _cornerness_score(gray: np.ndarray, corner: np.ndarray, radius: int) -> float:
    response = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    x0, y0, x1, y1 = _patch_bounds(gray.shape, corner, radius)
    local = response[y0:y1, x0:x1]
    if local.size == 0:
        return 0.0

    local_max = float(np.max(local))
    if local_max <= 1e-12:
        return 0.0

    x = int(np.clip(round(float(corner[0])), 0, gray.shape[1] - 1))
    y = int(np.clip(round(float(corner[1])), 0, gray.shape[0] - 1))
    return float(np.clip(float(response[y, x]) / local_max, 0.0, 1.0))


def _local_geometry_score(corners: np.ndarray, corner_index: int) -> float:
    previous_index = (corner_index - 1) % 4
    next_index = (corner_index + 1) % 4
    corner = corners[corner_index]
    v1 = _unit(corners[previous_index] - corner)
    v2 = _unit(corners[next_index] - corner)
    if np.linalg.norm(v1) <= 1e-12 or np.linalg.norm(v2) <= 1e-12:
        return 0.0

    cosine = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return float(np.exp(-((abs(angle - 90.0) / 45.0) ** 2)))


def compute_corner_quality(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    corner_index: int,
    config: CornerQualityConfig | None = None,
) -> CornerQualityResult:
    """
    Score one refined corner using only evidence from its own frame.

    This deliberately does not use temporal information yet. The experiment
    first asks whether the best-looking corner in a track can be identified
    reliably before we use it to improve other frames.
    """
    if config is None:
        config = CornerQualityConfig()

    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), received {corners.shape}")
    if not 0 <= corner_index < 4:
        raise ValueError("corner_index must be in [0, 3]")

    corner = corners[corner_index]
    previous_index = (corner_index - 1) % 4
    next_index = (corner_index + 1) % 4

    to_previous = _unit(corners[previous_index] - corner)
    to_next = _unit(corners[next_index] - corner)
    inward = _unit(to_previous + to_next)
    outward = -inward

    plate_height = max(_plate_height(corners), 1.0)
    patch_radius = int(
        round(
            np.clip(
                config.patch_radius_height_fraction * plate_height,
                config.min_patch_radius_px,
                config.max_patch_radius_px,
            )
        )
    )
    probe_distance = float(
        np.clip(
            config.probe_distance_height_fraction * plate_height,
            config.min_probe_distance_px,
            config.max_probe_distance_px,
        )
    )
    edge_probe_length = float(
        np.clip(
            config.edge_probe_length_height_fraction * plate_height,
            config.min_edge_probe_length_px,
            config.max_edge_probe_length_px,
        )
    )

    yellow = _yellow_mask(image_bgr, config)
    inside_probe = corner + probe_distance * inward
    outside_probe = corner + probe_distance * outward
    yellow_inside = _disk_mean(yellow, inside_probe, config.probe_radius_px)
    yellow_outside_fraction = _disk_mean(yellow, outside_probe, config.probe_radius_px)
    yellow_outside = 1.0 - yellow_outside_fraction
    yellow_boundary = float(np.clip(0.5 * (yellow_inside + yellow_outside), 0.0, 1.0))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    x0, y0, x1, y1 = _patch_bounds(magnitude.shape, corner, patch_radius)
    local_gradient = magnitude[y0:y1, x0:x1]
    scale = float(np.percentile(local_gradient, 90.0)) if local_gradient.size else 1.0
    scale = max(scale, 1e-6)

    edge1_end = corner + edge_probe_length * to_previous
    edge2_end = corner + edge_probe_length * to_next
    edge1_raw = _sample_line_band_mean(
        magnitude,
        corner,
        edge1_end,
        half_width_px=config.edge_band_half_width_px,
    )
    edge2_raw = _sample_line_band_mean(
        magnitude,
        corner,
        edge2_end,
        half_width_px=config.edge_band_half_width_px,
    )
    edge1 = float(np.clip(edge1_raw / scale, 0.0, 1.0))
    edge2 = float(np.clip(edge2_raw / scale, 0.0, 1.0))
    edge_gradient = float(np.sqrt(edge1 * edge2))

    cornerness = _cornerness_score(gray, corner, patch_radius)
    geometry = _local_geometry_score(corners, corner_index)

    weights = np.array(
        [
            config.yellow_boundary_weight,
            config.edge_gradient_weight,
            config.cornerness_weight,
            config.local_geometry_weight,
        ],
        dtype=np.float64,
    )
    components = np.array(
        [yellow_boundary, edge_gradient, cornerness, geometry],
        dtype=np.float64,
    )
    score = float(np.dot(weights, components) / max(float(np.sum(weights)), 1e-12))

    return CornerQualityResult(
        corner_index=corner_index,
        corner_name=CORNER_NAMES[corner_index],
        score=float(np.clip(score, 0.0, 1.0)),
        yellow_boundary_score=yellow_boundary,
        yellow_inside_score=float(np.clip(yellow_inside, 0.0, 1.0)),
        yellow_outside_score=float(np.clip(yellow_outside, 0.0, 1.0)),
        edge_gradient_score=edge_gradient,
        adjacent_edge_1_gradient_score=edge1,
        adjacent_edge_2_gradient_score=edge2,
        cornerness_score=cornerness,
        local_geometry_score=geometry,
        patch_radius_px=patch_radius,
        probe_distance_px=probe_distance,
    )


def compute_all_corner_qualities(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    config: CornerQualityConfig | None = None,
) -> list[CornerQualityResult]:
    return [
        compute_corner_quality(
            image_bgr,
            corners,
            corner_index,
            config=config,
        )
        for corner_index in range(4)
    ]
