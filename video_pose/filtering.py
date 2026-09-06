from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from video_pose.models import PlateDetection


@dataclass(frozen=True)
class PlateFilterConfig:
    """
    Lightweight sanity filtering for the VIDEO POSE product.

    Important:
    This is deliberately more permissive than the old video-calibration
    filter. A strongly yawed plate can legitimately have a small apparent
    aspect ratio or non-parallel projected side edges.

    The goal here is only to reject obviously unusable quadrilaterals.
    Pose quality should later be checked using solvePnP/reprojection metrics.
    """

    min_detection_confidence: float = 0.35
    min_quad_area_px: float = 1_000.0

    # Perspective can shrink the apparent width considerably at large yaw.
    min_quad_aspect_ratio: float = 1.20
    max_quad_aspect_ratio: float = 8.0

    min_edge_length_px: float = 12.0
    min_corner_distance_px: float = 8.0
    min_refined_height_px: float = 35.0

    # Keep only loose degeneracy checks here.
    max_opposite_width_ratio: float = 1.50
    max_opposite_height_ratio: float = 1.50

    max_opposite_width_angle_diff_deg: float = 15.0
    max_opposite_height_angle_diff_deg: float = 20.0

    min_quality_score: float = 0.25


def polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * (
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )


def quad_area(corners: np.ndarray) -> float:
    return abs(polygon_signed_area(corners))


def edge_lengths(corners: np.ndarray) -> np.ndarray:
    next_corners = np.roll(corners, -1, axis=0)
    return np.linalg.norm(next_corners - corners, axis=1)


def refined_plate_height(corners: np.ndarray) -> float:
    lengths = edge_lengths(corners)
    right = float(lengths[1])
    left = float(lengths[3])
    return 0.5 * (left + right)


def minimum_corner_distance(corners: np.ndarray) -> float:
    distances: list[float] = []

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            distances.append(
                float(
                    np.linalg.norm(
                        corners[first_index] - corners[second_index]
                    )
                )
            )

    return min(distances)


def quad_aspect_ratio(corners: np.ndarray) -> float:
    lengths = edge_lengths(corners)

    top = float(lengths[0])
    right = float(lengths[1])
    bottom = float(lengths[2])
    left = float(lengths[3])

    average_width = 0.5 * (top + bottom)
    average_height = 0.5 * (left + right)

    if average_height <= 1e-8:
        return float("inf")

    return average_width / average_height


def is_convex_quad(corners: np.ndarray) -> bool:
    contour = corners.astype(np.float32).reshape((-1, 1, 2))
    return bool(cv2.isContourConvex(contour))


def opposite_edge_ratios(corners: np.ndarray) -> tuple[float, float]:
    lengths = edge_lengths(corners)

    top = float(lengths[0])
    right = float(lengths[1])
    bottom = float(lengths[2])
    left = float(lengths[3])

    minimum_width = min(top, bottom)
    minimum_height = min(left, right)

    width_ratio = (
        float("inf")
        if minimum_width <= 1e-8
        else max(top, bottom) / minimum_width
    )

    height_ratio = (
        float("inf")
        if minimum_height <= 1e-8
        else max(left, right) / minimum_height
    )

    return float(width_ratio), float(height_ratio)


def line_orientation_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    delta = p2 - p1
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return float(angle % 180.0)


def orientation_difference_deg(angle1: float, angle2: float) -> float:
    difference = abs(angle1 - angle2)
    return float(min(difference, 180.0 - difference))


def opposite_edge_angle_differences(
    corners: np.ndarray,
) -> tuple[float, float]:
    top_angle = line_orientation_deg(corners[0], corners[1])
    bottom_angle = line_orientation_deg(corners[3], corners[2])
    left_angle = line_orientation_deg(corners[0], corners[3])
    right_angle = line_orientation_deg(corners[1], corners[2])

    return (
        orientation_difference_deg(top_angle, bottom_angle),
        orientation_difference_deg(left_angle, right_angle),
    )


def compute_quality_score(
    detection: PlateDetection,
    config: PlateFilterConfig,
) -> float:
    confidence_score = float(
        np.clip(detection.detection_confidence, 0.0, 1.0)
    )

    area_score = float(
        np.clip(
            quad_area(detection.corners)
            / (4.0 * config.min_quad_area_px),
            0.0,
            1.0,
        )
    )

    shortest_edge = float(np.min(edge_lengths(detection.corners)))

    edge_score = float(
        np.clip(
            shortest_edge / (2.0 * config.min_edge_length_px),
            0.0,
            1.0,
        )
    )

    quality_score = (
        0.35 * confidence_score
        + 0.40 * area_score
        + 0.25 * edge_score
    )

    return float(np.clip(quality_score, 0.0, 1.0))


def detection_geometry(detection: PlateDetection) -> dict[str, float]:
    corners = detection.corners
    lengths = edge_lengths(corners)
    width_ratio, height_ratio = opposite_edge_ratios(corners)
    width_angle_diff, height_angle_diff = opposite_edge_angle_differences(corners)

    return {
        "area": quad_area(corners),
        "height": refined_plate_height(corners),
        "aspect": quad_aspect_ratio(corners),
        "min_edge": float(np.min(lengths)),
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "width_angle": width_angle_diff,
        "height_angle": height_angle_diff,
    }


def validate_detection(
    detection: PlateDetection,
    config: PlateFilterConfig,
) -> tuple[bool, str | None]:
    corners = detection.corners

    if not np.all(np.isfinite(corners)):
        return False, "non_finite_corners"

    if detection.detection_confidence < config.min_detection_confidence:
        return False, "low_detection_confidence"

    if not is_convex_quad(corners):
        return False, "non_convex_quad"

    if quad_area(corners) < config.min_quad_area_px:
        return False, "small_quad_area"

    lengths = edge_lengths(corners)

    if refined_plate_height(corners) < config.min_refined_height_px:
        return False, "small_refined_plate_height"

    if float(np.min(lengths)) < config.min_edge_length_px:
        return False, "short_quad_edge"

    if minimum_corner_distance(corners) < config.min_corner_distance_px:
        return False, "corners_too_close"

    aspect_ratio = quad_aspect_ratio(corners)

    if not np.isfinite(aspect_ratio):
        return False, "invalid_quad_aspect_ratio"

    if aspect_ratio < config.min_quad_aspect_ratio:
        return False, "quad_too_tall"

    if aspect_ratio > config.max_quad_aspect_ratio:
        return False, "quad_too_wide"

    width_ratio, height_ratio = opposite_edge_ratios(corners)

    if not np.isfinite(width_ratio):
        return False, "invalid_opposite_width_ratio"

    if not np.isfinite(height_ratio):
        return False, "invalid_opposite_height_ratio"

    if width_ratio > config.max_opposite_width_ratio:
        return False, "opposite_width_edges_mismatch"

    if height_ratio > config.max_opposite_height_ratio:
        return False, "opposite_height_edges_mismatch"

    width_angle_diff, height_angle_diff = opposite_edge_angle_differences(corners)

    if width_angle_diff > config.max_opposite_width_angle_diff_deg:
        return False, "opposite_width_angle_mismatch"

    if height_angle_diff > config.max_opposite_height_angle_diff_deg:
        return False, "opposite_height_angle_mismatch"

    return True, None


def filter_detections(
    detections: list[PlateDetection],
    config: PlateFilterConfig | None = None,
) -> list[PlateDetection]:
    if config is None:
        config = PlateFilterConfig()

    for detection in detections:
        detection.quality_score = compute_quality_score(detection, config)

        is_valid, rejection_reason = validate_detection(detection, config)

        if not is_valid:
            detection.reject(rejection_reason or "unknown_rejection")
            continue

        if detection.quality_score < config.min_quality_score:
            detection.reject("low_quality_score")
            continue

        detection.accept()

    return detections


def get_accepted_detections(
    detections: list[PlateDetection],
) -> list[PlateDetection]:
    return [detection for detection in detections if detection.accepted]
