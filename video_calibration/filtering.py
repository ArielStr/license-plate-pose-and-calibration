from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from video_calibration.observation import PlateObservation


@dataclass(frozen=True)
class ObservationFilterConfig:
    """
    Thresholds used for the initial geometric filtering stage.

    These checks should reject clearly invalid observations without being
    overly strict about perspective distortion.
    """

    min_detection_confidence: float = 0.35

    # Minimum refined quadrilateral area in pixels.
    min_quad_area_px: float = 1_500.0

    # Reject highly degenerate or extremely stretched quadrilaterals.
    min_quad_aspect_ratio: float = 1.2
    max_quad_aspect_ratio: float = 8.0

    # Minimum edge length of the refined quadrilateral.
    min_edge_length_px: float = 15.0

    # Minimum distance between any two refined corners.
    min_corner_distance_px: float = 10.0

    max_opposite_width_ratio: float = 1.25
    max_opposite_height_ratio: float = 1.25
    # Allowed margin outside the image.
    # A small margin is useful because line intersections can occasionally
    # fall slightly outside the image boundary.
    outside_image_tolerance_px: float = 5.0

    # Score threshold after all checks.
    min_quality_score: float = 0.35


def polygon_signed_area(points: np.ndarray) -> float:
    """
    Compute the signed area of a 2D polygon.

    Positive or negative sign depends on corner ordering.
    The absolute value is the geometric area.
    """
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
    """Return the absolute area of the refined quadrilateral."""
    return abs(polygon_signed_area(corners))


def edge_lengths(corners: np.ndarray) -> np.ndarray:
    """
    Return the lengths of the four quadrilateral edges.

    Corner order is expected to be:

        top-left
        top-right
        bottom-right
        bottom-left
    """
    next_corners = np.roll(corners, -1, axis=0)
    return np.linalg.norm(next_corners - corners, axis=1)


def minimum_corner_distance(corners: np.ndarray) -> float:
    """
    Return the minimum Euclidean distance between any two corners.
    """
    distances: list[float] = []

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            distance = np.linalg.norm(
                corners[first_index] - corners[second_index]
            )
            distances.append(float(distance))

    return min(distances)


def quad_aspect_ratio(corners: np.ndarray) -> float:
    """
    Estimate a perspective-tolerant quadrilateral aspect ratio.

    We average opposite edge lengths:

        width  = average(top, bottom)
        height = average(left, right)
    """
    lengths = edge_lengths(corners)

    top = lengths[0]
    right = lengths[1]
    bottom = lengths[2]
    left = lengths[3]

    average_width = 0.5 * (top + bottom)
    average_height = 0.5 * (left + right)

    if average_height <= 1e-8:
        return float("inf")

    return float(average_width / average_height)


def is_convex_quad(corners: np.ndarray) -> bool:
    """
    Check whether the four corners form a convex quadrilateral.
    """
    contour = corners.astype(np.float32).reshape((-1, 1, 2))
    return bool(cv2.isContourConvex(contour))

def opposite_edge_ratios(
    corners: np.ndarray,
) -> tuple[float, float]:
    """
    Return relative differences between opposite quad edges.

    Corner order:
        0: top-left
        1: top-right
        2: bottom-right
        3: bottom-left
    """
    lengths = edge_lengths(corners)

    top = float(lengths[0])
    right = float(lengths[1])
    bottom = float(lengths[2])
    left = float(lengths[3])

    minimum_width = min(top, bottom)
    minimum_height = min(left, right)

    if minimum_width <= 1e-8:
        width_ratio = float("inf")
    else:
        width_ratio = max(top, bottom) / minimum_width

    if minimum_height <= 1e-8:
        height_ratio = float("inf")
    else:
        height_ratio = max(left, right) / minimum_height

    return float(width_ratio), float(height_ratio)

def compute_quality_score(
    observation: PlateObservation,
    config: ObservationFilterConfig,
) -> float:
    """
    Compute a simple MVP quality score in the range [0, 1].

    The initial score uses only measurements already available in
    PlateObservation. Mask quality and temporal consistency will be added
    in later versions.
    """
    confidence_score = np.clip(
        observation.detection_confidence,
        0.0,
        1.0,
    )

    area = quad_area(observation.corners)

    # Reaches 1 when the quadrilateral area is at least four times
    # the minimum required area.
    area_score = np.clip(
        area / (4.0 * config.min_quad_area_px),
        0.0,
        1.0,
    )

    lengths = edge_lengths(observation.corners)
    shortest_edge = float(np.min(lengths))

    edge_score = np.clip(
        shortest_edge / (2.0 * config.min_edge_length_px),
        0.0,
        1.0,
    )

    # Detection confidence is useful, but refined geometry is more important.
    quality_score = (
        0.35 * confidence_score
        + 0.40 * area_score
        + 0.25 * edge_score
    )

    return float(np.clip(quality_score, 0.0, 1.0))


def validate_observation(
    observation: PlateObservation,
    config: ObservationFilterConfig,
) -> tuple[bool, str | None]:
    """
    Validate one observation.

    Returns:
        (True, None) when valid.

        (False, rejection_reason) when invalid.
    """
    corners = observation.corners
    homography = observation.homography

    if not np.all(np.isfinite(corners)):
        return False, "non_finite_corners"

    if not np.all(np.isfinite(homography)):
        return False, "non_finite_homography"

    if observation.detection_confidence < config.min_detection_confidence:
        return False, "low_detection_confidence"

    if not is_convex_quad(corners):
        return False, "non_convex_quad"

    area = quad_area(corners)

    if area < config.min_quad_area_px:
        return False, "small_quad_area"

    lengths = edge_lengths(corners)

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

    # if not corners_inside_image(
    #     observation,
    #     tolerance_px=config.outside_image_tolerance_px,
    # ):
    #     return False, "corners_outside_image"

    if abs(float(np.linalg.det(homography))) < 1e-12:
        return False, "singular_homography"

    width_ratio, height_ratio = opposite_edge_ratios(
        corners
    )

    if not np.isfinite(width_ratio):
        return False, "invalid_opposite_width_ratio"

    if not np.isfinite(height_ratio):
        return False, "invalid_opposite_height_ratio"

    if width_ratio > config.max_opposite_width_ratio:
        return False, "opposite_width_edges_mismatch"

    if height_ratio > config.max_opposite_height_ratio:
        return False, "opposite_height_edges_mismatch"

    return True, None


def filter_observations(
    observations: list[PlateObservation],
    config: ObservationFilterConfig | None = None,
) -> list[PlateObservation]:
    """
    Evaluate all observations and mark rejected ones.

    The function preserves all observations in the returned list.
    Rejected observations remain available for debugging through:

        observation.accepted
        observation.rejection_reason

    Use get_accepted_observations() when only valid observations are needed.
    """
    if config is None:
        config = ObservationFilterConfig()

    for observation in observations:
        observation.quality_score = compute_quality_score(
            observation,
            config,
        )

        is_valid, rejection_reason = validate_observation(
            observation,
            config,
        )

        if not is_valid:
            observation.reject(rejection_reason or "unknown_rejection")
            continue

        if observation.quality_score < config.min_quality_score:
            observation.reject("low_quality_score")
            continue

        observation.accept()

    return observations


def get_accepted_observations(
    observations: list[PlateObservation],
) -> list[PlateObservation]:
    """Return only observations that passed the filtering stage."""
    return [
        observation
        for observation in observations
        if observation.accepted
    ]


def print_filter_summary(
    observations: list[PlateObservation],
) -> None:
    """Print accepted and rejected observation statistics."""
    accepted = [
        observation
        for observation in observations
        if observation.accepted
    ]

    rejected = [
        observation
        for observation in observations
        if not observation.accepted
    ]

    rejection_counts: dict[str, int] = {}

    for observation in rejected:
        reason = observation.rejection_reason or "unknown"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    print("\n========== OBSERVATION FILTER ==========")
    print(f"Total observations : {len(observations)}")
    print(f"Accepted           : {len(accepted)}")
    print(f"Rejected           : {len(rejected)}")

    if rejection_counts:
        print("\nRejection reasons:")

        for reason, count in sorted(rejection_counts.items()):
            print(f"  {reason}: {count}")

    print("========================================\n")


def print_observation_geometry(
    observations: list[PlateObservation],
) -> None:
    print("\n========== OBSERVATION GEOMETRY ==========")

    for observation in observations:
        area = quad_area(observation.corners)
        aspect_ratio = quad_aspect_ratio(
            observation.corners
        )
        lengths = edge_lengths(
            observation.corners
        )

        width_ratio, height_ratio = (
            opposite_edge_ratios(
                observation.corners
            )
        )

        status = (
            "ACCEPTED"
            if observation.accepted
            else "REJECTED"
        )

        print(
            f"Frame={observation.frame_index:6d} | "
            f"area={area:9.1f} | "
            f"aspect={aspect_ratio:6.3f} | "
            f"min_edge={np.min(lengths):6.1f} | "
            f"width_ratio={width_ratio:5.3f} | "
            f"height_ratio={height_ratio:5.3f} | "
            f"score={observation.quality_score:.3f} | "
            f"{status}"
        )

    print("==========================================\n")