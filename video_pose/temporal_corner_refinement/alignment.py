from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .window import TrackObservation


@dataclass(frozen=True)
class AlignmentConfig:
    max_features: int = 120
    quality_level: float = 0.01
    min_distance_px: float = 5.0
    block_size: int = 5

    lk_win_size: tuple[int, int] = (31, 31)
    lk_max_level: int = 4
    lk_max_iterations: int = 40
    lk_epsilon: float = 0.01

    max_forward_backward_error_px: float = 2.0
    ransac_reproj_threshold_px: float = 3.0

    min_tracked_features: int = 8
    min_inliers: int = 6
    min_inlier_ratio: float = 0.45

    bbox_pad_x: float = 0.12
    bbox_pad_y: float = 0.40


@dataclass(frozen=True)
class PairwiseAlignment:
    success: bool
    reason: str
    H_source_to_target: np.ndarray | None
    source_feature_count: int
    tracked_feature_count: int
    inlier_count: int
    inlier_ratio: float
    mean_fb_error_px: float | None


def _expanded_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, ...],
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    x1 -= pad_x * w
    x2 += pad_x * w
    y1 -= pad_y * h
    y2 += pad_y * h

    ih, iw = image_shape[:2]
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(iw - 1, int(np.ceil(x2))),
        min(ih - 1, int(np.ceil(y2))),
    )


def _feature_mask(
    image_shape: tuple[int, ...],
    bbox: np.ndarray,
    *,
    pad_x: float,
    pad_y: float,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = _expanded_bbox(
        bbox, image_shape, pad_x, pad_y
    )
    if x2 > x1 and y2 > y1:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def estimate_pairwise_homography(
    source: TrackObservation,
    target: TrackObservation,
    *,
    config: AlignmentConfig | None = None,
) -> PairwiseAlignment:
    config = config or AlignmentConfig()

    gray_src = cv2.cvtColor(source.image, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(target.image, cv2.COLOR_BGR2GRAY)

    mask = _feature_mask(
        source.image.shape,
        source.detection.bbox,
        pad_x=config.bbox_pad_x,
        pad_y=config.bbox_pad_y,
    )

    pts0 = cv2.goodFeaturesToTrack(
        gray_src,
        maxCorners=config.max_features,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance_px,
        mask=mask,
        blockSize=config.block_size,
        useHarrisDetector=False,
    )

    if pts0 is None:
        return PairwiseAlignment(
            False, "no_features", None, 0, 0, 0, 0.0, None
        )

    source_feature_count = len(pts0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.lk_max_iterations,
        config.lk_epsilon,
    )

    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(
        gray_src,
        gray_dst,
        pts0,
        None,
        winSize=config.lk_win_size,
        maxLevel=config.lk_max_level,
        criteria=criteria,
    )

    if pts1 is None or st1 is None:
        return PairwiseAlignment(
            False, "forward_lk_failed", None,
            source_feature_count, 0, 0, 0.0, None
        )

    pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray_dst,
        gray_src,
        pts1,
        None,
        winSize=config.lk_win_size,
        maxLevel=config.lk_max_level,
        criteria=criteria,
    )

    if pts0_back is None or st2 is None:
        return PairwiseAlignment(
            False, "backward_lk_failed", None,
            source_feature_count, 0, 0, 0.0, None
        )

    src = pts0.reshape(-1, 2).astype(np.float64)
    dst = pts1.reshape(-1, 2).astype(np.float64)
    back = pts0_back.reshape(-1, 2).astype(np.float64)

    fb = np.linalg.norm(back - src, axis=1)

    valid = (
        st1.reshape(-1).astype(bool)
        & st2.reshape(-1).astype(bool)
        & np.isfinite(fb)
        & (fb <= config.max_forward_backward_error_px)
    )

    src = src[valid]
    dst = dst[valid]
    fb = fb[valid]

    tracked_count = len(src)

    if tracked_count < config.min_tracked_features:
        return PairwiseAlignment(
            False, "too_few_tracks", None,
            source_feature_count, tracked_count, 0, 0.0,
            float(np.mean(fb)) if len(fb) else None,
        )

    H, inliers = cv2.findHomography(
        src.astype(np.float32),
        dst.astype(np.float32),
        cv2.RANSAC,
        config.ransac_reproj_threshold_px,
    )

    if H is None or inliers is None:
        return PairwiseAlignment(
            False, "homography_failed", None,
            source_feature_count, tracked_count, 0, 0.0,
            float(np.mean(fb)),
        )

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = float(inlier_count / max(tracked_count, 1))

    if (
        inlier_count < config.min_inliers
        or inlier_ratio < config.min_inlier_ratio
    ):
        return PairwiseAlignment(
            False, "weak_homography", H.astype(np.float64),
            source_feature_count, tracked_count,
            inlier_count, inlier_ratio,
            float(np.mean(fb)),
        )

    return PairwiseAlignment(
        True, "ok", H.astype(np.float64),
        source_feature_count, tracked_count,
        inlier_count, inlier_ratio,
        float(np.mean(fb)),
    )


def warp_quad(corners: np.ndarray, H: np.ndarray) -> np.ndarray:
    warped = cv2.perspectiveTransform(
        np.asarray(corners, dtype=np.float32).reshape(1, 4, 2),
        np.asarray(H, dtype=np.float64),
    )
    return warped.reshape(4, 2).astype(np.float64)


def compose_adjacent_homographies_to_reference(
    alignments: list[PairwiseAlignment],
    *,
    reference_index: int,
) -> list[np.ndarray | None]:
    """
    alignments[k] maps observation k -> k+1.

    Returns H_i_to_ref for every observation i.
    """
    n = len(alignments) + 1
    out: list[np.ndarray | None] = [None] * n
    out[reference_index] = np.eye(3, dtype=np.float64)

    # Left side: i -> i+1 -> ... -> ref
    H = np.eye(3, dtype=np.float64)
    for i in range(reference_index - 1, -1, -1):
        a = alignments[i]
        if not a.success or a.H_source_to_target is None:
            H = None
            break
        if H is None:
            break
        H = H @ a.H_source_to_target
        out[i] = H.copy()

    # Right side: i -> i-1 -> ... -> ref, so inverse adjacent transforms.
    H = np.eye(3, dtype=np.float64)
    for i in range(reference_index + 1, n):
        a = alignments[i - 1]
        if not a.success or a.H_source_to_target is None:
            H = None
            break
        if H is None:
            break
        try:
            inv = np.linalg.inv(a.H_source_to_target)
        except np.linalg.LinAlgError:
            H = None
            break
        H = H @ inv
        out[i] = H.copy()

    return out
