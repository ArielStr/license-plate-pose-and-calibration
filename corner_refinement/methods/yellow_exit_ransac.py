from __future__ import annotations

import math

import cv2
import numpy as np

from corner_refinement.yellow_utils import estimate_dynamic_yellow_hsv


# ============================================================
# FINAL YELLOW-EXIT + LOCAL RANSAC METHOD
# ============================================================

# Safe inner quad
SIDE_COVERAGE_SAMPLES = 120
SIDE_YELLOW_COVERAGE_REQUIRED = 0.98
SHRINK_STEP_PX = 0.5
MAX_SHRINK_PX = 15.0

RAW_SIDE_SAMPLE_COUNT = 140
RAW_YELLOW_FRACTION_REQUIRED = 0.78
RAW_YELLOW_INWARD_FRACTION_REQUIRED = 0.88
RAW_CHECK_INWARD_PX = 1.0
RAW_YELLOW_SOFT_MIN = 0.58

# Boundary color model
EDGE_LINE_SAMPLES_PER_SIDE = 160
EXTRA_INWARD_PX = 1.0
MIN_EDGE_COLOR_SAMPLES_PER_SIDE = 20
MIN_FALLBACK_VALUE = 35

BLACK_V_MAX = 35
BLACK_LAB_L_MAX = 45

LAB_SIGMA_FLOOR = np.asarray(
    [8.0, 7.0, 7.0],
    dtype=np.float64,
)

EDGE_COLOR_Z_THRESHOLD = 2.35

# Normal-profile yellow exit
PROFILE_TANGENTIAL_SAMPLES = 120
PROFILE_OUTWARD_MIN_PX = 0.0
PROFILE_OUTWARD_STEP_PX = 0.25
PROFILE_EXTRA_OUTWARD_MARGIN_PX = 2.5

PROFILE_SMOOTH_KERNEL = np.asarray(
    [0.15, 0.70, 0.15],
    dtype=np.float64,
)

PROFILE_YELLOW_SUPPORT_THRESHOLD = 0.12
PROFILE_MIN_YELLOW_RUN_PX = 1.0
PROFILE_MIN_NONYELLOW_RUN_PX = 0.75
PROFILE_MIN_DROP = 0.035

MIN_VALID_RIDGE_FRACTION = 0.30
MIN_RIDGE_POINTS = 12
RIDGE_OFFSET_MEDIAN_WINDOW = 5

# Local t-d RANSAC
RANSAC_THRESHOLDS_PX = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
]

RANSAC_ITERATIONS = 500
RANSAC_RANDOM_SEED = 7

MIN_RANSAC_TANGENTIAL_SPAN_FRACTION = 0.55

W_RANSAC_INLIER_FRACTION = 0.45
W_RANSAC_TANGENTIAL_SPAN = 0.45
W_RANSAC_RESIDUAL = 0.10

CORNER_SNAP_RADIUS_PX = 3.0


# ============================================================
# BASIC GEOMETRY
# ============================================================

def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + 1e-12)


def _order_quad_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float64,
    )


def _quad_center(quad: np.ndarray) -> np.ndarray:
    return np.mean(quad, axis=0)


def _line_inward_normal(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    direction = _normalize(p2 - p1)

    normal = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    midpoint = 0.5 * (p1 + p2)

    if np.dot(center - midpoint, normal) < 0:
        normal = -normal

    return normal


def _line_intersection(
    p: np.ndarray,
    d: np.ndarray,
    q: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    A = np.column_stack([d, -e])

    if abs(float(np.linalg.det(A))) < 1e-9:
        raise RuntimeError("Parallel / degenerate lines.")

    t, _ = np.linalg.solve(A, q - p)
    return p + float(t) * d


def _sample_line(
    p1: np.ndarray,
    p2: np.ndarray,
    count: int,
) -> np.ndarray:
    t = np.linspace(
        0.0,
        1.0,
        count,
        dtype=np.float64,
    )

    return (
        (1.0 - t[:, None]) * p1[None, :]
        + t[:, None] * p2[None, :]
    )


def _nearest_mask_values(
    mask: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    x = np.round(points[:, 0]).astype(np.int32)
    y = np.round(points[:, 1]).astype(np.int32)

    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < mask.shape[1])
        & (y < mask.shape[0])
    )

    values = np.zeros(len(points), dtype=np.float64)

    values[valid] = (
        mask[y[valid], x[valid]] > 0
    ).astype(np.float64)

    return values


def _bilinear_sample_multichannel(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    x = points[:, 0]
    y = points[:, 1]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (
        (x0 >= 0)
        & (y0 >= 0)
        & (x1 < image.shape[1])
        & (y1 < image.shape[0])
    )

    result = np.full(
        (len(points), image.shape[2]),
        np.nan,
        dtype=np.float64,
    )

    if not np.any(valid):
        return result

    xv = x[valid]
    yv = y[valid]

    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    wx = xv - x0v
    wy = yv - y0v

    result[valid] = (
        (1.0 - wx)[:, None]
        * (1.0 - wy)[:, None]
        * image[y0v, x0v]
        + wx[:, None]
        * (1.0 - wy)[:, None]
        * image[y0v, x1v]
        + (1.0 - wx)[:, None]
        * wy[:, None]
        * image[y1v, x0v]
        + wx[:, None]
        * wy[:, None]
        * image[y1v, x1v]
    )

    return result


def _sample_scalar_map(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return _bilinear_sample_multichannel(
        image[:, :, None],
        points,
    )[:, 0]


def _min_area_rect_quad(clean_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("No contour in clean yellow mask.")

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    return _order_quad_tl_tr_br_bl(box)


def _side_line_from_inward_offset(
    *,
    min_quad: np.ndarray,
    side_index: int,
    inward_offset_px: float,
):
    center = _quad_center(min_quad)

    p1 = min_quad[side_index]
    p2 = min_quad[(side_index + 1) % 4]

    direction = _normalize(p2 - p1)
    n_in = _line_inward_normal(p1, p2, center)

    shifted_p1 = p1 + float(inward_offset_px) * n_in
    shifted_p2 = p2 + float(inward_offset_px) * n_in

    return shifted_p1, shifted_p2, direction, n_in


def _build_quad_from_side_offsets(
    *,
    min_quad: np.ndarray,
    side_offsets: list[float],
) -> np.ndarray:
    center = _quad_center(min_quad)
    lines = []

    for side_index, offset_px in enumerate(side_offsets):
        p1 = min_quad[side_index]
        p2 = min_quad[(side_index + 1) % 4]

        direction = _normalize(p2 - p1)
        n_in = _line_inward_normal(p1, p2, center)

        line_point = p1 + float(offset_px) * n_in
        lines.append((line_point, direction))

    top, right, bottom, left = lines

    tl = _line_intersection(top[0], top[1], left[0], left[1])
    tr = _line_intersection(top[0], top[1], right[0], right[1])
    br = _line_intersection(bottom[0], bottom[1], right[0], right[1])
    bl = _line_intersection(bottom[0], bottom[1], left[0], left[1])

    return np.asarray([tl, tr, br, bl], dtype=np.float64)


def _side_local_frame(
    safe_quad: np.ndarray,
    side_index: int,
):
    center = _quad_center(safe_quad)

    p1 = np.asarray(
        safe_quad[side_index],
        dtype=np.float64,
    )

    p2 = np.asarray(
        safe_quad[(side_index + 1) % 4],
        dtype=np.float64,
    )

    t = _normalize(p2 - p1)
    n_in = _line_inward_normal(p1, p2, center)
    n_out = -n_in

    return p1, p2, t, n_in, n_out


# ============================================================
# RAW YELLOW SUPPORT + SAFE INNER QUAD
# ============================================================

def _build_raw_yellow_support(
    crop: np.ndarray,
):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    dynamic_result = estimate_dynamic_yellow_hsv(crop)

    if dynamic_result is None:
        raise RuntimeError(
            "Dynamic HSV estimation failed in yellow_exit_ransac."
        )

    lower_yellow, upper_yellow, _, _ = dynamic_result

    raw_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    hsv_f = hsv.astype(np.float64)

    lower = np.asarray(
        lower_yellow,
        dtype=np.float64,
    ).reshape(1, 1, 3)

    upper = np.asarray(
        upper_yellow,
        dtype=np.float64,
    ).reshape(1, 1, 3)

    below = np.maximum(lower - hsv_f, 0.0)
    above = np.maximum(hsv_f - upper, 0.0)
    outside = below + above

    channel_scales = np.asarray(
        [8.0, 35.0, 45.0],
        dtype=np.float64,
    ).reshape(1, 1, 3)

    normalized = outside / channel_scales

    distance = np.sqrt(
        np.sum(normalized * normalized, axis=2)
    )

    raw_soft = np.exp(
        -0.5 * distance * distance
    ).astype(np.float64)

    raw_soft[raw_mask > 0] = np.maximum(
        raw_soft[raw_mask > 0],
        0.95,
    )

    return raw_mask, raw_soft


def _evaluate_raw_yellow_side(
    *,
    min_quad: np.ndarray,
    side_index: int,
    inward_offset_px: float,
    clean_mask: np.ndarray,
    raw_mask: np.ndarray,
    raw_soft: np.ndarray,
) -> dict:
    p1, p2, _, n_in = _side_line_from_inward_offset(
        min_quad=min_quad,
        side_index=side_index,
        inward_offset_px=inward_offset_px,
    )

    line = _sample_line(
        p1,
        p2,
        RAW_SIDE_SAMPLE_COUNT,
    )

    line_inward = (
        line
        + RAW_CHECK_INWARD_PX
        * n_in[None, :]
    )

    clean_fraction = float(
        np.mean(
            _nearest_mask_values(
                clean_mask,
                line,
            )
        )
    )

    raw_hard_fraction = float(
        np.mean(
            _nearest_mask_values(
                raw_mask,
                line,
            )
        )
    )

    raw_hard_inward_fraction = float(
        np.mean(
            _nearest_mask_values(
                raw_mask,
                line_inward,
            )
        )
    )

    soft_values = _sample_scalar_map(
        raw_soft,
        line,
    )

    soft_inward_values = _sample_scalar_map(
        raw_soft,
        line_inward,
    )

    soft_fraction = float(
        np.mean(
            soft_values >= RAW_YELLOW_SOFT_MIN
        )
    )

    soft_inward_fraction = float(
        np.mean(
            soft_inward_values >= RAW_YELLOW_SOFT_MIN
        )
    )

    raw_fraction = max(
        raw_hard_fraction,
        soft_fraction,
    )

    raw_inward_fraction = max(
        raw_hard_inward_fraction,
        soft_inward_fraction,
    )

    valid = bool(
        clean_fraction
        >= SIDE_YELLOW_COVERAGE_REQUIRED
        and raw_fraction
        >= RAW_YELLOW_FRACTION_REQUIRED
        and raw_inward_fraction
        >= RAW_YELLOW_INWARD_FRACTION_REQUIRED
    )

    return {
        "valid": valid,
        "clean_fraction": clean_fraction,
        "raw_fraction": raw_fraction,
        "raw_inward_fraction": raw_inward_fraction,
    }


def _find_side_specific_safe_offsets(
    *,
    min_quad: np.ndarray,
    clean_mask: np.ndarray,
    raw_mask: np.ndarray,
    raw_soft: np.ndarray,
):
    offsets = []

    for side_index in range(4):
        selected_offset = None

        for offset_px in np.arange(
            0.0,
            MAX_SHRINK_PX
            + 0.5 * SHRINK_STEP_PX,
            SHRINK_STEP_PX,
            dtype=np.float64,
        ):
            metrics = _evaluate_raw_yellow_side(
                min_quad=min_quad,
                side_index=side_index,
                inward_offset_px=float(offset_px),
                clean_mask=clean_mask,
                raw_mask=raw_mask,
                raw_soft=raw_soft,
            )

            if metrics["valid"]:
                selected_offset = float(offset_px)
                break

        if selected_offset is None:
            selected_offset = float(MAX_SHRINK_PX)

        offsets.append(selected_offset)

    return offsets


# ============================================================
# BOUNDARY YELLOW COLOR MODEL
# ============================================================

def _build_dark_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    dark = (
        (hsv[:, :, 2] <= BLACK_V_MAX)
        | (lab[:, :, 0] <= BLACK_LAB_L_MAX)
    )

    return dark.astype(np.uint8)


def _sample_dark_flags(
    dark_mask: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    x = np.round(points[:, 0]).astype(np.int32)
    y = np.round(points[:, 1]).astype(np.int32)

    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < dark_mask.shape[1])
        & (y < dark_mask.shape[0])
    )

    flags = np.ones(len(points), dtype=bool)

    flags[valid] = (
        dark_mask[y[valid], x[valid]] > 0
    )

    return flags


def _collect_boundary_line_color_samples(
    *,
    crop: np.ndarray,
    clean_mask: np.ndarray,
    safe_quad: np.ndarray,
):
    lab_image = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float64)

    hsv_image = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    ).astype(np.float64)

    dark_mask = _build_dark_mask(crop)
    center = _quad_center(safe_quad)

    all_samples = []

    for i in range(4):
        p1 = safe_quad[i]
        p2 = safe_quad[(i + 1) % 4]

        n_in = _line_inward_normal(
            p1,
            p2,
            center,
        )

        line_points = _sample_line(
            p1,
            p2,
            EDGE_LINE_SAMPLES_PER_SIDE,
        )

        line_lab = _bilinear_sample_multichannel(
            lab_image,
            line_points,
        )

        line_hsv = _bilinear_sample_multichannel(
            hsv_image,
            line_points,
        )

        line_dark = _sample_dark_flags(
            dark_mask,
            line_points,
        )

        line_geom = (
            _nearest_mask_values(
                clean_mask,
                line_points,
            )
            > 0.5
        )

        line_finite = np.all(
            np.isfinite(line_lab),
            axis=1,
        )

        keep_line = (
            (~line_dark)
            & line_geom
            & line_finite
            & (line_hsv[:, 2] >= MIN_FALLBACK_VALUE)
        )

        side_samples = [line_lab[keep_line]]

        line_count = int(
            np.count_nonzero(keep_line)
        )

        if line_count < MIN_EDGE_COLOR_SAMPLES_PER_SIDE:
            inward_points = (
                line_points
                + EXTRA_INWARD_PX
                * n_in[None, :]
            )

            inward_lab = _bilinear_sample_multichannel(
                lab_image,
                inward_points,
            )

            inward_hsv = _bilinear_sample_multichannel(
                hsv_image,
                inward_points,
            )

            inward_dark = _sample_dark_flags(
                dark_mask,
                inward_points,
            )

            inward_geom = (
                _nearest_mask_values(
                    clean_mask,
                    inward_points,
                )
                > 0.5
            )

            inward_finite = np.all(
                np.isfinite(inward_lab),
                axis=1,
            )

            keep_inward = (
                (~inward_dark)
                & inward_geom
                & inward_finite
                & (inward_hsv[:, 2] >= MIN_FALLBACK_VALUE)
            )

            side_samples.append(
                inward_lab[keep_inward]
            )

        usable = [
            samples
            for samples in side_samples
            if len(samples) > 0
        ]

        if (
            not usable
            or sum(len(samples) for samples in usable)
            < MIN_EDGE_COLOR_SAMPLES_PER_SIDE
        ):
            fallback_keep = (
                line_geom
                & line_finite
                & (~line_dark)
            )

            fallback_lab = line_lab[
                fallback_keep
            ]

            if len(fallback_lab) > 0:
                usable.append(
                    fallback_lab
                )

        if not usable:
            raise RuntimeError(
                "No finite/non-black boundary color samples."
            )

        all_samples.append(
            np.vstack(usable)
        )

    combined = np.vstack(all_samples)

    return lab_image, dark_mask, combined


def _robust_lab_model(
    samples: np.ndarray,
) -> dict:
    samples = np.asarray(
        samples,
        dtype=np.float64,
    )

    samples = samples[
        np.all(
            np.isfinite(samples),
            axis=1,
        )
    ]

    if len(samples) < 20:
        raise RuntimeError(
            f"Too few boundary-color samples: {len(samples)}"
        )

    median = np.median(
        samples,
        axis=0,
    )

    mad = np.median(
        np.abs(
            samples
            - median[None, :]
        ),
        axis=0,
    )

    sigma = np.maximum(
        1.4826 * mad,
        LAB_SIGMA_FLOOR,
    )

    return {
        "median": median,
        "sigma": sigma,
    }


def _build_color_similarity_maps(
    *,
    lab_image: np.ndarray,
    dark_mask: np.ndarray,
    model: dict,
):
    delta = (
        lab_image
        - model["median"][None, None, :]
    )

    z = (
        delta
        / model["sigma"][None, None, :]
    )

    z_distance = np.sqrt(
        np.sum(
            z * z,
            axis=2,
        )
    )

    soft = np.exp(
        -0.5
        * z_distance
        * z_distance
    ).astype(np.float64)

    soft[dark_mask > 0] = 0.0

    soft_threshold = float(
        math.exp(
            -0.5
            * EDGE_COLOR_Z_THRESHOLD
            * EDGE_COLOR_Z_THRESHOLD
        )
    )

    hard = np.zeros(
        soft.shape,
        dtype=np.uint8,
    )

    hard[
        soft >= soft_threshold
    ] = 255

    hard[dark_mask > 0] = 0

    return soft, hard, soft_threshold


def _build_second_remainder_mask(
    *,
    hard_color_mask: np.ndarray,
    safe_quad: np.ndarray,
) -> np.ndarray:
    inner_mask = np.zeros_like(
        hard_color_mask
    )

    cv2.fillConvexPoly(
        inner_mask,
        np.round(
            safe_quad
        ).astype(np.int32),
        255,
    )

    remainder_region = np.where(
        inner_mask > 0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.bitwise_and(
        hard_color_mask,
        remainder_region,
    )


# ============================================================
# FINAL YELLOW EXIT
# ============================================================

def _find_final_yellow_exit(
    profile: np.ndarray,
    outward_offsets: np.ndarray,
):
    values = np.asarray(
        profile,
        dtype=np.float64,
    )

    offsets = np.asarray(
        outward_offsets,
        dtype=np.float64,
    )

    if len(values) < 5:
        return None

    smooth = np.convolve(
        values,
        PROFILE_SMOOTH_KERNEL,
        mode="same",
    )

    yellow = (
        smooth
        >= PROFILE_YELLOW_SUPPORT_THRESHOLD
    )

    min_yellow_samples = max(
        1,
        int(
            round(
                PROFILE_MIN_YELLOW_RUN_PX
                / PROFILE_OUTWARD_STEP_PX
            )
        ),
    )

    min_nonyellow_samples = max(
        1,
        int(
            round(
                PROFILE_MIN_NONYELLOW_RUN_PX
                / PROFILE_OUTWARD_STEP_PX
            )
        ),
    )

    candidates = []

    for i in range(
        min_yellow_samples,
        len(smooth) - min_nonyellow_samples,
    ):
        if not (
            yellow[i - 1]
            and not yellow[i]
        ):
            continue

        if not np.all(
            yellow[
                i - min_yellow_samples:i
            ]
        ):
            continue

        if not np.all(
            ~yellow[
                i:i + min_nonyellow_samples
            ]
        ):
            continue

        left_value = float(
            np.mean(
                smooth[
                    i - min_yellow_samples:i
                ]
            )
        )

        right_value = float(
            np.mean(
                smooth[
                    i:i + min_nonyellow_samples
                ]
            )
        )

        drop = float(
            left_value
            - right_value
        )

        if drop < PROFILE_MIN_DROP:
            continue

        boundary_offset = float(
            0.5
            * (
                offsets[i - 1]
                + offsets[i]
            )
        )

        candidates.append(
            {
                "offset_px": boundary_offset,
                "drop": drop,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda row: row["offset_px"],
    )


def _median_filter_offsets(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < 3 or window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    radius = window // 2
    out = values.copy()

    for i in range(len(values)):
        i0 = max(0, i - radius)
        i1 = min(
            len(values),
            i + radius + 1,
        )

        out[i] = np.median(
            values[i0:i1]
        )

    return out


def _extract_side_profile_rows(
    *,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
    safe_side_offset_px: float,
):
    p1, p2, t, _, n_out = _side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(p2 - p1)
    )

    tangential_positions = np.linspace(
        0.04 * side_length,
        0.96 * side_length,
        PROFILE_TANGENTIAL_SAMPLES,
        dtype=np.float64,
    )

    max_outward = float(
        safe_side_offset_px
        + PROFILE_EXTRA_OUTWARD_MARGIN_PX
    )

    outward_offsets = np.arange(
        PROFILE_OUTWARD_MIN_PX,
        max_outward
        + 0.5 * PROFILE_OUTWARD_STEP_PX,
        PROFILE_OUTWARD_STEP_PX,
        dtype=np.float64,
    )

    rows = []

    for tangential in tangential_positions:
        base = p1 + float(tangential) * t

        ray_points = (
            base[None, :]
            + outward_offsets[:, None]
            * n_out[None, :]
        )

        profile = _sample_scalar_map(
            soft_support,
            ray_points,
        )

        profile_safe = np.where(
            np.isfinite(profile),
            profile,
            0.0,
        )

        exit_info = _find_final_yellow_exit(
            profile_safe,
            outward_offsets,
        )

        if exit_info is None:
            continue

        rows.append(
            {
                "tangential_px": float(tangential),
                "selected_outward_px": float(
                    exit_info["offset_px"]
                ),
                "drop": float(
                    exit_info["drop"]
                ),
            }
        )

    if rows:
        offsets = np.asarray(
            [
                row["selected_outward_px"]
                for row in rows
            ],
            dtype=np.float64,
        )

        smoothed = _median_filter_offsets(
            offsets,
            RIDGE_OFFSET_MEDIAN_WINDOW,
        )

        for row, value in zip(
            rows,
            smoothed,
        ):
            row["selected_outward_px"] = float(value)

    valid_fraction = float(
        len(rows)
        / max(
            len(tangential_positions),
            1,
        )
    )

    return rows, valid_fraction


# ============================================================
# LOCAL t-d RANSAC
# ============================================================

def _fit_td_line_from_two_points(
    t1: float,
    d1: float,
    t2: float,
    d2: float,
):
    dt = float(t2 - t1)

    if abs(dt) < 1e-9:
        return None

    a = float(
        (d2 - d1)
        / dt
    )

    b = float(
        d1
        - a * t1
    )

    return a, b


def _fit_td_least_squares(
    t: np.ndarray,
    d: np.ndarray,
):
    A = np.column_stack(
        [
            np.asarray(
                t,
                dtype=np.float64,
            ),
            np.ones(
                len(t),
                dtype=np.float64,
            ),
        ]
    )

    solution, *_ = np.linalg.lstsq(
        A,
        np.asarray(
            d,
            dtype=np.float64,
        ),
        rcond=None,
    )

    return (
        float(solution[0]),
        float(solution[1]),
    )


def _evaluate_td_ransac_model(
    *,
    t: np.ndarray,
    d: np.ndarray,
    inliers: np.ndarray,
    a: float,
    b: float,
    side_length: float,
) -> dict:
    n_total = len(t)
    n_inliers = int(
        np.count_nonzero(inliers)
    )

    if n_inliers < 2:
        return {
            "score": -float("inf"),
        }

    t_in = t[inliers]
    d_in = d[inliers]

    predicted = (
        a * t_in
        + b
    )

    residuals = np.abs(
        d_in
        - predicted
    )

    t_span_px = float(
        np.max(t_in)
        - np.min(t_in)
    )

    t_span_fraction = float(
        np.clip(
            t_span_px
            / max(
                side_length,
                1e-9,
            ),
            0.0,
            1.0,
        )
    )

    inlier_fraction = float(
        n_inliers
        / max(
            n_total,
            1,
        )
    )

    median_abs_residual = float(
        np.median(residuals)
    )

    residual_score = float(
        math.exp(
            -median_abs_residual
        )
    )

    score = float(
        W_RANSAC_INLIER_FRACTION
        * inlier_fraction
        + W_RANSAC_TANGENTIAL_SPAN
        * t_span_fraction
        + W_RANSAC_RESIDUAL
        * residual_score
    )

    return {
        "inlier_fraction": inlier_fraction,
        "t_span_fraction": t_span_fraction,
        "median_abs_residual_px": median_abs_residual,
        "score": score,
    }


def _ransac_fit_td(
    *,
    t: np.ndarray,
    d: np.ndarray,
    threshold_px: float,
    side_length: float,
) -> dict:
    t = np.asarray(
        t,
        dtype=np.float64,
    )

    d = np.asarray(
        d,
        dtype=np.float64,
    )

    if len(t) < 2:
        raise RuntimeError(
            "Too few points for RANSAC."
        )

    rng = np.random.default_rng(
        RANSAC_RANDOM_SEED
    )

    best = None

    for _ in range(RANSAC_ITERATIONS):
        i, j = rng.choice(
            len(t),
            size=2,
            replace=False,
        )

        model = _fit_td_line_from_two_points(
            float(t[i]),
            float(d[i]),
            float(t[j]),
            float(d[j]),
        )

        if model is None:
            continue

        a, b = model

        residuals = np.abs(
            d - (a * t + b)
        )

        inliers = (
            residuals
            <= float(threshold_px)
        )

        if np.count_nonzero(inliers) < 2:
            continue

        a_refit, b_refit = _fit_td_least_squares(
            t[inliers],
            d[inliers],
        )

        residuals_refit = np.abs(
            d
            - (
                a_refit * t
                + b_refit
            )
        )

        inliers_refit = (
            residuals_refit
            <= float(threshold_px)
        )

        if np.count_nonzero(inliers_refit) < 2:
            continue

        a_final, b_final = _fit_td_least_squares(
            t[inliers_refit],
            d[inliers_refit],
        )

        metrics = _evaluate_td_ransac_model(
            t=t,
            d=d,
            inliers=inliers_refit,
            a=a_final,
            b=b_final,
            side_length=side_length,
        )

        candidate = {
            "threshold_px": float(threshold_px),
            "a": a_final,
            "b": b_final,
            "inliers": inliers_refit,
            **metrics,
        }

        if (
            best is None
            or candidate["score"]
            > best["score"]
        ):
            best = candidate

    if best is None:
        raise RuntimeError(
            f"RANSAC failed for threshold={threshold_px:.2f}"
        )

    return best


def _td_model_to_xy_line(
    *,
    safe_quad: np.ndarray,
    side_index: int,
    a: float,
    b: float,
):
    p1, p2, t_axis, _, n_out = _side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(p2 - p1)
    )

    q0 = (
        p1
        + b * n_out
    )

    q1 = (
        p1
        + side_length * t_axis
        + (
            a * side_length
            + b
        )
        * n_out
    )

    direction = _normalize(
        q1 - q0
    )

    return q0, direction


def _refine_one_side_ransac(
    *,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
    safe_side_offset_px: float,
):
    rows, valid_fraction = _extract_side_profile_rows(
        soft_support=soft_support,
        safe_quad=safe_quad,
        side_index=side_index,
        safe_side_offset_px=safe_side_offset_px,
    )

    trustworthy = bool(
        len(rows) >= MIN_RIDGE_POINTS
        and valid_fraction >= MIN_VALID_RIDGE_FRACTION
    )

    if not trustworthy:
        p1, _, t_axis, _, _ = _side_local_frame(
            safe_quad,
            side_index,
        )

        return {
            "fallback": True,
            "point": p1.copy(),
            "direction": t_axis.copy(),
        }

    t_values = np.asarray(
        [
            row["tangential_px"]
            for row in rows
        ],
        dtype=np.float64,
    )

    d_values = np.asarray(
        [
            row["selected_outward_px"]
            for row in rows
        ],
        dtype=np.float64,
    )

    p1, p2, _, _, _ = _side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(p2 - p1)
    )

    models = []

    for threshold_px in RANSAC_THRESHOLDS_PX:
        candidate = _ransac_fit_td(
            t=t_values,
            d=d_values,
            threshold_px=threshold_px,
            side_length=side_length,
        )

        candidate["passes_span_guard"] = bool(
            candidate["t_span_fraction"]
            >= MIN_RANSAC_TANGENTIAL_SPAN_FRACTION
        )

        models.append(candidate)

    eligible = [
        row
        for row in models
        if row["passes_span_guard"]
    ]

    best = max(
        eligible if eligible else models,
        key=lambda row: row["score"],
    )

    point, direction = _td_model_to_xy_line(
        safe_quad=safe_quad,
        side_index=side_index,
        a=best["a"],
        b=best["b"],
    )

    return {
        "fallback": False,
        "point": point,
        "direction": direction,
    }


def _refined_quad_from_lines(
    fitted_sides: list[dict],
) -> np.ndarray:
    top, right, bottom, left = fitted_sides

    tl = _line_intersection(
        top["point"],
        top["direction"],
        left["point"],
        left["direction"],
    )

    tr = _line_intersection(
        top["point"],
        top["direction"],
        right["point"],
        right["direction"],
    )

    br = _line_intersection(
        bottom["point"],
        bottom["direction"],
        right["point"],
        right["direction"],
    )

    bl = _line_intersection(
        bottom["point"],
        bottom["direction"],
        left["point"],
        left["direction"],
    )

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float64,
    )


def _snap_corner_to_second_mask(
    corner: np.ndarray,
    second_mask: np.ndarray,
) -> np.ndarray:
    ys, xs = np.nonzero(
        second_mask > 0
    )

    if len(xs) == 0:
        return corner.copy()

    pts = np.column_stack(
        [xs, ys]
    ).astype(np.float64)

    dist = np.linalg.norm(
        pts
        - corner[None, :],
        axis=1,
    )

    index = int(
        np.argmin(dist)
    )

    if dist[index] <= CORNER_SNAP_RADIUS_PX:
        return (
            0.75 * corner
            + 0.25 * pts[index]
        )

    return corner.copy()


# ============================================================
# PUBLIC METHOD
# ============================================================

def corners_from_yellow_exit_ransac(
    contour,
    x_offset,
    y_offset,
    crop=None,
    mask=None,
    debug=False,
    debug_name="yellow_exit_ransac",
):
    """
    Refine yellow license-plate corners using the final experiment:

      clean yellow mask
      -> side-specific safe inner quad
      -> learn boundary-yellow color from safe lines
      -> normal rays outward
      -> final stable yellow exit
      -> local t-d RANSAC per side
      -> line intersections
      -> tiny second-mask corner snap

    Parameters mirror the other corner_refinement methods so this function
    can be plugged directly into yellow_refinement.py.

    Returns points in FULL IMAGE coordinates, ordered:
        TL, TR, BR, BL
    """
    del contour, debug, debug_name

    if crop is None:
        raise ValueError(
            "yellow_exit_ransac requires crop"
        )

    if mask is None:
        raise ValueError(
            "yellow_exit_ransac requires clean yellow mask"
        )

    clean_mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    min_quad = _min_area_rect_quad(
        clean_mask
    )

    raw_mask, raw_soft = _build_raw_yellow_support(
        crop
    )

    side_offsets = _find_side_specific_safe_offsets(
        min_quad=min_quad,
        clean_mask=clean_mask,
        raw_mask=raw_mask,
        raw_soft=raw_soft,
    )

    safe_quad = _build_quad_from_side_offsets(
        min_quad=min_quad,
        side_offsets=side_offsets,
    )

    lab_image, dark_mask, boundary_samples = (
        _collect_boundary_line_color_samples(
            crop=crop,
            clean_mask=clean_mask,
            safe_quad=safe_quad,
        )
    )

    model = _robust_lab_model(
        boundary_samples
    )

    (
        soft_support,
        hard_color_mask,
        _soft_threshold,
    ) = _build_color_similarity_maps(
        lab_image=lab_image,
        dark_mask=dark_mask,
        model=model,
    )

    second_mask = _build_second_remainder_mask(
        hard_color_mask=hard_color_mask,
        safe_quad=safe_quad,
    )

    fitted_sides = []

    for side_index in range(4):
        fitted = _refine_one_side_ransac(
            soft_support=soft_support,
            safe_quad=safe_quad,
            side_index=side_index,
            safe_side_offset_px=float(
                side_offsets[side_index]
            ),
        )

        fitted_sides.append(fitted)

    refined_quad = _refined_quad_from_lines(
        fitted_sides
    )

    refined_quad = np.asarray(
        [
            _snap_corner_to_second_mask(
                corner,
                second_mask,
            )
            for corner in refined_quad
        ],
        dtype=np.float64,
    )

    refined_quad[:, 0] += float(x_offset)
    refined_quad[:, 1] += float(y_offset)

    return refined_quad.astype(np.float32)
