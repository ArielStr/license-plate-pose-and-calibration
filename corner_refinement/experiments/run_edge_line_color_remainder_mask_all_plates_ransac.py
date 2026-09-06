from __future__ import annotations

from pathlib import Path
import csv
import math
import sys

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

# Put this file at:
#   corner_refinement/experiments/run_edge_line_color_remainder_mask_two_plates.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from single_plate_pose.detection import load_roboflow_result
from multi_plate_new.detection import (
    extract_plate_detections,
    non_max_suppression,
)
from corner_refinement.common import (
    bbox_from_prediction_points_xy_padding,
    crop_from_bbox,
)
from corner_refinement.yellow_utils import (
    estimate_dynamic_yellow_hsv,
    choose_geometry_config,
)


# ============================================================
# CONFIG
# ============================================================

VIDEO_NAME = "test_video"

FRAME_NAMES = [

    "frame_000010",
    "frame_000040",
    "frame_000050",
]

BASE_INPUT_DIR = (
    PROJECT_ROOT
    / "video_calibration"
    / "debug"
    / "sampled_frames"
)

BASE_JSON_DIR = (
    PROJECT_ROOT
    / "roboflow_jsons"
    / VIDEO_NAME
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "corner_refinement"
    / "debug"
    / "final_yellow_exit_all_plates_two_frames"
)

MIN_CONFIDENCE = 0.35
MIN_DETECTION_AREA = 300.0
NMS_IOU_THRESHOLD = 0.5

CROP_PADDING_RATIO_X = 0.02
CROP_PADDING_RATIO_Y = 0.15


# ------------------------------------------------------------
# Fast safe inner rectangle:
# minAreaRect -> shift all four lines inward until each side
# lies almost entirely inside the cleaned yellow mask.
# ------------------------------------------------------------

SIDE_COVERAGE_SAMPLES = 120
SIDE_YELLOW_COVERAGE_REQUIRED = 0.98

SHRINK_STEP_PX = 0.5
MAX_SHRINK_PX = 15.0

# ============================================================
# SIDE-SPECIFIC SAFE-RECT VALIDATION USING RAW YELLOW COLOR
# ============================================================
#
# The previous safe rectangle was validated only against the cleaned
# morphological yellow mask. That can falsely accept a line that sits on
# filled/closed pixels rather than real yellow in the original crop.
#
# New rule for EACH side independently:
#   - enough points must belong to the cleaned yellow component
#   - enough points on the actual RGB/HSV crop must be raw yellow
#   - also check a second sample line +1 px inward
#
# Each side is shifted inward independently until it satisfies the rule.
RAW_SIDE_SAMPLE_COUNT = 140
RAW_YELLOW_FRACTION_REQUIRED = 0.78
RAW_YELLOW_INWARD_FRACTION_REQUIRED = 0.88
RAW_CHECK_INWARD_PX = 1.0

# Soft rescue around the dynamic HSV interval. This allows blurred yellow
# boundary pixels that fall slightly outside the hard inRange threshold.
RAW_YELLOW_SOFT_MIN = 0.58


# ------------------------------------------------------------
# Learn ONLY the yellow color on the safe-rectangle LINES.
#
# If a side has too few usable samples after dark-pixel removal,
# add exactly one extra line 1 px inward.
# ------------------------------------------------------------

EDGE_LINE_SAMPLES_PER_SIDE = 160
EXTRA_INWARD_PX = 1.0
MIN_EDGE_COLOR_SAMPLES_PER_SIDE = 20

# If the exact line + 1 px inward still gives too few raw-yellow samples,
# do NOT fail. Use the brightest/non-black samples that are geometrically
# on the cleaned yellow component. This prevents blurred/dim yellow borders
# from being thrown away just because they are darker than the plate center.
MIN_FALLBACK_VALUE = 35

BLACK_V_MAX = 35
BLACK_LAB_L_MAX = 45

# Robust Lab model.
LAB_SIGMA_FLOOR = np.asarray(
    [8.0, 7.0, 7.0],
    dtype=np.float64,
)

# Hard mask from distance to learned edge-yellow model.
EDGE_COLOR_Z_THRESHOLD = 2.35

# No broad blur: we want the selected color itself to remain visible.
SOFT_MAP_BLUR_KSIZE = 0


# ------------------------------------------------------------
# "SECOND MASK" requested in the experiment:
#
# 1) learn boundary-yellow from the 4 safe lines
# 2) classify the WHOLE crop by that color
# 3) remove the safe inner polygon
# 4) keep only the remainder outside it
#
# This explicitly visualizes:
#       second_mask = learned_color_mask AND outside_safe_rect
# ------------------------------------------------------------

INNER_EXCLUSION_EXTRA_PX = 0.0


# ------------------------------------------------------------
# NEW side refinement:
# use the SECOND MASK pixels themselves, not offset scanning.
#
# For each side:
#   1) build a narrow band around / just outside the safe side
#   2) keep SECOND_MASK pixels inside that band
#   3) remove tiny connected components
#   4) prefer components long in the side direction and close
#      to the safe side
#   5) fit a new line with cv2.fitLine
# ------------------------------------------------------------

SIDE_BAND_INWARD_PX = 1.5
SIDE_BAND_OUTWARD_PX = 12.0
SIDE_BAND_TANGENTIAL_PAD_PX = 4.0

MIN_COMPONENT_PIXELS = 8
MIN_COMPONENT_SPAN_FRACTION = 0.20

# Component ranking.
W_COMPONENT_SPAN = 0.55
W_COMPONENT_PIXEL_COUNT = 0.20
W_COMPONENT_PROXIMITY = 0.25

# Optional robust trimming before final fit.
FITLINE_DISTANCE_TRIM_PERCENTILE = 85.0

# ------------------------------------------------------------
# OUTER ENVELOPE EXTRACTION
# ------------------------------------------------------------
#
# After selecting the best SECOND_MASK component for a side,
# do NOT fit all component pixels. For each tangential bin along
# the side, keep only the OUTERMOST pixel in the outward-normal
# direction. Then fitLine only to that envelope.
#
# This should move the fitted line from the middle of a thick
# white component to its true exterior boundary.
ENVELOPE_BIN_SIZE_PX = 1.0
MIN_ENVELOPE_POINTS = 8

# Small median smoothing of outward envelope coordinate across
# neighboring bins. Set to 1 to disable.
ENVELOPE_MEDIAN_WINDOW = 3

# Corner snap sanity check.
CORNER_SNAP_RADIUS_PX = 3.0

# ------------------------------------------------------------
# NORMAL-PROFILE RIDGE EXTRACTION ON SOFT COLOR SIMILARITY
# ------------------------------------------------------------
# For many points along each safe side, cast a short ray OUTWARD.
# Along each ray:
#   1) sample the learned-color soft similarity
#   2) convert it to a local binary support using the SAME color threshold
#   3) find the FIRST stable support run outside the safe side
#   4) inside that first run, take the point with MAX soft similarity
# Then robustly fit one line through all such ridge points.
#
# This intentionally avoids global connected-component selection, so a large
# car region farther away cannot beat the actual first plate boundary.
PROFILE_TANGENTIAL_SAMPLES = 120
PROFILE_OUTWARD_MIN_PX = 0.0
PROFILE_OUTWARD_STEP_PX = 0.25

# Search range is now dynamic per side:
#   safe-side inward offset + this small exterior margin.
PROFILE_EXTRA_OUTWARD_MARGIN_PX = 2.5

# Smooth each 1D similarity profile before detecting the final yellow exit.
PROFILE_SMOOTH_KERNEL = np.asarray(
    [0.15, 0.70, 0.15],
    dtype=np.float64,
)

# We are looking for the LAST stable transition from plate-yellow-like
# support to non-yellow support.
PROFILE_YELLOW_SUPPORT_THRESHOLD = 0.12
PROFILE_MIN_YELLOW_RUN_PX = 1.0
PROFILE_MIN_NONYELLOW_RUN_PX = 0.75

# A valid boundary should show an actual drop in similarity.
PROFILE_MIN_DROP = 0.035

# Need enough successful normal cuts before trusting a fitted side.
MIN_VALID_RIDGE_FRACTION = 0.30
MIN_RIDGE_POINTS = 12

# Robust trimming of boundary points before final line fit.
RIDGE_LINE_TRIM_PERCENTILE = 82.0

# Mild median smoothing of selected outward offsets along the side.
RIDGE_OFFSET_MEDIAN_WINDOW = 5


# Debug
LINE_THICKNESS = 1
POINT_RADIUS = 2


# ============================================================
# INPUT
# ============================================================

def build_prediction_dict_from_detection(detection) -> dict:
    x_min, y_min, x_max, y_max = np.asarray(
        detection.bbox_xyxy,
        dtype=np.float64,
    )

    return {
        "x": float(0.5 * (x_min + x_max)),
        "y": float(0.5 * (y_min + y_max)),
        "width": float(x_max - x_min),
        "height": float(y_max - y_min),
        "points": [
            {"x": float(x_min), "y": float(y_min)},
            {"x": float(x_max), "y": float(y_min)},
            {"x": float(x_max), "y": float(y_max)},
            {"x": float(x_min), "y": float(y_max)},
        ],
    }


def get_frame_detections(frame_name: str):
    image_path = BASE_INPUT_DIR / f"{frame_name}.jpg"
    json_path = BASE_JSON_DIR / f"{frame_name}.json"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Cached JSON not found: {json_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    raw_result = load_roboflow_result(json_path)

    detections = extract_plate_detections(
        raw_result,
        min_confidence=MIN_CONFIDENCE,
        min_area=MIN_DETECTION_AREA,
    )

    detections = non_max_suppression(
        detections,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    return image, detections


def load_all_plate_indices(
    frame_name: str,
) -> list[int]:
    image_path = BASE_INPUT_DIR / f"{frame_name}.jpg"
    json_path = BASE_JSON_DIR / f"{frame_name}.json"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Cached JSON not found: {json_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    raw_result = load_roboflow_result(
        json_path
    )

    detections = extract_plate_detections(
        raw_result,
        min_confidence=MIN_CONFIDENCE,
        min_area=MIN_DETECTION_AREA,
    )

    detections = non_max_suppression(
        detections,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    print(
        f"{frame_name}: detected {len(detections)} plates after NMS"
    )

    return list(
        range(
            len(detections)
        )
    )


def load_target_plate(
    frame_name: str,
    plate_index: int,
):
    image, detections = get_frame_detections(frame_name)

    if not 0 <= plate_index < len(detections):
        raise IndexError(
            f"{frame_name}: plate_index={plate_index}, "
            f"detections={len(detections)}"
        )

    pred = build_prediction_dict_from_detection(
        detections[plate_index]
    )

    bbox = bbox_from_prediction_points_xy_padding(
        pred,
        image.shape,
        padding_x_ratio=CROP_PADDING_RATIO_X,
        padding_y_ratio=CROP_PADDING_RATIO_Y,
    )

    crop, x_offset, y_offset = crop_from_bbox(
        image,
        bbox,
    )

    return crop, x_offset, y_offset, pred


# ============================================================
# YELLOW MASK
# ============================================================

def create_clean_yellow_mask(
    crop: np.ndarray,
    pred_width: float,
):
    config = choose_geometry_config(pred_width)

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    dynamic_result = estimate_dynamic_yellow_hsv(crop)

    if dynamic_result is None:
        raise RuntimeError(
            "Dynamic HSV estimation failed."
        )

    lower_yellow, upper_yellow, _, _ = dynamic_result

    raw_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    mask = raw_mask.copy()

    if config["open_kernel"] is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            config["open_kernel"],
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config["open_iterations"],
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config["close_kernel"],
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config["close_iterations"],
    )

    if config["fill_kernel"] is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            config["fill_kernel"],
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config["fill_iterations"],
        )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        raise RuntimeError(
            "No connected yellow component."
        )

    largest_label = (
        1
        + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
    )

    clean_mask = np.zeros_like(mask)

    clean_mask[
        labels == largest_label
    ] = 255

    # --------------------------------------------------------
    # Soft RAW-yellow likelihood from the original HSV crop.
    # No morphology here.
    #
    # Pixels inside the dynamic HSV interval receive support ~1.
    # Near-boundary colors receive partial support according to their
    # distance from the interval. This is only for validating that the
    # shrunken side itself really lies on yellow plate pixels.
    # --------------------------------------------------------
    hsv_f = hsv.astype(np.float64)

    lower = np.asarray(
        lower_yellow,
        dtype=np.float64,
    ).reshape(1, 1, 3)

    upper = np.asarray(
        upper_yellow,
        dtype=np.float64,
    ).reshape(1, 1, 3)

    below = np.maximum(
        lower - hsv_f,
        0.0,
    )

    above = np.maximum(
        hsv_f - upper,
        0.0,
    )

    outside = below + above

    channel_scales = np.asarray(
        [8.0, 35.0, 45.0],
        dtype=np.float64,
    ).reshape(1, 1, 3)

    normalized = (
        outside
        / channel_scales
    )

    distance = np.sqrt(
        np.sum(
            normalized * normalized,
            axis=2,
        )
    )

    raw_soft = np.exp(
        -0.5
        * distance
        * distance
    ).astype(np.float64)

    raw_soft[
        raw_mask > 0
    ] = np.maximum(
        raw_soft[
            raw_mask > 0
        ],
        0.95,
    )

    return (
        clean_mask,
        raw_mask,
        raw_soft,
    )


# ============================================================
# GEOMETRY
# ============================================================

def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    return vector / (
        np.linalg.norm(vector) + 1e-12
    )


def order_quad_tl_tr_br_bl(
    points: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(
        points,
        dtype=np.float64,
    )

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


def min_area_rect_quad(
    clean_mask: np.ndarray,
) -> np.ndarray:
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "No contour in clean yellow mask."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    return order_quad_tl_tr_br_bl(box)


def quad_center(
    quad: np.ndarray,
) -> np.ndarray:
    return np.mean(
        quad,
        axis=0,
    )


def line_inward_normal(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    direction = normalize(
        p2 - p1
    )

    normal = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    midpoint = 0.5 * (p1 + p2)

    if np.dot(
        center - midpoint,
        normal,
    ) < 0:
        normal = -normal

    return normal


def sample_line(
    p1: np.ndarray,
    p2: np.ndarray,
    count: int,
    trim_fraction: float = 0.0,
) -> np.ndarray:
    t0 = float(trim_fraction)
    t1 = 1.0 - float(trim_fraction)

    t = np.linspace(
        t0,
        t1,
        count,
        dtype=np.float64,
    )

    return (
        (1.0 - t[:, None]) * p1[None, :]
        + t[:, None] * p2[None, :]
    )


def nearest_mask_values(
    mask: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    x = np.round(
        points[:, 0]
    ).astype(np.int32)

    y = np.round(
        points[:, 1]
    ).astype(np.int32)

    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < mask.shape[1])
        & (y < mask.shape[0])
    )

    values = np.zeros(
        len(points),
        dtype=np.float64,
    )

    values[valid] = (
        mask[
            y[valid],
            x[valid],
        ]
        > 0
    ).astype(np.float64)

    return values


def line_intersection(
    p: np.ndarray,
    d: np.ndarray,
    q: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    A = np.column_stack(
        [d, -e]
    )

    if abs(
        float(np.linalg.det(A))
    ) < 1e-9:
        raise RuntimeError(
            "Parallel / degenerate lines."
        )

    t, _s = np.linalg.solve(
        A,
        q - p,
    )

    return p + float(t) * d


def shift_quad_inward(
    quad: np.ndarray,
    distance_px: float,
) -> np.ndarray:
    center = quad_center(quad)

    lines = []

    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]

        d = normalize(
            p2 - p1
        )

        n_in = line_inward_normal(
            p1,
            p2,
            center,
        )

        lines.append(
            (
                p1
                + float(distance_px)
                * n_in,
                d,
            )
        )

    tl = line_intersection(
        lines[3][0],
        lines[3][1],
        lines[0][0],
        lines[0][1],
    )

    tr = line_intersection(
        lines[0][0],
        lines[0][1],
        lines[1][0],
        lines[1][1],
    )

    br = line_intersection(
        lines[1][0],
        lines[1][1],
        lines[2][0],
        lines[2][1],
    )

    bl = line_intersection(
        lines[2][0],
        lines[2][1],
        lines[3][0],
        lines[3][1],
    )

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float64,
    )


def side_coverages(
    quad: np.ndarray,
    clean_mask: np.ndarray,
) -> list[float]:
    coverages = []

    for i in range(4):
        points = sample_line(
            quad[i],
            quad[(i + 1) % 4],
            SIDE_COVERAGE_SAMPLES,
        )

        values = nearest_mask_values(
            clean_mask,
            points,
        )

        coverages.append(
            float(np.mean(values))
        )

    return coverages


def shrink_until_safe(
    min_quad: np.ndarray,
    clean_mask: np.ndarray,
):
    distances = np.arange(
        0.0,
        MAX_SHRINK_PX
        + 0.5 * SHRINK_STEP_PX,
        SHRINK_STEP_PX,
        dtype=np.float64,
    )

    last = None

    for distance_px in distances:
        safe_quad = shift_quad_inward(
            min_quad,
            float(distance_px),
        )

        coverages = side_coverages(
            safe_quad,
            clean_mask,
        )

        last = (
            safe_quad,
            float(distance_px),
            coverages,
        )

        if (
            min(coverages)
            >= SIDE_YELLOW_COVERAGE_REQUIRED
        ):
            return last

    return last



# ============================================================
# SIDE-SPECIFIC RAW-YELLOW SAFE RECTANGLE
# ============================================================

def sample_scalar_map(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return bilinear_sample_multichannel(
        image[:, :, None],
        points,
    )[:, 0]


def side_line_from_inward_offset(
    *,
    min_quad: np.ndarray,
    side_index: int,
    inward_offset_px: float,
):
    center = quad_center(min_quad)

    p1 = min_quad[side_index]
    p2 = min_quad[(side_index + 1) % 4]

    direction = normalize(
        p2 - p1
    )

    n_in = line_inward_normal(
        p1,
        p2,
        center,
    )

    shifted_p1 = (
        p1
        + float(inward_offset_px)
        * n_in
    )

    shifted_p2 = (
        p2
        + float(inward_offset_px)
        * n_in
    )

    return (
        shifted_p1,
        shifted_p2,
        direction,
        n_in,
    )


def evaluate_raw_yellow_side(
    *,
    min_quad: np.ndarray,
    side_index: int,
    inward_offset_px: float,
    clean_mask: np.ndarray,
    raw_mask: np.ndarray,
    raw_soft: np.ndarray,
) -> dict:
    p1, p2, _direction, n_in = side_line_from_inward_offset(
        min_quad=min_quad,
        side_index=side_index,
        inward_offset_px=inward_offset_px,
    )

    line = sample_line(
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
            nearest_mask_values(
                clean_mask,
                line,
            )
        )
    )

    raw_hard_fraction = float(
        np.mean(
            nearest_mask_values(
                raw_mask,
                line,
            )
        )
    )

    raw_hard_inward_fraction = float(
        np.mean(
            nearest_mask_values(
                raw_mask,
                line_inward,
            )
        )
    )

    soft_values = sample_scalar_map(
        raw_soft,
        line,
    )

    soft_inward_values = sample_scalar_map(
        raw_soft,
        line_inward,
    )

    soft_fraction = float(
        np.mean(
            soft_values
            >= RAW_YELLOW_SOFT_MIN
        )
    )

    soft_inward_fraction = float(
        np.mean(
            soft_inward_values
            >= RAW_YELLOW_SOFT_MIN
        )
    )

    # Accept either strong hard-HSV support or strong soft-HSV support.
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
        "raw_hard_fraction": raw_hard_fraction,
        "raw_soft_fraction": soft_fraction,
        "raw_fraction": raw_fraction,
        "raw_hard_inward_fraction": raw_hard_inward_fraction,
        "raw_soft_inward_fraction": soft_inward_fraction,
        "raw_inward_fraction": raw_inward_fraction,
        "p1": p1,
        "p2": p2,
        "n_in": n_in,
    }


def find_side_specific_safe_offsets(
    *,
    min_quad: np.ndarray,
    clean_mask: np.ndarray,
    raw_mask: np.ndarray,
    raw_soft: np.ndarray,
):
    side_names = [
        "top",
        "right",
        "bottom",
        "left",
    ]

    offsets = []
    metrics_by_side = []

    for side_index, side_name in enumerate(
        side_names
    ):
        selected = None

        for offset_px in np.arange(
            0.0,
            MAX_SHRINK_PX
            + 0.5 * SHRINK_STEP_PX,
            SHRINK_STEP_PX,
            dtype=np.float64,
        ):
            metrics = evaluate_raw_yellow_side(
                min_quad=min_quad,
                side_index=side_index,
                inward_offset_px=float(offset_px),
                clean_mask=clean_mask,
                raw_mask=raw_mask,
                raw_soft=raw_soft,
            )

            if metrics["valid"]:
                selected = {
                    "side": side_name,
                    "offset_px": float(offset_px),
                    **metrics,
                }
                break

        if selected is None:
            # Conservative fallback: use maximum shrink and report it.
            metrics = evaluate_raw_yellow_side(
                min_quad=min_quad,
                side_index=side_index,
                inward_offset_px=float(MAX_SHRINK_PX),
                clean_mask=clean_mask,
                raw_mask=raw_mask,
                raw_soft=raw_soft,
            )

            selected = {
                "side": side_name,
                "offset_px": float(MAX_SHRINK_PX),
                **metrics,
            }

        offsets.append(
            selected["offset_px"]
        )

        metrics_by_side.append(
            selected
        )

    return offsets, metrics_by_side


def build_quad_from_side_offsets(
    *,
    min_quad: np.ndarray,
    side_offsets: list[float],
) -> np.ndarray:
    center = quad_center(min_quad)

    lines = []

    for side_index, offset_px in enumerate(
        side_offsets
    ):
        p1 = min_quad[side_index]
        p2 = min_quad[(side_index + 1) % 4]

        direction = normalize(
            p2 - p1
        )

        n_in = line_inward_normal(
            p1,
            p2,
            center,
        )

        line_point = (
            p1
            + float(offset_px)
            * n_in
        )

        lines.append(
            (
                line_point,
                direction,
            )
        )

    top, right, bottom, left = lines

    tl = line_intersection(
        top[0], top[1],
        left[0], left[1],
    )

    tr = line_intersection(
        top[0], top[1],
        right[0], right[1],
    )

    br = line_intersection(
        bottom[0], bottom[1],
        right[0], right[1],
    )

    bl = line_intersection(
        bottom[0], bottom[1],
        left[0], left[1],
    )

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float64,
    )


def save_safe_side_validation_debug(
    *,
    case_dir: Path,
    crop: np.ndarray,
    min_quad: np.ndarray,
    safe_quad: np.ndarray,
    raw_mask: np.ndarray,
    raw_soft: np.ndarray,
    side_metrics: list[dict],
):
    safe_view = draw_quad(
        crop,
        safe_quad,
        (0, 255, 0),
        1,
    )

    raw_mask_bgr = cv2.cvtColor(
        raw_mask,
        cv2.COLOR_GRAY2BGR,
    )

    raw_soft_vis = np.clip(
        raw_soft * 255.0,
        0,
        255,
    ).astype(np.uint8)

    raw_soft_color = cv2.applyColorMap(
        raw_soft_vis,
        cv2.COLORMAP_TURBO,
    )

    # Overlay the individually validated side lines.
    side_view = crop.copy()

    side_colors = [
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 165, 255),
    ]

    for metric, color in zip(
        side_metrics,
        side_colors,
    ):
        p1 = np.asarray(
            metric["p1"],
            dtype=np.float64,
        )

        p2 = np.asarray(
            metric["p2"],
            dtype=np.float64,
        )

        cv2.line(
            side_view,
            tuple(
                np.round(p1).astype(np.int32)
            ),
            tuple(
                np.round(p2).astype(np.int32)
            ),
            color,
            2,
            cv2.LINE_AA,
        )

    views = [
        ("side-specific safe quad", safe_view),
        ("raw HSV yellow mask", raw_mask_bgr),
        ("raw yellow soft support", raw_soft_color),
        ("validated side lines", side_view),
    ]

    cell_w = 520
    rendered = []

    for title, image in views:
        scale = cell_w / image.shape[1]

        resized = cv2.resize(
            image,
            (
                cell_w,
                max(
                    1,
                    int(
                        round(
                            image.shape[0]
                            * scale
                        )
                    ),
                ),
            ),
            interpolation=(
                cv2.INTER_NEAREST
                if "mask" in title.lower()
                else cv2.INTER_LINEAR
            ),
        )

        bar = np.zeros(
            (42, cell_w, 3),
            dtype=np.uint8,
        )

        cv2.putText(
            bar,
            title,
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        rendered.append(
            np.vstack(
                [bar, resized]
            )
        )

    max_h = max(
        image.shape[0]
        for image in rendered
    )

    padded = []

    for image in rendered:
        canvas = np.zeros(
            (max_h, cell_w, 3),
            dtype=np.uint8,
        )

        canvas[
            :image.shape[0],
            :image.shape[1],
        ] = image

        padded.append(canvas)

    body = np.hstack(
        padded
    )

    header = np.zeros(
        (155, body.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        "SIDE-SPECIFIC SAFE RECT | validated against RAW yellow, not morphology only",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    offset_text = " | ".join(
        (
            f"{m['side']}={m['offset_px']:.1f}px "
            f"raw={m['raw_fraction']:.2f} "
            f"in={m['raw_inward_fraction']:.2f}"
        )
        for m in side_metrics
    )

    cv2.putText(
        header,
        offset_text,
        (12, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        (
            f"requirements | clean>={SIDE_YELLOW_COVERAGE_REQUIRED:.2f} | "
            f"raw line>={RAW_YELLOW_FRACTION_REQUIRED:.2f} | "
            f"raw +1px inward>={RAW_YELLOW_INWARD_FRACTION_REQUIRED:.2f}"
        ),
        (12, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (195, 195, 195),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        "Each side moves inward independently until these tests pass.",
        (12, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    panel = np.vstack(
        [header, body]
    )

    cv2.imwrite(
        str(
            case_dir
            / "SAFE_SIDE_VALIDATION_PANEL.jpg"
        ),
        panel,
    )

    csv_path = (
        case_dir
        / "safe_side_validation.csv"
    )

    csv_rows = []

    for metric in side_metrics:
        csv_rows.append(
            {
                key: value
                for key, value in metric.items()
                if key not in (
                    "p1",
                    "p2",
                    "n_in",
                )
            }
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                csv_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )


# ============================================================
# COLOR MODEL
# ============================================================

def bilinear_sample_multichannel(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    image = np.asarray(
        image,
        dtype=np.float64,
    )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

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
        (
            len(points),
            image.shape[2],
        ),
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


def bilinear_sample_scalar(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return bilinear_sample_multichannel(
        image[:, :, None],
        points,
    )[:, 0]


def build_dark_mask(
    crop: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB,
    )

    dark = (
        (hsv[:, :, 2] <= BLACK_V_MAX)
        | (lab[:, :, 0] <= BLACK_LAB_L_MAX)
    )

    return dark.astype(np.uint8)


def sample_dark_flags(
    dark_mask: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    x = np.round(
        points[:, 0]
    ).astype(np.int32)

    y = np.round(
        points[:, 1]
    ).astype(np.int32)

    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < dark_mask.shape[1])
        & (y < dark_mask.shape[0])
    )

    flags = np.ones(
        len(points),
        dtype=bool,
    )

    flags[valid] = (
        dark_mask[
            y[valid],
            x[valid],
        ]
        > 0
    )

    return flags


def collect_boundary_line_color_samples(
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

    # IMPORTANT:
    # The previous version used a very aggressive "dark" threshold.
    # That could classify the actual blurred/brownish yellow boundary itself
    # as black and delete an entire side. Here "dark" means genuinely black.
    dark_mask = build_dark_mask(crop)

    center = quad_center(safe_quad)

    all_samples = []
    debug = []
    counts = []

    side_names = [
        "top",
        "right",
        "bottom",
        "left",
    ]

    for i, side_name in enumerate(side_names):
        p1 = safe_quad[i]
        p2 = safe_quad[(i + 1) % 4]

        n_in = line_inward_normal(
            p1,
            p2,
            center,
        )

        # First: sample the FULL exact side line.
        line_points = sample_line(
            p1,
            p2,
            EDGE_LINE_SAMPLES_PER_SIDE,
        )

        line_lab = bilinear_sample_multichannel(
            lab_image,
            line_points,
        )

        line_hsv = bilinear_sample_multichannel(
            hsv_image,
            line_points,
        )

        line_dark = sample_dark_flags(
            dark_mask,
            line_points,
        )

        line_geom = (
            nearest_mask_values(
                clean_mask,
                line_points,
            )
            > 0.5
        )

        line_finite = np.all(
            np.isfinite(line_lab),
            axis=1,
        )

        # Keep line pixels that are geometrically on the yellow component
        # and are not genuinely black. We intentionally do NOT require the
        # original raw HSV yellow threshold here, because the boundary can be
        # blurred/darker than the interior yellow.
        keep_line = (
            (~line_dark)
            & line_geom
            & line_finite
            & (line_hsv[:, 2] >= MIN_FALLBACK_VALUE)
        )

        side_samples = [
            line_lab[keep_line]
        ]

        debug.append(
            {
                "side": side_name,
                "kind": "line",
                "points": line_points,
                "keep": keep_line,
            }
        )

        line_count = int(
            np.count_nonzero(keep_line)
        )

        extra_count = 0

        # If the exact edge line does not have enough usable yellow samples,
        # add exactly ONE line 1 px inward, as agreed.
        if line_count < MIN_EDGE_COLOR_SAMPLES_PER_SIDE:
            inward_points = (
                line_points
                + EXTRA_INWARD_PX
                * n_in[None, :]
            )

            inward_lab = bilinear_sample_multichannel(
                lab_image,
                inward_points,
            )

            inward_hsv = bilinear_sample_multichannel(
                hsv_image,
                inward_points,
            )

            inward_dark = sample_dark_flags(
                dark_mask,
                inward_points,
            )

            inward_geom = (
                nearest_mask_values(
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

            extra_count = int(
                np.count_nonzero(keep_inward)
            )

            debug.append(
                {
                    "side": side_name,
                    "kind": "+1px inward",
                    "points": inward_points,
                    "keep": keep_inward,
                }
            )

        usable = [
            samples
            for samples in side_samples
            if len(samples) > 0
        ]

        # Do not fail an entire plate because one side is dim.
        # If needed, use all finite geometrically-valid samples from the exact
        # line, except genuinely black pixels.
        fallback_count = 0

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

                fallback_count = int(
                    len(fallback_lab)
                )

                debug.append(
                    {
                        "side": side_name,
                        "kind": "fallback exact line",
                        "points": line_points,
                        "keep": fallback_keep,
                    }
                )

        if not usable:
            raise RuntimeError(
                f"No finite/non-black samples at all for {side_name}."
            )

        side_lab = np.vstack(
            usable
        )

        all_samples.append(
            side_lab
        )

        counts.append(
            {
                "side": side_name,
                "line_samples": line_count,
                "extra_inward_samples": extra_count,
                "fallback_samples": fallback_count,
                "total_samples": int(
                    len(side_lab)
                ),
            }
        )

    combined = np.vstack(
        all_samples
    )

    return (
        lab_image,
        dark_mask,
        combined,
        counts,
        debug,
    )



def robust_lab_model(
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
            f"Too few total boundary-color samples: {len(samples)}"
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
        "count": int(len(samples)),
    }


def lab_model_to_bgr_swatch(
    model: dict,
    size: int = 80,
) -> np.ndarray:
    lab_pixel = np.clip(
        np.round(
            model["median"]
        ),
        0,
        255,
    ).astype(np.uint8)

    lab_image = np.zeros(
        (1, 1, 3),
        dtype=np.uint8,
    )

    lab_image[0, 0] = lab_pixel

    bgr = cv2.cvtColor(
        lab_image,
        cv2.COLOR_LAB2BGR,
    )[0, 0]

    swatch = np.full(
        (size, size, 3),
        bgr,
        dtype=np.uint8,
    )

    return swatch


def build_color_similarity_maps(
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
    ).astype(np.float32)

    soft[
        dark_mask > 0
    ] = 0.0

    if (
        SOFT_MAP_BLUR_KSIZE
        and SOFT_MAP_BLUR_KSIZE >= 3
    ):
        k = int(
            SOFT_MAP_BLUR_KSIZE
        )

        if k % 2 == 0:
            k += 1

        soft = cv2.GaussianBlur(
            soft,
            (k, k),
            0,
        )

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

    hard[
        dark_mask > 0
    ] = 0

    return (
        soft.astype(np.float64),
        hard,
        z_distance,
        soft_threshold,
    )


def build_second_remainder_mask(
    *,
    hard_color_mask: np.ndarray,
    safe_quad: np.ndarray,
):
    inner_mask = np.zeros_like(
        hard_color_mask
    )

    quad_to_fill = safe_quad.copy()

    if INNER_EXCLUSION_EXTRA_PX > 0:
        quad_to_fill = shift_quad_inward(
            safe_quad,
            -float(INNER_EXCLUSION_EXTRA_PX),
        )

    cv2.fillConvexPoly(
        inner_mask,
        np.round(
            quad_to_fill
        ).astype(np.int32),
        255,
    )

    remainder_region = np.where(
        inner_mask > 0,
        0,
        255,
    ).astype(np.uint8)

    second_mask = cv2.bitwise_and(
        hard_color_mask,
        remainder_region,
    )

    return (
        second_mask,
        inner_mask,
        remainder_region,
    )


# ============================================================
# SOFT NORMAL-PROFILE RIDGE REFINEMENT
# ============================================================

def sample_scalar_bilinear(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return bilinear_sample_multichannel(
        image[:, :, None],
        points,
    )[:, 0]


def find_final_yellow_exit(
    profile: np.ndarray,
    outward_offsets: np.ndarray,
):
    """Find the final stable yellow-like -> non-yellow transition.

    Starting from the safe interior side, we move outward through the plate.
    We accept the LAST transition where:
      - there is a stable yellow-like run before it
      - there is a stable non-yellow run after it
      - soft similarity drops meaningfully across the boundary
    """
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
        # Transition must go from yellow-like to non-yellow-like.
        if not (
            yellow[i - 1]
            and not yellow[i]
        ):
            continue

        yellow_before = yellow[
            i - min_yellow_samples:i
        ]

        nonyellow_after = ~yellow[
            i:i + min_nonyellow_samples
        ]

        if not np.all(
            yellow_before
        ):
            continue

        if not np.all(
            nonyellow_after
        ):
            continue

        left_value = float(
            np.mean(
                smooth[
                    max(
                        0,
                        i - min_yellow_samples,
                    ):i
                ]
            )
        )

        right_value = float(
            np.mean(
                smooth[
                    i:min(
                        len(smooth),
                        i + min_nonyellow_samples,
                    )
                ]
            )
        )

        drop = float(
            left_value
            - right_value
        )

        if drop < PROFILE_MIN_DROP:
            continue

        # Boundary lies between samples i-1 and i.
        boundary_offset = float(
            0.5
            * (
                offsets[i - 1]
                + offsets[i]
            )
        )

        candidates.append(
            {
                "index": int(i),
                "offset_px": boundary_offset,
                "drop": drop,
                "left_support": left_value,
                "right_support": right_value,
                "smoothed_profile": smooth,
            }
        )

    if not candidates:
        return None

    # We explicitly want the LAST credible yellow exit.
    return max(
        candidates,
        key=lambda row: row["offset_px"],
    )



def median_filter_offsets(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    radius = window // 2
    out = values.copy()

    for i in range(len(values)):
        i0 = max(0, i - radius)
        i1 = min(len(values), i + radius + 1)
        out[i] = np.median(values[i0:i1])

    return out


def extract_side_ridge_points(
    *,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
    soft_threshold: float,
    safe_side_offset_px: float,
):
    # soft_threshold is kept for reporting only.
    p1, p2, t, _n_in, n_out = side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(
            p2 - p1
        )
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

    selected_t = []
    selected_offset = []
    selected_drop = []
    selected_left = []
    selected_right = []
    rows = []

    for tangential in tangential_positions:
        base = (
            p1
            + float(tangential) * t
        )

        ray_points = (
            base[None, :]
            + outward_offsets[:, None]
            * n_out[None, :]
        )

        profile = sample_scalar_bilinear(
            soft_support,
            ray_points,
        )

        finite = np.isfinite(
            profile
        )

        profile_safe = np.where(
            finite,
            profile,
            0.0,
        )

        exit_info = find_final_yellow_exit(
            profile_safe,
            outward_offsets,
        )

        row = {
            "tangential_px": float(tangential),
            "valid": False,
            "selected_outward_px": np.nan,
            "drop": 0.0,
            "left_support": 0.0,
            "right_support": 0.0,
            "search_max_outward_px": max_outward,
        }

        if exit_info is not None:
            selected_t.append(
                float(tangential)
            )

            selected_offset.append(
                float(
                    exit_info["offset_px"]
                )
            )

            selected_drop.append(
                float(
                    exit_info["drop"]
                )
            )

            selected_left.append(
                float(
                    exit_info["left_support"]
                )
            )

            selected_right.append(
                float(
                    exit_info["right_support"]
                )
            )

            row.update(
                {
                    "valid": True,
                    "selected_outward_px": float(
                        exit_info["offset_px"]
                    ),
                    "drop": float(
                        exit_info["drop"]
                    ),
                    "left_support": float(
                        exit_info["left_support"]
                    ),
                    "right_support": float(
                        exit_info["right_support"]
                    ),
                }
            )

        rows.append(
            row
        )

    selected_t = np.asarray(
        selected_t,
        dtype=np.float64,
    )

    selected_offset = np.asarray(
        selected_offset,
        dtype=np.float64,
    )

    selected_drop = np.asarray(
        selected_drop,
        dtype=np.float64,
    )

    selected_left = np.asarray(
        selected_left,
        dtype=np.float64,
    )

    selected_right = np.asarray(
        selected_right,
        dtype=np.float64,
    )

    valid_fraction = float(
        len(selected_t)
        / max(
            len(tangential_positions),
            1,
        )
    )

    if len(selected_t):
        order = np.argsort(
            selected_t
        )

        selected_t = selected_t[
            order
        ]

        selected_offset = selected_offset[
            order
        ]

        selected_drop = selected_drop[
            order
        ]

        selected_left = selected_left[
            order
        ]

        selected_right = selected_right[
            order
        ]

        selected_offset_smoothed = median_filter_offsets(
            selected_offset,
            RIDGE_OFFSET_MEDIAN_WINDOW,
        )

        ridge_points = (
            p1[None, :]
            + selected_t[:, None]
            * t[None, :]
            + selected_offset_smoothed[:, None]
            * n_out[None, :]
        )

    else:
        selected_offset_smoothed = np.empty(
            (0,),
            dtype=np.float64,
        )

        ridge_points = np.empty(
            (0, 2),
            dtype=np.float64,
        )

    stats = {
        "valid_fraction": valid_fraction,
        "ridge_point_count": int(
            len(ridge_points)
        ),
        "median_outward_px": (
            float(
                np.median(
                    selected_offset_smoothed
                )
            )
            if len(selected_offset_smoothed)
            else 0.0
        ),
        "min_outward_px": (
            float(
                np.min(
                    selected_offset_smoothed
                )
            )
            if len(selected_offset_smoothed)
            else 0.0
        ),
        "max_outward_px": (
            float(
                np.max(
                    selected_offset_smoothed
                )
            )
            if len(selected_offset_smoothed)
            else 0.0
        ),
        "median_drop": (
            float(
                np.median(
                    selected_drop
                )
            )
            if len(selected_drop)
            else 0.0
        ),
        "median_left_support": (
            float(
                np.median(
                    selected_left
                )
            )
            if len(selected_left)
            else 0.0
        ),
        "median_right_support": (
            float(
                np.median(
                    selected_right
                )
            )
            if len(selected_right)
            else 0.0
        ),
        "search_max_outward_px": max_outward,
        "threshold": float(
            soft_threshold
        ),
    }

    return (
        ridge_points,
        rows,
        stats,
    )



def fit_ridge_line_robust(
    ridge_points: np.ndarray,
):
    points = np.asarray(
        ridge_points,
        dtype=np.float64,
    )

    if len(points) < 2:
        raise RuntimeError('Too few ridge points for line fit.')

    pts = points.astype(np.float32).reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    direction = normalize(
        np.asarray([vx, vy], dtype=np.float64)
    )
    point = np.asarray([x0, y0], dtype=np.float64)

    normal = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    distances = np.abs(
        (points - point[None, :]) @ normal
    )

    threshold = float(
        np.percentile(
            distances,
            RIDGE_LINE_TRIM_PERCENTILE,
        )
    )

    keep = distances <= threshold
    trimmed = points[keep]

    if len(trimmed) >= 2:
        pts2 = trimmed.astype(np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(
            pts2,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)

        direction = normalize(
            np.asarray([vx, vy], dtype=np.float64)
        )
        point = np.asarray([x0, y0], dtype=np.float64)

    return point, direction, trimmed


def refine_one_side_soft_profile(
    *,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
    soft_threshold: float,
    safe_side_offset_px: float,
):
    ridge_points, profile_rows, stats = extract_side_ridge_points(
        soft_support=soft_support,
        safe_quad=safe_quad,
        side_index=side_index,
        soft_threshold=soft_threshold,
        safe_side_offset_px=safe_side_offset_px,
    )

    trustworthy = bool(
        len(ridge_points) >= MIN_RIDGE_POINTS
        and stats['valid_fraction'] >= MIN_VALID_RIDGE_FRACTION
    )

    if trustworthy:
        point, direction, fit_points = fit_ridge_line_robust(
            ridge_points
        )
        fallback = False
        fit_source = 'final_yellow_exit_ridge'
    else:
        p1, _p2, t, _n_in, _n_out = side_local_frame(
            safe_quad,
            side_index,
        )
        point = p1.copy()
        direction = t.copy()
        fit_points = np.empty((0, 2), dtype=np.float64)
        fallback = True
        fit_source = 'safe_side_fallback'

    return {
        'fallback': fallback,
        'fit_source': fit_source,
        'point': point,
        'direction': direction,
        'fit_points': fit_points,
        'ridge_points': ridge_points,
        'profile_rows': profile_rows,
        **stats,
    }


def refined_quad_from_profile_lines(
    fitted_sides: list[dict],
) -> np.ndarray:
    top, right, bottom, left = fitted_sides

    tl = line_intersection(
        top['point'], top['direction'],
        left['point'], left['direction'],
    )
    tr = line_intersection(
        top['point'], top['direction'],
        right['point'], right['direction'],
    )
    br = line_intersection(
        bottom['point'], bottom['direction'],
        right['point'], right['direction'],
    )
    bl = line_intersection(
        bottom['point'], bottom['direction'],
        left['point'], left['direction'],
    )

    return np.asarray([tl, tr, br, bl], dtype=np.float64)


def save_normal_profile_side_debug(
    *,
    case_dir: Path,
    crop: np.ndarray,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_name: str,
    side_index: int,
    fitted: dict,
) -> None:
    # Left: soft similarity, middle: selected ridge points, right: final fit.
    soft_vis = np.clip(
        soft_support * 255.0,
        0,
        255,
    ).astype(np.uint8)
    soft_color = cv2.applyColorMap(
        soft_vis,
        cv2.COLORMAP_TURBO,
    )

    ridge_view = crop.copy()
    fit_view = crop.copy()

    p1, p2, _t, _n_in, _n_out = side_local_frame(
        safe_quad,
        side_index,
    )

    cv2.line(
        ridge_view,
        tuple(np.round(p1).astype(np.int32)),
        tuple(np.round(p2).astype(np.int32)),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Red = all successful raw ridge points.
    for p in fitted['ridge_points']:
        cv2.circle(
            ridge_view,
            tuple(np.round(p).astype(np.int32)),
            1,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

    if not fitted['fallback']:
        # Green = points retained by robust line fit.
        for p in fitted['fit_points']:
            cv2.circle(
                fit_view,
                tuple(np.round(p).astype(np.int32)),
                1,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

        point = fitted['point']
        direction = fitted['direction']
        a = point - 1000.0 * direction
        b = point + 1000.0 * direction

        cv2.line(
            fit_view,
            tuple(np.round(a).astype(np.int32)),
            tuple(np.round(b).astype(np.int32)),
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    panel = np.hstack([
        soft_color,
        ridge_view,
        fit_view,
    ])

    bar = np.zeros(
        (54, panel.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        bar,
        (
            f'{side_name} | valid={fitted["valid_fraction"]:.2f} | '
            f'N={fitted["ridge_point_count"]} | '
            f'out={fitted["median_outward_px"]:.2f}px | '
            f'drop={fitted.get("median_drop", 0.0):.3f} | '
            f'search<={fitted.get("search_max_outward_px", 0.0):.1f}px | '
            f'fallback={fitted["fallback"]}'
        ),
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(case_dir / f'normal_profile_debug_{side_name}.jpg'),
        np.vstack([bar, panel]),
    )

    # CSV: one row per normal cut, so we can inspect exactly where each
    # intersection was selected.
    csv_path = case_dir / f'normal_profile_{side_name}.csv'

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fitted['profile_rows'][0].keys()),
        )
        writer.writeheader()
        writer.writerows(fitted['profile_rows'])


# ============================================================
# SIDE REFINEMENT USING SECOND MASK PIXELS + FITLINE
# ============================================================

def side_local_frame(
    safe_quad: np.ndarray,
    side_index: int,
):
    center = quad_center(safe_quad)

    p1 = np.asarray(
        safe_quad[side_index],
        dtype=np.float64,
    )

    p2 = np.asarray(
        safe_quad[(side_index + 1) % 4],
        dtype=np.float64,
    )

    t = normalize(p2 - p1)
    n_in = line_inward_normal(
        p1,
        p2,
        center,
    )
    n_out = -n_in

    return p1, p2, t, n_in, n_out


def points_in_side_band(
    *,
    points_xy: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
) -> np.ndarray:
    p1, p2, t, _n_in, n_out = side_local_frame(
        safe_quad,
        side_index,
    )

    rel = (
        np.asarray(points_xy, dtype=np.float64)
        - p1[None, :]
    )

    tangential = rel @ t
    outward = rel @ n_out

    side_length = float(
        np.linalg.norm(p2 - p1)
    )

    keep = (
        (tangential >= -SIDE_BAND_TANGENTIAL_PAD_PX)
        & (
            tangential
            <= side_length + SIDE_BAND_TANGENTIAL_PAD_PX
        )
        & (
            outward
            >= -SIDE_BAND_INWARD_PX
        )
        & (
            outward
            <= SIDE_BAND_OUTWARD_PX
        )
    )

    return keep


def make_side_band_mask(
    *,
    second_mask: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
) -> np.ndarray:
    ys, xs = np.nonzero(
        second_mask > 0
    )

    out = np.zeros_like(
        second_mask
    )

    if len(xs) == 0:
        return out

    pts = np.column_stack(
        [xs, ys]
    ).astype(np.float64)

    keep = points_in_side_band(
        points_xy=pts,
        safe_quad=safe_quad,
        side_index=side_index,
    )

    pts_keep = pts[keep]

    if len(pts_keep):
        xk = pts_keep[:, 0].astype(np.int32)
        yk = pts_keep[:, 1].astype(np.int32)
        out[yk, xk] = 255

    return out


def component_metrics(
    *,
    component_points_xy: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
) -> dict:
    p1, p2, t, _n_in, n_out = side_local_frame(
        safe_quad,
        side_index,
    )

    rel = (
        component_points_xy
        - p1[None, :]
    )

    tangential = rel @ t
    outward = rel @ n_out

    side_length = max(
        float(np.linalg.norm(p2 - p1)),
        1e-6,
    )

    span_px = float(
        np.max(tangential)
        - np.min(tangential)
    )

    span_fraction = float(
        np.clip(
            span_px / side_length,
            0.0,
            1.0,
        )
    )

    pixel_count = int(
        len(component_points_xy)
    )

    # Normalize pixel count relative to side length.
    pixel_density_score = float(
        np.clip(
            pixel_count
            / max(side_length * 1.5, 1.0),
            0.0,
            1.0,
        )
    )

    # Prefer components close to the safe side.
    median_outward = float(
        np.median(outward)
    )

    proximity_score = float(
        np.exp(
            -max(median_outward, 0.0)
            / 5.0
        )
    )

    score = float(
        W_COMPONENT_SPAN
        * span_fraction
        + W_COMPONENT_PIXEL_COUNT
        * pixel_density_score
        + W_COMPONENT_PROXIMITY
        * proximity_score
    )

    return {
        "span_px": span_px,
        "span_fraction": span_fraction,
        "pixel_count": pixel_count,
        "pixel_density_score": pixel_density_score,
        "median_outward_px": median_outward,
        "proximity_score": proximity_score,
        "component_score": score,
    }


def select_best_component_for_side(
    *,
    side_band_mask: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
):
    num_labels, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            side_band_mask,
            connectivity=8,
        )
    )

    candidates = []

    for label in range(1, num_labels):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < MIN_COMPONENT_PIXELS:
            continue

        ys, xs = np.nonzero(
            labels == label
        )

        points = np.column_stack(
            [xs, ys]
        ).astype(np.float64)

        metrics = component_metrics(
            component_points_xy=points,
            safe_quad=safe_quad,
            side_index=side_index,
        )

        if (
            metrics["span_fraction"]
            < MIN_COMPONENT_SPAN_FRACTION
        ):
            continue

        candidates.append(
            {
                "label": label,
                "points": points,
                **metrics,
            }
        )

    if not candidates:
        return None, []

    best = max(
        candidates,
        key=lambda row: row["component_score"],
    )

    return best, candidates



def moving_median_1d(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    window = int(window)

    if (
        window <= 1
        or len(values) < 3
    ):
        return values.copy()

    if window % 2 == 0:
        window += 1

    radius = window // 2
    out = np.empty_like(values)

    for i in range(len(values)):
        i0 = max(0, i - radius)
        i1 = min(len(values), i + radius + 1)

        out[i] = np.median(
            values[i0:i1]
        )

    return out


def extract_outer_envelope_points(
    *,
    component_points_xy: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
) -> tuple[np.ndarray, dict]:
    """Keep one OUTERMOST component point per tangential bin.

    Coordinates are expressed in the local side frame:
      tangential = along the safe side
      outward    = along the side's outward normal

    For every tangential bin, choose the point with maximum outward
    coordinate. This extracts the exterior envelope of a thick component.
    """
    p1, p2, t, _n_in, n_out = side_local_frame(
        safe_quad,
        side_index,
    )

    points = np.asarray(
        component_points_xy,
        dtype=np.float64,
    )

    rel = (
        points
        - p1[None, :]
    )

    tangential = rel @ t
    outward = rel @ n_out

    side_length = float(
        np.linalg.norm(
            p2 - p1
        )
    )

    # Only use the tangential extent corresponding to the actual side,
    # not the padded band beyond the two corners.
    valid = (
        (tangential >= 0.0)
        & (tangential <= side_length)
    )

    points = points[valid]
    tangential = tangential[valid]
    outward = outward[valid]

    if len(points) == 0:
        return (
            np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            {
                "envelope_bins": 0,
                "envelope_outward_median_px": 0.0,
                "envelope_outward_min_px": 0.0,
                "envelope_outward_max_px": 0.0,
            },
        )

    bin_ids = np.floor(
        tangential
        / max(
            float(ENVELOPE_BIN_SIZE_PX),
            1e-6,
        )
    ).astype(np.int32)

    selected_t = []
    selected_out = []

    for bin_id in np.unique(bin_ids):
        idx = np.where(
            bin_ids == bin_id
        )[0]

        if len(idx) == 0:
            continue

        # OUTERMOST point = maximum coordinate along n_out.
        local_best = idx[
            int(
                np.argmax(
                    outward[idx]
                )
            )
        ]

        selected_t.append(
            float(
                tangential[
                    local_best
                ]
            )
        )

        selected_out.append(
            float(
                outward[
                    local_best
                ]
            )
        )

    selected_t = np.asarray(
        selected_t,
        dtype=np.float64,
    )

    selected_out = np.asarray(
        selected_out,
        dtype=np.float64,
    )

    order = np.argsort(
        selected_t
    )

    selected_t = selected_t[order]
    selected_out = selected_out[order]

    if ENVELOPE_MEDIAN_WINDOW > 1:
        selected_out = moving_median_1d(
            selected_out,
            ENVELOPE_MEDIAN_WINDOW,
        )

    envelope_points = (
        p1[None, :]
        + selected_t[:, None] * t[None, :]
        + selected_out[:, None] * n_out[None, :]
    )

    stats = {
        "envelope_bins": int(
            len(envelope_points)
        ),
        "envelope_outward_median_px": (
            float(
                np.median(
                    selected_out
                )
            )
            if len(selected_out)
            else 0.0
        ),
        "envelope_outward_min_px": (
            float(
                np.min(
                    selected_out
                )
            )
            if len(selected_out)
            else 0.0
        ),
        "envelope_outward_max_px": (
            float(
                np.max(
                    selected_out
                )
            )
            if len(selected_out)
            else 0.0
        ),
    }

    return envelope_points, stats


def fit_line_robust(
    points_xy: np.ndarray,
):
    pts = np.asarray(
        points_xy,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    if len(pts) < 2:
        raise RuntimeError(
            "Too few points for fitLine."
        )

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    direction = normalize(
        np.asarray(
            [vx, vy],
            dtype=np.float64,
        )
    )

    point = np.asarray(
        [x0, y0],
        dtype=np.float64,
    )

    # Robust second pass:
    # remove the farthest point-to-line outliers and refit.
    rel = (
        np.asarray(points_xy, dtype=np.float64)
        - point[None, :]
    )

    normal = np.asarray(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    distances = np.abs(
        rel @ normal
    )

    threshold = float(
        np.percentile(
            distances,
            FITLINE_DISTANCE_TRIM_PERCENTILE,
        )
    )

    keep = (
        distances
        <= threshold
    )

    trimmed = np.asarray(
        points_xy,
        dtype=np.float64,
    )[keep]

    if len(trimmed) >= 2:
        pts2 = trimmed.astype(
            np.float32
        ).reshape(-1, 1, 2)

        vx, vy, x0, y0 = cv2.fitLine(
            pts2,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)

        direction = normalize(
            np.asarray(
                [vx, vy],
                dtype=np.float64,
            )
        )

        point = np.asarray(
            [x0, y0],
            dtype=np.float64,
        )

    return point, direction, trimmed


def refine_one_side_fitline(
    *,
    second_mask: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
):
    side_band_mask = make_side_band_mask(
        second_mask=second_mask,
        safe_quad=safe_quad,
        side_index=side_index,
    )

    best_component, candidates = (
        select_best_component_for_side(
            side_band_mask=side_band_mask,
            safe_quad=safe_quad,
            side_index=side_index,
        )
    )

    if best_component is None:
        # Conservative fallback: keep the safe side itself.
        p1, p2, t, _n_in, _n_out = side_local_frame(
            safe_quad,
            side_index,
        )

        return {
            "fallback": True,
            "point": p1.copy(),
            "direction": t.copy(),
            "fit_points": np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            "fit_source_points": np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            "fit_source": "safe_side_fallback",
            "envelope_points": np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            "envelope_bins": 0,
            "envelope_outward_median_px": 0.0,
            "envelope_outward_min_px": 0.0,
            "envelope_outward_max_px": 0.0,
            "band_mask": side_band_mask,
            "component_mask": np.zeros_like(
                second_mask
            ),
            "component_score": 0.0,
            "span_fraction": 0.0,
            "median_outward_px": 0.0,
        }, candidates

    envelope_points, envelope_stats = extract_outer_envelope_points(
        component_points_xy=best_component["points"],
        safe_quad=safe_quad,
        side_index=side_index,
    )

    # Prefer the outer envelope. If the component is too short/sparse,
    # conservatively fall back to fitting the full selected component.
    if len(envelope_points) >= MIN_ENVELOPE_POINTS:
        fit_source_points = envelope_points
        fit_source = "outer_envelope"
    else:
        fit_source_points = best_component["points"]
        fit_source = "full_component_fallback"

    point, direction, fit_points = fit_line_robust(
        fit_source_points
    )

    component_mask = np.zeros_like(
        second_mask
    )

    component_points = best_component[
        "points"
    ].astype(np.int32)

    component_mask[
        component_points[:, 1],
        component_points[:, 0],
    ] = 255

    return {
        "fallback": False,
        "point": point,
        "direction": direction,
        "fit_points": fit_points,
        "fit_source_points": fit_source_points,
        "fit_source": fit_source,
        "envelope_points": envelope_points,
        "band_mask": side_band_mask,
        "component_mask": component_mask,
        **envelope_stats,
        **{
            key: value
            for key, value in best_component.items()
            if key not in ("points", "label")
        },
    }, candidates


def refined_quad_from_fitlines(
    fitted_sides: list[dict],
) -> np.ndarray:
    top = fitted_sides[0]
    right = fitted_sides[1]
    bottom = fitted_sides[2]
    left = fitted_sides[3]

    tl = line_intersection(
        top["point"],
        top["direction"],
        left["point"],
        left["direction"],
    )

    tr = line_intersection(
        top["point"],
        top["direction"],
        right["point"],
        right["direction"],
    )

    br = line_intersection(
        bottom["point"],
        bottom["direction"],
        right["point"],
        right["direction"],
    )

    bl = line_intersection(
        bottom["point"],
        bottom["direction"],
        left["point"],
        left["direction"],
    )

    return np.asarray(
        [tl, tr, br, bl],
        dtype=np.float64,
    )


def snap_corner_to_second_mask(
    corner: np.ndarray,
    second_mask: np.ndarray,
) -> np.ndarray:
    x0 = int(round(float(corner[0])))
    y0 = int(round(float(corner[1])))

    ys, xs = np.nonzero(
        second_mask > 0
    )

    if len(xs) == 0:
        return corner.copy()

    pts = np.column_stack(
        [xs, ys]
    ).astype(np.float64)

    diff = (
        pts
        - corner[None, :]
    )

    dist = np.linalg.norm(
        diff,
        axis=1,
    )

    index = int(
        np.argmin(dist)
    )

    if (
        dist[index]
        <= CORNER_SNAP_RADIUS_PX
    ):
        # Tiny stabilization only, not a large relocation.
        return (
            0.75 * corner
            + 0.25 * pts[index]
        )

    return corner.copy()


def snap_quad_corners(
    quad: np.ndarray,
    second_mask: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            snap_corner_to_second_mask(
                corner,
                second_mask,
            )
            for corner in quad
        ],
        dtype=np.float64,
    )


# ============================================================
# DEBUG
# ============================================================

def draw_quad(
    image: np.ndarray,
    quad: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = LINE_THICKNESS,
) -> np.ndarray:
    out = image.copy()

    pts = np.round(
        quad
    ).astype(np.int32)

    cv2.polylines(
        out,
        [pts.reshape(-1, 1, 2)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )

    for p in pts:
        cv2.circle(
            out,
            tuple(p),
            POINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )

    return out


def colorize_soft_map(
    soft_support: np.ndarray,
) -> np.ndarray:
    gray = np.clip(
        soft_support * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )


def build_sampling_view(
    crop: np.ndarray,
    debug_rows: list[dict],
) -> np.ndarray:
    view = crop.copy()

    side_colors = {
        "top": (255, 255, 0),
        "right": (0, 255, 255),
        "bottom": (255, 0, 255),
        "left": (0, 165, 255),
    }

    for row in debug_rows:
        color = side_colors[
            row["side"]
        ]

        points = row["points"]
        keep = row["keep"]

        for point, is_kept in zip(
            points,
            keep,
        ):
            if not is_kept:
                continue

            x = int(
                round(
                    float(point[0])
                )
            )

            y = int(
                round(
                    float(point[1])
                )
            )

            if (
                0 <= x < view.shape[1]
                and 0 <= y < view.shape[0]
            ):
                cv2.circle(
                    view,
                    (x, y),
                    1,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

    return view


def make_debug_panel(
    *,
    crop: np.ndarray,
    clean_mask: np.ndarray,
    min_quad: np.ndarray,
    safe_quad: np.ndarray,
    sampling_view: np.ndarray,
    model: dict,
    hard_color_mask: np.ndarray,
    soft_support: np.ndarray,
    second_mask: np.ndarray,
    refined_quad: np.ndarray,
    frame_name: str,
    plate_index: int,
    shrink_px: float,
    coverages: list[float],
    selected_sides: list[dict],
) -> np.ndarray:
    min_view = draw_quad(
        crop,
        min_quad,
        (0, 255, 255),
        1,
    )

    safe_view = draw_quad(
        crop,
        safe_quad,
        (0, 255, 0),
        1,
    )

    refined_view = draw_quad(
        crop,
        refined_quad,
        (255, 0, 255),
        1,
    )

    clean_bgr = cv2.cvtColor(
        clean_mask,
        cv2.COLOR_GRAY2BGR,
    )

    hard_bgr = cv2.cvtColor(
        hard_color_mask,
        cv2.COLOR_GRAY2BGR,
    )

    second_bgr = cv2.cvtColor(
        second_mask,
        cv2.COLOR_GRAY2BGR,
    )

    soft_bgr = colorize_soft_map(
        soft_support
    )

    selected_pixels = np.zeros_like(crop)
    selected_pixels[
        hard_color_mask > 0
    ] = crop[
        hard_color_mask > 0
    ]

    remainder_pixels = np.zeros_like(crop)
    remainder_pixels[
        second_mask > 0
    ] = crop[
        second_mask > 0
    ]

    # Fit the color swatch to the actual crop height.
    # The previous fixed 90x90 swatch crashed on short crops.
    max_swatch = max(
        16,
        min(
            90,
            crop.shape[0] - 16,
            crop.shape[1] - 16,
        ),
    )

    swatch = lab_model_to_bgr_swatch(
        model,
        size=max_swatch,
    )

    swatch_canvas = np.zeros_like(crop)

    h, w = swatch.shape[:2]

    y0 = min(
        8,
        max(0, crop.shape[0] - h),
    )

    x0 = min(
        8,
        max(0, crop.shape[1] - w),
    )

    swatch_canvas[
        y0:y0+h,
        x0:x0+w,
    ] = swatch

    cv2.putText(
        swatch_canvas,
        "selected boundary-yellow",
        (8, min(crop.shape[0] - 8, h + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    median = model["median"]
    sigma = model["sigma"]

    cv2.putText(
        swatch_canvas,
        f"Lab=({median[0]:.1f},{median[1]:.1f},{median[2]:.1f})",
        (8, min(crop.shape[0] - 8, h + 50)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        swatch_canvas,
        f"sigma=({sigma[0]:.1f},{sigma[1]:.1f},{sigma[2]:.1f})",
        (8, min(crop.shape[0] - 8, h + 68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    views = [
        ("1. crop + minAreaRect", min_view),
        ("2. safe inner rectangle", safe_view),
        ("3. exact line samples used for color", sampling_view),
        ("4. chosen boundary-yellow color", swatch_canvas),
        ("5. original clean yellow mask", clean_bgr),
        ("6. learned-color soft similarity", soft_bgr),
        ("7. learned-color hard mask - whole crop", hard_bgr),
        ("8. RGB pixels selected by learned color", selected_pixels),
        ("9. SECOND MASK = learned color only in remainder", second_bgr),
        ("10. RGB pixels surviving SECOND MASK", remainder_pixels),
        ("11. refined lines from SECOND MASK", refined_view),
    ]

    # Add all-overlays view as the 12th cell.
    overlay = crop.copy()
    overlay = draw_quad(
        overlay,
        min_quad,
        (0, 255, 255),
        1,
    )
    overlay = draw_quad(
        overlay,
        safe_quad,
        (0, 255, 0),
        1,
    )
    overlay = draw_quad(
        overlay,
        refined_quad,
        (255, 0, 255),
        1,
    )

    views.append(
        ("12. all overlays", overlay)
    )

    cell_w = 500

    def render_cell(
        title: str,
        image: np.ndarray,
    ) -> np.ndarray:
        scale = (
            cell_w
            / image.shape[1]
        )

        resized = cv2.resize(
            image,
            (
                cell_w,
                max(
                    1,
                    int(
                        round(
                            image.shape[0]
                            * scale
                        )
                    ),
                ),
            ),
            interpolation=cv2.INTER_NEAREST
            if "mask" in title.lower()
            else cv2.INTER_LINEAR,
        )

        bar = np.zeros(
            (38, cell_w, 3),
            dtype=np.uint8,
        )

        cv2.putText(
            bar,
            title,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return np.vstack(
            [bar, resized]
        )

    rendered = [
        render_cell(
            title,
            image,
        )
        for title, image in views
    ]

    max_h = max(
        image.shape[0]
        for image in rendered
    )

    padded = []

    for image in rendered:
        canvas = np.zeros(
            (max_h, cell_w, 3),
            dtype=np.uint8,
        )

        canvas[
            :image.shape[0],
            :image.shape[1],
        ] = image

        padded.append(canvas)

    rows = []

    for start in range(
        0,
        len(padded),
        3,
    ):
        rows.append(
            np.hstack(
                padded[
                    start:start+3
                ]
            )
        )

    body = np.vstack(rows)

    header = np.zeros(
        (
            190,
            body.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        (
            f"{frame_name} | plate {plate_index} | "
            "EDGE-LINE COLOR -> REMAINDER MASK DEBUG"
        ),
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        (
            f"safe shrink={shrink_px:.1f}px | "
            f"side coverages="
            f"{', '.join(f'{x:.2f}' for x in coverages)} | "
            f"model N={model['count']}"
        ),
        (12, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    fit_text = ", ".join(
        (
            f"{name}:valid={row.get('valid_fraction', 0.0):.2f},"
            f"out={row.get('median_outward_px', 0.0):.1f}"
        )
        for name, row in zip(
            ["top", "right", "bottom", "left"],
            selected_sides,
        )
    )

    cv2.putText(
        header,
        (
            "normal-profile ridge fits | "
            + fit_text
        ),
        (12, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        (
            "IMPORTANT: color model is learned ONLY from yellow pixels on the 4 green lines; "
            "+1px inward is used only when a side lacks samples."
        ),
        (12, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (195, 195, 195),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        header,
        (
            "SOFT refinement: safe interior -> move outward through yellow -> FINAL stable yellow exit -> robust cv2.fitLine -> intersections."
        ),
        (12, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return np.vstack(
        [header, body]
    )



def save_side_score_profiles(
    *,
    case_dir: Path,
    side_score_rows: list[dict],
) -> None:
    """Save a compact diagnostic image for each side: score vs outward offset."""
    side_names = ["top", "right", "bottom", "left"]

    for side_name in side_names:
        rows = [
            row
            for row in side_score_rows
            if row["side"] == side_name
        ]

        if not rows:
            continue

        width = 900
        height = 520
        margin_l = 75
        margin_r = 30
        margin_t = 55
        margin_b = 70

        canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        offsets = np.asarray(
            [row["offset_px"] for row in rows],
            dtype=np.float64,
        )

        coverage = np.asarray(
            [row["second_mask_coverage"] for row in rows],
            dtype=np.float64,
        )

        runs = np.asarray(
            [row["longest_contiguous_run"] for row in rows],
            dtype=np.float64,
        )

        scores = np.asarray(
            [row["score"] for row in rows],
            dtype=np.float64,
        )

        x0 = margin_l
        x1 = width - margin_r
        y0 = margin_t
        y1 = height - margin_b

        cv2.rectangle(
            canvas,
            (x0, y0),
            (x1, y1),
            (110, 110, 110),
            1,
        )

        min_x = float(np.min(offsets))
        max_x = float(np.max(offsets))
        x_span = max(max_x - min_x, 1e-6)

        def to_xy(x, y):
            px = int(
                round(
                    x0
                    + (float(x) - min_x)
                    / x_span
                    * (x1 - x0)
                )
            )

            py = int(
                round(
                    y1
                    - np.clip(float(y), 0.0, 1.0)
                    * (y1 - y0)
                )
            )

            return px, py

        def draw_curve(values, color):
            pts = np.asarray(
                [
                    to_xy(x, y)
                    for x, y in zip(offsets, values)
                ],
                dtype=np.int32,
            )

            if len(pts) >= 2:
                cv2.polylines(
                    canvas,
                    [pts.reshape(-1, 1, 2)],
                    False,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        draw_curve(
            coverage,
            (255, 255, 255),
        )

        draw_curve(
            runs,
            (0, 255, 255),
        )

        draw_curve(
            scores,
            (255, 0, 255),
        )

        best_index = int(
            np.argmax(scores)
        )

        bx, by = to_xy(
            offsets[best_index],
            scores[best_index],
        )

        cv2.circle(
            canvas,
            (bx, by),
            5,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"{side_name} | SECOND MASK side scan",
            (20, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                f"best offset={offsets[best_index]:.1f}px | "
                f"coverage={coverage[best_index]:.3f} | "
                f"run={runs[best_index]:.3f} | "
                f"score={scores[best_index]:.3f}"
            ),
            (20, 505),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "white=coverage | yellow=longest contiguous run | magenta=final score | green=selected",
            (225, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (185, 185, 185),
            1,
            cv2.LINE_AA,
        )

        for tick in np.linspace(
            min_x,
            max_x,
            7,
        ):
            tx, _ = to_xy(
                tick,
                0.0,
            )

            cv2.line(
                canvas,
                (tx, y1),
                (tx, y1 + 5),
                (120, 120, 120),
                1,
            )

            cv2.putText(
                canvas,
                f"{tick:.1f}",
                (tx - 13, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            str(
                case_dir
                / f"side_profile_{side_name}.jpg"
            ),
            canvas,
        )


# ============================================================
# MAIN
# ============================================================

def process_case(
    frame_name: str,
    plate_index: int,
) -> dict:
    print(
        f"\n--- {frame_name} | plate {plate_index} ---"
    )

    case_dir = (
        OUTPUT_ROOT
        / f"{frame_name}_plate_{plate_index}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop, _x_offset, _y_offset, pred = (
        load_target_plate(
            frame_name,
            plate_index,
        )
    )

    (
        clean_mask,
        raw_yellow_mask,
        raw_yellow_soft,
    ) = create_clean_yellow_mask(
        crop,
        float(pred["width"]),
    )

    min_quad = min_area_rect_quad(
        clean_mask
    )

    (
        side_offsets,
        safe_side_metrics,
    ) = find_side_specific_safe_offsets(
        min_quad=min_quad,
        clean_mask=clean_mask,
        raw_mask=raw_yellow_mask,
        raw_soft=raw_yellow_soft,
    )

    safe_quad = build_quad_from_side_offsets(
        min_quad=min_quad,
        side_offsets=side_offsets,
    )

    # Keep legacy summary variables for the existing panel.
    shrink_px = float(
        np.mean(
            side_offsets
        )
    )

    coverages = [
        float(
            metric["clean_fraction"]
        )
        for metric in safe_side_metrics
    ]

    save_safe_side_validation_debug(
        case_dir=case_dir,
        crop=crop,
        min_quad=min_quad,
        safe_quad=safe_quad,
        raw_mask=raw_yellow_mask,
        raw_soft=raw_yellow_soft,
        side_metrics=safe_side_metrics,
    )

    (
        lab_image,
        dark_mask,
        boundary_samples,
        sample_counts,
        debug_sample_rows,
    ) = collect_boundary_line_color_samples(
        crop=crop,
        clean_mask=clean_mask,
        safe_quad=safe_quad,
    )

    model = robust_lab_model(
        boundary_samples
    )

    (
        soft_support,
        hard_color_mask,
        _z_distance,
        soft_threshold,
    ) = build_color_similarity_maps(
        lab_image=lab_image,
        dark_mask=dark_mask,
        model=model,
    )

    (
        second_mask,
        inner_mask,
        remainder_region,
    ) = build_second_remainder_mask(
        hard_color_mask=hard_color_mask,
        safe_quad=safe_quad,
    )

    side_names = [
        "top",
        "right",
        "bottom",
        "left",
    ]

    fitted_sides = []

    for side_index, side_name in enumerate(side_names):
        fitted = refine_one_side_soft_profile(
            soft_support=soft_support,
            safe_quad=safe_quad,
            side_index=side_index,
            soft_threshold=soft_threshold,
            safe_side_offset_px=float(
                side_offsets[
                    side_index
                ]
            ),
        )

        fitted_sides.append(fitted)

        print(
            f"{side_name:>6}: "
            f"fallback={fitted['fallback']} | "
            f"valid={fitted['valid_fraction']:.3f} | "
            f"N={fitted['ridge_point_count']} | "
            f"out={fitted['median_outward_px']:.2f}px | "
            f"drop={fitted.get('median_drop', 0.0):.3f} | "
            f"search<= {fitted.get('search_max_outward_px', 0.0):.1f}px"
        )

        save_normal_profile_side_debug(
            case_dir=case_dir,
            crop=crop,
            soft_support=soft_support,
            safe_quad=safe_quad,
            side_name=side_name,
            side_index=side_index,
            fitted=fitted,
        )

    # Save one summary row per fitted side.
    fitted_side_summary_path = (
        case_dir
        / "fitted_side_summary.csv"
    )

    fitted_side_rows = []

    for side_name, fitted in zip(
        side_names,
        fitted_sides,
    ):
        fitted_side_rows.append(
            {
                "side": side_name,
                "fallback": fitted["fallback"],
                "fit_source": fitted["fit_source"],
                "valid_fraction": fitted["valid_fraction"],
                "ridge_point_count": fitted["ridge_point_count"],
                "median_outward_px": fitted["median_outward_px"],
                "min_outward_px": fitted["min_outward_px"],
                "max_outward_px": fitted["max_outward_px"],
                "median_drop": fitted.get("median_drop", 0.0),
                "median_left_support": fitted.get("median_left_support", 0.0),
                "median_right_support": fitted.get("median_right_support", 0.0),
                "search_max_outward_px": fitted.get("search_max_outward_px", 0.0),
                "threshold": fitted["threshold"],
            }
        )

    with fitted_side_summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                fitted_side_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(fitted_side_rows)

    refined_quad = refined_quad_from_profile_lines(
        fitted_sides
    )

    refined_quad = snap_quad_corners(
        refined_quad,
        second_mask,
    )

    sampling_view = build_sampling_view(
        crop,
        debug_sample_rows,
    )

    panel = make_debug_panel(
        crop=crop,
        clean_mask=clean_mask,
        min_quad=min_quad,
        safe_quad=safe_quad,
        sampling_view=sampling_view,
        model=model,
        hard_color_mask=hard_color_mask,
        soft_support=soft_support,
        second_mask=second_mask,
        refined_quad=refined_quad,
        frame_name=frame_name,
        plate_index=plate_index,
        shrink_px=shrink_px,
        coverages=coverages,
        selected_sides=fitted_sides,
    )

    panel_path = (
        case_dir
        / "final_yellow_exit_boundary_refinement_panel.jpg"
    )

    cv2.imwrite(
        str(panel_path),
        panel,
    )

    # Save each important stage separately too.
    cv2.imwrite(
        str(
            case_dir
            / "00_crop.jpg"
        ),
        crop,
    )

    cv2.imwrite(
        str(
            case_dir
            / "01_clean_yellow_mask.png"
        ),
        clean_mask,
    )

    cv2.imwrite(
        str(
            case_dir
            / "02_edge_line_samples.jpg"
        ),
        sampling_view,
    )

    cv2.imwrite(
        str(
            case_dir
            / "03_learned_color_hard_mask_whole_crop.png"
        ),
        hard_color_mask,
    )

    soft_vis = np.clip(
        soft_support * 255.0,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            case_dir
            / "04_learned_color_soft_support.png"
        ),
        soft_vis,
    )

    cv2.imwrite(
        str(
            case_dir
            / "05_safe_inner_polygon_mask.png"
        ),
        inner_mask,
    )

    cv2.imwrite(
        str(
            case_dir
            / "06_remainder_region.png"
        ),
        remainder_region,
    )

    cv2.imwrite(
        str(
            case_dir
            / "07_SECOND_MASK_learned_color_in_remainder.png"
        ),
        second_mask,
    )

    refined_overlay = draw_quad(
        crop,
        refined_quad,
        (255, 0, 255),
        1,
    )

    cv2.imwrite(
        str(
            case_dir
            / "08_refined_quad.jpg"
        ),
        refined_overlay,
    )

    counts_path = (
        case_dir
        / "boundary_color_sample_counts.csv"
    )

    with counts_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                sample_counts[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            sample_counts
        )

    print(
        f"model Lab median = "
        f"{model['median'][0]:.1f}, "
        f"{model['median'][1]:.1f}, "
        f"{model['median'][2]:.1f}"
    )

    print(
        f"model sigma      = "
        f"{model['sigma'][0]:.1f}, "
        f"{model['sigma'][1]:.1f}, "
        f"{model['sigma'][2]:.1f}"
    )

    print(
        f"model samples    = {model['count']}"
    )

    print(
        f"soft threshold   = {soft_threshold:.6f}"
    )

    return {
        "frame_name": frame_name,
        "plate_index": plate_index,
        "status": "ok",
        "shrink_px": shrink_px,
        "safe_top_offset_px": side_offsets[0],
        "safe_right_offset_px": side_offsets[1],
        "safe_bottom_offset_px": side_offsets[2],
        "safe_left_offset_px": side_offsets[3],
        "model_sample_count": model["count"],
        "model_L": float(model["median"][0]),
        "model_a": float(model["median"][1]),
        "model_b": float(model["median"][2]),
        "top_valid_fraction": fitted_sides[0]["valid_fraction"],
        "right_valid_fraction": fitted_sides[1]["valid_fraction"],
        "bottom_valid_fraction": fitted_sides[2]["valid_fraction"],
        "left_valid_fraction": fitted_sides[3]["valid_fraction"],
        "panel_path": str(panel_path),
    }


def main() -> None:
    print(
        "\n========== FINAL YELLOW EXIT | ALL PLATES IN TWO FRAMES =========="
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for frame_name in FRAME_NAMES:
        try:
            plate_indices = load_all_plate_indices(
                frame_name
            )
        except Exception as exc:
            print(
                f"FAILED loading detections for {frame_name}: {exc}"
            )

            summaries.append(
                {
                    "frame_name": frame_name,
                    "plate_index": "",
                    "status": f"frame error: {exc}",
                }
            )
            continue

        if not plate_indices:
            print(
                f"{frame_name}: no plates detected"
            )
            summaries.append(
                {
                    "frame_name": frame_name,
                    "plate_index": "",
                    "status": "no detections",
                }
            )
            continue

        for plate_index in plate_indices:
            try:
                summary = process_case(
                    frame_name,
                    plate_index,
                )

                if "status" not in summary:
                    summary["status"] = "ok"

                summaries.append(
                    summary
                )

            except Exception as exc:
                print(
                    f"FAILED {frame_name} | "
                    f"plate {plate_index}: {exc}"
                )

                summaries.append(
                    {
                        "frame_name": frame_name,
                        "plate_index": plate_index,
                        "status": f"error: {exc}",
                    }
                )

    summary_path = (
        OUTPUT_ROOT
        / "all_plates_two_frames_summary.csv"
    )

    fieldnames = []

    for row in summaries:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print(
        f"\nSaved summary: {summary_path}"
    )

    print(
        f"Saved debug  : {OUTPUT_ROOT}"
    )

    print(
        "=================================================================\n"
    )






# ============================================================
# RANSAC LINE FIT FOR ALL SIDES
# ============================================================
#
# This version keeps the existing yellow-exit ridge extraction exactly as-is,
# but replaces the final free XY cv2.fitLine with a robust fit in the local
# safe-side coordinates:
#
#       outward_offset d(t) = a*t + b
#
# This is intended to reject local disturbances (e.g. screws) without forcing
# the refined side to remain parallel to the safe side.
# ============================================================

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


def fit_td_line_from_two_points(
    t1: float,
    d1: float,
    t2: float,
    d2: float,
):
    dt = float(t2 - t1)

    if abs(dt) < 1e-9:
        return None

    a = float((d2 - d1) / dt)
    b = float(d1 - a * t1)

    return a, b


def fit_td_least_squares(
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


def evaluate_td_ransac_model(
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
        np.count_nonzero(
            inliers
        )
    )

    if n_inliers < 2:
        return {
            "inlier_count": n_inliers,
            "inlier_fraction": 0.0,
            "t_span_px": 0.0,
            "t_span_fraction": 0.0,
            "median_abs_residual_px": float("inf"),
            "mean_abs_residual_px": float("inf"),
            "score": -float("inf"),
        }

    t_in = t[inliers]
    d_in = d[inliers]

    predicted = (
        a * t_in
        + b
    )

    residuals = np.abs(
        d_in - predicted
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
        np.median(
            residuals
        )
    )

    mean_abs_residual = float(
        np.mean(
            residuals
        )
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
        "inlier_count": n_inliers,
        "inlier_fraction": inlier_fraction,
        "t_span_px": t_span_px,
        "t_span_fraction": t_span_fraction,
        "median_abs_residual_px": median_abs_residual,
        "mean_abs_residual_px": mean_abs_residual,
        "score": score,
    }


def ransac_fit_td(
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

    for _ in range(
        RANSAC_ITERATIONS
    ):
        i, j = rng.choice(
            len(t),
            size=2,
            replace=False,
        )

        model = (
            fit_td_line_from_two_points(
                float(t[i]),
                float(d[i]),
                float(t[j]),
                float(d[j]),
            )
        )

        if model is None:
            continue

        a, b = model

        residuals = np.abs(
            d
            - (
                a * t
                + b
            )
        )

        inliers = (
            residuals
            <= float(
                threshold_px
            )
        )

        if np.count_nonzero(
            inliers
        ) < 2:
            continue

        a_refit, b_refit = (
            fit_td_least_squares(
                t[inliers],
                d[inliers],
            )
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
            <= float(
                threshold_px
            )
        )

        if np.count_nonzero(
            inliers_refit
        ) < 2:
            continue

        a_final, b_final = (
            fit_td_least_squares(
                t[inliers_refit],
                d[inliers_refit],
            )
        )

        metrics = (
            evaluate_td_ransac_model(
                t=t,
                d=d,
                inliers=inliers_refit,
                a=a_final,
                b=b_final,
                side_length=side_length,
            )
        )

        candidate = {
            "threshold_px": float(
                threshold_px
            ),
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


def td_model_to_xy_line(
    *,
    safe_quad: np.ndarray,
    side_index: int,
    a: float,
    b: float,
):
    (
        p1,
        p2,
        t_axis,
        _n_in,
        n_out,
    ) = side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(
            p2 - p1
        )
    )

    q0 = (
        p1
        + b
        * n_out
    )

    q1 = (
        p1
        + side_length
        * t_axis
        + (
            a
            * side_length
            + b
        )
        * n_out
    )

    direction = normalize(
        q1 - q0
    )

    return (
        q0,
        direction,
    )


def refine_one_side_soft_profile_ransac(
    *,
    soft_support: np.ndarray,
    safe_quad: np.ndarray,
    side_index: int,
    soft_threshold: float,
    safe_side_offset_px: float,
):
    ridge_points, profile_rows, stats = (
        extract_side_ridge_points(
            soft_support=soft_support,
            safe_quad=safe_quad,
            side_index=side_index,
            soft_threshold=soft_threshold,
            safe_side_offset_px=safe_side_offset_px,
        )
    )

    valid_rows = [
        row
        for row in profile_rows
        if bool(
            row["valid"]
        )
        and np.isfinite(
            row[
                "selected_outward_px"
            ]
        )
    ]

    trustworthy = bool(
        len(valid_rows)
        >= MIN_RIDGE_POINTS
        and stats[
            "valid_fraction"
        ]
        >= MIN_VALID_RIDGE_FRACTION
    )

    if not trustworthy:
        (
            p1,
            _p2,
            t,
            _n_in,
            _n_out,
        ) = side_local_frame(
            safe_quad,
            side_index,
        )

        return {
            "fallback": True,
            "fit_source": "safe_side_fallback",
            "point": p1.copy(),
            "direction": t.copy(),
            "fit_points": np.empty(
                (0, 2),
                dtype=np.float64,
            ),
            "ridge_points": ridge_points,
            "profile_rows": profile_rows,
            "ransac_threshold_px": np.nan,
            "ransac_inlier_fraction": 0.0,
            "ransac_t_span_fraction": 0.0,
            "ransac_median_residual_px": np.nan,
            **stats,
        }

    t_values = np.asarray(
        [
            row[
                "tangential_px"
            ]
            for row in valid_rows
        ],
        dtype=np.float64,
    )

    d_values = np.asarray(
        [
            row[
                "selected_outward_px"
            ]
            for row in valid_rows
        ],
        dtype=np.float64,
    )

    (
        p1,
        p2,
        t_axis,
        _n_in,
        n_out,
    ) = side_local_frame(
        safe_quad,
        side_index,
    )

    side_length = float(
        np.linalg.norm(
            p2 - p1
        )
    )

    models = []

    for threshold_px in (
        RANSAC_THRESHOLDS_PX
    ):
        candidate = (
            ransac_fit_td(
                t=t_values,
                d=d_values,
                threshold_px=threshold_px,
                side_length=side_length,
            )
        )

        candidate[
            "passes_span_guard"
        ] = bool(
            candidate[
                "t_span_fraction"
            ]
            >= MIN_RANSAC_TANGENTIAL_SPAN_FRACTION
        )

        models.append(
            candidate
        )

    eligible = [
        row
        for row in models
        if row[
            "passes_span_guard"
        ]
    ]

    best = max(
        (
            eligible
            if eligible
            else models
        ),
        key=lambda row: row[
            "score"
        ],
    )

    point, direction = (
        td_model_to_xy_line(
            safe_quad=safe_quad,
            side_index=side_index,
            a=best["a"],
            b=best["b"],
        )
    )

    raw_profile_points_xy = (
        p1[None, :]
        + t_values[:, None]
        * t_axis[None, :]
        + d_values[:, None]
        * n_out[None, :]
    )

    fit_points = (
        raw_profile_points_xy[
            best["inliers"]
        ]
    )

    return {
        "fallback": False,
        "fit_source": "ransac_td_final_yellow_exit",
        "point": point,
        "direction": direction,
        "fit_points": fit_points,
        "ridge_points": raw_profile_points_xy,
        "profile_rows": profile_rows,
        "ransac_threshold_px": float(
            best[
                "threshold_px"
            ]
        ),
        "ransac_inlier_fraction": float(
            best[
                "inlier_fraction"
            ]
        ),
        "ransac_t_span_fraction": float(
            best[
                "t_span_fraction"
            ]
        ),
        "ransac_median_residual_px": float(
            best[
                "median_abs_residual_px"
            ]
        ),
        **stats,
    }


def process_case_ransac(
    frame_name: str,
    plate_index: int,
) -> dict:
    print(
        f"\n--- {frame_name} | plate {plate_index} | RANSAC ---"
    )

    case_dir = (
        OUTPUT_ROOT_RANSAC
        / f"{frame_name}_plate_{plate_index}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop, _x_offset, _y_offset, pred = (
        load_target_plate(
            frame_name,
            plate_index,
        )
    )

    (
        clean_mask,
        raw_yellow_mask,
        raw_yellow_soft,
    ) = create_clean_yellow_mask(
        crop,
        float(
            pred["width"]
        ),
    )

    min_quad = min_area_rect_quad(
        clean_mask
    )

    side_offsets, safe_side_metrics = (
        find_side_specific_safe_offsets(
            min_quad=min_quad,
            clean_mask=clean_mask,
            raw_mask=raw_yellow_mask,
            raw_soft=raw_yellow_soft,
        )
    )

    safe_quad = (
        build_quad_from_side_offsets(
            min_quad=min_quad,
            side_offsets=side_offsets,
        )
    )

    shrink_px = float(
        np.mean(
            side_offsets
        )
    )

    coverages = [
        float(
            metric[
                "clean_fraction"
            ]
        )
        for metric in safe_side_metrics
    ]

    save_safe_side_validation_debug(
        case_dir=case_dir,
        crop=crop,
        min_quad=min_quad,
        safe_quad=safe_quad,
        raw_mask=raw_yellow_mask,
        raw_soft=raw_yellow_soft,
        side_metrics=safe_side_metrics,
    )

    (
        lab_image,
        dark_mask,
        boundary_samples,
        sample_counts,
        debug_sample_rows,
    ) = collect_boundary_line_color_samples(
        crop=crop,
        clean_mask=clean_mask,
        safe_quad=safe_quad,
    )

    model = robust_lab_model(
        boundary_samples
    )

    (
        soft_support,
        hard_color_mask,
        _z_distance,
        soft_threshold,
    ) = build_color_similarity_maps(
        lab_image=lab_image,
        dark_mask=dark_mask,
        model=model,
    )

    (
        second_mask,
        inner_mask,
        remainder_region,
    ) = build_second_remainder_mask(
        hard_color_mask=hard_color_mask,
        safe_quad=safe_quad,
    )

    side_names = [
        "top",
        "right",
        "bottom",
        "left",
    ]

    fitted_sides = []

    for side_index, side_name in enumerate(
        side_names
    ):
        fitted = (
            refine_one_side_soft_profile_ransac(
                soft_support=soft_support,
                safe_quad=safe_quad,
                side_index=side_index,
                soft_threshold=soft_threshold,
                safe_side_offset_px=float(
                    side_offsets[
                        side_index
                    ]
                ),
            )
        )

        fitted_sides.append(
            fitted
        )

        print(
            f"{side_name:>6}: "
            f"fallback={fitted['fallback']} | "
            f"valid={fitted['valid_fraction']:.3f} | "
            f"N={fitted['ridge_point_count']} | "
            f"out={fitted['median_outward_px']:.2f}px | "
            f"ransac_thr={fitted.get('ransac_threshold_px', np.nan):.2f} | "
            f"inliers={fitted.get('ransac_inlier_fraction', 0.0):.2f} | "
            f"span={fitted.get('ransac_t_span_fraction', 0.0):.2f}"
        )

        save_normal_profile_side_debug(
            case_dir=case_dir,
            crop=crop,
            soft_support=soft_support,
            safe_quad=safe_quad,
            side_name=side_name,
            side_index=side_index,
            fitted=fitted,
        )

    refined_quad = (
        refined_quad_from_profile_lines(
            fitted_sides
        )
    )

    refined_quad = snap_quad_corners(
        refined_quad,
        second_mask,
    )

    sampling_view = build_sampling_view(
        crop,
        debug_sample_rows,
    )

    panel = make_debug_panel(
        crop=crop,
        clean_mask=clean_mask,
        min_quad=min_quad,
        safe_quad=safe_quad,
        sampling_view=sampling_view,
        model=model,
        hard_color_mask=hard_color_mask,
        soft_support=soft_support,
        second_mask=second_mask,
        refined_quad=refined_quad,
        frame_name=frame_name,
        plate_index=plate_index,
        shrink_px=shrink_px,
        coverages=coverages,
        selected_sides=fitted_sides,
    )

    panel_path = (
        case_dir
        / "final_yellow_exit_boundary_refinement_panel_RANSAC.jpg"
    )

    cv2.imwrite(
        str(
            panel_path
        ),
        panel,
    )

    cv2.imwrite(
        str(
            case_dir
            / "00_crop.jpg"
        ),
        crop,
    )

    cv2.imwrite(
        str(
            case_dir
            / "01_clean_yellow_mask.png"
        ),
        clean_mask,
    )

    cv2.imwrite(
        str(
            case_dir
            / "02_edge_line_samples.jpg"
        ),
        sampling_view,
    )

    cv2.imwrite(
        str(
            case_dir
            / "03_learned_color_hard_mask_whole_crop.png"
        ),
        hard_color_mask,
    )

    soft_vis = np.clip(
        soft_support * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(
            case_dir
            / "04_learned_color_soft_support.png"
        ),
        soft_vis,
    )

    cv2.imwrite(
        str(
            case_dir
            / "05_safe_inner_polygon_mask.png"
        ),
        inner_mask,
    )

    cv2.imwrite(
        str(
            case_dir
            / "06_remainder_region.png"
        ),
        remainder_region,
    )

    cv2.imwrite(
        str(
            case_dir
            / "07_SECOND_MASK_learned_color_in_remainder.png"
        ),
        second_mask,
    )

    refined_overlay = draw_quad(
        crop,
        refined_quad,
        (255, 0, 255),
        1,
    )

    cv2.imwrite(
        str(
            case_dir
            / "08_refined_quad_RANSAC.jpg"
        ),
        refined_overlay,
    )

    fitted_side_summary_path = (
        case_dir
        / "fitted_side_summary_RANSAC.csv"
    )

    rows = []

    for side_name, fitted in zip(
        side_names,
        fitted_sides,
    ):
        rows.append(
            {
                "side": side_name,
                "fallback": fitted["fallback"],
                "fit_source": fitted["fit_source"],
                "valid_fraction": fitted["valid_fraction"],
                "ridge_point_count": fitted["ridge_point_count"],
                "median_outward_px": fitted["median_outward_px"],
                "min_outward_px": fitted["min_outward_px"],
                "max_outward_px": fitted["max_outward_px"],
                "ransac_threshold_px": fitted.get(
                    "ransac_threshold_px",
                    np.nan,
                ),
                "ransac_inlier_fraction": fitted.get(
                    "ransac_inlier_fraction",
                    0.0,
                ),
                "ransac_t_span_fraction": fitted.get(
                    "ransac_t_span_fraction",
                    0.0,
                ),
                "ransac_median_residual_px": fitted.get(
                    "ransac_median_residual_px",
                    np.nan,
                ),
            }
        )

    with fitted_side_summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return {
        "frame_name": frame_name,
        "plate_index": plate_index,
        "status": "ok",
        "panel_path": str(
            panel_path
        ),
    }


OUTPUT_ROOT_RANSAC = (
    PROJECT_ROOT
    / "corner_refinement"
    / "debug"
    / "final_yellow_exit_all_plates_ransac"
)


def main_ransac_all_plates() -> None:
    print(
        "\n========== FINAL YELLOW EXIT | ALL PLATES | RANSAC FIT =========="
    )

    OUTPUT_ROOT_RANSAC.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for frame_name in FRAME_NAMES:
        try:
            plate_indices = (
                load_all_plate_indices(
                    frame_name
                )
            )
        except Exception as exc:
            print(
                f"FAILED loading detections for {frame_name}: {exc}"
            )

            summaries.append(
                {
                    "frame_name": frame_name,
                    "plate_index": "",
                    "status": f"frame error: {exc}",
                }
            )
            continue

        for plate_index in plate_indices:
            try:
                summary = (
                    process_case_ransac(
                        frame_name,
                        plate_index,
                    )
                )

                summaries.append(
                    summary
                )

            except Exception as exc:
                print(
                    f"FAILED {frame_name} | "
                    f"plate {plate_index}: {exc}"
                )

                summaries.append(
                    {
                        "frame_name": frame_name,
                        "plate_index": plate_index,
                        "status": f"error: {exc}",
                    }
                )

    summary_path = (
        OUTPUT_ROOT_RANSAC
        / "all_plates_ransac_summary.csv"
    )

    fieldnames = []

    for row in summaries:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print(
        f"\nSaved summary: {summary_path}"
    )

    print(
        f"Saved debug  : {OUTPUT_ROOT_RANSAC}"
    )

    print(
        "================================================================\n"
    )


if __name__ == "__main__":
    main_ransac_all_plates()
