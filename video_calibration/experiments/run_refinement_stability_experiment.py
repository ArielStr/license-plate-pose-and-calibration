from __future__ import annotations

from pathlib import Path
import csv
import math
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


from video_calibration.frame_processor import process_sampled_frame
from video_calibration.frame_sampler import SampledFrame
from video_calibration.tracking import (
    PlateTrackerConfig,
    build_tracks,
    get_long_tracks,
    print_track_summary,
)


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "test_video2.mov"
)

API_KEY = os.getenv(
    "ROBOFLOW_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set."
    )

# Controlled short sequence.
START_FRAME = 0
NUM_SAMPLED_FRAMES = 30
FRAME_STEP = 1

# We only analyze tracks that survive for at least this many observations.
MIN_TRACK_LENGTH = 5

# Number of refinements shown in each visual panel.
PANEL_SIZE = 10

USE_CACHED_JSON = True

MIN_CONFIDENCE = 0.35
MIN_DETECTION_AREA = 300.0
MIN_WIDTH = 80.0
MIN_HEIGHT = 20.0
NMS_IOU_THRESHOLD = 0.5

REFINEMENT_METHOD = "robust_mask_lines"

# The custom panels below are the panels for this experiment.
# Keep the normal per-frame debug panels disabled.
REFINEMENT_DEBUG = False
REFINEMENT_PANEL_DEBUG = False

TRACKER_CONFIG = PlateTrackerConfig(
    min_iou=0.10,
    max_normalized_center_distance=1.50,
    max_area_ratio=2.50,
    max_frame_gap=max(
        3,
        FRAME_STEP + 1,
    ),
)

VIDEO_CALIBRATION_DIR = (
    PROJECT_ROOT
    / "video_calibration"
)

OUTPUT_DIR = (
    VIDEO_CALIBRATION_DIR
    / "outputs"
    / "corner_tracking"
)

PANEL_DIR = (
    OUTPUT_DIR
    / "refinement_panels"
)

OBSERVATION_CSV_PATH = (
    OUTPUT_DIR
    / "refinement_stability_observations.csv"
)

TRACK_SUMMARY_CSV_PATH = (
    OUTPUT_DIR
    / "refinement_stability_summary.csv"
)


# ============================================================
# VIDEO READING
# ============================================================

def read_sampled_frames(
    video_path: Path,
    *,
    start_frame: int,
    num_sampled_frames: int,
    frame_step: int,
) -> list[SampledFrame]:
    if frame_step <= 0:
        raise ValueError(
            "frame_step must be positive."
        )

    if num_sampled_frames <= 0:
        raise ValueError(
            "num_sampled_frames must be positive."
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    sampled_frames: list[SampledFrame] = []

    try:
        for sample_index in range(
            num_sampled_frames
        ):
            frame_index = (
                start_frame
                + sample_index * frame_step
            )

            if frame_index >= total_frames:
                break

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, image = capture.read()

            if not success:
                print(
                    f"Warning: could not read "
                    f"frame {frame_index}"
                )
                continue

            timestamp_sec = (
                frame_index / fps
                if fps > 0.0
                else 0.0
            )

            sampled_frames.append(
                SampledFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image=image,
                )
            )

    finally:
        capture.release()

    return sampled_frames


# ============================================================
# OBSERVATION COLLECTION
# ============================================================

def collect_observations(
    sampled_frames: list[SampledFrame],
) -> list:
    all_observations = []

    for sampled_frame in sampled_frames:
        observations = process_sampled_frame(
            sampled_frame,
            api_key=API_KEY,
            video_name=VIDEO_PATH.stem,
            use_cached_json=USE_CACHED_JSON,
            min_confidence=MIN_CONFIDENCE,
            min_detection_area=MIN_DETECTION_AREA,
            min_width=MIN_WIDTH,
            min_height=MIN_HEIGHT,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            refinement_method=REFINEMENT_METHOD,
            refinement_debug=REFINEMENT_DEBUG,
            refinement_panel_debug=REFINEMENT_PANEL_DEBUG,
        )

        all_observations.extend(
            observations
        )

    return all_observations


# ============================================================
# GEOMETRY / STABILITY METRICS
# ============================================================

def polygon_area(
    corners: np.ndarray,
) -> float:
    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    return abs(
        float(
            cv2.contourArea(
                corners.astype(
                    np.float32
                )
            )
        )
    )


def edge_lengths(
    corners: np.ndarray,
) -> np.ndarray:
    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    return np.asarray(
        [
            np.linalg.norm(
                corners[
                    (index + 1) % 4
                ]
                - corners[index]
            )
            for index in range(4)
        ],
        dtype=np.float64,
    )


def quad_geometry(
    corners: np.ndarray,
) -> dict:
    lengths = edge_lengths(
        corners
    )

    top = float(lengths[0])
    right = float(lengths[1])
    bottom = float(lengths[2])
    left = float(lengths[3])

    mean_width = 0.5 * (
        top + bottom
    )

    mean_height = 0.5 * (
        left + right
    )

    aspect_ratio = (
        mean_width / mean_height
        if mean_height > 1e-9
        else float("nan")
    )

    return {
        "top_edge_px": top,
        "right_edge_px": right,
        "bottom_edge_px": bottom,
        "left_edge_px": left,
        "mean_width_px": mean_width,
        "mean_height_px": mean_height,
        "aspect_ratio": aspect_ratio,
        "quad_area_px2": polygon_area(
            corners
        ),
    }


def normalize_corners_to_bbox(
    corners: np.ndarray,
    bbox,
) -> np.ndarray:
    """
    Remove most image translation/scale effects.

    The result describes where each refined corner lies inside its
    detection bbox. This is a diagnostic stability coordinate system:
    it is not a ground-truth corner error.
    """
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = [
        float(value)
        for value in bbox
    ]

    width = max(
        x_max - x_min,
        1e-9,
    )

    height = max(
        y_max - y_min,
        1e-9,
    )

    normalized = np.asarray(
        corners,
        dtype=np.float64,
    ).copy()

    normalized[:, 0] = (
        normalized[:, 0] - x_min
    ) / width

    normalized[:, 1] = (
        normalized[:, 1] - y_min
    ) / height

    return normalized


def rms_corner_distance(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
) -> float:
    differences = (
        np.asarray(
            corners_a,
            dtype=np.float64,
        )
        - np.asarray(
            corners_b,
            dtype=np.float64,
        )
    )

    squared_distances = np.sum(
        differences ** 2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                squared_distances
            )
        )
    )


def mean_corner_distance(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
) -> float:
    differences = (
        np.asarray(
            corners_a,
            dtype=np.float64,
        )
        - np.asarray(
            corners_b,
            dtype=np.float64,
        )
    )

    distances = np.linalg.norm(
        differences,
        axis=1,
    )

    return float(
        np.mean(
            distances
        )
    )


def max_corner_distance(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
) -> float:
    differences = (
        np.asarray(
            corners_a,
            dtype=np.float64,
        )
        - np.asarray(
            corners_b,
            dtype=np.float64,
        )
    )

    distances = np.linalg.norm(
        differences,
        axis=1,
    )

    return float(
        np.max(
            distances
        )
    )


def build_track_metric_rows(
    track,
) -> list[dict]:
    observations = sorted(
        track.observations,
        key=lambda observation: (
            observation.frame_index
        ),
    )

    normalized_corners = [
        normalize_corners_to_bbox(
            observation.corners,
            observation.bbox,
        )
        for observation in observations
    ]

    # Robust reference shape for the complete track.
    median_normalized_corners = np.median(
        np.stack(
            normalized_corners,
            axis=0,
        ),
        axis=0,
    )

    rows: list[dict] = []

    previous_observation = None
    previous_normalized = None
    previous_geometry = None

    for (
        observation_index,
        observation,
    ) in enumerate(
        observations
    ):
        corners = np.asarray(
            observation.corners,
            dtype=np.float64,
        )

        normalized = (
            normalized_corners[
                observation_index
            ]
        )

        geometry = quad_geometry(
            corners
        )

        (
            bbox_x_min,
            bbox_y_min,
            bbox_x_max,
            bbox_y_max,
        ) = [
            float(value)
            for value in observation.bbox
        ]

        bbox_width = max(
            bbox_x_max - bbox_x_min,
            1e-9,
        )

        bbox_height = max(
            bbox_y_max - bbox_y_min,
            1e-9,
        )

        bbox_diagonal = math.hypot(
            bbox_width,
            bbox_height,
        )

        row = {
            "track_id": track.track_id,
            "track_observation_index": observation_index,
            "frame_index": observation.frame_index,
            "timestamp_sec": observation.timestamp_sec,
            "accepted": observation.accepted,
            "rejection_reason": (
                observation.rejection_reason
                or ""
            ),
            "quality_score": observation.quality_score,
            "bbox_width_px": bbox_width,
            "bbox_height_px": bbox_height,
            **geometry,
            # Shape jitter relative to the median shape of the track.
            "normalized_shape_rms_to_track_median": (
                rms_corner_distance(
                    normalized,
                    median_normalized_corners,
                )
            ),
            "normalized_shape_mean_to_track_median": (
                mean_corner_distance(
                    normalized,
                    median_normalized_corners,
                )
            ),
        }

        if previous_observation is None:
            row.update(
                {
                    "frame_gap": "",
                    "raw_corner_mean_step_px": "",
                    "raw_corner_max_step_px": "",
                    "raw_corner_rms_step_px": "",
                    "raw_corner_mean_step_over_bbox_diag": "",
                    "normalized_shape_mean_step": "",
                    "normalized_shape_max_step": "",
                    "normalized_shape_rms_step": "",
                    "width_change_pct": "",
                    "height_change_pct": "",
                    "area_change_pct": "",
                    "aspect_ratio_change": "",
                }
            )

        else:
            previous_corners = np.asarray(
                previous_observation.corners,
                dtype=np.float64,
            )

            raw_mean = mean_corner_distance(
                corners,
                previous_corners,
            )

            raw_max = max_corner_distance(
                corners,
                previous_corners,
            )

            raw_rms = rms_corner_distance(
                corners,
                previous_corners,
            )

            normalized_mean = mean_corner_distance(
                normalized,
                previous_normalized,
            )

            normalized_max = max_corner_distance(
                normalized,
                previous_normalized,
            )

            normalized_rms = rms_corner_distance(
                normalized,
                previous_normalized,
            )

            previous_width = max(
                previous_geometry[
                    "mean_width_px"
                ],
                1e-9,
            )

            previous_height = max(
                previous_geometry[
                    "mean_height_px"
                ],
                1e-9,
            )

            previous_area = max(
                previous_geometry[
                    "quad_area_px2"
                ],
                1e-9,
            )

            row.update(
                {
                    "frame_gap": (
                        observation.frame_index
                        - previous_observation.frame_index
                    ),
                    "raw_corner_mean_step_px": raw_mean,
                    "raw_corner_max_step_px": raw_max,
                    "raw_corner_rms_step_px": raw_rms,
                    "raw_corner_mean_step_over_bbox_diag": (
                        raw_mean / bbox_diagonal
                    ),
                    "normalized_shape_mean_step": normalized_mean,
                    "normalized_shape_max_step": normalized_max,
                    "normalized_shape_rms_step": normalized_rms,
                    "width_change_pct": (
                        100.0
                        * (
                            geometry["mean_width_px"]
                            - previous_geometry[
                                "mean_width_px"
                            ]
                        )
                        / previous_width
                    ),
                    "height_change_pct": (
                        100.0
                        * (
                            geometry["mean_height_px"]
                            - previous_geometry[
                                "mean_height_px"
                            ]
                        )
                        / previous_height
                    ),
                    "area_change_pct": (
                        100.0
                        * (
                            geometry["quad_area_px2"]
                            - previous_geometry[
                                "quad_area_px2"
                            ]
                        )
                        / previous_area
                    ),
                    "aspect_ratio_change": (
                        geometry["aspect_ratio"]
                        - previous_geometry[
                            "aspect_ratio"
                        ]
                    ),
                }
            )

        for corner_index in range(4):
            row[
                f"corner_{corner_index}_x"
            ] = float(
                corners[
                    corner_index,
                    0,
                ]
            )

            row[
                f"corner_{corner_index}_y"
            ] = float(
                corners[
                    corner_index,
                    1,
                ]
            )

            row[
                f"corner_{corner_index}_norm_x"
            ] = float(
                normalized[
                    corner_index,
                    0,
                ]
            )

            row[
                f"corner_{corner_index}_norm_y"
            ] = float(
                normalized[
                    corner_index,
                    1,
                ]
            )

        rows.append(
            row
        )

        previous_observation = observation
        previous_normalized = normalized
        previous_geometry = geometry

    return rows


def finite_values(
    rows: list[dict],
    key: str,
) -> list[float]:
    values = []

    for row in rows:
        value = row.get(
            key,
            "",
        )

        if value == "":
            continue

        value = float(
            value
        )

        if math.isfinite(
            value
        ):
            values.append(
                value
            )

    return values


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return float(
        np.mean(
            values
        )
    )


def safe_median(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return float(
        np.median(
            values
        )
    )


def safe_max(
    values: list[float],
) -> float:
    if not values:
        return float("nan")

    return float(
        np.max(
            values
        )
    )


def summarize_track_rows(
    track,
    rows: list[dict],
) -> dict:
    normalized_steps = finite_values(
        rows,
        "normalized_shape_mean_step",
    )

    normalized_rms_to_median = finite_values(
        rows,
        "normalized_shape_rms_to_track_median",
    )

    raw_steps = finite_values(
        rows,
        "raw_corner_mean_step_px",
    )

    width_changes = [
        abs(value)
        for value in finite_values(
            rows,
            "width_change_pct",
        )
    ]

    height_changes = [
        abs(value)
        for value in finite_values(
            rows,
            "height_change_pct",
        )
    ]

    area_changes = [
        abs(value)
        for value in finite_values(
            rows,
            "area_change_pct",
        )
    ]

    aspect_ratios = finite_values(
        rows,
        "aspect_ratio",
    )

    accepted_count = sum(
        bool(
            observation.accepted
        )
        for observation in track.observations
    )

    return {
        "track_id": track.track_id,
        "length": track.length,
        "first_frame": track.first_frame_index,
        "last_frame": track.last_frame_index,
        "accepted_count": accepted_count,
        "accepted_fraction": (
            accepted_count
            / max(
                track.length,
                1,
            )
        ),
        "mean_raw_corner_step_px": (
            safe_mean(
                raw_steps
            )
        ),
        "median_raw_corner_step_px": (
            safe_median(
                raw_steps
            )
        ),
        "max_raw_corner_step_px": (
            safe_max(
                raw_steps
            )
        ),
        "mean_normalized_shape_step": (
            safe_mean(
                normalized_steps
            )
        ),
        "median_normalized_shape_step": (
            safe_median(
                normalized_steps
            )
        ),
        "max_normalized_shape_step": (
            safe_max(
                normalized_steps
            )
        ),
        "mean_normalized_rms_to_track_median": (
            safe_mean(
                normalized_rms_to_median
            )
        ),
        "max_normalized_rms_to_track_median": (
            safe_max(
                normalized_rms_to_median
            )
        ),
        "mean_abs_width_change_pct": (
            safe_mean(
                width_changes
            )
        ),
        "mean_abs_height_change_pct": (
            safe_mean(
                height_changes
            )
        ),
        "mean_abs_area_change_pct": (
            safe_mean(
                area_changes
            )
        ),
        "aspect_ratio_mean": (
            safe_mean(
                aspect_ratios
            )
        ),
        "aspect_ratio_std": (
            float(
                np.std(
                    aspect_ratios
                )
            )
            if aspect_ratios
            else float("nan")
        ),
    }


# ============================================================
# CSV
# ============================================================

def save_rows_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(
            f"No rows to save: {output_path}"
        )
        return

    all_fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in all_fieldnames:
                all_fieldnames.append(
                    key
                )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Saved CSV: {output_path}"
    )


# ============================================================
# REFINEMENT PANELS
# ============================================================

def read_video_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray | None:
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, image = capture.read()

    if not success:
        return None

    return image


def make_refinement_crop(
    image: np.ndarray,
    observation,
    *,
    target_width: int = 360,
    target_height: int = 210,
) -> np.ndarray:
    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = [
        float(value)
        for value in observation.bbox
    ]

    bbox_width = max(
        x_max - x_min,
        1.0,
    )

    bbox_height = max(
        y_max - y_min,
        1.0,
    )

    # Extra margin so the refinement can be judged visually.
    pad_x = 0.20 * bbox_width
    pad_y = 0.35 * bbox_height

    crop_x_min = max(
        0,
        int(
            math.floor(
                x_min - pad_x
            )
        ),
    )

    crop_y_min = max(
        0,
        int(
            math.floor(
                y_min - pad_y
            )
        ),
    )

    crop_x_max = min(
        image.shape[1],
        int(
            math.ceil(
                x_max + pad_x
            )
        ),
    )

    crop_y_max = min(
        image.shape[0],
        int(
            math.ceil(
                y_max + pad_y
            )
        ),
    )

    crop = image[
        crop_y_min:crop_y_max,
        crop_x_min:crop_x_max,
    ].copy()

    if crop.size == 0:
        return np.zeros(
            (
                target_height,
                target_width,
                3,
            ),
            dtype=np.uint8,
        )

    local_corners = np.asarray(
        observation.corners,
        dtype=np.float64,
    ).copy()

    local_corners[:, 0] -= (
        crop_x_min
    )

    local_corners[:, 1] -= (
        crop_y_min
    )

    polyline = np.round(
        local_corners
    ).astype(
        np.int32
    )

    cv2.polylines(
        crop,
        [
            polyline.reshape(
                -1,
                1,
                2,
            )
        ],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    for (
        corner_index,
        point,
    ) in enumerate(
        polyline
    ):
        point_tuple = (
            int(point[0]),
            int(point[1]),
        )

        cv2.circle(
            crop,
            point_tuple,
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            crop,
            str(
                corner_index
            ),
            (
                point_tuple[0] + 7,
                point_tuple[1] - 7,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    scale = min(
        target_width / crop.shape[1],
        target_height / crop.shape[0],
    )

    resized_width = max(
        1,
        int(
            round(
                crop.shape[1]
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                crop.shape[0]
                * scale
            )
        ),
    )

    resized = cv2.resize(
        crop,
        (
            resized_width,
            resized_height,
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        ),
    )

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        target_width
        - resized_width
    ) // 2

    y_offset = (
        target_height
        - resized_height
    ) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


def create_track_refinement_panels(
    track,
    rows: list[dict],
    *,
    video_path: Path,
    output_dir: Path,
    panel_size: int,
) -> None:
    observations = sorted(
        track.observations,
        key=lambda observation: (
            observation.frame_index
        ),
    )

    row_by_frame = {
        int(
            row["frame_index"]
        ): row
        for row in rows
    }

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video for panel generation: "
            f"{video_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cell_width = 360
    image_height = 210
    header_height = 62
    cell_height = (
        header_height
        + image_height
    )

    try:
        for panel_start in range(
            0,
            len(observations),
            panel_size,
        ):
            panel_observations = (
                observations[
                    panel_start:
                    panel_start + panel_size
                ]
            )

            columns = min(
                5,
                len(
                    panel_observations
                ),
            )

            rows_count = int(
                math.ceil(
                    len(
                        panel_observations
                    )
                    / columns
                )
            )

            title_height = 55

            panel = np.zeros(
                (
                    title_height
                    + rows_count
                    * cell_height,
                    columns
                    * cell_width,
                    3,
                ),
                dtype=np.uint8,
            )

            panel_number = (
                panel_start
                // panel_size
                + 1
            )

            cv2.putText(
                panel,
                (
                    f"Track {track.track_id} | "
                    f"refinements "
                    f"{panel_start + 1}-"
                    f"{panel_start + len(panel_observations)}"
                ),
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for (
                local_index,
                observation,
            ) in enumerate(
                panel_observations
            ):
                grid_row = (
                    local_index
                    // columns
                )

                grid_column = (
                    local_index
                    % columns
                )

                cell_x = (
                    grid_column
                    * cell_width
                )

                cell_y = (
                    title_height
                    + grid_row
                    * cell_height
                )

                frame = read_video_frame(
                    capture,
                    observation.frame_index,
                )

                if frame is None:
                    continue

                crop = make_refinement_crop(
                    frame,
                    observation,
                    target_width=cell_width,
                    target_height=image_height,
                )

                metric_row = row_by_frame[
                    observation.frame_index
                ]

                normalized_step = (
                    metric_row[
                        "normalized_shape_mean_step"
                    ]
                )

                if normalized_step == "":
                    step_text = "shape step: --"
                else:
                    step_text = (
                        "shape step: "
                        f"{float(normalized_step):.4f}"
                    )

                status_text = (
                    "A"
                    if observation.accepted
                    else "R"
                )

                header_line_1 = (
                    f"frame {observation.frame_index} | "
                    f"{status_text}"
                )

                cv2.putText(
                    panel,
                    header_line_1,
                    (
                        cell_x + 10,
                        cell_y + 23,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    panel,
                    step_text,
                    (
                        cell_x + 10,
                        cell_y + 49,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )

                image_y = (
                    cell_y
                    + header_height
                )

                panel[
                    image_y:
                    image_y + image_height,
                    cell_x:
                    cell_x + cell_width,
                ] = crop

                cv2.rectangle(
                    panel,
                    (
                        cell_x,
                        cell_y,
                    ),
                    (
                        cell_x
                        + cell_width
                        - 1,
                        cell_y
                        + cell_height
                        - 1,
                    ),
                    (90, 90, 90),
                    1,
                )

            output_path = (
                output_dir
                / (
                    f"track_{track.track_id:03d}"
                    f"_panel_{panel_number:02d}.jpg"
                )
            )

            success = cv2.imwrite(
                str(
                    output_path
                ),
                panel,
            )

            if not success:
                raise RuntimeError(
                    f"Could not save panel: "
                    f"{output_path}"
                )

            print(
                f"Saved refinement panel: "
                f"{output_path}"
            )

    finally:
        capture.release()


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_stability_summary(
    summaries: list[dict],
) -> None:
    print(
        "\n========== REFINEMENT STABILITY =========="
    )

    for summary in summaries:
        print(
            f"Track {summary['track_id']:3d} | "
            f"N={summary['length']:3d} | "
            f"accepted="
            f"{100.0 * summary['accepted_fraction']:5.1f}% | "
            f"mean shape step="
            f"{summary['mean_normalized_shape_step']:.5f} | "
            f"max shape step="
            f"{summary['max_normalized_shape_step']:.5f} | "
            f"mean |dAR|="
            f"{summary['aspect_ratio_std']:.5f}"
        )

    print(
        "==========================================\n"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n========== REFINEMENT STABILITY EXPERIMENT =========="
    )
    print(
        f"Video              : {VIDEO_PATH}"
    )
    print(
        f"Start frame        : {START_FRAME}"
    )
    print(
        f"Sampled frames     : {NUM_SAMPLED_FRAMES}"
    )
    print(
        f"Frame step         : {FRAME_STEP}"
    )
    print(
        f"Panel size         : {PANEL_SIZE}"
    )

    sampled_frames = read_sampled_frames(
        VIDEO_PATH,
        start_frame=START_FRAME,
        num_sampled_frames=NUM_SAMPLED_FRAMES,
        frame_step=FRAME_STEP,
    )

    print(
        f"Frames actually read: "
        f"{len(sampled_frames)}"
    )

    if not sampled_frames:
        raise RuntimeError(
            "No frames were read from the video."
        )

    all_observations = collect_observations(
        sampled_frames
    )

    print(
        f"Built observations : "
        f"{len(all_observations)}"
    )

    if not all_observations:
        raise RuntimeError(
            "No plate observations were built."
        )

    tracks = build_tracks(
        all_observations,
        config=TRACKER_CONFIG,
    )

    print_track_summary(
        tracks
    )

    long_tracks = get_long_tracks(
        tracks,
        min_length=MIN_TRACK_LENGTH,
    )

    print(
        f"Tracks with length >= "
        f"{MIN_TRACK_LENGTH}: "
        f"{len(long_tracks)}"
    )

    if not long_tracks:
        raise RuntimeError(
            "No sufficiently long tracks were found."
        )

    all_metric_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for track in long_tracks:
        track_rows = build_track_metric_rows(
            track
        )

        all_metric_rows.extend(
            track_rows
        )

        summary = summarize_track_rows(
            track,
            track_rows,
        )

        all_summary_rows.append(
            summary
        )

        create_track_refinement_panels(
            track,
            track_rows,
            video_path=VIDEO_PATH,
            output_dir=PANEL_DIR,
            panel_size=PANEL_SIZE,
        )

    save_rows_csv(
        all_metric_rows,
        OBSERVATION_CSV_PATH,
    )

    save_rows_csv(
        all_summary_rows,
        TRACK_SUMMARY_CSV_PATH,
    )

    print_stability_summary(
        all_summary_rows
    )

    print(
        "Important interpretation:"
    )
    print(
        "- raw_corner_* contains real plate motion + refinement motion."
    )
    print(
        "- normalized_shape_* removes most bbox translation/scale and "
        "is the main diagnostic for refinement jitter."
    )
    print(
        "- These are temporal consistency metrics, not ground-truth "
        "corner errors."
    )

    print(
        "\nOutputs:"
    )
    print(
        f"  Panels      : {PANEL_DIR}"
    )
    print(
        f"  Observations: {OBSERVATION_CSV_PATH}"
    )
    print(
        f"  Summary     : {TRACK_SUMMARY_CSV_PATH}"
    )
    print(
        "=====================================================\n"
    )


if __name__ == "__main__":
    main()
