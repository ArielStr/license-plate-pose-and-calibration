from __future__ import annotations

from pathlib import Path
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from video_calibration.corner_quality import (
    CornerQualityConfig,
    compute_all_corner_qualities,
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

VIDEO_PATH = PROJECT_ROOT / "cars_photos" / "test_video2.mov"

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set."
    )

START_FRAME = 0
NUM_SAMPLED_FRAMES = 30
FRAME_STEP = 1

MIN_TRACK_LENGTH = 10

TARGET_TRACK_ID = 3
NUM_TARGET_OBSERVATIONS = 10
TARGET_START_POSITION_IN_TRACK = 0

TOP_K_ANCHORS = 5
MIN_VALID_ANCHORS = 3
MAX_CONSENSUS_SPREAD_PX = 3.0

USE_CACHED_JSON = True

MIN_CONFIDENCE = 0.35
MIN_DETECTION_AREA = 300.0
MIN_WIDTH = 80.0
MIN_HEIGHT = 20.0
NMS_IOU_THRESHOLD = 0.5

REFINEMENT_METHOD = "yellow_exit_ransac"
REFINEMENT_DEBUG = False
REFINEMENT_PANEL_DEBUG = False

TRACKER_CONFIG = PlateTrackerConfig(
    min_iou=0.10,
    max_normalized_center_distance=1.50,
    max_area_ratio=2.50,
    max_frame_gap=max(3, FRAME_STEP + 1),
)

CORNER_QUALITY_CONFIG = CornerQualityConfig()

LK_WIN_SIZE = (31, 31)
LK_MAX_LEVEL = 4
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    40,
    0.01,
)

MAX_FORWARD_BACKWARD_ERROR_PX = 2.0


VIDEO_CALIBRATION_DIR = PROJECT_ROOT / "video_calibration"

OUTPUT_DIR = (
    VIDEO_CALIBRATION_DIR
    / "outputs"
    / "corner_consensus_10_frames_fast"
)

CSV_PATH = OUTPUT_DIR / "corner_corrections_fast.csv"

PANEL_PATH = (
    OUTPUT_DIR
    / f"track_{TARGET_TRACK_ID:03d}_before_after_10_frames_fast.jpg"
)


# ============================================================
# VIDEO / PIPELINE
# ============================================================

def read_sampled_frames(
    video_path: Path,
    *,
    start_frame: int,
    num_sampled_frames: int,
    frame_step: int,
) -> list[SampledFrame]:
    """
    Read the diagnostic sequence once.

    Important:
    These images are then reused for:
        - frame processing
        - corner-quality scoring
        - optical flow
        - visualization

    No random VideoCapture seeking is performed later.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_frames: list[SampledFrame] = []

    try:
        for sample_index in range(num_sampled_frames):
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
                    f"Warning: could not read frame {frame_index}"
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

        all_observations.extend(observations)

    return all_observations


def build_frame_maps(
    sampled_frames: list[SampledFrame],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """
    Cache BGR and grayscale frames once.
    """
    color_frames = {
        sampled_frame.frame_index: sampled_frame.image
        for sampled_frame in sampled_frames
    }

    gray_frames = {
        frame_index: cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        for frame_index, image in color_frames.items()
    }

    return color_frames, gray_frames


# ============================================================
# CORNER QUALITY
# ============================================================

def score_all_track_corners(
    track,
    *,
    color_frames: dict[int, np.ndarray],
) -> dict[int, list[dict]]:
    candidates_by_corner: dict[int, list[dict]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    for observation in sorted(
        track.observations,
        key=lambda item: item.frame_index,
    ):
        image = color_frames.get(
            observation.frame_index
        )

        if image is None:
            raise RuntimeError(
                f"Frame {observation.frame_index} "
                "is missing from the frame cache."
            )

        results = compute_all_corner_qualities(
            image,
            observation.corners,
            config=CORNER_QUALITY_CONFIG,
        )

        for result in results:
            candidates_by_corner[
                result.corner_index
            ].append(
                {
                    "frame_index": observation.frame_index,
                    "observation": observation,
                    "quality_score": result.score,
                    "yellow_score": result.yellow_boundary_score,
                    "gradient_score": result.edge_gradient_score,
                    "cornerness_score": result.cornerness_score,
                }
            )

    for corner_index in range(4):
        candidates_by_corner[
            corner_index
        ].sort(
            key=lambda row: (
                -float(row["quality_score"]),
                int(row["frame_index"]),
            )
        )

    return candidates_by_corner


# ============================================================
# BATCH OPTICAL FLOW
# ============================================================

def batch_lk_with_forward_backward(
    gray_from: np.ndarray,
    gray_to: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Track many points in one OpenCV call.

    Returns:
        tracked_points : (N, 2)
        valid          : (N,) bool
        fb_errors      : (N,)
    """
    points_xy = np.asarray(
        points_xy,
        dtype=np.float32,
    )

    if points_xy.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=np.float64),
        )

    points = points_xy.reshape(
        -1,
        1,
        2,
    )

    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_from,
        gray_to,
        points,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )

    count = points_xy.shape[0]

    if forward is None or status_forward is None:
        return (
            np.full((count, 2), np.nan, dtype=np.float64),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, dtype=np.float64),
        )

    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray_to,
        gray_from,
        forward,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )

    if backward is None or status_backward is None:
        return (
            forward.reshape(-1, 2).astype(np.float64),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, dtype=np.float64),
        )

    original = points.reshape(
        -1,
        2,
    ).astype(
        np.float64
    )

    returned = backward.reshape(
        -1,
        2,
    ).astype(
        np.float64
    )

    tracked = forward.reshape(
        -1,
        2,
    ).astype(
        np.float64
    )

    fb_errors = np.linalg.norm(
        returned - original,
        axis=1,
    )

    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(fb_errors)
        & (
            fb_errors
            <= MAX_FORWARD_BACKWARD_ERROR_PX
        )
    )

    return (
        tracked,
        valid,
        fb_errors.astype(np.float64),
    )


def build_corner_trajectory_table(
    track,
    *,
    corner_index: int,
    gray_frames: dict[int, np.ndarray],
) -> dict[
    tuple[int, int],
    dict,
]:
    """
    Precompute propagation from EVERY observation of one corner
    to EVERY reachable observation frame in the same track.

    The key is:
        (source_frame_index, target_frame_index)

    Efficiency:
    For each adjacent frame pair we call LK once with all active source
    anchors together, rather than once per source/target query.
    """
    observations = sorted(
        track.observations,
        key=lambda item: item.frame_index,
    )

    frame_indices = [
        observation.frame_index
        for observation in observations
    ]

    observation_by_frame = {
        observation.frame_index: observation
        for observation in observations
    }

    frame_position = {
        frame_index: position
        for position, frame_index in enumerate(frame_indices)
    }

    table: dict[
        tuple[int, int],
        dict,
    ] = {}

    # Identity entries.
    for observation in observations:
        frame_index = observation.frame_index
        point = np.asarray(
            observation.corners[
                corner_index
            ],
            dtype=np.float64,
        )

        table[
            (
                frame_index,
                frame_index,
            )
        ] = {
            "success": True,
            "point": point.copy(),
            "max_fb_error_px": 0.0,
            "transitions": 0,
        }

    # --------------------------------------------------------
    # FORWARD PROPAGATION
    # --------------------------------------------------------

    # State for every source anchor that has entered the sequence.
    forward_state: dict[int, dict] = {}

    for transition_position in range(
        len(frame_indices) - 1
    ):
        current_frame = frame_indices[
            transition_position
        ]

        next_frame = frame_indices[
            transition_position + 1
        ]

        # Only true consecutive frames are propagated.
        if next_frame != current_frame + 1:
            forward_state.clear()
            continue

        # Add the anchor whose native frame is current_frame.
        native_observation = observation_by_frame[
            current_frame
        ]

        forward_state[
            current_frame
        ] = {
            "point": np.asarray(
                native_observation.corners[
                    corner_index
                ],
                dtype=np.float64,
            ),
            "max_fb_error_px": 0.0,
            "transitions": 0,
            "valid": True,
        }

        active_sources = [
            source_frame
            for source_frame, state in forward_state.items()
            if state["valid"]
        ]

        if not active_sources:
            continue

        points = np.stack(
            [
                forward_state[
                    source_frame
                ]["point"]
                for source_frame in active_sources
            ],
            axis=0,
        )

        (
            tracked_points,
            valid_flags,
            fb_errors,
        ) = batch_lk_with_forward_backward(
            gray_frames[
                current_frame
            ],
            gray_frames[
                next_frame
            ],
            points,
        )

        for local_index, source_frame in enumerate(
            active_sources
        ):
            state = forward_state[
                source_frame
            ]

            if not bool(
                valid_flags[
                    local_index
                ]
            ):
                state["valid"] = False

                table[
                    (
                        source_frame,
                        next_frame,
                    )
                ] = {
                    "success": False,
                    "point": None,
                    "max_fb_error_px": float(
                        max(
                            state[
                                "max_fb_error_px"
                            ],
                            fb_errors[
                                local_index
                            ],
                        )
                    ),
                    "transitions": int(
                        state["transitions"]
                    ),
                }
                continue

            state[
                "point"
            ] = tracked_points[
                local_index
            ]

            state[
                "max_fb_error_px"
            ] = max(
                float(
                    state[
                        "max_fb_error_px"
                    ]
                ),
                float(
                    fb_errors[
                        local_index
                    ]
                ),
            )

            state[
                "transitions"
            ] += 1

            table[
                (
                    source_frame,
                    next_frame,
                )
            ] = {
                "success": True,
                "point": state[
                    "point"
                ].copy(),
                "max_fb_error_px": float(
                    state[
                        "max_fb_error_px"
                    ]
                ),
                "transitions": int(
                    state["transitions"]
                ),
            }

    # --------------------------------------------------------
    # BACKWARD PROPAGATION
    # --------------------------------------------------------

    backward_state: dict[int, dict] = {}

    for transition_position in range(
        len(frame_indices) - 1,
        0,
        -1,
    ):
        current_frame = frame_indices[
            transition_position
        ]

        previous_frame = frame_indices[
            transition_position - 1
        ]

        if previous_frame != current_frame - 1:
            backward_state.clear()
            continue

        native_observation = observation_by_frame[
            current_frame
        ]

        backward_state[
            current_frame
        ] = {
            "point": np.asarray(
                native_observation.corners[
                    corner_index
                ],
                dtype=np.float64,
            ),
            "max_fb_error_px": 0.0,
            "transitions": 0,
            "valid": True,
        }

        active_sources = [
            source_frame
            for source_frame, state in backward_state.items()
            if state["valid"]
        ]

        if not active_sources:
            continue

        points = np.stack(
            [
                backward_state[
                    source_frame
                ]["point"]
                for source_frame in active_sources
            ],
            axis=0,
        )

        (
            tracked_points,
            valid_flags,
            fb_errors,
        ) = batch_lk_with_forward_backward(
            gray_frames[
                current_frame
            ],
            gray_frames[
                previous_frame
            ],
            points,
        )

        for local_index, source_frame in enumerate(
            active_sources
        ):
            state = backward_state[
                source_frame
            ]

            if not bool(
                valid_flags[
                    local_index
                ]
            ):
                state["valid"] = False

                table[
                    (
                        source_frame,
                        previous_frame,
                    )
                ] = {
                    "success": False,
                    "point": None,
                    "max_fb_error_px": float(
                        max(
                            state[
                                "max_fb_error_px"
                            ],
                            fb_errors[
                                local_index
                            ],
                        )
                    ),
                    "transitions": int(
                        state["transitions"]
                    ),
                }
                continue

            state[
                "point"
            ] = tracked_points[
                local_index
            ]

            state[
                "max_fb_error_px"
            ] = max(
                float(
                    state[
                        "max_fb_error_px"
                    ]
                ),
                float(
                    fb_errors[
                        local_index
                    ]
                ),
            )

            state[
                "transitions"
            ] += 1

            table[
                (
                    source_frame,
                    previous_frame,
                )
            ] = {
                "success": True,
                "point": state[
                    "point"
                ].copy(),
                "max_fb_error_px": float(
                    state[
                        "max_fb_error_px"
                    ]
                ),
                "transitions": int(
                    state["transitions"]
                ),
            }

    return table


def build_all_trajectory_tables(
    track,
    *,
    gray_frames: dict[int, np.ndarray],
) -> dict[
    int,
    dict[
        tuple[int, int],
        dict,
    ],
]:
    tables = {}

    for corner_index in range(4):
        start_time = time.perf_counter()

        tables[
            corner_index
        ] = build_corner_trajectory_table(
            track,
            corner_index=corner_index,
            gray_frames=gray_frames,
        )

        elapsed = time.perf_counter() - start_time

        print(
            f"Precomputed trajectories for corner "
            f"{corner_index}: {elapsed:.3f}s"
        )

    return tables


# ============================================================
# CONSENSUS
# ============================================================

def weighted_consensus(
    propagated_rows: list[dict],
) -> np.ndarray:
    valid_rows = [
        row
        for row in propagated_rows
        if row["success"]
    ]

    points = np.stack(
        [
            np.asarray(
                row["point"],
                dtype=np.float64,
            )
            for row in valid_rows
        ],
        axis=0,
    )

    weights = np.asarray(
        [
            max(
                float(row["anchor_quality"]),
                1e-6,
            )
            / (
                1.0
                + float(
                    row["max_fb_error_px"]
                )
            )
            for row in valid_rows
        ],
        dtype=np.float64,
    )

    weights /= np.sum(weights)

    return np.sum(
        points * weights[:, None],
        axis=0,
    )


def consensus_spread(
    propagated_rows: list[dict],
    consensus: np.ndarray,
) -> float:
    distances = [
        float(
            np.linalg.norm(
                np.asarray(
                    row["point"],
                    dtype=np.float64,
                )
                - consensus
            )
        )
        for row in propagated_rows
        if row["success"]
    ]

    if not distances:
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                np.square(distances)
            )
        )
    )


def correct_one_corner_fast(
    *,
    target_observation,
    corner_index: int,
    ranked_candidates: list[dict],
    trajectory_table: dict[
        tuple[int, int],
        dict,
    ],
) -> tuple[np.ndarray, dict]:
    target_frame_index = (
        target_observation.frame_index
    )

    anchors = [
        row
        for row in ranked_candidates
        if int(row["frame_index"])
        != target_frame_index
    ][
        :TOP_K_ANCHORS
    ]

    propagated_rows: list[dict] = []

    for anchor_rank, anchor_row in enumerate(
        anchors,
        start=1,
    ):
        anchor_frame_index = int(
            anchor_row[
                "frame_index"
            ]
        )

        propagation = trajectory_table.get(
            (
                anchor_frame_index,
                target_frame_index,
            )
        )

        if propagation is None:
            success = False
            point = None
            max_fb_error = float("inf")
            transitions = 0

        else:
            success = bool(
                propagation[
                    "success"
                ]
            )

            point = propagation[
                "point"
            ]

            max_fb_error = float(
                propagation[
                    "max_fb_error_px"
                ]
            )

            transitions = int(
                propagation[
                    "transitions"
                ]
            )

        propagated_rows.append(
            {
                "anchor_rank": anchor_rank,
                "anchor_frame_index": (
                    anchor_frame_index
                ),
                "anchor_quality": float(
                    anchor_row[
                        "quality_score"
                    ]
                ),
                "success": success,
                "point": point,
                "max_fb_error_px": max_fb_error,
                "transitions": transitions,
            }
        )

    valid_rows = [
        row
        for row in propagated_rows
        if row["success"]
    ]

    original_corner = np.asarray(
        target_observation.corners[
            corner_index
        ],
        dtype=np.float64,
    )

    if len(valid_rows) < MIN_VALID_ANCHORS:
        return (
            original_corner.copy(),
            {
                "used_correction": False,
                "reason": "too_few_valid_anchors",
                "valid_anchor_count": len(valid_rows),
                "consensus_spread_px": float("nan"),
                "original_to_consensus_px": 0.0,
            },
        )

    consensus = weighted_consensus(
        valid_rows
    )

    spread = consensus_spread(
        valid_rows,
        consensus,
    )

    original_to_consensus = float(
        np.linalg.norm(
            original_corner
            - consensus
        )
    )

    if spread > MAX_CONSENSUS_SPREAD_PX:
        return (
            original_corner.copy(),
            {
                "used_correction": False,
                "reason": "consensus_spread_too_large",
                "valid_anchor_count": len(valid_rows),
                "consensus_spread_px": spread,
                "original_to_consensus_px": (
                    original_to_consensus
                ),
            },
        )

    return (
        consensus.copy(),
        {
            "used_correction": True,
            "reason": "temporal_consensus",
            "valid_anchor_count": len(valid_rows),
            "consensus_spread_px": spread,
            "original_to_consensus_px": (
                original_to_consensus
            ),
        },
    )


# ============================================================
# VISUALIZATION
# ============================================================

def crop_around_plate(
    image: np.ndarray,
    bbox,
    *,
    pad_x_fraction: float = 0.25,
    pad_y_fraction: float = 0.50,
) -> tuple[np.ndarray, int, int]:
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
        1.0,
    )

    height = max(
        y_max - y_min,
        1.0,
    )

    pad_x = (
        pad_x_fraction * width
    )

    pad_y = (
        pad_y_fraction * height
    )

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

    return (
        crop,
        crop_x_min,
        crop_y_min,
    )


def draw_quad_on_crop(
    crop: np.ndarray,
    corners_global: np.ndarray,
    *,
    crop_x_min: int,
    crop_y_min: int,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
) -> np.ndarray:
    result = crop.copy()

    local = np.asarray(
        corners_global,
        dtype=np.float64,
    ).copy()

    local[:, 0] -= crop_x_min
    local[:, 1] -= crop_y_min

    points = np.round(
        local
    ).astype(
        np.int32
    )

    cv2.polylines(
        result,
        [
            points.reshape(
                -1,
                1,
                2,
            )
        ],
        True,
        line_color,
        2,
        cv2.LINE_AA,
    )

    for point in points:
        cv2.circle(
            result,
            (
                int(point[0]),
                int(point[1]),
            ),
            3,
            point_color,
            -1,
            cv2.LINE_AA,
        )

    return result


def fit_to_cell(
    image: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scale = min(
        target_width / image.shape[1],
        target_height / image.shape[0],
    )

    resized_width = max(
        1,
        int(
            round(
                image.shape[1] * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                image.shape[0] * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
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
        target_width - resized_width
    ) // 2

    y_offset = (
        target_height - resized_height
    ) // 2

    canvas[
        y_offset:
        y_offset + resized_height,
        x_offset:
        x_offset + resized_width,
    ] = resized

    return canvas


def create_before_after_panel(
    results: list[dict],
    *,
    color_frames: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    cell_width = 330
    image_height = 180
    header_height = 55

    frames_per_row = 2
    frame_block_width = 2 * cell_width
    frame_block_height = (
        header_height + image_height
    )

    num_rows = int(
        math.ceil(
            len(results)
            / frames_per_row
        )
    )

    title_height = 55

    panel = np.zeros(
        (
            title_height
            + num_rows
            * frame_block_height,
            frames_per_row
            * frame_block_width,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        panel,
        (
            f"Track {TARGET_TRACK_ID} | "
            f"FAST 10-frame temporal corner correction"
        ),
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for result_index, result in enumerate(results):
        grid_row = (
            result_index // frames_per_row
        )

        grid_column = (
            result_index % frames_per_row
        )

        block_x = (
            grid_column * frame_block_width
        )

        block_y = (
            title_height
            + grid_row * frame_block_height
        )

        observation = result["observation"]
        frame_index = observation.frame_index

        image = color_frames[
            frame_index
        ]

        (
            crop,
            crop_x_min,
            crop_y_min,
        ) = crop_around_plate(
            image,
            observation.bbox,
        )

        before = draw_quad_on_crop(
            crop,
            observation.corners,
            crop_x_min=crop_x_min,
            crop_y_min=crop_y_min,
            line_color=(0, 255, 255),
            point_color=(0, 0, 255),
        )

        after = draw_quad_on_crop(
            crop,
            result["corrected_corners"],
            crop_x_min=crop_x_min,
            crop_y_min=crop_y_min,
            line_color=(0, 255, 0),
            point_color=(0, 255, 0),
        )

        before = fit_to_cell(
            before,
            target_width=cell_width,
            target_height=image_height,
        )

        after = fit_to_cell(
            after,
            target_width=cell_width,
            target_height=image_height,
        )

        shifts = np.linalg.norm(
            result["corrected_corners"]
            - observation.corners,
            axis=1,
        )

        mean_shift = float(
            np.mean(shifts)
        )

        max_shift = float(
            np.max(shifts)
        )

        corrected_count = sum(
            bool(item["used_correction"])
            for item in result["corner_details"]
        )

        cv2.putText(
            panel,
            (
                f"frame {frame_index} | "
                f"corrected {corrected_count}/4 | "
                f"mean shift={mean_shift:.2f}px | "
                f"max={max_shift:.2f}px"
            ),
            (
                block_x + 8,
                block_y + 22,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            "BEFORE",
            (
                block_x + 8,
                block_y + 47,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            "AFTER",
            (
                block_x + cell_width + 8,
                block_y + 47,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        image_y = (
            block_y + header_height
        )

        panel[
            image_y:
            image_y + image_height,
            block_x:
            block_x + cell_width,
        ] = before

        panel[
            image_y:
            image_y + image_height,
            block_x + cell_width:
            block_x + 2 * cell_width,
        ] = after

        cv2.rectangle(
            panel,
            (
                block_x,
                block_y,
            ),
            (
                block_x
                + frame_block_width
                - 1,
                block_y
                + frame_block_height
                - 1,
            ),
            (80, 80, 80),
            1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        panel,
    )

    if not success:
        raise RuntimeError(
            f"Could not save panel: {output_path}"
        )

    print(
        f"Saved before/after panel: "
        f"{output_path}"
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved CSV: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    total_start = time.perf_counter()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    print(
        "\n========== FAST 10-FRAME CORNER CORRECTION =========="
    )

    stage_start = time.perf_counter()

    sampled_frames = read_sampled_frames(
        VIDEO_PATH,
        start_frame=START_FRAME,
        num_sampled_frames=NUM_SAMPLED_FRAMES,
        frame_step=FRAME_STEP,
    )

    color_frames, gray_frames = (
        build_frame_maps(
            sampled_frames
        )
    )

    print(
        f"Frame loading/cache : "
        f"{time.perf_counter() - stage_start:.3f}s"
    )

    stage_start = time.perf_counter()

    all_observations = collect_observations(
        sampled_frames
    )

    print(
        f"Plate pipeline      : "
        f"{time.perf_counter() - stage_start:.3f}s"
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

    target_tracks = [
        track
        for track in long_tracks
        if track.track_id == TARGET_TRACK_ID
    ]

    if not target_tracks:
        raise RuntimeError(
            f"Track {TARGET_TRACK_ID} "
            "was not found among the long tracks."
        )

    track = target_tracks[0]

    observations = sorted(
        track.observations,
        key=lambda item: item.frame_index,
    )

    target_observations = observations[
        TARGET_START_POSITION_IN_TRACK:
        TARGET_START_POSITION_IN_TRACK
        + NUM_TARGET_OBSERVATIONS
    ]

    if len(target_observations) < NUM_TARGET_OBSERVATIONS:
        raise RuntimeError(
            "The selected track does not contain "
            f"{NUM_TARGET_OBSERVATIONS} observations "
            "from the requested start position."
        )

    print(
        f"Target track        : {TARGET_TRACK_ID}"
    )

    print(
        "Target frames       : "
        + ", ".join(
            str(observation.frame_index)
            for observation in target_observations
        )
    )

    stage_start = time.perf_counter()

    ranked_candidates = score_all_track_corners(
        track,
        color_frames=color_frames,
    )

    print(
        f"Corner scoring      : "
        f"{time.perf_counter() - stage_start:.3f}s"
    )

    stage_start = time.perf_counter()

    trajectory_tables = build_all_trajectory_tables(
        track,
        gray_frames=gray_frames,
    )

    print(
        f"All flow precompute : "
        f"{time.perf_counter() - stage_start:.3f}s"
    )

    stage_start = time.perf_counter()

    results: list[dict] = []
    csv_rows: list[dict] = []

    for observation in target_observations:
        corrected_corners = (
            observation.corners.copy()
        )

        corner_details: list[dict] = []

        for corner_index in range(4):
            (
                corrected_corner,
                details,
            ) = correct_one_corner_fast(
                target_observation=observation,
                corner_index=corner_index,
                ranked_candidates=(
                    ranked_candidates[
                        corner_index
                    ]
                ),
                trajectory_table=(
                    trajectory_tables[
                        corner_index
                    ]
                ),
            )

            corrected_corners[
                corner_index
            ] = corrected_corner

            details = dict(details)
            details["corner_index"] = (
                corner_index
            )

            corner_details.append(
                details
            )

            original_corner = (
                observation.corners[
                    corner_index
                ]
            )

            csv_rows.append(
                {
                    "track_id": TARGET_TRACK_ID,
                    "frame_index": observation.frame_index,
                    "corner_index": corner_index,
                    "original_x": float(
                        original_corner[0]
                    ),
                    "original_y": float(
                        original_corner[1]
                    ),
                    "corrected_x": float(
                        corrected_corner[0]
                    ),
                    "corrected_y": float(
                        corrected_corner[1]
                    ),
                    "shift_px": float(
                        np.linalg.norm(
                            corrected_corner
                            - original_corner
                        )
                    ),
                    "used_correction": details[
                        "used_correction"
                    ],
                    "reason": details["reason"],
                    "valid_anchor_count": details[
                        "valid_anchor_count"
                    ],
                    "consensus_spread_px": details[
                        "consensus_spread_px"
                    ],
                    "original_to_consensus_px": details[
                        "original_to_consensus_px"
                    ],
                }
            )

        results.append(
            {
                "observation": observation,
                "corrected_corners": corrected_corners,
                "corner_details": corner_details,
            }
        )

    print(
        f"Consensus/correction: "
        f"{time.perf_counter() - stage_start:.3f}s"
    )

    save_csv(
        csv_rows,
        CSV_PATH,
    )

    create_before_after_panel(
        results,
        color_frames=color_frames,
        output_path=PANEL_PATH,
    )

    print(
        "\n========== FRAME SUMMARY =========="
    )

    for result in results:
        observation = result["observation"]

        shifts = np.linalg.norm(
            result["corrected_corners"]
            - observation.corners,
            axis=1,
        )

        corrected_count = sum(
            bool(item["used_correction"])
            for item in result["corner_details"]
        )

        print(
            f"Frame {observation.frame_index:3d} | "
            f"corrected={corrected_count}/4 | "
            f"mean shift={float(np.mean(shifts)):.3f}px | "
            f"max shift={float(np.max(shifts)):.3f}px"
        )

    print(
        "===================================\n"
    )

    print(
        f"TOTAL runtime       : "
        f"{time.perf_counter() - total_start:.3f}s"
    )

    print(
        "\nCompare this output with the previous slow experiment:"
    )
    print(
        "- same TARGET_TRACK_ID"
    )
    print(
        "- same 10 target observations"
    )
    print(
        "- same top-K anchor selection"
    )
    print(
        "- same LK parameters and forward/backward threshold"
    )
    print(
        "- same consensus formula and spread rejection"
    )
    print(
        "- only the implementation is batched/cached"
    )

    print(
        f"\nPanel: {PANEL_PATH}"
    )
    print(
        f"CSV  : {CSV_PATH}"
    )

    print(
        "=========================================================\n"
    )


if __name__ == "__main__":
    main()
