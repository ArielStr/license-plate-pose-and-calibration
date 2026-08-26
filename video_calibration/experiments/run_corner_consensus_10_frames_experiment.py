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
FRAME_STEP = 5

MIN_TRACK_LENGTH = 10

# Controlled larger experiment.
TARGET_TRACK_ID = 3

# Ten consecutive observations from the selected track.
NUM_TARGET_OBSERVATIONS = 10
TARGET_START_POSITION_IN_TRACK = 0

# Number of high-quality anchors used independently for each corner.
TOP_K_ANCHORS = 5

# Minimum successful propagated anchors needed to use the consensus.
MIN_VALID_ANCHORS = 3

# If the anchors disagree too much, keep the original corner.
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

# Lucas-Kanade optical flow.
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
    / "corner_consensus_10_frames"
)

CSV_PATH = OUTPUT_DIR / "corner_corrections.csv"

PANEL_PATH = (
    OUTPUT_DIR
    / f"track_{TARGET_TRACK_ID:03d}_before_after_10_frames.jpg"
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


def open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    return capture


def read_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, image = capture.read()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    return image


# ============================================================
# CORNER QUALITY
# ============================================================

def score_all_track_corners(
    track,
    *,
    video_path: Path,
) -> dict[int, list[dict]]:
    """
    Returns one ranked candidate list per corner index.

    Each candidate is still the original single-frame refinement.
    """
    candidates_by_corner: dict[int, list[dict]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    capture = open_video(video_path)

    try:
        for observation in sorted(
            track.observations,
            key=lambda item: item.frame_index,
        ):
            image = read_frame(
                capture,
                observation.frame_index,
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

    finally:
        capture.release()

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
# OPTICAL FLOW
# ============================================================

def track_point_one_step(
    image_from: np.ndarray,
    image_to: np.ndarray,
    point_xy: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    gray_from = cv2.cvtColor(
        image_from,
        cv2.COLOR_BGR2GRAY,
    )

    gray_to = cv2.cvtColor(
        image_to,
        cv2.COLOR_BGR2GRAY,
    )

    point = np.asarray(
        point_xy,
        dtype=np.float32,
    ).reshape(1, 1, 2)

    forward_point, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_from,
        gray_to,
        point,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
    )

    if (
        forward_point is None
        or status is None
        or int(status[0, 0]) != 1
    ):
        return None, float("inf")

    backward_point, backward_status, _ = (
        cv2.calcOpticalFlowPyrLK(
            gray_to,
            gray_from,
            forward_point,
            None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA,
        )
    )

    if (
        backward_point is None
        or backward_status is None
        or int(backward_status[0, 0]) != 1
    ):
        return None, float("inf")

    original = point.reshape(2).astype(np.float64)
    returned = backward_point.reshape(2).astype(np.float64)

    fb_error = float(
        np.linalg.norm(returned - original)
    )

    return (
        forward_point.reshape(2).astype(np.float64),
        fb_error,
    )


def propagate_corner_between_frames(
    capture: cv2.VideoCapture,
    *,
    start_frame_index: int,
    target_frame_index: int,
    start_point_xy: np.ndarray,
) -> tuple[np.ndarray | None, float, int]:
    if start_frame_index == target_frame_index:
        return (
            np.asarray(
                start_point_xy,
                dtype=np.float64,
            ),
            0.0,
            0,
        )

    direction = (
        1
        if target_frame_index > start_frame_index
        else -1
    )

    current_frame_index = start_frame_index
    current_point = np.asarray(
        start_point_xy,
        dtype=np.float64,
    )

    max_fb_error = 0.0
    transitions = 0

    while current_frame_index != target_frame_index:
        next_frame_index = (
            current_frame_index + direction
        )

        current_image = read_frame(
            capture,
            current_frame_index,
        )

        next_image = read_frame(
            capture,
            next_frame_index,
        )

        tracked_point, fb_error = track_point_one_step(
            current_image,
            next_image,
            current_point,
        )

        if tracked_point is None:
            return (
                None,
                float("inf"),
                transitions,
            )

        max_fb_error = max(
            max_fb_error,
            fb_error,
        )

        if fb_error > MAX_FORWARD_BACKWARD_ERROR_PX:
            return (
                None,
                max_fb_error,
                transitions,
            )

        current_point = tracked_point
        current_frame_index = next_frame_index
        transitions += 1

    return (
        current_point,
        max_fb_error,
        transitions,
    )


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

    if not valid_rows:
        raise RuntimeError(
            "No valid propagated anchors."
        )

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
                + float(row["max_fb_error_px"])
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


def correct_one_corner(
    capture: cv2.VideoCapture,
    *,
    target_observation,
    corner_index: int,
    ranked_candidates: list[dict],
) -> tuple[np.ndarray, dict]:
    """
    Estimate one corrected corner using top-quality anchors from other frames.

    If the temporal evidence is weak, the original corner is preserved.
    """
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
        anchor_observation = (
            anchor_row["observation"]
        )

        start_point = (
            anchor_observation.corners[
                corner_index
            ]
        )

        (
            propagated_point,
            max_fb_error,
            transitions,
        ) = propagate_corner_between_frames(
            capture,
            start_frame_index=(
                anchor_observation.frame_index
            ),
            target_frame_index=(
                target_frame_index
            ),
            start_point_xy=start_point,
        )

        success = (
            propagated_point is not None
        )

        propagated_rows.append(
            {
                "anchor_rank": anchor_rank,
                "anchor_frame_index": (
                    anchor_observation.frame_index
                ),
                "anchor_quality": float(
                    anchor_row["quality_score"]
                ),
                "success": success,
                "point": propagated_point,
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
                "consensus": original_corner.copy(),
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
            original_corner - consensus
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
                "consensus": consensus,
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
            "consensus": consensus,
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
        point_tuple = (
            int(point[0]),
            int(point[1]),
        )

        cv2.circle(
            result,
            point_tuple,
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
    if image.size == 0:
        return np.zeros(
            (
                target_height,
                target_width,
                3,
            ),
            dtype=np.uint8,
        )

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
    video_path: Path,
    output_path: Path,
) -> None:
    capture = open_video(
        video_path
    )

    cell_width = 330
    image_height = 180
    header_height = 55

    # Each frame gets two columns: BEFORE | AFTER
    total_columns = 4
    frames_per_row = 2

    frame_block_width = (
        2 * cell_width
    )

    frame_block_height = (
        header_height
        + image_height
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
            f"10-frame temporal corner correction"
        ),
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    try:
        for result_index, result in enumerate(
            results
        ):
            grid_row = (
                result_index
                // frames_per_row
            )

            grid_column = (
                result_index
                % frames_per_row
            )

            block_x = (
                grid_column
                * frame_block_width
            )

            block_y = (
                title_height
                + grid_row
                * frame_block_height
            )

            observation = (
                result["observation"]
            )

            frame_index = (
                observation.frame_index
            )

            image = read_frame(
                capture,
                frame_index,
            )

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

            mean_shift = float(
                np.mean(
                    np.linalg.norm(
                        result[
                            "corrected_corners"
                        ]
                        - observation.corners,
                        axis=1,
                    )
                )
            )

            max_shift = float(
                np.max(
                    np.linalg.norm(
                        result[
                            "corrected_corners"
                        ]
                        - observation.corners,
                        axis=1,
                    )
                )
            )

            corrected_count = sum(
                bool(
                    item[
                        "used_correction"
                    ]
                )
                for item in result[
                    "corner_details"
                ]
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
                    block_x
                    + cell_width
                    + 8,
                    block_y + 47,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

            image_y = (
                block_y
                + header_height
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

    finally:
        capture.release()

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
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    print(
        "\n========== 10-FRAME CORNER CORRECTION EXPERIMENT =========="
    )

    sampled_frames = read_sampled_frames(
        VIDEO_PATH,
        start_frame=START_FRAME,
        num_sampled_frames=NUM_SAMPLED_FRAMES,
        frame_step=FRAME_STEP,
    )

    all_observations = collect_observations(
        sampled_frames
    )

    tracks = build_tracks(
        all_observations,
        config=TRACKER_CONFIG,
    )

    print_track_summary(tracks)

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

    ranked_candidates = score_all_track_corners(
        track,
        video_path=VIDEO_PATH,
    )

    capture = open_video(
        VIDEO_PATH
    )

    results: list[dict] = []
    csv_rows: list[dict] = []

    try:
        for observation in target_observations:
            corrected_corners = (
                observation.corners.copy()
            )

            corner_details: list[dict] = []

            for corner_index in range(4):
                (
                    corrected_corner,
                    details,
                ) = correct_one_corner(
                    capture,
                    target_observation=observation,
                    corner_index=corner_index,
                    ranked_candidates=(
                        ranked_candidates[
                            corner_index
                        ]
                    ),
                )

                corrected_corners[
                    corner_index
                ] = corrected_corner

                details = dict(details)
                details[
                    "corner_index"
                ] = corner_index

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
                        "used_correction": (
                            details[
                                "used_correction"
                            ]
                        ),
                        "reason": details[
                            "reason"
                        ],
                        "valid_anchor_count": (
                            details[
                                "valid_anchor_count"
                            ]
                        ),
                        "consensus_spread_px": (
                            details[
                                "consensus_spread_px"
                            ]
                        ),
                        "original_to_consensus_px": (
                            details[
                                "original_to_consensus_px"
                            ]
                        ),
                    }
                )

            results.append(
                {
                    "observation": observation,
                    "corrected_corners": corrected_corners,
                    "corner_details": corner_details,
                }
            )

    finally:
        capture.release()

    save_csv(
        csv_rows,
        CSV_PATH,
    )

    create_before_after_panel(
        results,
        video_path=VIDEO_PATH,
        output_path=PANEL_PATH,
    )

    print(
        "\n========== FRAME SUMMARY =========="
    )

    for result in results:
        observation = result[
            "observation"
        ]

        shifts = np.linalg.norm(
            result[
                "corrected_corners"
            ]
            - observation.corners,
            axis=1,
        )

        corrected_count = sum(
            bool(
                item[
                    "used_correction"
                ]
            )
            for item in result[
                "corner_details"
            ]
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
        "Important:"
    )
    print(
        "- This experiment now ACTUALLY changes the four corners."
    )
    print(
        "- A corner is replaced by temporal consensus only when "
        "at least MIN_VALID_ANCHORS survive optical-flow checks."
    )
    print(
        "- If consensus spread is larger than "
        f"{MAX_CONSENSUS_SPREAD_PX:.1f}px, the original corner is kept."
    )
    print(
        "- BEFORE = original robust_mask_lines quad."
    )
    print(
        "- AFTER  = quad after temporal multi-frame correction."
    )

    print(
        "\nOutputs:"
    )
    print(
        f"  Panel: {PANEL_PATH}"
    )
    print(
        f"  CSV  : {CSV_PATH}"
    )

    print(
        "===========================================================\n"
    )


if __name__ == "__main__":
    main()
