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


from video_calibration.corner_quality import (
    CORNER_NAMES,
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

START_FRAME = 0
NUM_SAMPLED_FRAMES = 20
FRAME_STEP = 1
MIN_TRACK_LENGTH = 5
PANEL_SIZE = 10

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
    max_frame_gap=max(
        3,
        FRAME_STEP + 1,
    ),
)

CORNER_QUALITY_CONFIG = CornerQualityConfig()

VIDEO_CALIBRATION_DIR = (
    PROJECT_ROOT
    / "video_calibration"
)

OUTPUT_DIR = (
    VIDEO_CALIBRATION_DIR
    / "outputs"
    / "corner_quality"
)

PANEL_DIR = (
    OUTPUT_DIR
    / "panels"
)

CSV_PATH = (
    OUTPUT_DIR
    / "corner_quality.csv"
)

BEST_CORNERS_CSV_PATH = (
    OUTPUT_DIR
    / "best_corners_per_track.csv"
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
                + sample_index
                * frame_step
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
# OBSERVATIONS / TRACKS
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
# IMAGE ACCESS
# ============================================================

def open_video(
    video_path: Path,
) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    return capture


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


# ============================================================
# CORNER QUALITY
# ============================================================

def evaluate_tracks(
    tracks,
    *,
    video_path: Path,
) -> list[dict]:
    rows: list[dict] = []

    capture = open_video(
        video_path
    )

    try:
        for track in tracks:
            observations = sorted(
                track.observations,
                key=lambda observation: (
                    observation.frame_index
                ),
            )

            for observation in observations:
                image = read_video_frame(
                    capture,
                    observation.frame_index,
                )

                if image is None:
                    print(
                        f"Warning: could not read "
                        f"frame {observation.frame_index} "
                        "for corner-quality scoring"
                    )
                    continue

                quality_results = compute_all_corner_qualities(
                    image,
                    observation.corners,
                    config=CORNER_QUALITY_CONFIG,
                )

                for result in quality_results:
                    corner = observation.corners[
                        result.corner_index
                    ]

                    rows.append(
                        {
                            "track_id": track.track_id,
                            "frame_index": observation.frame_index,
                            "timestamp_sec": observation.timestamp_sec,
                            "accepted": observation.accepted,
                            "rejection_reason": (
                                observation.rejection_reason
                                or ""
                            ),
                            "observation_quality_score": (
                                observation.quality_score
                            ),
                            "corner_index": result.corner_index,
                            "corner_name": result.corner_name,
                            "corner_x": float(
                                corner[0]
                            ),
                            "corner_y": float(
                                corner[1]
                            ),
                            "corner_quality_score": result.score,
                            "yellow_boundary_score": (
                                result.yellow_boundary_score
                            ),
                            "yellow_inside_score": (
                                result.yellow_inside_score
                            ),
                            "yellow_outside_score": (
                                result.yellow_outside_score
                            ),
                            "edge_gradient_score": (
                                result.edge_gradient_score
                            ),
                            "adjacent_edge_1_gradient_score": (
                                result.adjacent_edge_1_gradient_score
                            ),
                            "adjacent_edge_2_gradient_score": (
                                result.adjacent_edge_2_gradient_score
                            ),
                            "cornerness_score": (
                                result.cornerness_score
                            ),
                            "local_geometry_score": (
                                result.local_geometry_score
                            ),
                            "patch_radius_px": result.patch_radius_px,
                            "probe_distance_px": (
                                result.probe_distance_px
                            ),
                        }
                    )

    finally:
        capture.release()

    return rows


def rank_rows(
    rows: list[dict],
) -> list[dict]:
    grouped: dict[
        tuple[int, int],
        list[dict],
    ] = {}

    for row in rows:
        key = (
            int(
                row["track_id"]
            ),
            int(
                row["corner_index"]
            ),
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    ranked_rows: list[dict] = []

    for group_rows in grouped.values():
        sorted_rows = sorted(
            group_rows,
            key=lambda row: (
                -float(
                    row[
                        "corner_quality_score"
                    ]
                ),
                int(
                    row["frame_index"]
                ),
            ),
        )

        for rank, row in enumerate(
            sorted_rows,
            start=1,
        ):
            copied = dict(row)
            copied[
                "rank_within_track_corner"
            ] = rank
            copied[
                "num_candidates_in_track_corner"
            ] = len(
                sorted_rows
            )
            ranked_rows.append(
                copied
            )

    return ranked_rows


def build_best_corner_rows(
    ranked_rows: list[dict],
) -> list[dict]:
    best_rows = [
        row
        for row in ranked_rows
        if int(
            row[
                "rank_within_track_corner"
            ]
        )
        == 1
    ]

    return sorted(
        best_rows,
        key=lambda row: (
            int(
                row["track_id"]
            ),
            int(
                row["corner_index"]
            ),
        ),
    )


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

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

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
        writer.writerows(
            rows
        )

    print(
        f"Saved CSV: {output_path}"
    )


# ============================================================
# PANELS
# ============================================================

def draw_corner_zoom(
    image: np.ndarray,
    observation,
    corner_index: int,
    *,
    output_size: int = 300,
) -> np.ndarray:
    corners = np.asarray(
        observation.corners,
        dtype=np.float64,
    )

    corner = corners[
        corner_index
    ]

    right_height = float(
        np.linalg.norm(
            corners[2]
            - corners[1]
        )
    )

    left_height = float(
        np.linalg.norm(
            corners[3]
            - corners[0]
        )
    )

    plate_height = max(
        0.5
        * (
            left_height
            + right_height
        ),
        1.0,
    )

    radius = int(
        round(
            np.clip(
                1.60 * plate_height,
                28.0,
                85.0,
            )
        )
    )

    x = int(
        round(
            float(
                corner[0]
            )
        )
    )

    y = int(
        round(
            float(
                corner[1]
            )
        )
    )

    x_min = max(
        0,
        x - radius,
    )

    y_min = max(
        0,
        y - radius,
    )

    x_max = min(
        image.shape[1],
        x + radius + 1,
    )

    y_max = min(
        image.shape[0],
        y + radius + 1,
    )

    crop = image[
        y_min:y_max,
        x_min:x_max,
    ].copy()

    if crop.size == 0:
        return np.zeros(
            (
                output_size,
                output_size,
                3,
            ),
            dtype=np.uint8,
        )

    local_corners = corners.copy()
    local_corners[:, 0] -= x_min
    local_corners[:, 1] -= y_min

    previous_index = (
        corner_index - 1
    ) % 4

    next_index = (
        corner_index + 1
    ) % 4

    corner_point = tuple(
        np.round(
            local_corners[
                corner_index
            ]
        ).astype(
            int
        )
    )

    previous_point = tuple(
        np.round(
            local_corners[
                previous_index
            ]
        ).astype(
            int
        )
    )

    next_point = tuple(
        np.round(
            local_corners[
                next_index
            ]
        ).astype(
            int
        )
    )

    cv2.line(
        crop,
        corner_point,
        previous_point,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        crop,
        corner_point,
        next_point,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        crop,
        corner_point,
        5,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        crop,
        corner_point,
        9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    height, width = crop.shape[:2]

    scale = min(
        output_size / width,
        output_size / height,
    )

    resized_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                height * scale
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
            output_size,
            output_size,
            3,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        output_size
        - resized_width
    ) // 2

    y_offset = (
        output_size
        - resized_height
    ) // 2

    canvas[
        y_offset:
        y_offset + resized_height,
        x_offset:
        x_offset + resized_width,
    ] = resized

    return canvas


def create_ranked_corner_panels(
    tracks,
    ranked_rows: list[dict],
    *,
    video_path: Path,
    output_dir: Path,
    panel_size: int,
) -> None:
    observation_by_track_frame = {}

    for track in tracks:
        for observation in track.observations:
            observation_by_track_frame[
                (
                    track.track_id,
                    observation.frame_index,
                )
            ] = observation

    grouped: dict[
        tuple[int, int],
        list[dict],
    ] = {}

    for row in ranked_rows:
        key = (
            int(
                row["track_id"]
            ),
            int(
                row["corner_index"]
            ),
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    capture = open_video(
        video_path
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cell_width = 340
    cell_height = 405
    image_size = 300
    header_height = (
        cell_height
        - image_size
    )

    try:
        for (
            track_id,
            corner_index,
        ), group_rows in grouped.items():
            sorted_rows = sorted(
                group_rows,
                key=lambda row: int(
                    row[
                        "rank_within_track_corner"
                    ]
                ),
            )

            corner_name = CORNER_NAMES[
                corner_index
            ]

            corner_output_dir = (
                output_dir
                / f"track_{track_id:03d}"
                / corner_name
            )

            corner_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for panel_start in range(
                0,
                len(
                    sorted_rows
                ),
                panel_size,
            ):
                panel_rows = sorted_rows[
                    panel_start:
                    panel_start
                    + panel_size
                ]

                columns = min(
                    5,
                    len(
                        panel_rows
                    ),
                )

                num_rows = int(
                    math.ceil(
                        len(
                            panel_rows
                        )
                        / columns
                    )
                )

                title_height = 58

                panel = np.zeros(
                    (
                        title_height
                        + num_rows
                        * cell_height,
                        columns
                        * cell_width,
                        3,
                    ),
                    dtype=np.uint8,
                )

                first_rank = int(
                    panel_rows[0][
                        "rank_within_track_corner"
                    ]
                )

                last_rank = int(
                    panel_rows[-1][
                        "rank_within_track_corner"
                    ]
                )

                cv2.putText(
                    panel,
                    (
                        f"Track {track_id} | "
                        f"{corner_name} | "
                        f"best -> worst | "
                        f"ranks {first_rank}-{last_rank}"
                    ),
                    (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                for local_index, row in enumerate(
                    panel_rows
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

                    frame_index = int(
                        row[
                            "frame_index"
                        ]
                    )

                    observation = (
                        observation_by_track_frame[
                            (
                                track_id,
                                frame_index,
                            )
                        ]
                    )

                    image = read_video_frame(
                        capture,
                        frame_index,
                    )

                    if image is None:
                        continue

                    zoom = draw_corner_zoom(
                        image,
                        observation,
                        corner_index,
                        output_size=image_size,
                    )

                    rank = int(
                        row[
                            "rank_within_track_corner"
                        ]
                    )

                    score = float(
                        row[
                            "corner_quality_score"
                        ]
                    )

                    yellow = float(
                        row[
                            "yellow_boundary_score"
                        ]
                    )

                    gradient = float(
                        row[
                            "edge_gradient_score"
                        ]
                    )

                    cornerness = float(
                        row[
                            "cornerness_score"
                        ]
                    )

                    status = (
                        "A"
                        if bool(
                            row[
                                "accepted"
                            ]
                        )
                        else "R"
                    )

                    cv2.putText(
                        panel,
                        (
                            f"rank {rank} | "
                            f"frame {frame_index} | "
                            f"{status}"
                        ),
                        (
                            cell_x + 9,
                            cell_y + 23,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        panel,
                        (
                            f"Q={score:.3f} | "
                            f"Y={yellow:.3f}"
                        ),
                        (
                            cell_x + 9,
                            cell_y + 48,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.49,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        panel,
                        (
                            f"G={gradient:.3f} | "
                            f"C={cornerness:.3f}"
                        ),
                        (
                            cell_x + 9,
                            cell_y + 72,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.49,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )

                    image_x = (
                        cell_x
                        + (
                            cell_width
                            - image_size
                        ) // 2
                    )

                    image_y = (
                        cell_y
                        + header_height
                    )

                    panel[
                        image_y:
                        image_y + image_size,
                        image_x:
                        image_x + image_size,
                    ] = zoom

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
                        (85, 85, 85),
                        1,
                    )

                page_index = (
                    panel_start
                    // panel_size
                    + 1
                )

                output_path = (
                    corner_output_dir
                    / (
                        f"page_{page_index:02d}"
                        f"_ranked.jpg"
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
                        f"Could not save panel: {output_path}"
                    )

                print(
                    f"Saved corner-quality panel: {output_path}"
                )

    finally:
        capture.release()


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_best_corners(
    best_rows: list[dict],
) -> None:
    print(
        "\n========== BEST CORNER CANDIDATES =========="
    )

    for row in best_rows:
        print(
            f"Track {int(row['track_id']):3d} | "
            f"{row['corner_name']:12s} | "
            f"frame={int(row['frame_index']):3d} | "
            f"Q={float(row['corner_quality_score']):.3f} | "
            f"yellow={float(row['yellow_boundary_score']):.3f} | "
            f"gradient={float(row['edge_gradient_score']):.3f} | "
            f"cornerness={float(row['cornerness_score']):.3f}"
        )

    print(
        "============================================\n"
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
        "\n========== CORNER QUALITY EXPERIMENT =========="
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
        f"Minimum track len  : {MIN_TRACK_LENGTH}"
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

    rows = evaluate_tracks(
        long_tracks,
        video_path=VIDEO_PATH,
    )

    ranked_rows = rank_rows(
        rows
    )

    best_rows = build_best_corner_rows(
        ranked_rows
    )

    save_rows_csv(
        ranked_rows,
        CSV_PATH,
    )

    save_rows_csv(
        best_rows,
        BEST_CORNERS_CSV_PATH,
    )

    create_ranked_corner_panels(
        long_tracks,
        ranked_rows,
        video_path=VIDEO_PATH,
        output_dir=PANEL_DIR,
        panel_size=PANEL_SIZE,
    )

    print_best_corners(
        best_rows
    )

    print(
        "Interpretation:"
    )
    print(
        "- The experiment does NOT correct corners."
    )
    print(
        "- It ranks each corner type independently inside each track."
    )
    print(
        "- Q = total local corner-quality score."
    )
    print(
        "- Y = yellow-inside / non-yellow-outside boundary evidence."
    )
    print(
        "- G = gradient support on both adjacent refined edges."
    )
    print(
        "- C = local Shi-Tomasi cornerness."
    )
    print(
        "- First validation is visual: do the top-ranked corners "
        "really look better than the bottom-ranked ones?"
    )

    print(
        "\nOutputs:"
    )
    print(
        f"  Ranked CSV : {CSV_PATH}"
    )
    print(
        f"  Best CSV   : {BEST_CORNERS_CSV_PATH}"
    )
    print(
        f"  Panels     : {PANEL_DIR}"
    )
    print(
        "=================================================\n"
    )


if __name__ == "__main__":
    main()
