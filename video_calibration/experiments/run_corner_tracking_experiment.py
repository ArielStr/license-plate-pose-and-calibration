from __future__ import annotations

from pathlib import Path
import csv
import os
import sys

import cv2
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

START_FRAME = 0
NUM_SAMPLED_FRAMES = 15
FRAME_STEP = 1
MIN_TRACK_LENGTH = 5

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

VIDEO_CALIBRATION_DIR = (
    PROJECT_ROOT
    / "video_calibration"
)

OUTPUT_DIR = (
    VIDEO_CALIBRATION_DIR
    / "outputs"
    / "corner_tracking"
)

TRACK_CSV_PATH = (
    OUTPUT_DIR
    / "track_observations.csv"
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
# OUTPUT
# ============================================================

def save_track_observations_csv(
    tracks,
    output_path: Path,
) -> None:
    rows: list[dict] = []

    for track in tracks:
        for observation in track.observations:
            (
                x_min,
                y_min,
                x_max,
                y_max,
            ) = observation.bbox

            row = {
                "track_id": track.track_id,
                "frame_index": observation.frame_index,
                "timestamp_sec": observation.timestamp_sec,
                "accepted": observation.accepted,
                "rejection_reason": (
                    observation.rejection_reason
                    or ""
                ),
                "quality_score": observation.quality_score,
                "bbox_x_min": float(x_min),
                "bbox_y_min": float(y_min),
                "bbox_x_max": float(x_max),
                "bbox_y_max": float(y_max),
            }

            for (
                corner_index,
                corner,
            ) in enumerate(
                observation.corners
            ):
                row[
                    f"corner_{corner_index}_x"
                ] = float(
                    corner[0]
                )

                row[
                    f"corner_{corner_index}_y"
                ] = float(
                    corner[1]
                )

            rows.append(
                row
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(
            "No track observations to save."
        )
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Saved tracking CSV: "
        f"{output_path}"
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
        "\n========== PLATE TRACKING DIAGNOSTIC =========="
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

    save_track_observations_csv(
        tracks,
        TRACK_CSV_PATH,
    )

    print(
        "\nCurrent goal:"
    )
    print(
        "Verify that each physical plate keeps the same "
        "track_id throughout the selected sequence."
    )
    print(
        "Once association is correct, we will add corner "
        "prediction / optical flow and temporal error metrics."
    )
    print(
        "=================================================\n"
    )


if __name__ == "__main__":
    main()
