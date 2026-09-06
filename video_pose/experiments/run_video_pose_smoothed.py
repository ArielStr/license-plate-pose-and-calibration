from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from video_pose.camera_model import load_camera_model
from video_pose.frame_processor import process_sampled_frame
from video_pose.frame_sampler import read_video_metadata, sample_video_frames
from video_pose.filtering import get_accepted_detections
from video_pose.pose_estimator import PlatePoseEstimator, estimate_frame_results
from video_pose.temporal.adaptive_pose_filter import (
    AdaptivePoseFilterConfig,
    AdaptiveTemporalPoseFilter,
)
from video_pose.tracking import PlateTrackerConfig, associate_frame_detections
from video_pose.visualization.overlay import draw_frame_results
from video_pose.visualization.zoom_panel import ZoomPanelConfig, ZoomPanelRenderer


API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set.")

VIDEO_PATH = PROJECT_ROOT / "cars_photos" / "test_video.mov"
CALIBRATION_PATH = PROJECT_ROOT / "calibration" / "calibration_results_video.npz"

OUTPUT_DIR = PROJECT_ROOT / "video_pose" / "outputs" / "temporal_pose_v1"
OUTPUT_PATH = OUTPUT_DIR / "test_video_pose_temporal_v1.mp4"
ZOOM_PANEL_OUTPUT_PATH = OUTPUT_DIR / "test_video_pose_zoom_panel_temporal_v1.mp4"
CSV_PATH = OUTPUT_DIR / "temporal_pose_v1.csv"

PROCESS_EVERY_N_FRAMES = 1
USE_DISTORTION = True
USE_CACHED_ROBOFLOW_JSON = True

MIN_CONFIDENCE = 0.35
MIN_DETECTION_AREA = 300.0
MIN_WIDTH = 80.0
MIN_HEIGHT = 20.0
NMS_IOU_THRESHOLD = 0.50

REFINEMENT_METHOD = "robust_mask_lines"
REFINEMENT_DEBUG = False
REFINEMENT_PANEL_DEBUG = False

MAX_FRAMES = None
OUTPUT_FPS_OVERRIDE = None

TEMPORAL_CONFIG = AdaptivePoseFilterConfig(
    history_size=5,
    alpha_max=0.45,
    alpha_min=0.05,
    deviation_strength=0.70,
    distance_mad_floor_m=0.03,
    yaw_mad_floor_deg=0.75,
    min_history_for_adaptive_weight=3,
)

ZOOM_PANEL_CONFIG = ZoomPanelConfig(
    max_tracks=6,
    columns=3,
    cell_width=640,
    cell_height=360,
    header_height=70,
)


def create_video_writer(
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")
    return writer


def main() -> None:
    metadata = read_video_metadata(VIDEO_PATH)

    print("\n========== TEMPORAL POSE V1 | ADAPTIVE EMA ==========")
    print(f"Video       : {VIDEO_PATH}")
    print(f"Resolution  : {metadata.width}x{metadata.height}")
    print(f"FPS         : {metadata.fps:.3f}")
    print(f"Frames      : {metadata.total_frames}")
    print(f"History     : {TEMPORAL_CONFIG.history_size}")
    print(f"Alpha range : {TEMPORAL_CONFIG.alpha_min:.2f} -> {TEMPORAL_CONFIG.alpha_max:.2f}")
    print("======================================================\n")

    camera_model = load_camera_model(
        CALIBRATION_PATH,
        use_distortion=USE_DISTORTION,
    )
    pose_estimator = PlatePoseEstimator(camera_model)
    temporal_filter = AdaptiveTemporalPoseFilter(TEMPORAL_CONFIG)

    source_timeline_fps = metadata.fps / PROCESS_EVERY_N_FRAMES
    output_fps = (
        OUTPUT_FPS_OVERRIDE
        if OUTPUT_FPS_OVERRIDE is not None
        else source_timeline_fps
    )

    main_writer = create_video_writer(
        OUTPUT_PATH,
        width=metadata.width,
        height=metadata.height,
        fps=output_fps,
    )

    zoom_renderer = ZoomPanelRenderer(ZOOM_PANEL_CONFIG)
    zoom_writer = create_video_writer(
        ZOOM_PANEL_OUTPUT_PATH,
        width=zoom_renderer.width,
        height=zoom_renderer.height,
        fps=output_fps,
    )

    tracks = []
    next_track_id = 0
    tracker_config = PlateTrackerConfig(
        max_frame_gap=3 * PROCESS_EVERY_N_FRAMES,
    )

    csv_rows: list[dict[str, object]] = []
    processed_count = 0
    total_pose_results = 0

    try:
        frames = sample_video_frames(
            VIDEO_PATH,
            every_n_frames=PROCESS_EVERY_N_FRAMES,
            max_sampled_frames=MAX_FRAMES,
        )

        for sampled_frame in frames:
            print("\n------------------------------------")
            print(f"Processing frame {sampled_frame.frame_index}")

            detections = process_sampled_frame(
                sampled_frame,
                API_KEY,
                video_name=VIDEO_PATH.name,
                use_cached_json=USE_CACHED_ROBOFLOW_JSON,
                min_confidence=MIN_CONFIDENCE,
                min_detection_area=MIN_DETECTION_AREA,
                min_width=MIN_WIDTH,
                min_height=MIN_HEIGHT,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
                refinement_method=REFINEMENT_METHOD,
                refinement_debug=REFINEMENT_DEBUG,
                refinement_panel_debug=REFINEMENT_PANEL_DEBUG,
            )

            accepted_detections = get_accepted_detections(detections)

            tracks, next_track_id = associate_frame_detections(
                accepted_detections,
                tracks,
                next_track_id=next_track_id,
                config=tracker_config,
            )

            # Normal per-frame PnP. Corners are untouched.
            raw_results = estimate_frame_results(
                accepted_detections,
                pose_estimator,
            )

            # V1: only distance/yaw are temporally stabilized.
            frame_results = temporal_filter.filter_frame_results(raw_results)

            for debug in temporal_filter.last_debug:
                csv_rows.append(
                    {
                        "frame_index": sampled_frame.frame_index,
                        "track_id": debug.track_id,
                        "raw_distance_m": debug.raw_distance_m,
                        "filtered_distance_m": debug.filtered_distance_m,
                        "raw_yaw_deg": debug.raw_yaw_deg,
                        "filtered_yaw_deg": debug.filtered_yaw_deg,
                        "alpha_distance": debug.alpha_distance,
                        "alpha_yaw": debug.alpha_yaw,
                        "distance_deviation": debug.distance_deviation,
                        "yaw_deviation": debug.yaw_deviation,
                    }
                )
                print(
                    f"Track {debug.track_id}: "
                    f"D {debug.raw_distance_m:.3f}->{debug.filtered_distance_m:.3f} m "
                    f"(a={debug.alpha_distance:.3f}) | "
                    f"Yaw {debug.raw_yaw_deg:+.2f}->{debug.filtered_yaw_deg:+.2f} deg "
                    f"(a={debug.alpha_yaw:.3f})"
                )

            total_pose_results += len(frame_results)

            annotated_frame = draw_frame_results(
                sampled_frame.image,
                frame_results,
                frame_index=sampled_frame.frame_index,
            )
            main_writer.write(annotated_frame)

            timestamp_sec = (
                sampled_frame.frame_index / metadata.fps
                if metadata.fps > 0
                else 0.0
            )

            zoom_panel_frame = zoom_renderer.render(
                annotated_frame,
                frame_results,
                timestamp_sec=timestamp_sec,
                duration_sec=metadata.duration_sec,
            )
            zoom_writer.write(zoom_panel_frame)

            processed_count += 1

    finally:
        main_writer.release()
        zoom_writer.release()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "track_id",
        "raw_distance_m",
        "filtered_distance_m",
        "raw_yaw_deg",
        "filtered_yaw_deg",
        "alpha_distance",
        "alpha_yaw",
        "distance_deviation",
        "yaw_deviation",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========== TEMPORAL POSE V1 COMPLETE ==========")
    print(f"Processed frames : {processed_count}")
    print(f"Pose results     : {total_pose_results}")
    print(f"Tracks created   : {next_track_id}")
    print(f"Main video       : {OUTPUT_PATH}")
    print(f"Zoom panel       : {ZOOM_PANEL_OUTPUT_PATH}")
    print(f"Debug CSV        : {CSV_PATH}")
    print("================================================\n")


if __name__ == "__main__":
    main()
