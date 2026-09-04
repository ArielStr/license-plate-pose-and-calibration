from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from video_pose.camera_model import load_camera_model
from video_pose.frame_processor import process_sampled_frame
from video_pose.frame_sampler import read_video_metadata, sample_video_frames
from video_pose.filtering import get_accepted_detections
from video_pose.models import PlateDetection
from video_pose.pose_estimator import PlatePoseEstimator, estimate_frame_results
from video_pose.temporal.adaptive_pose_filter import (
    AdaptivePoseFilterConfig,
    AdaptiveTemporalPoseFilter,
)
from video_pose.tracking import PlateTrackerConfig, associate_frame_detections
from video_pose.visualization.overlay import draw_frame_results
from video_pose.visualization.zoom_panel import ZoomPanelConfig, ZoomPanelRenderer

from video_pose.temporal_corner_refinement.window import SlidingTrackWindowBuilder
from video_pose.temporal_corner_refinement.alignment import warp_quad
from video_pose.temporal_corner_refinement.methods.geometry_alignment import (
    run_geometry_baseline,
)
from video_pose.temporal_corner_refinement.methods.corner_color import (
    CornerColorConfig,
    run_corner_color,
)


API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set.")

CALIBRATION_PATH = PROJECT_ROOT / "calibration" / "calibration_results_video.npz"

VIDEO_PATH = PROJECT_ROOT / "cars_photos" / "test_video2.mov"

VIDEO_STEM = VIDEO_PATH.stem

OUTPUT_DIR = (
    PROJECT_ROOT
    / "video_pose"
    / "outputs"
    / "temporal_corner_refined_pose"
    / VIDEO_STEM
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / f"{VIDEO_STEM}_pose_temporal_corner_refined.mp4"
)

ZOOM_PANEL_OUTPUT_PATH = (
    OUTPUT_DIR
    / f"{VIDEO_STEM}_zoom_panel_temporal_corner_refined.mp4"
)

CSV_PATH = (
    OUTPUT_DIR
    / f"{VIDEO_STEM}_pose.csv"
)

CORNER_DEBUG_CSV_PATH = (
    OUTPUT_DIR
    / f"{VIDEO_STEM}_corner_debug.csv"
)


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


# ---------------------------------------------------------------------------
# Same pose smoothing config as the previous Adaptive EMA baseline.
# ---------------------------------------------------------------------------
TEMPORAL_POSE_CONFIG = AdaptivePoseFilterConfig(
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


# ---------------------------------------------------------------------------
# Latest tested temporal-corner method:
#
# A full 7-observation sliding window is used:
#
#     obs0 obs1 obs2 [obs3] obs4 obs5 obs6
#                       ^
#                    only this
#                    center observation
#                    receives refined corners
#
# For each valid window:
#   1) align the seven RAW quads into the center/reference frame
#   2) run the latest CORNER-wise color method
#   3) allow TL/TR/BR/BL to come from different source frames
#   4) use the mixed refined quad only for the CENTER frame
#
# If the temporal method cannot be applied, keep the ordinary
# robust_mask_lines corners for that frame.
# ---------------------------------------------------------------------------
CORNER_WINDOW_SIZE = 7
CORNER_CONFIG = CornerColorConfig()

# Same safety rule used in the experiment: do not temporally refine a window
# if any observation is touching/leaving the image border.
WINDOW_BORDER_MARGIN_PX = 4.0


def create_video_writer(
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    return writer


def clone_detection_with_corners(
    detection: PlateDetection,
    corners: np.ndarray,
) -> PlateDetection:
    """
    Preserve all baseline detection / tracking metadata and replace only the
    corners used by PnP.
    """
    return PlateDetection(
        frame_index=int(detection.frame_index),
        timestamp_sec=float(detection.timestamp_sec),
        bbox=np.asarray(detection.bbox, dtype=np.float64).copy(),
        corners=np.asarray(corners, dtype=np.float64).copy(),
        detection_confidence=float(detection.detection_confidence),
        quality_score=float(detection.quality_score),
        accepted=bool(detection.accepted),
        rejection_reason=detection.rejection_reason,
        track_id=(
            int(detection.track_id)
            if detection.track_id is not None
            else None
        ),
    )


def observation_has_full_plate_visibility(obs) -> bool:
    q = np.asarray(obs.detection.corners, dtype=np.float64)

    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False

    h, w = obs.image.shape[:2]

    min_x = float(np.min(q[:, 0]))
    max_x = float(np.max(q[:, 0]))
    min_y = float(np.min(q[:, 1]))
    max_y = float(np.max(q[:, 1]))

    return bool(
        min_x >= WINDOW_BORDER_MARGIN_PX
        and max_x <= (w - 1 - WINDOW_BORDER_MARGIN_PX)
        and min_y >= WINDOW_BORDER_MARGIN_PX
        and max_y <= (h - 1 - WINDOW_BORDER_MARGIN_PX)
    )


def window_has_full_plate_visibility(window) -> bool:
    return all(
        observation_has_full_plate_visibility(obs)
        for obs in window.observations
    )


def build_temporal_corner_overrides(
    metadata,
) -> tuple[
    dict[int, list[PlateDetection]],
    dict[tuple[int, int], np.ndarray],
    list[dict[str, object]],
]:
    """
    PASS 1

    Run the ordinary image-space pipeline and tracking once.

    For every Track, SlidingTrackWindowBuilder emits an overlapping
    7-observation window as soon as it becomes available. Only the CENTER
    observation gets a corner-wise temporal override.

    Returns:
      detections_by_frame:
          original tracked accepted detections used again in PASS 2.

      corner_override_by_key:
          (frame_index, track_id) -> temporal corners for that frame.

      debug_rows:
          diagnostics describing whether each center observation was refined.
    """
    tracks = []
    next_track_id = 0

    tracker_config = PlateTrackerConfig(
        max_frame_gap=3 * PROCESS_EVERY_N_FRAMES,
    )

    window_builder = SlidingTrackWindowBuilder(
        CORNER_WINDOW_SIZE
    )

    detections_by_frame: dict[int, list[PlateDetection]] = {}
    corner_override_by_key: dict[
        tuple[int, int],
        np.ndarray,
    ] = {}

    debug_rows: list[dict[str, object]] = []

    frames = sample_video_frames(
        VIDEO_PATH,
        every_n_frames=PROCESS_EVERY_N_FRAMES,
        max_sampled_frames=MAX_FRAMES,
    )

    for sampled_frame in frames:
        print("\n------------------------------------")
        print(
            f"PASS 1 | Processing frame "
            f"{sampled_frame.frame_index}"
        )

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

        accepted = get_accepted_detections(detections)

        tracks, next_track_id = associate_frame_detections(
            accepted,
            tracks,
            next_track_id=next_track_id,
            config=tracker_config,
        )

        # Save lightweight detection copies only. Images remain only in the
        # per-track 7-observation window builder.
        detections_by_frame[
            sampled_frame.frame_index
        ] = [
            clone_detection_with_corners(
                detection,
                detection.corners,
            )
            for detection in accepted
        ]

        for detection in accepted:
            window = window_builder.add(
                image=sampled_frame.image,
                detection=detection,
            )

            if window is None:
                continue

            center_obs = window.center
            center_track_id = int(window.track_id)
            center_key = (
                int(center_obs.frame_index),
                center_track_id,
            )

            if not window_has_full_plate_visibility(window):
                debug_rows.append(
                    {
                        "frame_index": center_obs.frame_index,
                        "track_id": center_track_id,
                        "window_start_frame": window.start_frame,
                        "window_end_frame": window.end_frame,
                        "temporal_applied": 0,
                        "reason": "partial_visibility_in_window",
                        "corner_tl_source_frame": None,
                        "corner_tr_source_frame": None,
                        "corner_br_source_frame": None,
                        "corner_bl_source_frame": None,
                        "corner_tl_score": None,
                        "corner_tr_score": None,
                        "corner_br_score": None,
                        "corner_bl_score": None,
                        "corner_joint_score": None,
                        "corner_mixed_source_count": None,
                    }
                )
                continue

            geometry = run_geometry_baseline(window)

            if not geometry.consensus.success:
                debug_rows.append(
                    {
                        "frame_index": center_obs.frame_index,
                        "track_id": center_track_id,
                        "window_start_frame": window.start_frame,
                        "window_end_frame": window.end_frame,
                        "temporal_applied": 0,
                        "reason": (
                            "alignment_or_geometry_failed:"
                            + geometry.consensus.reason
                        ),
                        "corner_tl_source_frame": None,
                        "corner_tr_source_frame": None,
                        "corner_br_source_frame": None,
                        "corner_bl_source_frame": None,
                        "corner_tl_score": None,
                        "corner_tr_score": None,
                        "corner_br_score": None,
                        "corner_bl_score": None,
                        "corner_joint_score": None,
                        "corner_mixed_source_count": None,
                    }
                )
                continue

            corner_result = run_corner_color(
                window,
                geometry.H_to_reference,
                config=CORNER_CONFIG,
            )

            if not corner_result.success:
                debug_rows.append(
                    {
                        "frame_index": center_obs.frame_index,
                        "track_id": center_track_id,
                        "window_start_frame": window.start_frame,
                        "window_end_frame": window.end_frame,
                        "temporal_applied": 0,
                        "reason": "corner_color_failed",
                        "corner_tl_source_frame": None,
                        "corner_tr_source_frame": None,
                        "corner_br_source_frame": None,
                        "corner_bl_source_frame": None,
                        "corner_tl_score": None,
                        "corner_tr_score": None,
                        "corner_br_score": None,
                        "corner_bl_score": None,
                        "corner_joint_score": None,
                        "corner_mixed_source_count": None,
                    }
                )
                continue

            # corner_quad_in_reference is already in the CENTER/reference
            # frame coordinate system. This is exactly the quad PnP should
            # receive for the center observation.
            refined_center_corners = np.asarray(
                corner_result.corner_quad_in_reference,
                dtype=np.float64,
            ).copy()

            if (
                refined_center_corners.shape != (4, 2)
                or not np.all(np.isfinite(refined_center_corners))
            ):
                debug_rows.append(
                    {
                        "frame_index": center_obs.frame_index,
                        "track_id": center_track_id,
                        "window_start_frame": window.start_frame,
                        "window_end_frame": window.end_frame,
                        "temporal_applied": 0,
                        "reason": "invalid_corner_quad",
                        "corner_tl_source_frame": None,
                        "corner_tr_source_frame": None,
                        "corner_br_source_frame": None,
                        "corner_bl_source_frame": None,
                        "corner_tl_score": None,
                        "corner_tr_score": None,
                        "corner_br_score": None,
                        "corner_bl_score": None,
                        "corner_joint_score": None,
                        "corner_mixed_source_count": None,
                    }
                )
                continue

            corner_override_by_key[center_key] = refined_center_corners

            source_frames = tuple(
                int(x) for x in corner_result.corner_source_frames
            )
            corner_scores = tuple(
                float(x) for x in corner_result.corner_scores
            )

            debug_rows.append(
                {
                    "frame_index": center_obs.frame_index,
                    "track_id": center_track_id,
                    "window_start_frame": window.start_frame,
                    "window_end_frame": window.end_frame,
                    "temporal_applied": 1,
                    "reason": "ok",
                    "corner_tl_source_frame": source_frames[0],
                    "corner_tr_source_frame": source_frames[1],
                    "corner_br_source_frame": source_frames[2],
                    "corner_bl_source_frame": source_frames[3],
                    "corner_tl_score": corner_scores[0],
                    "corner_tr_score": corner_scores[1],
                    "corner_br_score": corner_scores[2],
                    "corner_bl_score": corner_scores[3],
                    "corner_joint_score": float(corner_result.joint_score),
                    "corner_mixed_source_count": int(
                        corner_result.mixed_source_count
                    ),
                }
            )

            print(
                f"  Track {center_track_id} | "
                f"center F{center_obs.frame_index} | "
                f"CORNER sources "
                f"TL={source_frames[0]} "
                f"TR={source_frames[1]} "
                f"BR={source_frames[2]} "
                f"BL={source_frames[3]} | "
                f"joint={corner_result.joint_score:.4f}"
            )

    print("\n========== PASS 1 COMPLETE ==========")
    print(f"Tracks created        : {next_track_id}")
    print(
        f"Temporal center quads : "
        f"{len(corner_override_by_key)}"
    )
    print("=====================================\n")

    return (
        detections_by_frame,
        corner_override_by_key,
        debug_rows,
    )


def main() -> None:
    metadata = read_video_metadata(VIDEO_PATH)

    print("\n======================================================")
    print(" TEMPORAL CORNER-WISE REFINEMENT -> PnP -> ADAPTIVE EMA")
    print("======================================================")
    print(f"Video        : {VIDEO_PATH}")
    print(f"Resolution   : {metadata.width}x{metadata.height}")
    print(f"FPS          : {metadata.fps:.3f}")
    print(f"Frames       : {metadata.total_frames}")
    print(f"Corner window: {CORNER_WINDOW_SIZE} observations")
    print(
        "Corner rule  : tested CORNER-wise color selection on overlapping "
        "7-observation windows; refine CENTER only"
    )
    print(
        "Fallback     : RAW robust_mask_lines corners when no valid "
        "temporal window/result"
    )
    print(
        "Pose filter  : identical Adaptive EMA config to previous baseline"
    )
    print("======================================================\n")

    camera_model = load_camera_model(
        CALIBRATION_PATH,
        use_distortion=USE_DISTORTION,
    )
    pose_estimator = PlatePoseEstimator(camera_model)

    # PASS 1: detect / track / build centered temporal corner overrides.
    (
        detections_by_frame,
        corner_override_by_key,
        corner_debug_rows,
    ) = build_temporal_corner_overrides(metadata)

    # PASS 2: identical product output path as the previous pose baseline,
    # except PnP sees temporal corners when an override exists.
    temporal_filter = AdaptiveTemporalPoseFilter(
        TEMPORAL_POSE_CONFIG
    )

    source_timeline_fps = (
        metadata.fps / PROCESS_EVERY_N_FRAMES
    )
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

    zoom_renderer = ZoomPanelRenderer(
        ZOOM_PANEL_CONFIG
    )

    zoom_writer = create_video_writer(
        ZOOM_PANEL_OUTPUT_PATH,
        width=zoom_renderer.width,
        height=zoom_renderer.height,
        fps=output_fps,
    )

    csv_rows: list[dict[str, object]] = []

    processed_count = 0
    total_pose_results = 0
    total_temporal_detections = 0
    total_raw_fallback_detections = 0

    try:
        frames = sample_video_frames(
            VIDEO_PATH,
            every_n_frames=PROCESS_EVERY_N_FRAMES,
            max_sampled_frames=MAX_FRAMES,
        )

        for sampled_frame in frames:
            print("\n------------------------------------")
            print(
                f"PASS 2 | Rendering frame "
                f"{sampled_frame.frame_index}"
            )

            baseline_detections = detections_by_frame.get(
                sampled_frame.frame_index,
                [],
            )

            pose_detections: list[PlateDetection] = []

            for detection in baseline_detections:
                if detection.track_id is None:
                    continue

                key = (
                    int(detection.frame_index),
                    int(detection.track_id),
                )

                temporal_corners = (
                    corner_override_by_key.get(key)
                )

                if temporal_corners is None:
                    total_raw_fallback_detections += 1
                    pose_detection = (
                        clone_detection_with_corners(
                            detection,
                            detection.corners,
                        )
                    )
                else:
                    total_temporal_detections += 1
                    pose_detection = (
                        clone_detection_with_corners(
                            detection,
                            temporal_corners,
                        )
                    )

                pose_detections.append(
                    pose_detection
                )

            # Same meaning as "raw_results" in the previous Adaptive EMA
            # baseline: pose before EMA. The only difference is that PnP now
            # receives temporal corners whenever available.
            raw_results = estimate_frame_results(
                pose_detections,
                pose_estimator,
            )

            # EXACT SAME pose-domain temporal filter as baseline.
            frame_results = temporal_filter.filter_frame_results(
                raw_results
            )

            for debug in temporal_filter.last_debug:
                csv_rows.append(
                    {
                        "frame_index": sampled_frame.frame_index,
                        "track_id": debug.track_id,
                        "raw_distance_m": debug.raw_distance_m,
                        "filtered_distance_m": (
                            debug.filtered_distance_m
                        ),
                        "raw_yaw_deg": debug.raw_yaw_deg,
                        "filtered_yaw_deg": (
                            debug.filtered_yaw_deg
                        ),
                        "alpha_distance": debug.alpha_distance,
                        "alpha_yaw": debug.alpha_yaw,
                        "distance_deviation": (
                            debug.distance_deviation
                        ),
                        "yaw_deviation": (
                            debug.yaw_deviation
                        ),
                    }
                )

                print(
                    f"Track {debug.track_id}: "
                    f"D {debug.raw_distance_m:.3f}"
                    f"->{debug.filtered_distance_m:.3f} m "
                    f"(a={debug.alpha_distance:.3f}) | "
                    f"Yaw {debug.raw_yaw_deg:+.2f}"
                    f"->{debug.filtered_yaw_deg:+.2f} deg "
                    f"(a={debug.alpha_yaw:.3f})"
                )

            total_pose_results += len(
                frame_results
            )

            annotated_frame = draw_frame_results(
                sampled_frame.image,
                frame_results,
                frame_index=sampled_frame.frame_index,
            )
            main_writer.write(
                annotated_frame
            )

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
            zoom_writer.write(
                zoom_panel_frame
            )

            processed_count += 1

    finally:
        main_writer.release()
        zoom_writer.release()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # IMPORTANT: identical CSV schema / column order to the previous
    # Adaptive EMA baseline.
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

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # Separate debug file so the main CSV remains directly comparable
    # to the old baseline.
    corner_debug_fieldnames = [
        "frame_index",
        "track_id",
        "window_start_frame",
        "window_end_frame",
        "temporal_applied",
        "reason",
        "corner_tl_source_frame",
        "corner_tr_source_frame",
        "corner_br_source_frame",
        "corner_bl_source_frame",
        "corner_tl_score",
        "corner_tr_score",
        "corner_br_score",
        "corner_bl_score",
        "corner_joint_score",
        "corner_mixed_source_count",
    ]

    with CORNER_DEBUG_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=corner_debug_fieldnames,
        )
        writer.writeheader()
        writer.writerows(corner_debug_rows)

    print("\n========== COMPLETE ==========")
    print(f"Processed frames      : {processed_count}")
    print(f"Pose results          : {total_pose_results}")
    print(
        f"Temporal-corner poses : "
        f"{total_temporal_detections}"
    )
    print(
        f"RAW fallback poses    : "
        f"{total_raw_fallback_detections}"
    )
    print(f"Main video            : {OUTPUT_PATH}")
    print(
        f"Zoom panel            : "
        f"{ZOOM_PANEL_OUTPUT_PATH}"
    )
    print(f"Pose CSV              : {CSV_PATH}")
    print(
        f"Corner debug CSV      : "
        f"{CORNER_DEBUG_CSV_PATH}"
    )
    print("==============================\n")


if __name__ == "__main__":
    main()
