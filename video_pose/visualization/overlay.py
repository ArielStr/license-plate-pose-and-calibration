from __future__ import annotations

import cv2
import numpy as np

from video_pose.models import PlateFrameResult


_TRACK_COLORS = (
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
)


def _track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 255, 255)
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def draw_plate_result(image: np.ndarray, result: PlateFrameResult) -> np.ndarray:
    """Draw one tracked plate result: refined quad, track id, distance and yaw."""
    output = image
    detection = result.detection
    pose = result.pose
    corners = np.asarray(detection.corners, dtype=np.float64)
    color = _track_color(detection.track_id)

    points = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        output,
        [points],
        isClosed=True,
        color=color,
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    x_min = int(np.min(corners[:, 0]))
    y_min = int(np.min(corners[:, 1]))

    track_text = (
        f"Track {detection.track_id}"
        if detection.track_id is not None
        else "Track ?"
    )

    if pose is None:
        pose_text = "Pose unavailable"
    else:
        pose_text = f"{pose.distance_m:.2f} m | yaw {pose.yaw_deg:+.1f} deg"

    text_x = max(10, x_min)
    first_line_y = max(34, y_min - 44)
    second_line_y = first_line_y + 28

    cv2.putText(
        output,
        track_text,
        (text_x, first_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        pose_text,
        (text_x, second_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def draw_frame_results(
    image: np.ndarray,
    results: list[PlateFrameResult],
    *,
    frame_index: int | None = None,
) -> np.ndarray:
    """Draw all product results for one frame."""
    output = image.copy()

    for result in results:
        draw_plate_result(output, result)

    if frame_index is not None:
        cv2.putText(
            output,
            f"Frame {frame_index}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output
