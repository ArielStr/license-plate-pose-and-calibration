from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from single_plate_pose.detection import (
    detect_plate_with_roboflow,
    load_roboflow_result,
    save_roboflow_result,
)

from multi_plate_new.detection import (
    extract_plate_detections,
    non_max_suppression,
)

from multi_plate_new.refinement import (
    refine_all_detections_yellow,
)

from multi_plate_new.homographies import (
    compute_plate_homography,
)

from video_calibration.frame_sampler import SampledFrame
from video_calibration.observation import PlateObservation
from video_calibration.filtering import filter_observations

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROBOFLOW_JSON_DIR = PROJECT_ROOT / "roboflow_jsons"
VIDEO_CALIBRATION_DIR = Path(__file__).resolve().parent

DETECTION_DEBUG_DIR = (
    VIDEO_CALIBRATION_DIR
    / "debug"
    / "detection_panels"
)

REFINEMENT_DEBUG_DIR = (
    VIDEO_CALIBRATION_DIR
    / "debug"
    / "refinement_panels"
)


def roboflow_json_path_for_frame(
    video_name: str,
    frame_index: int,
) -> Path:
    """
    Return a stable cache path for one video frame.

    Example:
        roboflow_jsons/test_video/frame_000010.json
    """
    safe_video_name = Path(video_name).stem

    video_cache_dir = ROBOFLOW_JSON_DIR / safe_video_name
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    return video_cache_dir / f"frame_{frame_index:06d}.json"


def save_detection_crops_panel(
    image: np.ndarray,
    detections: list,
    output_path: str | Path,
    *,
    frame_index: int,
    max_columns: int = 3,
    cell_width: int = 420,
    cell_height: int = 260,
) -> None:
    """
    Save a grid containing all detections used for refinement.

    Each cell contains:
        - cropped detection
        - detection index
        - confidence
        - bounding-box dimensions
        - bounding-box coordinates
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = image.shape[:2]

    if not detections:
        panel = np.zeros(
            (160, 700, 3),
            dtype=np.uint8,
        )

        cv2.putText(
            panel,
            f"Frame {frame_index}: no detections after filtering",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(output_path), panel)
        print(f"Saved empty detection panel: {output_path}")
        return

    num_columns = min(max_columns, len(detections))
    num_rows = int(
        np.ceil(len(detections) / num_columns)
    )

    panel = np.zeros(
        (
            num_rows * cell_height,
            num_columns * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    header_height = 58
    inner_margin = 10

    available_width = cell_width - 2 * inner_margin
    available_height = (
        cell_height
        - header_height
        - 2 * inner_margin
    )

    for detection_index, detection in enumerate(detections):
        row = detection_index // num_columns
        column = detection_index % num_columns

        cell_x = column * cell_width
        cell_y = row * cell_height

        bbox = np.asarray(
            detection.bbox_xyxy,
            dtype=np.float64,
        )

        x_min, y_min, x_max, y_max = bbox

        # Clip the bounding box to the actual image.
        x_min_int = max(0, int(np.floor(x_min)))
        y_min_int = max(0, int(np.floor(y_min)))
        x_max_int = min(
            image_width,
            int(np.ceil(x_max)),
        )
        y_max_int = min(
            image_height,
            int(np.ceil(y_max)),
        )

        crop = image[
            y_min_int:y_max_int,
            x_min_int:x_max_int,
        ]

        confidence = float(detection.confidence)

        title = (
            f"Detection {detection_index} | "
            f"conf={confidence:.3f}"
        )

        geometry_text = (
            f"bbox=({x_min_int},{y_min_int})-"
            f"({x_max_int},{y_max_int}) | "
            f"{x_max_int - x_min_int}x"
            f"{y_max_int - y_min_int}"
        )

        cv2.putText(
            panel,
            title,
            (
                cell_x + inner_margin,
                cell_y + 22,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            geometry_text,
            (
                cell_x + inner_margin,
                cell_y + 46,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if crop.size == 0:
            cv2.putText(
                panel,
                "INVALID / EMPTY CROP",
                (
                    cell_x + 35,
                    cell_y + header_height + 80,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            continue

        crop_height, crop_width = crop.shape[:2]

        resize_scale = min(
            available_width / crop_width,
            available_height / crop_height,
        )

        resized_width = max(
            1,
            int(round(crop_width * resize_scale)),
        )
        resized_height = max(
            1,
            int(round(crop_height * resize_scale)),
        )

        resized_crop = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=(
                cv2.INTER_AREA
                if resize_scale < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        crop_x = (
            cell_x
            + inner_margin
            + (available_width - resized_width) // 2
        )

        crop_y = (
            cell_y
            + header_height
            + inner_margin
            + (available_height - resized_height) // 2
        )

        panel[
            crop_y:crop_y + resized_height,
            crop_x:crop_x + resized_width,
        ] = resized_crop

        cv2.rectangle(
            panel,
            (crop_x, crop_y),
            (
                crop_x + resized_width - 1,
                crop_y + resized_height - 1,
            ),
            (255, 255, 255),
            1,
        )

        # Border around the entire grid cell.
        cv2.rectangle(
            panel,
            (cell_x, cell_y),
            (
                cell_x + cell_width - 1,
                cell_y + cell_height - 1,
            ),
            (80, 80, 80),
            1,
        )

    frame_title = f"Frame {frame_index} | detections={len(detections)}"

    title_bar_height = 45

    final_panel = np.zeros(
        (
            panel.shape[0] + title_bar_height,
            panel.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    final_panel[
        title_bar_height:,
        :,
    ] = panel

    cv2.putText(
        final_panel,
        frame_title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    write_success = cv2.imwrite(
        str(output_path),
        final_panel,
    )

    if not write_success:
        raise RuntimeError(
            f"Could not save detection panel: {output_path}"
        )

    print(f"Saved detection panel: {output_path}")
def detect_plates_in_frame(
    image: np.ndarray,
    api_key: str,
    *,
    video_name: str,
    frame_index: int,
    use_cached_json: bool = True,
    jpeg_quality: int = 95,
) -> object | None:
    """
    Run Roboflow on an in-memory video frame, with JSON caching.

    Cache structure:

        roboflow_jsons/
            <video_name>/
                frame_000000.json
                frame_000010.json
                ...

    When a cached JSON exists, Roboflow is not called again.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot detect plates in an empty image")

    json_path = roboflow_json_path_for_frame(
        video_name=video_name,
        frame_index=frame_index,
    )

    if use_cached_json and json_path.exists():
        print(
            f"Loading cached Roboflow result: {json_path}"
        )
        return load_roboflow_result(json_path)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        write_success = cv2.imwrite(
            str(temporary_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        if not write_success:
            raise RuntimeError(
                f"Could not write temporary frame: {temporary_path}"
            )

        print(
            f"Running Roboflow for frame {frame_index}..."
        )

        try:
            _, raw_result = detect_plate_with_roboflow(
                str(temporary_path),
                api_key,
            )

        except RuntimeError as error:
            if "No license plate detected" in str(error):
                print(
                    f"Frame {frame_index}: "
                    "Roboflow returned no detections"
                )
                return None

            raise

        save_roboflow_result(
            raw_result,
            json_path,
        )

        print(
            f"Saved Roboflow result: {json_path}"
        )

        return raw_result

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_observations(
    sampled_frame: SampledFrame,
    detections: list,
    refinement_results: list[dict],
) -> list[PlateObservation]:
    """
    Convert successful refinement results into PlateObservation objects.
    """
    observations: list[PlateObservation] = []

    for result in refinement_results:
        if not result["success"]:
            continue

        detection_index = int(result["index"])

        if not 0 <= detection_index < len(detections):
            print(
                "Skipping refinement with invalid detection index: "
                f"{detection_index}"
            )
            continue

        detection = detections[detection_index]

        corners = np.asarray(
            result["image_points"],
            dtype=np.float64,
        )

        try:
            homography = compute_plate_homography(corners)
        except cv2.error as error:
            print(
                f"Frame {sampled_frame.frame_index}, "
                f"plate {detection_index}: "
                f"homography computation failed: {error}"
            )
            continue

        if homography is None:
            print(
                f"Frame {sampled_frame.frame_index}, "
                f"plate {detection_index}: "
                "OpenCV returned no homography"
            )
            continue

        if not np.all(np.isfinite(homography)):
            print(
                f"Frame {sampled_frame.frame_index}, "
                f"plate {detection_index}: "
                "homography contains non-finite values"
            )
            continue

        observation = PlateObservation(
            frame_index=sampled_frame.frame_index,
            timestamp_sec=sampled_frame.timestamp_sec,
            bbox=detection.bbox_xyxy,
            corners=corners,
            homography=homography,
            detection_confidence=float(detection.confidence),
        )

        observations.append(observation)

    return observations


def process_sampled_frame(
    sampled_frame: SampledFrame,
    api_key: str,
    *,
    video_name: str,
    use_cached_json: bool = True,
    min_confidence: float = 0.35,
    min_detection_area: float = 300.0,
    min_width: float = 80.0,
    min_height: float = 20.0,
    nms_iou_threshold: float = 0.5,
    refinement_method: str = "robust_mask_lines",
    refinement_debug: bool = False,
    refinement_panel_debug: bool = False,
    refinement_panel_debug_dir: str | Path | None = None,
) -> list[PlateObservation]:
    """
    Process one sampled video frame through the plate pipeline.

    Flow:

        sampled frame
            -> Roboflow
            -> detection extraction
            -> non-maximum suppression
            -> size filtering
            -> yellow corner refinement
            -> homography
            -> PlateObservation
            -> observation filtering
            -> final refinement/debug panel

    The returned list preserves both accepted and rejected observations.
    Filtering status is stored in:

        observation.accepted
        observation.rejection_reason
    """
    raw_result = detect_plates_in_frame(
        image=sampled_frame.image,
        api_key=api_key,
        video_name=video_name,
        frame_index=sampled_frame.frame_index,
        use_cached_json=use_cached_json,
    )

    if raw_result is None:
        print(
            f"Frame {sampled_frame.frame_index}: "
            "no plate detections"
        )
        return []

    detections = extract_plate_detections(
        raw_result,
        min_confidence=min_confidence,
        min_area=min_detection_area,
    )

    detections = non_max_suppression(
        detections,
        iou_threshold=nms_iou_threshold,
    )

    detections = [
        detection
        for detection in detections
        if detection.width >= min_width
        and detection.height >= min_height
    ]

    if not detections:
        print(
            f"Frame {sampled_frame.frame_index}: "
            "all detections were filtered out"
        )
        return []

    refinement_results = refine_all_detections_yellow(
        sampled_frame.image,
        detections,
        debug=refinement_debug,
        method=refinement_method,
    )

    observations = build_observations(
        sampled_frame=sampled_frame,
        detections=detections,
        refinement_results=refinement_results,
    )

    # Apply the calibration-observation filters BEFORE writing the panel,
    # so the same panel can show ACCEPTED / REJECTED status.
    filter_observations(observations)

    if refinement_panel_debug:
        if refinement_panel_debug_dir is None:
            panel_directory = REFINEMENT_DEBUG_DIR
        else:
            panel_directory = Path(
                refinement_panel_debug_dir
            )

        panel_path = (
            panel_directory
            / (
                f"frame_{sampled_frame.frame_index:06d}"
                f"_refinement.jpg"
            )
        )

        save_refinement_crops_panel(
            image=sampled_frame.image,
            detections=detections,
            refinement_results=refinement_results,
            observations=observations,
            output_path=panel_path,
            frame_index=sampled_frame.frame_index,
        )

    accepted_count = sum(
        observation.accepted
        for observation in observations
    )

    print(
        f"Frame {sampled_frame.frame_index}: "
        f"{len(detections)} detections, "
        f"{len(observations)} valid observations, "
        f"{accepted_count} accepted"
    )

    return observations


def save_refinement_crops_panel(
    image: np.ndarray,
    detections: list,
    refinement_results: list[dict],
    output_path: str | Path,
    *,
    frame_index: int,
    max_columns: int = 3,
    cell_width: int = 430,
    cell_height: int = 310,
    crop_padding_px: int = 12,
    observations: list[PlateObservation] | None = None,
) -> None:
    """
    Save a panel showing the refined corners selected for every detection.

    The refinement points are stored in full-image coordinates, so they are
    converted into crop-local coordinates before drawing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = image.shape[:2]

    if not detections:
        return

    result_by_detection_index = {
        int(result["index"]): result
        for result in refinement_results
    }
    observation_by_detection_index: dict[int, PlateObservation] = {}

    if observations is not None:
        # Match observations back to detections using the bbox copied into
        # PlateObservation. This stays correct even if a successful
        # refinement later fails homography construction.
        for detection_index, detection in enumerate(detections):
            detection_bbox = np.asarray(
                detection.bbox_xyxy,
                dtype=np.float64,
            )

            for observation in observations:
                if np.allclose(
                    observation.bbox,
                    detection_bbox,
                    rtol=0.0,
                    atol=1e-6,
                ):
                    observation_by_detection_index[
                        detection_index
                    ] = observation
                    break
    num_columns = min(max_columns, len(detections))
    num_rows = int(np.ceil(len(detections) / num_columns))

    title_bar_height = 45

    panel = np.zeros(
        (
            title_bar_height + num_rows * cell_height,
            num_columns * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        panel,
        (
            f"Frame {frame_index} | "
            f"refinements={len(refinement_results)}"
        ),
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    header_height = 54
    margin = 10

    drawing_width = cell_width - 2 * margin
    drawing_height = (
        cell_height
        - header_height
        - 2 * margin
    )

    for detection_index, detection in enumerate(detections):
        row = detection_index // num_columns
        column = detection_index % num_columns

        cell_x = column * cell_width
        cell_y = title_bar_height + row * cell_height

        x_min, y_min, x_max, y_max = np.asarray(
            detection.bbox_xyxy,
            dtype=np.float64,
        )

        # Add a little padding so the refined corners are not clipped.
        crop_x_min = max(
            0,
            int(np.floor(x_min)) - crop_padding_px,
        )
        crop_y_min = max(
            0,
            int(np.floor(y_min)) - crop_padding_px,
        )
        crop_x_max = min(
            image_width,
            int(np.ceil(x_max)) + crop_padding_px,
        )
        crop_y_max = min(
            image_height,
            int(np.ceil(y_max)) + crop_padding_px,
        )

        crop = image[
            crop_y_min:crop_y_max,
            crop_x_min:crop_x_max,
        ].copy()

        result = result_by_detection_index.get(
            detection_index
        )

        success = bool(
            result is not None
            and result.get("success", False)
        )

        if not success:
            status_text = "REFINEMENT FAILED"
            status_color = (0, 0, 255)

        else:
            observation = observation_by_detection_index.get(
                detection_index
            )

            if observation is None:
                status_text = "REFINEMENT SUCCESS"
                status_color = (0, 255, 255)

            elif observation.accepted:
                status_text = "ACCEPTED"
                status_color = (0, 255, 0)

            else:
                status_text = "REJECTED"
                status_color = (0, 0, 255)

        cv2.putText(
            panel,
            (
                f"Plate {detection_index} | "
                f"conf={float(detection.confidence):.3f} | "
                f"{status_text}"
            ),
            (cell_x + margin, cell_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            status_color,
            1,
            cv2.LINE_AA,
        )

        second_line = (
            f"bbox={crop_x_max - crop_x_min}x"
            f"{crop_y_max - crop_y_min}"
        )

        observation = (
            observation_by_detection_index.get(
                detection_index
            )
        )

        if (
                observation is not None
                and not observation.accepted
                and observation.rejection_reason
        ):
            second_line += (
                f" | {observation.rejection_reason}"
            )

        cv2.putText(
            panel,
            second_line,
            (cell_x + margin, cell_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if crop.size == 0:
            continue

        if success:
            image_points = np.asarray(
                result["image_points"],
                dtype=np.float64,
            )

            # Convert global image coordinates to coordinates inside the crop.
            local_points = image_points.copy()
            local_points[:, 0] -= crop_x_min
            local_points[:, 1] -= crop_y_min

            integer_points = np.round(
                local_points
            ).astype(np.int32)

            # Draw the quadrilateral used to compute the homography.
            for point_index in range(4):
                next_index = (point_index + 1) % 4

                point = tuple(integer_points[point_index])
                next_point = tuple(integer_points[next_index])

                observation = observation_by_detection_index.get(
                    detection_index
                )

                if observation is None:
                    quad_color = (0, 255, 255)
                elif observation.accepted:
                    quad_color = (0, 255, 0)
                else:
                    quad_color = (0, 0, 255)

                cv2.line(
                    crop,
                    point,
                    next_point,
                    quad_color,
                    2,
                    cv2.LINE_AA,
                )

            # Draw and label each selected corner.
            for point_index, point in enumerate(integer_points):
                point_tuple = tuple(point)

                cv2.circle(
                    crop,
                    point_tuple,
                    6,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    crop,
                    point_tuple,
                    9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    crop,
                    str(point_index),
                    (
                        point_tuple[0] + 8,
                        point_tuple[1] - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        crop_height, crop_width = crop.shape[:2]

        resize_scale = min(
            drawing_width / crop_width,
            drawing_height / crop_height,
        )

        resized_width = max(
            1,
            int(round(crop_width * resize_scale)),
        )
        resized_height = max(
            1,
            int(round(crop_height * resize_scale)),
        )

        resized_crop = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=(
                cv2.INTER_AREA
                if resize_scale < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        target_x = (
            cell_x
            + margin
            + (drawing_width - resized_width) // 2
        )

        target_y = (
            cell_y
            + header_height
            + margin
            + (drawing_height - resized_height) // 2
        )

        panel[
            target_y:target_y + resized_height,
            target_x:target_x + resized_width,
        ] = resized_crop

        cv2.rectangle(
            panel,
            (target_x, target_y),
            (
                target_x + resized_width - 1,
                target_y + resized_height - 1,
            ),
            (255, 255, 255),
            1,
        )

        cv2.rectangle(
            panel,
            (cell_x, cell_y),
            (
                cell_x + cell_width - 1,
                cell_y + cell_height - 1,
            ),
            (80, 80, 80),
            1,
        )

    write_success = cv2.imwrite(
        str(output_path),
        panel,
    )

    if not write_success:
        raise RuntimeError(
            f"Could not save refinement panel: {output_path}"
        )

    print(f"Saved refinement panel: {output_path}")