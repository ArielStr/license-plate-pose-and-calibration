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

from video_pose.frame_sampler import SampledFrame
from video_pose.models import PlateDetection
from video_pose.filtering import filter_detections


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROBOFLOW_JSON_DIR = PROJECT_ROOT / "roboflow_jsons"
VIDEO_POSE_DIR = Path(__file__).resolve().parent

REFINEMENT_DEBUG_DIR = (
    VIDEO_POSE_DIR
    / "debug"
    / "refinement_panels"
)


def roboflow_json_path_for_frame(
    video_name: str,
    frame_index: int,
) -> Path:
    safe_video_name = Path(video_name).stem

    video_cache_dir = ROBOFLOW_JSON_DIR / safe_video_name
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    return video_cache_dir / f"frame_{frame_index:06d}.json"


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
    Run the existing Roboflow detector on one in-memory video frame.

    Cached JSON is deliberately kept outside video_pose so changing
    the product package does not invalidate detector results.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot detect plates in an empty image")

    json_path = roboflow_json_path_for_frame(
        video_name=video_name,
        frame_index=frame_index,
    )

    if use_cached_json and json_path.exists():
        print(f"Loading cached Roboflow result: {json_path}")
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
                f"Could not write temporary Roboflow image: {temporary_path}"
            )

        print(f"Running Roboflow for frame {frame_index}...")

        _, raw_result = detect_plate_with_roboflow(
            str(temporary_path),
            api_key,
        )

        save_roboflow_result(
            raw_result,
            json_path,
        )

        print(f"Saved Roboflow result: {json_path}")

        return raw_result

    except RuntimeError as error:
        if "No license plate detected" in str(error):
            print(
                f"Frame {frame_index}: "
                "Roboflow returned no detections"
            )
            return None

        raise

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_plate_detections(
    sampled_frame: SampledFrame,
    detections: list,
    refinement_results: list[dict],
) -> list[PlateDetection]:
    """
    Convert successful corner-refinement results into the product's
    image-space PlateDetection objects.

    No homography and no calibration-specific state are created here.
    """
    plate_detections: list[PlateDetection] = []

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

        if corners.shape != (4, 2):
            print(
                f"Frame {sampled_frame.frame_index}, "
                f"plate {detection_index}: "
                f"invalid corner shape {corners.shape}"
            )
            continue

        if not np.all(np.isfinite(corners)):
            print(
                f"Frame {sampled_frame.frame_index}, "
                f"plate {detection_index}: "
                "corners contain non-finite values"
            )
            continue

        plate_detections.append(
            PlateDetection(
                frame_index=sampled_frame.frame_index,
                timestamp_sec=sampled_frame.timestamp_sec,
                bbox=detection.bbox_xyxy,
                corners=corners,
                detection_confidence=float(detection.confidence),
            )
        )

    return plate_detections


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
) -> list[PlateDetection]:
    """
    Process one video frame through the image-space part of the product.

    Flow:

        frame
          -> Roboflow detection
          -> detection extraction
          -> NMS / coarse size filtering
          -> corner refinement
          -> PlateDetection
          -> geometric sanity filtering

    Pose estimation is intentionally NOT performed here yet.
    The next product layer will take accepted PlateDetection objects
    and run solvePnP using a known camera calibration.
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

    plate_detections = build_plate_detections(
        sampled_frame=sampled_frame,
        detections=detections,
        refinement_results=refinement_results,
    )

    filter_detections(plate_detections)

    if refinement_panel_debug:
        if refinement_panel_debug_dir is None:
            panel_directory = REFINEMENT_DEBUG_DIR
        else:
            panel_directory = Path(refinement_panel_debug_dir)

        panel_path = (
            panel_directory
            / f"frame_{sampled_frame.frame_index:06d}_refinement.jpg"
        )

        save_refinement_crops_panel(
            image=sampled_frame.image,
            detections=detections,
            refinement_results=refinement_results,
            plate_detections=plate_detections,
            output_path=panel_path,
            frame_index=sampled_frame.frame_index,
        )

    accepted_count = sum(
        detection.accepted
        for detection in plate_detections
    )

    print(
        f"Frame {sampled_frame.frame_index}: "
        f"{len(detections)} detector boxes, "
        f"{len(plate_detections)} refined plates, "
        f"{accepted_count} accepted"
    )

    return plate_detections


def save_refinement_crops_panel(
    image: np.ndarray,
    detections: list,
    refinement_results: list[dict],
    plate_detections: list[PlateDetection],
    output_path: str | Path,
    *,
    frame_index: int,
    max_columns: int = 3,
    cell_width: int = 430,
    cell_height: int = 310,
    crop_padding_px: int = 12,
) -> None:
    """
    Lightweight debug panel retained from the old calibration pipeline,
    but now displays product-level PlateDetection acceptance/rejection.
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

    plate_detection_by_detector_index: dict[int, PlateDetection] = {}

    for detector_index, detection in enumerate(detections):
        detection_bbox = np.asarray(
            detection.bbox_xyxy,
            dtype=np.float64,
        )

        for plate_detection in plate_detections:
            if np.allclose(
                plate_detection.bbox,
                detection_bbox,
                rtol=0.0,
                atol=1e-6,
            ):
                plate_detection_by_detector_index[
                    detector_index
                ] = plate_detection
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
        f"Frame {frame_index} | refined plates={len(plate_detections)}",
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
    drawing_height = cell_height - header_height - 2 * margin

    for detection_index, detection in enumerate(detections):
        row = detection_index // num_columns
        column = detection_index % num_columns

        cell_x = column * cell_width
        cell_y = title_bar_height + row * cell_height

        x_min, y_min, x_max, y_max = np.asarray(
            detection.bbox_xyxy,
            dtype=np.float64,
        )

        crop_x_min = max(0, int(np.floor(x_min)) - crop_padding_px)
        crop_y_min = max(0, int(np.floor(y_min)) - crop_padding_px)
        crop_x_max = min(image_width, int(np.ceil(x_max)) + crop_padding_px)
        crop_y_max = min(image_height, int(np.ceil(y_max)) + crop_padding_px)

        crop = image[
            crop_y_min:crop_y_max,
            crop_x_min:crop_x_max,
        ].copy()

        result = result_by_detection_index.get(detection_index)
        success = bool(result is not None and result.get("success", False))

        plate_detection = plate_detection_by_detector_index.get(
            detection_index
        )

        if not success:
            status_text = "REFINEMENT FAILED"
            status_color = (0, 0, 255)
        elif plate_detection is None:
            status_text = "REFINEMENT SUCCESS"
            status_color = (0, 255, 255)
        elif plate_detection.accepted:
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

        if (
            plate_detection is not None
            and not plate_detection.accepted
            and plate_detection.rejection_reason
        ):
            second_line += f" | {plate_detection.rejection_reason}"

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

            local_points = image_points.copy()
            local_points[:, 0] -= crop_x_min
            local_points[:, 1] -= crop_y_min

            integer_points = np.round(local_points).astype(np.int32)

            quad_color = status_color

            for point_index in range(4):
                next_index = (point_index + 1) % 4

                cv2.line(
                    crop,
                    tuple(integer_points[point_index]),
                    tuple(integer_points[next_index]),
                    quad_color,
                    2,
                    cv2.LINE_AA,
                )

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
                    (point_tuple[0] + 8, point_tuple[1] - 8),
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

        resized_width = max(1, int(round(crop_width * resize_scale)))
        resized_height = max(1, int(round(crop_height * resize_scale)))

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
