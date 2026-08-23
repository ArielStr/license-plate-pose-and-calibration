from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from multi_plate_new.experiments.corner_optimization_to_reference import (
    CornerOptimizationResult,
)


def _draw_arrow(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    start_point = tuple(
        np.round(start).astype(int)
    )

    end_point = tuple(
        np.round(end).astype(int)
    )

    cv2.arrowedLine(
        image,
        start_point,
        end_point,
        (255, 0, 255),
        4,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_corner_optimization_overview(
    image: np.ndarray,
    optimization_result: CornerOptimizationResult,
    save_path: str | Path,
    *,
    show: bool = False,
) -> np.ndarray:
    """
    Draw all original and optimized plate corners on the full image.

    Colors:
        original corners   - red
        optimized corners  - green
        movement arrows    - magenta
    """
    visualization = image.copy()

    for plate in optimization_result.plate_results:
        original = plate.original_corners
        optimized = plate.optimized_corners

        original_contour = np.round(
            original
        ).astype(np.int32).reshape((-1, 1, 2))

        optimized_contour = np.round(
            optimized
        ).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            visualization,
            [original_contour],
            isClosed=True,
            color=(0, 0, 255),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

        cv2.polylines(
            visualization,
            [optimized_contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=4,
            lineType=cv2.LINE_AA,
        )

        for corner_index in range(4):
            original_point = original[corner_index]
            optimized_point = optimized[corner_index]

            _draw_arrow(
                visualization,
                original_point,
                optimized_point,
            )

            original_int = tuple(
                np.round(
                    original_point
                ).astype(int)
            )

            optimized_int = tuple(
                np.round(
                    optimized_point
                ).astype(int)
            )

            cv2.circle(
                visualization,
                original_int,
                9,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                visualization,
                optimized_int,
                9,
                (0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            shift_distance = (
                plate.shift_distances[corner_index]
            )

            label = (
                f"{plate.plate_index}:{corner_index} "
                f"{shift_distance:.1f}px"
            )

            label_position = (
                optimized_int[0] + 12,
                optimized_int[1] - 12,
            )

            cv2.putText(
                visualization,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                visualization,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(save_path),
        visualization,
    )

    if not success:
        raise RuntimeError(
            f"Could not save optimization overview: {save_path}"
        )

    if show:
        preview = cv2.resize(
            visualization,
            None,
            fx=0.3,
            fy=0.3,
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow(
            "Corner optimization overview",
            preview,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return visualization


def draw_corner_optimization_grid(
    image: np.ndarray,
    optimization_result: CornerOptimizationResult,
    save_path: str | Path,
    *,
    padding_px: int = 80,
    crop_width: int = 700,
    show: bool = False,
) -> np.ndarray:
    """
    Create one enlarged crop per plate showing original corners,
    optimized corners and movement arrows.
    """
    crops: list[np.ndarray] = []

    for plate in optimization_result.plate_results:
        all_points = np.vstack(
            [
                plate.original_corners,
                plate.optimized_corners,
            ]
        )

        x_min = int(
            np.floor(
                np.min(all_points[:, 0])
                - padding_px
            )
        )

        y_min = int(
            np.floor(
                np.min(all_points[:, 1])
                - padding_px
            )
        )

        x_max = int(
            np.ceil(
                np.max(all_points[:, 0])
                + padding_px
            )
        )

        y_max = int(
            np.ceil(
                np.max(all_points[:, 1])
                + padding_px
            )
        )

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        crop = image[
            y_min:y_max,
            x_min:x_max,
        ].copy()

        if crop.size == 0:
            continue

        original_local = (
            plate.original_corners
            - np.array([x_min, y_min])
        )

        optimized_local = (
            plate.optimized_corners
            - np.array([x_min, y_min])
        )

        original_contour = np.round(
            original_local
        ).astype(np.int32).reshape((-1, 1, 2))

        optimized_contour = np.round(
            optimized_local
        ).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            crop,
            [original_contour],
            True,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.polylines(
            crop,
            [optimized_contour],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        for corner_index in range(4):
            _draw_arrow(
                crop,
                original_local[corner_index],
                optimized_local[corner_index],
            )

            original_int = tuple(
                np.round(
                    original_local[corner_index]
                ).astype(int)
            )

            optimized_int = tuple(
                np.round(
                    optimized_local[corner_index]
                ).astype(int)
            )

            cv2.circle(
                crop,
                original_int,
                7,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                crop,
                optimized_int,
                7,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

            distance = (
                plate.shift_distances[corner_index]
            )

            shift = plate.shifts[corner_index]

            label = (
                f"{corner_index}: "
                f"dx={shift[0]:+.2f}, "
                f"dy={shift[1]:+.2f}, "
                f"d={distance:.2f}"
            )

            label_position = (
                optimized_int[0] + 8,
                optimized_int[1] - 8,
            )

            cv2.putText(
                crop,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                crop,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        title = (
            f"Plate {plate.plate_index} | "
            f"RMS "
            f"{np.sqrt(np.mean(plate.initial_constraint_residuals ** 2)):.5f}"
            f" -> "
            f"{np.sqrt(np.mean(plate.final_constraint_residuals ** 2)):.5f}"
        )

        cv2.putText(
            crop,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            crop,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        scale = crop_width / crop.shape[1]

        resized_height = max(
            1,
            int(
                round(
                    crop.shape[0] * scale
                )
            ),
        )

        crop = cv2.resize(
            crop,
            (crop_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )

        crops.append(crop)

    if not crops:
        raise RuntimeError(
            "No valid crops were generated"
        )

    max_height = max(
        crop.shape[0]
        for crop in crops
    )

    normalized_crops: list[np.ndarray] = []

    for crop in crops:
        if crop.shape[0] < max_height:
            bottom_padding = (
                max_height - crop.shape[0]
            )

            crop = cv2.copyMakeBorder(
                crop,
                0,
                bottom_padding,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(30, 30, 30),
            )

        normalized_crops.append(crop)

    grid = np.hstack(
        normalized_crops
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(save_path),
        grid,
    )

    if not success:
        raise RuntimeError(
            f"Could not save optimization grid: {save_path}"
        )

    if show:
        preview_scale = min(
            1.0,
            1600.0 / grid.shape[1],
        )

        preview = cv2.resize(
            grid,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow(
            "Corner optimization grid",
            preview,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid