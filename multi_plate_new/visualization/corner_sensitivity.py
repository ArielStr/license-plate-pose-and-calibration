from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# SMALL DRAWING HELPERS
# ============================================================

ORIGINAL_COLOR = (0, 0, 255)      # red, BGR
OPTIMIZED_COLOR = (0, 200, 0)     # green
ARROW_COLOR = (255, 0, 255)       # magenta
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)
QUAD_ORIGINAL_COLOR = (0, 0, 255)
QUAD_OPTIMIZED_COLOR = (0, 255, 0)


def _save_image(
    save_path: str | Path,
    image: np.ndarray,
) -> None:
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(
        str(save_path),
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Could not save: {save_path}"
        )


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_SHADOW,
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def _arrow(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    thickness: int = 2,
) -> None:
    start_int = tuple(
        np.round(start).astype(int)
    )

    end_int = tuple(
        np.round(end).astype(int)
    )

    cv2.arrowedLine(
        image,
        start_int,
        end_int,
        ARROW_COLOR,
        thickness,
        cv2.LINE_AA,
        tipLength=0.25,
    )


# ============================================================
# FULL-IMAGE OVERVIEW
# ============================================================

def draw_corner_shift_overview(
    image: np.ndarray,
    optimization_result,
    save_path: str | Path,
) -> np.ndarray:
    """
    Full-scene context view.

    Original:
        red

    Optimized:
        green

    Movement:
        magenta arrows

    Note:
    sub-pixel shifts may be visually almost invisible here.
    The zoom-grid output is the important presentation asset.
    """
    vis = image.copy()

    for plate in optimization_result.plate_results:
        original = np.asarray(
            plate.original_corners,
            dtype=np.float64,
        )

        optimized = np.asarray(
            plate.optimized_corners,
            dtype=np.float64,
        )

        original_contour = (
            np.round(original)
            .astype(np.int32)
            .reshape((-1, 1, 2))
        )

        optimized_contour = (
            np.round(optimized)
            .astype(np.int32)
            .reshape((-1, 1, 2))
        )

        cv2.polylines(
            vis,
            [original_contour],
            True,
            QUAD_ORIGINAL_COLOR,
            4,
            cv2.LINE_AA,
        )

        cv2.polylines(
            vis,
            [optimized_contour],
            True,
            QUAD_OPTIMIZED_COLOR,
            3,
            cv2.LINE_AA,
        )

        for corner_index in range(4):
            p0 = original[
                corner_index
            ]

            p1 = optimized[
                corner_index
            ]

            _arrow(
                vis,
                p0,
                p1,
                thickness=3,
            )

            cv2.circle(
                vis,
                tuple(
                    np.round(p0)
                    .astype(int)
                ),
                8,
                ORIGINAL_COLOR,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                vis,
                tuple(
                    np.round(p1)
                    .astype(int)
                ),
                6,
                OPTIMIZED_COLOR,
                -1,
                cv2.LINE_AA,
            )

        centroid = np.mean(
            optimized,
            axis=0,
        )

        _draw_text(
            vis,
            f"Plate {plate.plate_index}",
            (
                int(centroid[0]),
                int(centroid[1] - 25),
            ),
            scale=0.8,
            thickness=2,
        )

    _save_image(
        save_path,
        vis,
    )

    return vis


# ============================================================
# ZOOM-IN VISUALIZATION
# ============================================================

def _extract_plate_crop(
    image: np.ndarray,
    original: np.ndarray,
    optimized: np.ndarray,
    *,
    padding_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_points = np.vstack(
        [original, optimized]
    )

    x_min, y_min = np.min(
        all_points,
        axis=0,
    )

    x_max, y_max = np.max(
        all_points,
        axis=0,
    )

    width = max(
        x_max - x_min,
        1.0,
    )

    height = max(
        y_max - y_min,
        1.0,
    )

    pad_x = padding_ratio * width
    pad_y = padding_ratio * height

    x1 = max(
        0,
        int(np.floor(x_min - pad_x)),
    )

    y1 = max(
        0,
        int(np.floor(y_min - pad_y)),
    )

    x2 = min(
        image.shape[1],
        int(np.ceil(x_max + pad_x)),
    )

    y2 = min(
        image.shape[0],
        int(np.ceil(y_max + pad_y)),
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    offset = np.array(
        [x1, y1],
        dtype=np.float64,
    )

    return (
        crop,
        original - offset,
        optimized - offset,
    )


def _draw_plate_crop(
    crop: np.ndarray,
    original_local: np.ndarray,
    optimized_local: np.ndarray,
    plate_index: int,
) -> np.ndarray:
    vis = crop.copy()

    original_contour = (
        np.round(original_local)
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )

    optimized_contour = (
        np.round(optimized_local)
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )

    cv2.polylines(
        vis,
        [original_contour],
        True,
        QUAD_ORIGINAL_COLOR,
        3,
        cv2.LINE_AA,
    )

    cv2.polylines(
        vis,
        [optimized_contour],
        True,
        QUAD_OPTIMIZED_COLOR,
        3,
        cv2.LINE_AA,
    )

    for i in range(4):
        p0 = original_local[i]
        p1 = optimized_local[i]

        _arrow(
            vis,
            p0,
            p1,
            thickness=2,
        )

        cv2.circle(
            vis,
            tuple(
                np.round(p0)
                .astype(int)
            ),
            5,
            ORIGINAL_COLOR,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            vis,
            tuple(
                np.round(p1)
                .astype(int)
            ),
            4,
            OPTIMIZED_COLOR,
            -1,
            cv2.LINE_AA,
        )

        _draw_text(
            vis,
            str(i),
            (
                int(round(p1[0])) + 6,
                int(round(p1[1])) - 6,
            ),
            scale=0.45,
        )

    _draw_text(
        vis,
        f"Plate {plate_index}",
        (10, 24),
        scale=0.7,
        thickness=2,
    )

    return vis


def _corner_zoom(
    image: np.ndarray,
    original_point: np.ndarray,
    optimized_point: np.ndarray,
    shift: np.ndarray,
    shift_distance: float,
    *,
    crop_radius_px: int = 12,
    output_size: int = 190,
    corner_index: int,
) -> np.ndarray:
    """
    Crop a small true image neighborhood around one corner and enlarge it.

    The shift itself is NOT artificially scaled.
    The entire local image crop is enlarged, exactly like a microscope zoom.
    """
    center = 0.5 * (
        original_point
        + optimized_point
    )

    cx = float(center[0])
    cy = float(center[1])

    x1 = max(
        0,
        int(np.floor(
            cx - crop_radius_px
        )),
    )

    y1 = max(
        0,
        int(np.floor(
            cy - crop_radius_px
        )),
    )

    x2 = min(
        image.shape[1],
        int(np.ceil(
            cx + crop_radius_px + 1
        )),
    )

    y2 = min(
        image.shape[0],
        int(np.ceil(
            cy + crop_radius_px + 1
        )),
    )

    local = image[
        y1:y2,
        x1:x2,
    ].copy()

    if local.size == 0:
        local = np.zeros(
            (25, 25, 3),
            dtype=np.uint8,
        )

        x1 = 0
        y1 = 0

    local_h, local_w = (
        local.shape[:2]
    )

    scale_x = (
        output_size
        / max(local_w, 1)
    )

    scale_y = (
        output_size
        / max(local_h, 1)
    )

    zoom = cv2.resize(
        local,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    original_local = np.array(
        [
            (original_point[0] - x1)
            * scale_x,

            (original_point[1] - y1)
            * scale_y,
        ],
        dtype=np.float64,
    )

    optimized_local = np.array(
        [
            (optimized_point[0] - x1)
            * scale_x,

            (optimized_point[1] - y1)
            * scale_y,
        ],
        dtype=np.float64,
    )

    # Draw line first so both point centers remain readable.
    _arrow(
        zoom,
        original_local,
        optimized_local,
        thickness=2,
    )

    cv2.circle(
        zoom,
        tuple(
            np.round(original_local)
            .astype(int)
        ),
        6,
        ORIGINAL_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        zoom,
        tuple(
            np.round(optimized_local)
            .astype(int)
        ),
        5,
        OPTIMIZED_COLOR,
        -1,
        cv2.LINE_AA,
    )

    label_height = 66

    canvas = np.zeros(
        (
            output_size
            + label_height,
            output_size,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :output_size,
        :,
    ] = zoom

    _draw_text(
        canvas,
        f"C{corner_index}",
        (6, 20),
        scale=0.55,
        thickness=2,
    )

    _draw_text(
        canvas,
        (
            f"dx={shift[0]:+.3f}  "
            f"dy={shift[1]:+.3f}"
        ),
        (
            8,
            output_size + 24,
        ),
        scale=0.42,
    )

    _draw_text(
        canvas,
        (
            f"|dp|={shift_distance:.3f} px"
        ),
        (
            8,
            output_size + 50,
        ),
        scale=0.48,
        thickness=2,
    )

    return canvas


def draw_corner_shift_zoom_grid(
    image: np.ndarray,
    optimization_result,
    save_path: str | Path,
    *,
    plate_width: int = 760,
    corner_zoom_size: int = 190,
    corner_crop_radius_px: int = 12,
) -> np.ndarray:
    """
    Presentation-oriented visualization.

    One row per plate:
        [large plate crop]
        [C0 zoom]
        [C1 zoom]
        [C2 zoom]
        [C3 zoom]

    Each corner zoom displays the exact sub-pixel:
        dx
        dy
        Euclidean displacement.
    """
    rows: list[np.ndarray] = []

    for plate in optimization_result.plate_results:
        original = np.asarray(
            plate.original_corners,
            dtype=np.float64,
        )

        optimized = np.asarray(
            plate.optimized_corners,
            dtype=np.float64,
        )

        crop, original_local, optimized_local = (
            _extract_plate_crop(
                image,
                original,
                optimized,
            )
        )

        crop_vis = _draw_plate_crop(
            crop,
            original_local,
            optimized_local,
            plate.plate_index,
        )

        crop_h, crop_w = (
            crop_vis.shape[:2]
        )

        scale = (
            plate_width
            / max(crop_w, 1)
        )

        resized_h = max(
            1,
            int(round(
                crop_h * scale
            )),
        )

        crop_vis = cv2.resize(
            crop_vis,
            (
                plate_width,
                resized_h,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        zooms: list[np.ndarray] = []

        for corner_index in range(4):
            zoom = _corner_zoom(
                image=image,
                original_point=
                    original[corner_index],
                optimized_point=
                    optimized[corner_index],
                shift=
                    plate.shifts[
                        corner_index
                    ],
                shift_distance=
                    float(
                        plate.shift_distances[
                            corner_index
                        ]
                    ),
                crop_radius_px=
                    corner_crop_radius_px,
                output_size=
                    corner_zoom_size,
                corner_index=
                    corner_index,
            )

            zooms.append(zoom)

        zoom_row = np.hstack(
            zooms
        )

        target_h = max(
            crop_vis.shape[0],
            zoom_row.shape[0],
        )

        if crop_vis.shape[0] < target_h:
            crop_vis = cv2.copyMakeBorder(
                crop_vis,
                0,
                target_h
                - crop_vis.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )

        if zoom_row.shape[0] < target_h:
            zoom_row = cv2.copyMakeBorder(
                zoom_row,
                0,
                target_h
                - zoom_row.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )

        separator = np.full(
            (
                target_h,
                16,
                3,
            ),
            245,
            dtype=np.uint8,
        )

        row = np.hstack(
            [
                crop_vis,
                separator,
                zoom_row,
            ]
        )

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No optimized plates found."
        )

    max_width = max(
        row.shape[1]
        for row in rows
    )

    normalized_rows: list[np.ndarray] = []

    for row in rows:
        if row.shape[1] < max_width:
            row = cv2.copyMakeBorder(
                row,
                0,
                0,
                0,
                max_width
                - row.shape[1],
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )

        normalized_rows.append(row)

    row_separator = np.full(
        (
            14,
            max_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    assembled: list[np.ndarray] = []

    for idx, row in enumerate(
        normalized_rows
    ):
        if idx > 0:
            assembled.append(
                row_separator
            )

        assembled.append(row)

    grid = np.vstack(
        assembled
    )

    # Legend
    legend_height = 70

    legend = np.full(
        (
            legend_height,
            max_width,
            3,
        ),
        250,
        dtype=np.uint8,
    )

    cv2.circle(
        legend,
        (35, 35),
        8,
        ORIGINAL_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        legend,
        "Original corner",
        (55, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        legend,
        (300, 35),
        8,
        OPTIMIZED_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        legend,
        "Optimized corner",
        (320, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        legend,
        "Zooms show the true sub-pixel displacement",
        (600, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    final = np.vstack(
        [
            grid,
            legend,
        ]
    )

    _save_image(
        save_path,
        final,
    )

    return final
