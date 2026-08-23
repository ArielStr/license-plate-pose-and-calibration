from __future__ import annotations

from itertools import combinations
from pathlib import Path
import csv
import sys

import cv2
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_plate_new.pipeline import calibrate_from_image
from multi_plate_new.homographies import compute_all_homographies
from multi_plate_new.intrinsics import estimate_intrinsics_from_homographies


# ============================================================
# CONFIG
# ============================================================

# Chessboard reference calibration used throughout the project.
K_REF = np.array(
    [
        [2816.53888, 0.0, 1498.87479],
        [0.0, 2845.88995, 2051.34557],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Run ONE image at a time.
#
# Example:
# IMAGE_PATH = PROJECT_ROOT / "cars_photos" / "4_cars" / "IMG_4425.jpeg"
IMAGE_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "4_cars"
    / "IMG_4442.jpeg"
)

METHOD = "robust_mask_lines"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "multi_plate_new"
    / "outputs"
    / "plate_count_experiment"
)

# We need at least 4 successful plates in the image.
# If only 3 survive the pipeline, there is no plate-count experiment to run.
MIN_SUCCESSFUL_PLATES = 4

# Debug panel settings
DEBUG_PANEL_PADDING_PX = 12
DEBUG_POINT_RADIUS_PX = 3
DEBUG_TEXT_SCALE = 0.5
DEBUG_TEXT_THICKNESS = 1


# ============================================================
# METRICS
# ============================================================

def relative_error_percent(
    value: float,
    reference: float,
) -> float:
    return float(
        (value - reference)
        / reference
        * 100.0
    )


def absolute_relative_error_percent(
    value: float,
    reference: float,
) -> float:
    return abs(
        relative_error_percent(
            value,
            reference,
        )
    )


def principal_point_error_px(
    value: float,
    reference: float,
) -> float:
    return float(
        value - reference
    )


def summarize_K(
    K: np.ndarray,
    K_ref: np.ndarray,
) -> dict:
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "skew": float(K[0, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),

        "fx_error_pct":
            relative_error_percent(
                float(K[0, 0]),
                float(K_ref[0, 0]),
            ),

        "fy_error_pct":
            relative_error_percent(
                float(K[1, 1]),
                float(K_ref[1, 1]),
            ),

        "fx_abs_error_pct":
            absolute_relative_error_percent(
                float(K[0, 0]),
                float(K_ref[0, 0]),
            ),

        "fy_abs_error_pct":
            absolute_relative_error_percent(
                float(K[1, 1]),
                float(K_ref[1, 1]),
            ),

        "cx_error_px":
            principal_point_error_px(
                float(K[0, 2]),
                float(K_ref[0, 2]),
            ),

        "cy_error_px":
            principal_point_error_px(
                float(K[1, 2]),
                float(K_ref[1, 2]),
            ),

        "cx_abs_error_px":
            abs(
                principal_point_error_px(
                    float(K[0, 2]),
                    float(K_ref[0, 2]),
                )
            ),

        "cy_abs_error_px":
            abs(
                principal_point_error_px(
                    float(K[1, 2]),
                    float(K_ref[1, 2]),
                )
            ),
    }


def mean_focal_abs_error_pct(
    metrics: dict,
) -> float:
    return float(
        0.5
        * (
            metrics["fx_abs_error_pct"]
            + metrics["fy_abs_error_pct"]
        )
    )


def mean_principal_point_abs_error_px(
    metrics: dict,
) -> float:
    return float(
        0.5
        * (
            metrics["cx_abs_error_px"]
            + metrics["cy_abs_error_px"]
        )
    )


# ============================================================
# HELPERS
# ============================================================

def extract_successful_refinements(
    refinement_results: list[dict],
) -> list[dict]:
    successful = []

    for result in refinement_results:
        if not result.get("success", False):
            continue

        if "image_points" not in result:
            continue

        corners = np.asarray(
            result["image_points"],
            dtype=np.float64,
        )

        if corners.shape != (4, 2):
            continue

        if not np.all(np.isfinite(corners)):
            continue

        successful.append(result)

    return successful


def get_plate_id(
    refinement_result: dict,
    fallback_index: int,
) -> int:
    return int(
        refinement_result.get(
            "index",
            fallback_index,
        )
    )


def calibrate_from_subset(
    subset: list[dict],
) -> dict:
    homographies = compute_all_homographies(
        subset
    )

    return estimate_intrinsics_from_homographies(
        homographies
    )


def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
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
        writer.writerows(rows)


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def _crop_with_padding(
    image: np.ndarray,
    corners: np.ndarray,
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop a padded region around one plate's detected corners.

    Returns:
        crop
        (x_min, y_min) offset of the crop in the full image
    """
    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    x_min = int(
        np.floor(
            np.min(corners[:, 0])
            - padding_px
        )
    )

    x_max = int(
        np.ceil(
            np.max(corners[:, 0])
            + padding_px
        )
    )

    y_min = int(
        np.floor(
            np.min(corners[:, 1])
            - padding_px
        )
    )

    y_max = int(
        np.ceil(
            np.max(corners[:, 1])
            + padding_px
        )
    )

    height, width = image.shape[:2]

    x_min = max(
        0,
        min(x_min, width - 1),
    )

    x_max = max(
        x_min + 1,
        min(x_max, width),
    )

    y_min = max(
        0,
        min(y_min, height - 1),
    )

    y_max = max(
        y_min + 1,
        min(y_max, height),
    )

    crop = image[
        y_min:y_max,
        x_min:x_max,
    ].copy()

    return crop, (x_min, y_min)


def _draw_corners_on_crop(
    crop: np.ndarray,
    corners: np.ndarray,
    offset: tuple[int, int],
    plate_id: int,
) -> np.ndarray:
    """
    Draw the four ordered image corners and their indices on one crop.

    Corner convention:
        0 -> 1 -> 2 -> 3 -> 0
    """
    output = crop.copy()

    x_offset, y_offset = offset

    local_corners = np.asarray(
        corners,
        dtype=np.float64,
    ).copy()

    local_corners[:, 0] -= x_offset
    local_corners[:, 1] -= y_offset

    local_points = np.round(
        local_corners
    ).astype(np.int32)

    # Connect corners so the actual estimated quadrilateral is obvious.
    polygon = local_points.reshape(
        -1,
        1,
        2,
    )

    cv2.polylines(
        output,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    for corner_index, point in enumerate(
        local_points
    ):
        x, y = int(point[0]), int(point[1])

        cv2.circle(
            output,
            (x, y),
            DEBUG_POINT_RADIUS_PX,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (x, y),
            DEBUG_POINT_RADIUS_PX + 1,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            output,
            str(corner_index),
            (
                x + DEBUG_POINT_RADIUS_PX + 5,
                y - DEBUG_POINT_RADIUS_PX - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEBUG_TEXT_SCALE,
            (255, 255, 255),
            DEBUG_TEXT_THICKNESS + 2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            str(corner_index),
            (
                x + DEBUG_POINT_RADIUS_PX + 5,
                y - DEBUG_POINT_RADIUS_PX - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEBUG_TEXT_SCALE,
            (0, 0, 0),
            DEBUG_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return output


def save_detected_plate_debug_panel(
    image: np.ndarray,
    successful_refinements: list[dict],
    output_path: Path,
) -> None:
    """
    Save a grid containing every successfully refined plate.

    Each crop shows:
        - the exact four detected image corners,
        - corner ordering 0,1,2,3,
        - the quadrilateral implied by those corners,
        - the plate ID used later in the subset experiment.

    This is intended as a quick sanity check before interpreting
    calibration results.
    """
    if not successful_refinements:
        return

    panels: list[np.ndarray] = []

    for fallback_index, refinement in enumerate(
        successful_refinements
    ):
        corners = np.asarray(
            refinement["image_points"],
            dtype=np.float64,
        )

        plate_id = get_plate_id(
            refinement,
            fallback_index,
        )

        crop, offset = _crop_with_padding(
            image,
            corners,
            DEBUG_PANEL_PADDING_PX,
        )

        annotated = _draw_corners_on_crop(
            crop,
            corners,
            offset,
            plate_id,
        )

        header_h = 28

        panel_with_title = np.full(
            (
                annotated.shape[0] + header_h,
                annotated.shape[1],
                3,
            ),
            245,
            dtype=np.uint8,
        )

        panel_with_title[
        header_h:,
        :
        ] = annotated

        cv2.putText(
            panel_with_title,
            f"Plate {plate_id}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        panels.append(
            panel_with_title
        )

    # Normalize panel heights while preserving aspect ratio.
    target_height = max(
        panel.shape[0]
        for panel in panels
    )

    resized_panels = []

    for panel in panels:
        height, width = panel.shape[:2]

        scale = (
            target_height
            / max(height, 1)
        )

        resized_width = max(
            1,
            int(round(width * scale)),
        )

        resized = cv2.resize(
            panel,
            (resized_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        resized_panels.append(
            resized
        )

    # Use at most 3 panels per row so each plate remains large enough
    # to inspect corner placement.
    columns = min(
        3,
        len(resized_panels),
    )

    rows = int(
        np.ceil(
            len(resized_panels)
            / columns
        )
    )

    cell_width = max(
        panel.shape[1]
        for panel in resized_panels
    )

    gap = 24
    header_height = 60

    canvas_width = (
        columns * cell_width
        + (columns + 1) * gap
    )

    canvas_height = (
        header_height
        + rows * target_height
        + (rows + 1) * gap
    )

    canvas = np.full(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        canvas,
        (
            "Successful plate refinements "
            "- exact corners used for calibration"
        ),
        (gap, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    for panel_index, panel in enumerate(
        resized_panels
    ):
        row = panel_index // columns
        column = panel_index % columns

        x = (
            gap
            + column * (cell_width + gap)
        )

        y = (
            header_height
            + gap
            + row * (target_height + gap)
        )

        panel_height, panel_width = (
            panel.shape[:2]
        )

        # Center narrower panels inside their grid cell.
        x += (
            cell_width
            - panel_width
        ) // 2

        canvas[
            y:y + panel_height,
            x:x + panel_width,
        ] = panel

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        canvas,
    )

    if not success:
        raise RuntimeError(
            f"Failed saving debug panel to {output_path}"
        )



# ============================================================
# PRINTING
# ============================================================

def print_header(
    image_name: str,
    num_successful: int,
) -> None:
    print("\n" + "=" * 78)
    print("PLATE COUNT EXPERIMENT")
    print("=" * 78)

    print(f"\nImage: {image_name}")
    print(
        f"Successful refined plates: "
        f"{num_successful}"
    )

    print("\nK_ref:")
    print(K_REF)


def print_count_summary(
    row: dict,
) -> None:
    print("\n" + "-" * 78)

    print(
        f"N = {row['num_plates']} plates"
    )

    print(
        f"Subsets evaluated: "
        f"{row['num_subsets']}"
    )

    print(
        f"Valid calibrations: "
        f"{row['num_valid_subsets']}"
    )

    print(
        "\nMean |focal error| [%]: "
        f"{row['mean_focal_abs_error_pct']:.3f}"
        f" ± "
        f"{row['std_focal_abs_error_pct']:.3f}"
    )

    print(
        "Mean |principal-point error| [px]: "
        f"{row['mean_principal_abs_error_px']:.3f}"
        f" ± "
        f"{row['std_principal_abs_error_px']:.3f}"
    )

    print(
        "Median condition number: "
        f"{row['median_condition_number']:.6e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image_name = IMAGE_PATH.stem

    case_dir = (
        OUTPUT_ROOT
        / image_name
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Run the normal Part 2 pipeline ONCE.
    # --------------------------------------------------------
    result = calibrate_from_image(
        image_path=IMAGE_PATH,
        use_cached_json=True,
        method=METHOD,
        debug_refinement=False,
        save_visualizations=True,
    )

    refinement_results = (
        result["refinement_results"]
    )

    image = np.asarray(
        result["image"]
    )

    successful = extract_successful_refinements(
        refinement_results
    )

    num_successful = len(successful)

    debug_panel_path = (
        case_dir
        / "detected_plates_debug.jpg"
    )

    save_detected_plate_debug_panel(
        image=image,
        successful_refinements=successful,
        output_path=debug_panel_path,
    )

    print(
        "\nSaved detected-plate debug panel to:"
    )
    print(debug_panel_path)

    print_header(
        IMAGE_PATH.name,
        num_successful,
    )

    # --------------------------------------------------------
    # 2. Abort if the image does not give us > 3 good plates.
    # --------------------------------------------------------
    if num_successful < MIN_SUCCESSFUL_PLATES:
        print(
            "\nSTOPPING EXPERIMENT:"
        )

        print(
            f"Only {num_successful} successful plates "
            f"were recovered."
        )

        print(
            f"At least {MIN_SUCCESSFUL_PLATES} are needed "
            f"to compare calibration using different "
            f"numbers of plates."
        )

        return

    # Preserve explicit plate IDs for reporting.
    plate_ids = [
        get_plate_id(
            refinement,
            fallback_index=i,
        )
        for i, refinement
        in enumerate(successful)
    ]

    print(
        "\nSuccessful plate IDs:",
        plate_ids,
    )

    # --------------------------------------------------------
    # 3. Evaluate EVERY subset for every N:
    #
    #       N = 3, 4, ..., number_of_successful_plates
    #
    # This prevents the result from depending on one arbitrary
    # choice such as "the first 3 plates".
    # --------------------------------------------------------
    subset_rows: list[dict] = []
    summary_rows: list[dict] = []

    for num_plates in range(
        3,
        num_successful + 1,
    ):
        current_rows: list[dict] = []

        subset_indices_list = list(
            combinations(
                range(num_successful),
                num_plates,
            )
        )

        print(
            "\n"
            + "#" * 78
        )

        print(
            f"TESTING N = {num_plates} "
            f"({len(subset_indices_list)} subsets)"
        )

        print(
            "#" * 78
        )

        for subset_number, subset_indices in enumerate(
            subset_indices_list,
            start=1,
        ):
            subset = [
                successful[i]
                for i in subset_indices
            ]

            subset_plate_ids = [
                plate_ids[i]
                for i in subset_indices
            ]

            try:
                calibration = calibrate_from_subset(
                    subset
                )

                K_est = np.asarray(
                    calibration["K"],
                    dtype=np.float64,
                )

                condition_number = float(
                    calibration[
                        "condition_number"
                    ]
                )

                metrics = summarize_K(
                    K_est,
                    K_REF,
                )

                row = {
                    "image":
                        IMAGE_PATH.name,

                    "num_plates":
                        num_plates,

                    "subset_number":
                        subset_number,

                    "plate_ids":
                        "-".join(
                            str(x)
                            for x in subset_plate_ids
                        ),

                    "success":
                        True,

                    "condition_number":
                        condition_number,

                    "fx":
                        metrics["fx"],

                    "fy":
                        metrics["fy"],

                    "cx":
                        metrics["cx"],

                    "cy":
                        metrics["cy"],

                    "skew":
                        metrics["skew"],

                    "fx_error_pct":
                        metrics["fx_error_pct"],

                    "fy_error_pct":
                        metrics["fy_error_pct"],

                    "cx_error_px":
                        metrics["cx_error_px"],

                    "cy_error_px":
                        metrics["cy_error_px"],

                    "mean_focal_abs_error_pct":
                        mean_focal_abs_error_pct(
                            metrics
                        ),

                    "mean_principal_abs_error_px":
                        mean_principal_point_abs_error_px(
                            metrics
                        ),
                }

                print(
                    f"N={num_plates} | "
                    f"plates={subset_plate_ids} | "
                    f"focal err="
                    f"{row['mean_focal_abs_error_pct']:.2f}% | "
                    f"pp err="
                    f"{row['mean_principal_abs_error_px']:.2f}px | "
                    f"cond="
                    f"{condition_number:.3e}"
                )

            except Exception as error:
                row = {
                    "image":
                        IMAGE_PATH.name,

                    "num_plates":
                        num_plates,

                    "subset_number":
                        subset_number,

                    "plate_ids":
                        "-".join(
                            str(x)
                            for x in subset_plate_ids
                        ),

                    "success":
                        False,

                    "condition_number":
                        np.nan,

                    "fx":
                        np.nan,

                    "fy":
                        np.nan,

                    "cx":
                        np.nan,

                    "cy":
                        np.nan,

                    "skew":
                        np.nan,

                    "fx_error_pct":
                        np.nan,

                    "fy_error_pct":
                        np.nan,

                    "cx_error_px":
                        np.nan,

                    "cy_error_px":
                        np.nan,

                    "mean_focal_abs_error_pct":
                        np.nan,

                    "mean_principal_abs_error_px":
                        np.nan,
                }

                print(
                    f"N={num_plates} | "
                    f"plates={subset_plate_ids} | "
                    f"FAILED: {error}"
                )

            subset_rows.append(row)

            if row["success"]:
                current_rows.append(row)

        # ----------------------------------------------------
        # 4. Aggregate all subsets for this N.
        # ----------------------------------------------------
        if not current_rows:
            print(
                f"\nNo valid calibration for N={num_plates}"
            )
            continue

        focal_errors = np.asarray(
            [
                row[
                    "mean_focal_abs_error_pct"
                ]
                for row in current_rows
            ],
            dtype=np.float64,
        )

        principal_errors = np.asarray(
            [
                row[
                    "mean_principal_abs_error_px"
                ]
                for row in current_rows
            ],
            dtype=np.float64,
        )

        condition_numbers = np.asarray(
            [
                row["condition_number"]
                for row in current_rows
            ],
            dtype=np.float64,
        )

        summary_row = {
            "image":
                IMAGE_PATH.name,

            "num_plates":
                num_plates,

            "num_subsets":
                len(subset_indices_list),

            "num_valid_subsets":
                len(current_rows),

            "mean_focal_abs_error_pct":
                float(
                    np.mean(focal_errors)
                ),

            "std_focal_abs_error_pct":
                float(
                    np.std(focal_errors)
                ),

            "min_focal_abs_error_pct":
                float(
                    np.min(focal_errors)
                ),

            "max_focal_abs_error_pct":
                float(
                    np.max(focal_errors)
                ),

            "mean_principal_abs_error_px":
                float(
                    np.mean(principal_errors)
                ),

            "std_principal_abs_error_px":
                float(
                    np.std(principal_errors)
                ),

            "min_principal_abs_error_px":
                float(
                    np.min(principal_errors)
                ),

            "max_principal_abs_error_px":
                float(
                    np.max(principal_errors)
                ),

            "mean_condition_number":
                float(
                    np.mean(condition_numbers)
                ),

            "median_condition_number":
                float(
                    np.median(condition_numbers)
                ),

            "min_condition_number":
                float(
                    np.min(condition_numbers)
                ),

            "max_condition_number":
                float(
                    np.max(condition_numbers)
                ),
        }

        summary_rows.append(
            summary_row
        )

        print_count_summary(
            summary_row
        )

    # --------------------------------------------------------
    # 5. Save experiment outputs.
    # --------------------------------------------------------
    subset_csv = (
        case_dir
        / "all_subsets.csv"
    )

    summary_csv = (
        case_dir
        / "summary_by_plate_count.csv"
    )

    save_csv(
        subset_rows,
        subset_csv,
    )

    save_csv(
        summary_rows,
        summary_csv,
    )

    print(
        "\n\nSaved detailed subset results to:"
    )

    print(subset_csv)

    print(
        "\nSaved plate-count summary to:"
    )

    print(summary_csv)


if __name__ == "__main__":
    main()
