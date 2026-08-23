from __future__ import annotations

from pathlib import Path
import csv
import sys

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
from multi_plate_new.experiments.corner_optimization_to_reference import (
    CornerOptimizationConfig,
    optimize_corners_to_reference,
)
from multi_plate_new.visualization.corner_sensitivity import (
    draw_corner_shift_overview,
    draw_corner_shift_zoom_grid,
)


# ============================================================
# CONFIG
# ============================================================

# Chessboard calibration reference used throughout the project.
K_REF = np.array(
    [
        [2816.53888, 0.0, 1498.87479],
        [0.0, 2845.88995, 2051.34557],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "cars_photos"
    / "multi_cars"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "multi_plate_new"
    / "outputs"
    / "corner_sensitivity"
)

CASES = [
    {
        "name": "case_1",
        "image": IMAGE_DIR / "IMG_4375.jpeg",
    },
    {
        "name": "case_2",
        "image": IMAGE_DIR / "IMG_4349.jpeg",
    },
    {
        "name": "case_3",
        "image": IMAGE_DIR / "IMG_4443.jpeg",
    },
]

METHOD = "robust_mask_lines"

OPTIMIZATION_CONFIG = CornerOptimizationConfig(
    max_shift_px=10.0,
    movement_weight=1e-3,
    max_function_evaluations=2000,
    verbose=0,
)


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

        "fx_error_pct": relative_error_percent(
            float(K[0, 0]),
            float(K_ref[0, 0]),
        ),

        "fy_error_pct": relative_error_percent(
            float(K[1, 1]),
            float(K_ref[1, 1]),
        ),

        "cx_error_px": principal_point_error_px(
            float(K[0, 2]),
            float(K_ref[0, 2]),
        ),

        "cy_error_px": principal_point_error_px(
            float(K[1, 2]),
            float(K_ref[1, 2]),
        ),
    }


def recover_K_from_refinements(
    refinement_results: list[dict],
) -> dict:
    homographies = compute_all_homographies(
        refinement_results
    )

    return estimate_intrinsics_from_homographies(
        homographies
    )


# ============================================================
# CSV
# ============================================================

def save_corner_csv(
    case_name: str,
    optimization_result,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case",
        "plate_index",
        "corner_index",
        "original_x",
        "original_y",
        "optimized_x",
        "optimized_y",
        "dx_px",
        "dy_px",
        "shift_px",
    ]

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

        for plate in optimization_result.plate_results:
            for corner_index in range(4):
                original = (
                    plate.original_corners[
                        corner_index
                    ]
                )

                optimized = (
                    plate.optimized_corners[
                        corner_index
                    ]
                )

                shift = (
                    plate.shifts[
                        corner_index
                    ]
                )

                writer.writerow(
                    {
                        "case": case_name,
                        "plate_index":
                            plate.plate_index,
                        "corner_index":
                            corner_index,

                        "original_x":
                            float(original[0]),
                        "original_y":
                            float(original[1]),

                        "optimized_x":
                            float(optimized[0]),
                        "optimized_y":
                            float(optimized[1]),

                        "dx_px":
                            float(shift[0]),
                        "dy_px":
                            float(shift[1]),

                        "shift_px":
                            float(
                                plate.shift_distances[
                                    corner_index
                                ]
                            ),
                    }
                )


def save_summary_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

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
# PRINTING
# ============================================================

def print_case_result(
    *,
    case_name: str,
    K_before: np.ndarray,
    K_after: np.ndarray,
    before_metrics: dict,
    after_metrics: dict,
    optimization_result,
    condition_before: float,
    condition_after: float,
) -> None:
    print("\n" + "=" * 76)
    print(
        f"CORNER SENSITIVITY — {case_name}"
    )
    print("=" * 76)

    print("\nK_ref:")
    print(K_REF)

    print("\nK BEFORE:")
    print(K_before)

    print("\nK AFTER:")
    print(K_after)

    print("\nFOCAL ERROR")
    print(
        f"  fx: "
        f"{before_metrics['fx_error_pct']:+.2f}%"
        f" -> "
        f"{after_metrics['fx_error_pct']:+.2f}%"
    )

    print(
        f"  fy: "
        f"{before_metrics['fy_error_pct']:+.2f}%"
        f" -> "
        f"{after_metrics['fy_error_pct']:+.2f}%"
    )

    print("\nPRINCIPAL POINT ERROR")
    print(
        f"  cx: "
        f"{before_metrics['cx_error_px']:+.2f} px"
        f" -> "
        f"{after_metrics['cx_error_px']:+.2f} px"
    )

    print(
        f"  cy: "
        f"{before_metrics['cy_error_px']:+.2f} px"
        f" -> "
        f"{after_metrics['cy_error_px']:+.2f} px"
    )

    all_shifts = np.concatenate(
        [
            plate.shift_distances
            for plate
            in optimization_result.plate_results
        ]
    )

    print("\nCORNER MOVEMENT")
    print(
        f"  mean   : "
        f"{np.mean(all_shifts):.4f} px"
    )

    print(
        f"  median : "
        f"{np.median(all_shifts):.4f} px"
    )

    print(
        f"  max    : "
        f"{np.max(all_shifts):.4f} px"
    )

    print("\nZHANG CONSTRAINT RMS")
    print(
        f"  "
        f"{optimization_result.initial_constraint_rms:.8f}"
        f" -> "
        f"{optimization_result.final_constraint_rms:.8f}"
    )

    print("\nCONDITION NUMBER")
    print(
        f"  {condition_before:.6e}"
        f" -> "
        f"{condition_after:.6e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict] = []

    for case in CASES:
        case_name = case["name"]
        image_path = case["image"]

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            f"RUNNING {case_name}: "
            f"{image_path.name}"
        )

        print(
            "#" * 80
        )

        # ----------------------------------------------------
        # 1. Baseline Part 2 calibration.
        # ----------------------------------------------------
        baseline = calibrate_from_image(
            image_path=image_path,
            use_cached_json=True,
            method=METHOD,
            debug_refinement=False,
            save_visualizations=False,
        )

        image = baseline["image"]

        original_refinements = (
            baseline["refinement_results"]
        )

        K_before = np.asarray(
            baseline["K_est"],
            dtype=np.float64,
        )

        condition_before = float(
            baseline[
                "constraint_condition_number"
            ]
        )

        # ----------------------------------------------------
        # 2. Diagnostic optimization toward K_ref.
        # ----------------------------------------------------
        optimization = (
            optimize_corners_to_reference(
                refinement_results=
                    original_refinements,
                K_ref=K_REF,
                config=OPTIMIZATION_CONFIG,
            )
        )

        optimized_refinements = (
            optimization.
            optimized_refinement_results
        )

        # ----------------------------------------------------
        # 3. Re-estimate K after tiny corner shifts.
        # ----------------------------------------------------
        after_result = (
            recover_K_from_refinements(
                optimized_refinements
            )
        )

        K_after = np.asarray(
            after_result["K"],
            dtype=np.float64,
        )

        condition_after = float(
            after_result["condition_number"]
        )

        before_metrics = summarize_K(
            K_before,
            K_REF,
        )

        after_metrics = summarize_K(
            K_after,
            K_REF,
        )

        # ----------------------------------------------------
        # 4. Shift statistics.
        # ----------------------------------------------------
        all_shifts = np.concatenate(
            [
                plate.shift_distances
                for plate
                in optimization.plate_results
            ]
        )

        mean_shift = float(
            np.mean(all_shifts)
        )

        median_shift = float(
            np.median(all_shifts)
        )

        max_shift = float(
            np.max(all_shifts)
        )

        # ----------------------------------------------------
        # 5. Save visualizations.
        # ----------------------------------------------------
        case_dir = (
            OUTPUT_ROOT
            / case_name
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        draw_corner_shift_overview(
            image=image,
            optimization_result=optimization,
            save_path=(
                case_dir
                / "overview.jpg"
            ),
        )

        draw_corner_shift_zoom_grid(
            image=image,
            optimization_result=optimization,
            save_path=(
                case_dir
                / "corner_shift_zoom.jpg"
            ),
        )

        save_corner_csv(
            case_name=case_name,
            optimization_result=optimization,
            output_path=(
                case_dir
                / "corner_shifts.csv"
            ),
        )

        # ----------------------------------------------------
        # 6. Print and save summary.
        # ----------------------------------------------------
        print_case_result(
            case_name=case_name,
            K_before=K_before,
            K_after=K_after,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            optimization_result=optimization,
            condition_before=condition_before,
            condition_after=condition_after,
        )

        summary_rows.append(
            {
                "case": case_name,

                "mean_shift_px":
                    mean_shift,

                "median_shift_px":
                    median_shift,

                "max_shift_px":
                    max_shift,

                "constraint_rms_before":
                    optimization.
                    initial_constraint_rms,

                "constraint_rms_after":
                    optimization.
                    final_constraint_rms,

                "condition_before":
                    condition_before,

                "condition_after":
                    condition_after,

                "fx_before":
                    before_metrics["fx"],

                "fx_after":
                    after_metrics["fx"],

                "fx_ref":
                    float(K_REF[0, 0]),

                "fx_error_before_pct":
                    before_metrics[
                        "fx_error_pct"
                    ],

                "fx_error_after_pct":
                    after_metrics[
                        "fx_error_pct"
                    ],

                "fy_before":
                    before_metrics["fy"],

                "fy_after":
                    after_metrics["fy"],

                "fy_ref":
                    float(K_REF[1, 1]),

                "fy_error_before_pct":
                    before_metrics[
                        "fy_error_pct"
                    ],

                "fy_error_after_pct":
                    after_metrics[
                        "fy_error_pct"
                    ],

                "cx_before":
                    before_metrics["cx"],

                "cx_after":
                    after_metrics["cx"],

                "cx_ref":
                    float(K_REF[0, 2]),

                "cx_error_before_px":
                    before_metrics[
                        "cx_error_px"
                    ],

                "cx_error_after_px":
                    after_metrics[
                        "cx_error_px"
                    ],

                "cy_before":
                    before_metrics["cy"],

                "cy_after":
                    after_metrics["cy"],

                "cy_ref":
                    float(K_REF[1, 2]),

                "cy_error_before_px":
                    before_metrics[
                        "cy_error_px"
                    ],

                "cy_error_after_px":
                    after_metrics[
                        "cy_error_px"
                    ],

                "skew_before":
                    before_metrics["skew"],

                "skew_after":
                    after_metrics["skew"],
            }
        )

    summary_path = (
        OUTPUT_ROOT
        / "corner_sensitivity_summary.csv"
    )

    save_summary_csv(
        summary_rows,
        summary_path,
    )

    print(
        "\n\nSaved experiment summary to:"
    )

    print(summary_path)


if __name__ == "__main__":
    main()
