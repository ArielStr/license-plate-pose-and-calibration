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
from multi_plate_new.experiments.corner_optimization_to_reference import (
    CornerOptimizationConfig,
    optimize_corners_to_reference,
)
from multi_plate_new.experiments.corner_optimization_pose_analysis import (
    compare_pose_before_after,
)


# ============================================================
# CONFIG
# ============================================================

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
    / "direct_pose_sensitivity"
)

# Same cases used in the corner-sensitivity experiment.
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

# We intentionally use the same K in both pose estimates.
DIST_COEFFS = None


# ============================================================
# HELPERS
# ============================================================

def _successful_by_index(
    refinement_results: list[dict],
) -> dict[int, dict]:
    output: dict[int, dict] = {}

    for position, result in enumerate(
        refinement_results
    ):
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

        plate_index = int(
            result.get(
                "index",
                position,
            )
        )

        output[plate_index] = result

    return output


def corner_shift_stats(
    original_refinement_results: list[dict],
    optimized_refinement_results: list[dict],
) -> dict[int, dict]:
    original = _successful_by_index(
        original_refinement_results
    )

    optimized = _successful_by_index(
        optimized_refinement_results
    )

    common = sorted(
        set(original.keys())
        & set(optimized.keys())
    )

    stats: dict[int, dict] = {}

    for plate_index in common:
        p0 = np.asarray(
            original[
                plate_index
            ]["image_points"],
            dtype=np.float64,
        )

        p1 = np.asarray(
            optimized[
                plate_index
            ]["image_points"],
            dtype=np.float64,
        )

        distances = np.linalg.norm(
            p1 - p0,
            axis=1,
        )

        stats[plate_index] = {
            "mean_shift_px": float(
                np.mean(distances)
            ),

            "max_shift_px": float(
                np.max(distances)
            ),
        }

    return stats


def save_rows(
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
# PRINTING
# ============================================================

def print_case_results(
    case_name: str,
    rows: list[dict],
) -> None:
    print(
        "\n"
        + "=" * 76
    )

    print(
        f"DIRECT POSE SENSITIVITY — {case_name}"
    )

    print(
        "=" * 76
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  K_ref is fixed in BOTH pose estimations."
    )

    print(
        "  Only the detected corner locations change."
    )

    for row in rows:
        print(
            f"\nPlate {row['plate_index']}"
        )

        print(
            f"  Mean corner shift : "
            f"{row['mean_corner_shift_px']:.4f} px"
        )

        print(
            f"  Max corner shift  : "
            f"{row['max_corner_shift_px']:.4f} px"
        )

        print(
            f"  Distance          : "
            f"{row['original_distance_cm']:.2f}"
            f" -> "
            f"{row['optimized_distance_cm']:.2f} cm"
        )

        print(
            f"  Delta distance    : "
            f"{row['distance_change_cm']:+.4f} cm"
        )

        print(
            f"  Yaw               : "
            f"{row['original_yaw_deg']:.4f}"
            f" -> "
            f"{row['optimized_yaw_deg']:.4f} deg"
        )

        print(
            f"  Delta yaw         : "
            f"{row['yaw_change_deg']:+.5f} deg"
        )

    if not rows:
        return

    mean_shifts = np.asarray(
        [
            row["mean_corner_shift_px"]
            for row in rows
        ],
        dtype=np.float64,
    )

    max_shifts = np.asarray(
        [
            row["max_corner_shift_px"]
            for row in rows
        ],
        dtype=np.float64,
    )

    distance_changes = np.asarray(
        [
            abs(row["distance_change_cm"])
            for row in rows
        ],
        dtype=np.float64,
    )

    yaw_changes = np.asarray(
        [
            abs(row["yaw_change_deg"])
            for row in rows
        ],
        dtype=np.float64,
    )

    print(
        "\n---------- CASE SUMMARY ----------"
    )

    print(
        f"Mean corner shift             : "
        f"{np.mean(mean_shifts):.4f} px"
    )

    print(
        f"Maximum corner shift          : "
        f"{np.max(max_shifts):.4f} px"
    )

    print(
        f"Mean absolute distance change : "
        f"{np.mean(distance_changes):.4f} cm"
    )

    print(
        f"Max absolute distance change  : "
        f"{np.max(distance_changes):.4f} cm"
    )

    print(
        f"Mean absolute yaw change      : "
        f"{np.mean(yaw_changes):.5f} deg"
    )

    print(
        f"Max absolute yaw change       : "
        f"{np.max(yaw_changes):.5f} deg"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_plate_rows: list[dict] = []
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
        # 1. Run the normal multi-plate pipeline.
        # ----------------------------------------------------
        baseline = calibrate_from_image(
            image_path=image_path,
            use_cached_json=True,
            method=METHOD,
            debug_refinement=False,
            save_visualizations=False,
        )

        original_refinements = (
            baseline["refinement_results"]
        )

        # ----------------------------------------------------
        # 2. Optimize the corners toward K_ref exactly as in
        #    the corner-sensitivity experiment.
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
        # 3. Pose before/after using the SAME K_ref.
        # ----------------------------------------------------
        comparisons = compare_pose_before_after(
            original_refinement_results=
                original_refinements,
            optimized_refinement_results=
                optimized_refinements,
            K_ref=K_REF,
            dist_coeffs=DIST_COEFFS,
        )

        shift_stats = corner_shift_stats(
            original_refinement_results=
                original_refinements,
            optimized_refinement_results=
                optimized_refinements,
        )

        case_rows: list[dict] = []

        for comparison in comparisons:
            stats = shift_stats[
                comparison.plate_index
            ]

            row = {
                "case":
                    case_name,

                "image":
                    image_path.name,

                "plate_index":
                    comparison.plate_index,

                "mean_corner_shift_px":
                    stats["mean_shift_px"],

                "max_corner_shift_px":
                    stats["max_shift_px"],

                "original_distance_cm":
                    comparison.
                    original_distance_cm,

                "optimized_distance_cm":
                    comparison.
                    optimized_distance_cm,

                "distance_change_cm":
                    comparison.
                    distance_change_cm,

                "abs_distance_change_cm":
                    abs(
                        comparison.
                        distance_change_cm
                    ),

                "original_yaw_deg":
                    comparison.
                    original_yaw_deg,

                "optimized_yaw_deg":
                    comparison.
                    optimized_yaw_deg,

                "yaw_change_deg":
                    comparison.
                    yaw_change_deg,

                "abs_yaw_change_deg":
                    abs(
                        comparison.
                        yaw_change_deg
                    ),
            }

            case_rows.append(row)
            all_plate_rows.append(row)

        print_case_results(
            case_name,
            case_rows,
        )

        # ----------------------------------------------------
        # 4. Per-case summary for the presentation.
        # ----------------------------------------------------
        if case_rows:
            mean_shifts = np.asarray(
                [
                    row[
                        "mean_corner_shift_px"
                    ]
                    for row in case_rows
                ],
                dtype=np.float64,
            )

            max_shifts = np.asarray(
                [
                    row[
                        "max_corner_shift_px"
                    ]
                    for row in case_rows
                ],
                dtype=np.float64,
            )

            distance_changes = np.asarray(
                [
                    row[
                        "abs_distance_change_cm"
                    ]
                    for row in case_rows
                ],
                dtype=np.float64,
            )

            yaw_changes = np.asarray(
                [
                    row[
                        "abs_yaw_change_deg"
                    ]
                    for row in case_rows
                ],
                dtype=np.float64,
            )

            summary_rows.append(
                {
                    "case":
                        case_name,

                    "image":
                        image_path.name,

                    "num_plates":
                        len(case_rows),

                    "mean_corner_shift_px":
                        float(
                            np.mean(
                                mean_shifts
                            )
                        ),

                    "max_corner_shift_px":
                        float(
                            np.max(
                                max_shifts
                            )
                        ),

                    "mean_abs_distance_change_cm":
                        float(
                            np.mean(
                                distance_changes
                            )
                        ),

                    "max_abs_distance_change_cm":
                        float(
                            np.max(
                                distance_changes
                            )
                        ),

                    "mean_abs_yaw_change_deg":
                        float(
                            np.mean(
                                yaw_changes
                            )
                        ),

                    "max_abs_yaw_change_deg":
                        float(
                            np.max(
                                yaw_changes
                            )
                        ),
                }
            )

    # --------------------------------------------------------
    # 5. Save final CSVs.
    # --------------------------------------------------------
    detailed_path = (
        OUTPUT_ROOT
        / "direct_pose_sensitivity_per_plate.csv"
    )

    summary_path = (
        OUTPUT_ROOT
        / "direct_pose_sensitivity_summary.csv"
    )

    save_rows(
        all_plate_rows,
        detailed_path,
    )

    save_rows(
        summary_rows,
        summary_path,
    )

    print(
        "\n\n"
        + "=" * 76
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        "\nDetailed results:"
    )

    print(
        detailed_path
    )

    print(
        "\nSummary results:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()
