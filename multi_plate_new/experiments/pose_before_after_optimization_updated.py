from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from single_plate_pose.pose_estimation import (
    estimate_plate_pose_from_K,
)


@dataclass
class PlatePoseComparison:
    plate_index: int

    mean_corner_shift_px: float
    max_corner_shift_px: float

    original_distance_cm: float
    optimized_distance_cm: float
    distance_change_cm: float

    original_yaw_deg: float
    optimized_yaw_deg: float
    yaw_change_deg: float

    original_normal: np.ndarray
    optimized_normal: np.ndarray


def _successful_results_by_index(
        refinement_results: list[dict],
) -> dict[int, dict]:
    """
    Keep only successful refinement results that contain valid corners,
    and index them by plate index.
    """

    results_by_index: dict[int, dict] = {}

    for position, result in enumerate(refinement_results):
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

        results_by_index[plate_index] = result

    return results_by_index


def compare_pose_before_after(
        original_refinement_results: list[dict],
        optimized_refinement_results: list[dict],
        K_ref: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
) -> list[PlatePoseComparison]:
    """
    Compare pose estimation before and after corner optimization.

    Important:
    Both pose estimations use exactly the same camera intrinsic matrix K_ref.

    Therefore, the only changing input is the detected corner location.
    """

    K_ref = np.asarray(
        K_ref,
        dtype=np.float64,
    )

    original_by_index = _successful_results_by_index(
        original_refinement_results
    )

    optimized_by_index = _successful_results_by_index(
        optimized_refinement_results
    )

    common_indices = sorted(
        set(original_by_index.keys())
        & set(optimized_by_index.keys())
    )

    comparisons: list[PlatePoseComparison] = []

    for plate_index in common_indices:
        original_corners = np.asarray(
            original_by_index[
                plate_index
            ]["image_points"],
            dtype=np.float64,
        )

        optimized_corners = np.asarray(
            optimized_by_index[
                plate_index
            ]["image_points"],
            dtype=np.float64,
        )

        corner_shift_magnitudes = np.linalg.norm(
            optimized_corners - original_corners,
            axis=1,
        )

        original_pose = estimate_plate_pose_from_K(
            image_points=original_corners,
            K=K_ref,
            dist_coeffs=dist_coeffs,
        )

        optimized_pose = estimate_plate_pose_from_K(
            image_points=optimized_corners,
            K=K_ref,
            dist_coeffs=dist_coeffs,
        )

        original_distance_cm = float(
            original_pose["distance_to_center_cm"]
        )

        optimized_distance_cm = float(
            optimized_pose["distance_to_center_cm"]
        )

        original_yaw_deg = float(
            original_pose["yaw_deg"]
        )

        optimized_yaw_deg = float(
            optimized_pose["yaw_deg"]
        )

        comparisons.append(
            PlatePoseComparison(
                plate_index=plate_index,

                mean_corner_shift_px=float(np.mean(corner_shift_magnitudes)),
                max_corner_shift_px=float(np.max(corner_shift_magnitudes)),

                original_distance_cm=original_distance_cm,
                optimized_distance_cm=optimized_distance_cm,
                distance_change_cm=(
                    optimized_distance_cm
                    - original_distance_cm
                ),

                original_yaw_deg=original_yaw_deg,
                optimized_yaw_deg=optimized_yaw_deg,
                yaw_change_deg=(
                    optimized_yaw_deg
                    - original_yaw_deg
                ),

                original_normal=np.asarray(
                    original_pose["normal_camera"],
                    dtype=np.float64,
                ).copy(),

                optimized_normal=np.asarray(
                    optimized_pose["normal_camera"],
                    dtype=np.float64,
                ).copy(),
            )
        )

    return comparisons



def save_pose_comparison_csv(
        comparisons: list[PlatePoseComparison],
        output_csv: str | Path,
) -> Path:
    """
    Save presentation-friendly per-plate results and one summary row.

    K_ref is fixed in both pose estimations, so the pose deltas measure
    only the direct sensitivity to the optimized subpixel corner shifts.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    distance_changes = np.asarray(
        [abs(c.distance_change_cm) for c in comparisons],
        dtype=np.float64,
    )
    yaw_changes = np.asarray(
        [abs(c.yaw_change_deg) for c in comparisons],
        dtype=np.float64,
    )
    mean_corner_shifts = np.asarray(
        [c.mean_corner_shift_px for c in comparisons],
        dtype=np.float64,
    )
    max_corner_shifts = np.asarray(
        [c.max_corner_shift_px for c in comparisons],
        dtype=np.float64,
    )

    fieldnames = [
        "row_type",
        "plate_index",
        "mean_corner_shift_px",
        "max_corner_shift_px",
        "original_distance_cm",
        "optimized_distance_cm",
        "distance_change_cm",
        "abs_distance_change_cm",
        "original_yaw_deg",
        "optimized_yaw_deg",
        "yaw_change_deg",
        "abs_yaw_change_deg",
        "mean_abs_distance_change_cm",
        "max_abs_distance_change_cm",
        "mean_abs_yaw_change_deg",
        "max_abs_yaw_change_deg",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in comparisons:
            writer.writerow({
                "row_type": "plate",
                "plate_index": c.plate_index,
                "mean_corner_shift_px": c.mean_corner_shift_px,
                "max_corner_shift_px": c.max_corner_shift_px,
                "original_distance_cm": c.original_distance_cm,
                "optimized_distance_cm": c.optimized_distance_cm,
                "distance_change_cm": c.distance_change_cm,
                "abs_distance_change_cm": abs(c.distance_change_cm),
                "original_yaw_deg": c.original_yaw_deg,
                "optimized_yaw_deg": c.optimized_yaw_deg,
                "yaw_change_deg": c.yaw_change_deg,
                "abs_yaw_change_deg": abs(c.yaw_change_deg),
                "mean_abs_distance_change_cm": "",
                "max_abs_distance_change_cm": "",
                "mean_abs_yaw_change_deg": "",
                "max_abs_yaw_change_deg": "",
            })

        if comparisons:
            writer.writerow({
                "row_type": "summary",
                "plate_index": "",
                "mean_corner_shift_px": float(np.mean(mean_corner_shifts)),
                "max_corner_shift_px": float(np.max(max_corner_shifts)),
                "original_distance_cm": "",
                "optimized_distance_cm": "",
                "distance_change_cm": "",
                "abs_distance_change_cm": "",
                "original_yaw_deg": "",
                "optimized_yaw_deg": "",
                "yaw_change_deg": "",
                "abs_yaw_change_deg": "",
                "mean_abs_distance_change_cm": float(np.mean(distance_changes)),
                "max_abs_distance_change_cm": float(np.max(distance_changes)),
                "mean_abs_yaw_change_deg": float(np.mean(yaw_changes)),
                "max_abs_yaw_change_deg": float(np.max(yaw_changes)),
            })

    return output_csv

def print_pose_comparison(
        comparisons: list[PlatePoseComparison],
) -> None:
    print(
        "\n========== POSE BEFORE / AFTER CORNER OPTIMIZATION =========="
    )

    if not comparisons:
        print("No comparable plates found.")
        print(
            "============================================================\n"
        )
        return

    for comparison in comparisons:
        print(
            f"\nPlate {comparison.plate_index}"
        )

        print(
            "  Corner shift:"
        )
        print(
            f"    mean      : "
            f"{comparison.mean_corner_shift_px:.4f} px"
        )
        print(
            f"    max       : "
            f"{comparison.max_corner_shift_px:.4f} px"
        )

        print(
            "  Distance to center:"
        )

        print(
            f"    original : "
            f"{comparison.original_distance_cm:.2f} cm"
        )

        print(
            f"    optimized: "
            f"{comparison.optimized_distance_cm:.2f} cm"
        )

        print(
            f"    change   : "
            f"{comparison.distance_change_cm:+.2f} cm"
        )

        print(
            "  Yaw:"
        )

        print(
            f"    original : "
            f"{comparison.original_yaw_deg:.3f} deg"
        )

        print(
            f"    optimized: "
            f"{comparison.optimized_yaw_deg:.3f} deg"
        )

        print(
            f"    change   : "
            f"{comparison.yaw_change_deg:+.3f} deg"
        )

        print(
            "  Plate normal:"
        )

        print(
            "    original : "
            f"{comparison.original_normal}"
        )

        print(
            "    optimized: "
            f"{comparison.optimized_normal}"
        )

    distance_changes = np.array(
        [
            abs(c.distance_change_cm)
            for c in comparisons
        ],
        dtype=np.float64,
    )

    yaw_changes = np.array(
        [
            abs(c.yaw_change_deg)
            for c in comparisons
        ],
        dtype=np.float64,
    )

    print(
        "\n---------- SUMMARY ----------"
    )

    print(
        f"Mean absolute distance change: "
        f"{np.mean(distance_changes):.3f} cm"
    )

    print(
        f"Max absolute distance change : "
        f"{np.max(distance_changes):.3f} cm"
    )

    print(
        f"Mean absolute yaw change     : "
        f"{np.mean(yaw_changes):.4f} deg"
    )

    print(
        f"Max absolute yaw change      : "
        f"{np.max(yaw_changes):.4f} deg"
    )

    print(
        "================================"
        "============================\n"
    )