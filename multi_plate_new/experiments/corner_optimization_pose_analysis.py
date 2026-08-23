from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from single_plate_pose.pose_estimation import (
    estimate_plate_pose_from_K,
)


@dataclass
class PlatePoseComparison:
    plate_index: int

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