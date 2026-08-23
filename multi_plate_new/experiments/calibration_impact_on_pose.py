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

from calibration.calibration_utils import load_calibration_results
from multi_plate_new.pipeline import calibrate_from_image
from single_plate_pose.pose_estimation import estimate_plate_pose_from_K


# ============================================================
# CONFIG
# ============================================================

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results.npz"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "cars_photos"
    / "multi_cars"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "multi_plate_new"
    / "outputs"
    / "calibration_impact_on_pose"
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

# Keep this consistent with the Part 1 experiments.
USE_DISTORTION = False


# ============================================================
# HELPERS
# ============================================================

def signed_angle_difference_deg(
    angle_a_deg: float,
    angle_b_deg: float,
) -> float:
    """
    Return angle_a - angle_b wrapped to [-180, 180).
    """
    return float(
        (angle_a_deg - angle_b_deg + 180.0)
        % 360.0
        - 180.0
    )


def rotation_difference_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
) -> float:
    """
    Geodesic angle between two rotation matrices.
    """
    R_delta = R_a @ R_b.T

    cos_theta = (
        np.trace(R_delta) - 1.0
    ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cos_theta)
        )
    )


def load_reference_intrinsics():
    """
    Load the chessboard calibration used as the reference K.

    Important:
    K_ref is used only for evaluation, not by the Part 2
    calibration pipeline.
    """
    calibration = load_calibration_results(
        CALIBRATION_PATH
    )

    K_ref = np.asarray(
        calibration["K_cv"],
        dtype=np.float64,
    )

    dist_coeffs_ref = np.asarray(
        calibration["dist_coeffs"],
        dtype=np.float64,
    )

    if not USE_DISTORTION:
        dist_coeffs_ref = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

    return K_ref, dist_coeffs_ref


def evaluate_plate(
    *,
    case_name: str,
    plate_index: int,
    image_points: np.ndarray,
    K_est: np.ndarray,
    K_ref: np.ndarray,
    dist_coeffs_ref: np.ndarray,
) -> dict:
    """
    Estimate the SAME plate pose twice using the SAME detected corners:

        1. K_ref  -> reference pose
        2. K_est  -> pose using the Part 2 estimated intrinsics

    Therefore, differences isolate the downstream effect of the
    intrinsic-calibration error.
    """

    zero_distortion = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    pose_ref = estimate_plate_pose_from_K(
        image_points=image_points,
        K=K_ref,
        dist_coeffs=dist_coeffs_ref,
    )

    pose_est = estimate_plate_pose_from_K(
        image_points=image_points,
        K=K_est,
        dist_coeffs=zero_distortion,
    )

    distance_ref_cm = float(
        pose_ref["distance_to_center_cm"]
    )

    distance_est_cm = float(
        pose_est["distance_to_center_cm"]
    )

    yaw_ref_deg = float(
        pose_ref["yaw_deg"]
    )

    yaw_est_deg = float(
        pose_est["yaw_deg"]
    )

    delta_distance_cm = (
        distance_est_cm - distance_ref_cm
    )

    delta_yaw_deg = signed_angle_difference_deg(
        yaw_est_deg,
        yaw_ref_deg,
    )

    t_ref = np.asarray(
        pose_ref["tvec"],
        dtype=np.float64,
    ).reshape(3)

    t_est = np.asarray(
        pose_est["tvec"],
        dtype=np.float64,
    ).reshape(3)

    translation_difference_cm = float(
        np.linalg.norm(
            t_est - t_ref
        )
    )

    rotation_difference = rotation_difference_deg(
        pose_est["R"],
        pose_ref["R"],
    )

    return {
        "case": case_name,
        "plate_index": plate_index,

        "distance_ref_cm": distance_ref_cm,
        "distance_est_cm": distance_est_cm,
        "delta_distance_cm": delta_distance_cm,
        "abs_delta_distance_cm": abs(delta_distance_cm),

        "yaw_ref_deg": yaw_ref_deg,
        "yaw_est_deg": yaw_est_deg,
        "delta_yaw_deg": delta_yaw_deg,
        "abs_delta_yaw_deg": abs(delta_yaw_deg),

        "translation_difference_cm":
            translation_difference_cm,

        "rotation_difference_deg":
            rotation_difference,
    }


def print_case_summary(
    case_name: str,
    K_est: np.ndarray,
    rows: list[dict],
) -> None:
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"CALIBRATION IMPACT ON POSE — {case_name}"
    )

    print(
        "=" * 70
    )

    print("\nEstimated K:")
    print(K_est)

    print("\nPer-plate comparison:")

    for row in rows:
        print(
            f"\nPlate {row['plate_index']}:"
        )

        print(
            "  Distance [cm] "
            f"K_ref={row['distance_ref_cm']:.2f}, "
            f"K_est={row['distance_est_cm']:.2f}, "
            f"delta={row['delta_distance_cm']:+.2f}"
        )

        print(
            "  Yaw [deg]     "
            f"K_ref={row['yaw_ref_deg']:+.2f}, "
            f"K_est={row['yaw_est_deg']:+.2f}, "
            f"delta={row['delta_yaw_deg']:+.2f}"
        )

        print(
            "  ||delta t||   "
            f"{row['translation_difference_cm']:.2f} cm"
        )

        print(
            "  delta R       "
            f"{row['rotation_difference_deg']:.2f} deg"
        )

    mean_abs_distance = float(
        np.mean(
            [
                row["abs_delta_distance_cm"]
                for row in rows
            ]
        )
    )

    max_abs_distance = float(
        np.max(
            [
                row["abs_delta_distance_cm"]
                for row in rows
            ]
        )
    )

    mean_abs_yaw = float(
        np.mean(
            [
                row["abs_delta_yaw_deg"]
                for row in rows
            ]
        )
    )

    max_abs_yaw = float(
        np.max(
            [
                row["abs_delta_yaw_deg"]
                for row in rows
            ]
        )
    )

    mean_translation_difference = float(
        np.mean(
            [
                row["translation_difference_cm"]
                for row in rows
            ]
        )
    )

    mean_rotation_difference = float(
        np.mean(
            [
                row["rotation_difference_deg"]
                for row in rows
            ]
        )
    )

    print("\nCASE SUMMARY")
    print(
        f"  Mean |delta distance|: "
        f"{mean_abs_distance:.2f} cm"
    )
    print(
        f"  Max  |delta distance|: "
        f"{max_abs_distance:.2f} cm"
    )
    print(
        f"  Mean |delta yaw|     : "
        f"{mean_abs_yaw:.2f} deg"
    )
    print(
        f"  Max  |delta yaw|     : "
        f"{max_abs_yaw:.2f} deg"
    )
    print(
        f"  Mean ||delta t||     : "
        f"{mean_translation_difference:.2f} cm"
    )
    print(
        f"  Mean delta R         : "
        f"{mean_rotation_difference:.2f} deg"
    )


def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case",
        "plate_index",
        "distance_ref_cm",
        "distance_est_cm",
        "delta_distance_cm",
        "abs_delta_distance_cm",
        "yaw_ref_deg",
        "yaw_est_deg",
        "delta_yaw_deg",
        "abs_delta_yaw_deg",
        "translation_difference_cm",
        "rotation_difference_deg",
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
        writer.writerows(rows)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    K_ref, dist_coeffs_ref = (
        load_reference_intrinsics()
    )

    print(
        "\n========== REFERENCE INTRINSICS =========="
    )
    print(K_ref)

    if USE_DISTORTION:
        print("\nUsing reference distortion:")
        print(dist_coeffs_ref.reshape(-1))
    else:
        print(
            "\nDistortion disabled for both "
            "reference and estimated-K pose calculations."
        )

    all_rows: list[dict] = []

    for case in CASES:
        case_name = case["name"]
        image_path = case["image"]

        print(
            "\n\n"
            + "#" * 80
        )
        print(
            f"RUNNING {case_name}: {image_path.name}"
        )
        print(
            "#" * 80
        )

        # ----------------------------------------------------
        # Part 2: estimate K from this multi-plate image.
        # ----------------------------------------------------
        calibration_result = calibrate_from_image(
            image_path=image_path,
            use_cached_json=True,
            method=METHOD,
            debug_refinement=False,
        )

        K_est = np.asarray(
            calibration_result["K_est"],
            dtype=np.float64,
        )

        successful_refinements = (
            calibration_result[
                "successful_refinements"
            ]
        )

        case_rows: list[dict] = []

        # ----------------------------------------------------
        # Part 1: estimate each plate pose twice.
        # Same corners, only K changes.
        # ----------------------------------------------------
        for refinement in successful_refinements:
            row = evaluate_plate(
                case_name=case_name,
                plate_index=int(
                    refinement["index"]
                ),
                image_points=np.asarray(
                    refinement["image_points"],
                    dtype=np.float64,
                ),
                K_est=K_est,
                K_ref=K_ref,
                dist_coeffs_ref=dist_coeffs_ref,
            )

            case_rows.append(row)
            all_rows.append(row)

        print_case_summary(
            case_name,
            K_est,
            case_rows,
        )

    # --------------------------------------------------------
    # Save all per-plate results for later presentation plots.
    # --------------------------------------------------------
    csv_path = (
        OUTPUT_DIR
        / "calibration_impact_on_pose.csv"
    )

    save_csv(
        all_rows,
        csv_path,
    )

    print(
        "\n\nSaved experiment results to:"
    )
    print(csv_path)


if __name__ == "__main__":
    main()
