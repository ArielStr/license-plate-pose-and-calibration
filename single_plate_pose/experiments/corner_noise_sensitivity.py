from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from calibration.calibration_utils import load_calibration_results
from single_plate_pose.detection import get_predictions
from corner_refinement import refine_yellow_inner_corners
from single_plate_pose.pose_estimation import estimate_plate_pose_from_K


# ============================================================
# CONFIG
# ============================================================
VISUALIZATION_SCALE = 20.0
SAVE_VISUALIZATIONS = True
API_KEY = "yuT6cPYTPnHnS2qKxXQ4"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results.npz"
)

# Change this to the image taken from 3 meters.
IMAGE_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "depth"
    / "3_meter.jpeg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "debug"
    / "single_plate_pose"
    / "corner_noise_sensitivity"
)

METHOD = "robust_mask_lines"

# Ground-truth distance measured physically.
GROUND_TRUTH_DISTANCE_CM = 300.0

# Noise is applied independently to every x/y coordinate.
NOISE_SIGMAS_PX = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
]

NUM_TRIALS = 300

# Fixed seed => experiment is reproducible.
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def add_gaussian_corner_noise(
        corners: np.ndarray,
        sigma_px: float,
        rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add independent Gaussian noise to every x/y coordinate.

    Returns
    -------
    noisy_corners:
        Perturbed corners.

    noise:
        Actual (dx, dy) added to every corner.
    """

    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    noise = rng.normal(
        loc=0.0,
        scale=sigma_px,
        size=corners.shape,
    )

    noisy_corners = corners + noise

    return noisy_corners, noise


def compute_corner_displacement(
        noise: np.ndarray,
) -> tuple[float, float]:
    """
    Return:
        mean corner displacement
        maximum corner displacement
    """

    distances = np.linalg.norm(
        noise,
        axis=1,
    )

    return (
        float(np.mean(distances)),
        float(np.max(distances)),
    )


def detect_single_plate_corners(
        image: np.ndarray,
        image_path: Path,
) -> np.ndarray:
    """
    Detect one plate and return its refined yellow corners.
    """

    predictions = get_predictions(
        str(image_path),
        API_KEY,
    )

    if not predictions:
        raise RuntimeError(
            "No plate predictions were found."
        )

    print(
        f"Found {len(predictions)} plate prediction(s)"
    )

    # For the first experiment we expect one main plate.
    prediction = predictions[0]

    yellow_points, _, _ = refine_yellow_inner_corners(
        image,
        prediction,
        debug=False,
        method=METHOD,
        debug_name="corner_noise_baseline",
    )

    return np.asarray(
        yellow_points,
        dtype=np.float64,
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_noise_experiment(
        baseline_corners: np.ndarray,
        K: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """
    Perturb the detected corners and measure the resulting pose change.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    baseline_pose = estimate_plate_pose_from_K(
        image_points=baseline_corners,
        K=K,
        dist_coeffs=None,
    )

    baseline_distance_cm = float(
        baseline_pose["distance_to_center_cm"]
    )

    baseline_yaw_deg = float(
        baseline_pose["yaw_deg"]
    )

    print(
        "\n========== BASELINE =========="
    )

    print(
        f"Distance: "
        f"{baseline_distance_cm:.3f} cm"
    )

    print(
        f"Ground truth distance: "
        f"{GROUND_TRUTH_DISTANCE_CM:.3f} cm"
    )

    print(
        f"Baseline distance error: "
        f"{baseline_distance_cm - GROUND_TRUTH_DISTANCE_CM:+.3f} cm"
    )

    print(
        f"Yaw: "
        f"{baseline_yaw_deg:.4f} deg"
    )

    print(
        "==============================\n"
    )

    rows = []

    for sigma_px in NOISE_SIGMAS_PX:

        trials_for_sigma = (
            1
            if sigma_px == 0.0
            else NUM_TRIALS
        )

        for trial_index in range(
                trials_for_sigma
        ):

            noisy_corners, noise = (
                add_gaussian_corner_noise(
                    baseline_corners,
                    sigma_px,
                    rng,
                )
            )

            try:
                pose = estimate_plate_pose_from_K(
                    image_points=noisy_corners,
                    K=K,
                    dist_coeffs=None,
                )

            except Exception as exc:
                print(
                    f"Pose failed for sigma={sigma_px}, "
                    f"trial={trial_index}: {exc}"
                )

                continue

            distance_cm = float(
                pose["distance_to_center_cm"]
            )

            yaw_deg = float(
                pose["yaw_deg"]
            )

            mean_corner_shift_px, max_corner_shift_px = (
                compute_corner_displacement(
                    noise
                )
            )

            rows.append(
                {
                    "sigma_px": sigma_px,
                    "trial": trial_index,

                    "mean_corner_shift_px":
                        mean_corner_shift_px,

                    "max_corner_shift_px":
                        max_corner_shift_px,

                    "distance_cm":
                        distance_cm,

                    "distance_change_from_baseline_cm":
                        distance_cm
                        - baseline_distance_cm,

                    "distance_error_from_ground_truth_cm":
                        distance_cm
                        - GROUND_TRUTH_DISTANCE_CM,

                    "yaw_deg":
                        yaw_deg,

                    "yaw_change_from_baseline_deg":
                        yaw_deg
                        - baseline_yaw_deg,
                    "corner_0_dx": noise[0, 0],
                    "corner_0_dy": noise[0, 1],

                    "corner_1_dx": noise[1, 0],
                    "corner_1_dy": noise[1, 1],

                    "corner_2_dx": noise[2, 0],
                    "corner_2_dy": noise[2, 1],

                    "corner_3_dx": noise[3, 0],
                    "corner_3_dy": noise[3, 1],
                }
            )

    dataframe = pd.DataFrame(
        rows
    )

    baseline_info = {
        "distance_cm": baseline_distance_cm,
        "yaw_deg": baseline_yaw_deg,
    }

    return dataframe, baseline_info
def save_representative_visualizations(
        image: np.ndarray,
        baseline_corners: np.ndarray,
        dataframe: pd.DataFrame,
        output_dir: Path,
) -> None:

    visualization_dir = (
        output_dir
        / "noise_visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for sigma_px in NOISE_SIGMAS_PX:

        if sigma_px == 0.0:
            continue

        representative = (
            find_representative_trial(
                dataframe,
                sigma_px,
            )
        )

        noise = np.array(
            [
                [
                    representative["corner_0_dx"],
                    representative["corner_0_dy"],
                ],
                [
                    representative["corner_1_dx"],
                    representative["corner_1_dy"],
                ],
                [
                    representative["corner_2_dx"],
                    representative["corner_2_dy"],
                ],
                [
                    representative["corner_3_dx"],
                    representative["corner_3_dy"],
                ],
            ],
            dtype=np.float64,
        )

        noisy_corners = (
            baseline_corners + noise
        )

        save_path = (
            visualization_dir
            / f"sigma_{sigma_px:.2f}_representative.jpg"
        )

        draw_noise_visualization(
            image=image,
            original_corners=baseline_corners,
            noisy_corners=noisy_corners,
            sigma_px=sigma_px,
            save_path=save_path,
        )

        print(
            f"Saved visualization: "
            f"{save_path}"
        )

def summarize_results(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize sensitivity for every noise level.
    """

    summary_rows = []

    for sigma_px, group in dataframe.groupby(
            "sigma_px",
            sort=True,
    ):

        abs_distance_change = np.abs(
            group[
                "distance_change_from_baseline_cm"
            ].to_numpy()
        )

        abs_yaw_change = np.abs(
            group[
                "yaw_change_from_baseline_deg"
            ].to_numpy()
        )

        ground_truth_error = np.abs(
            group[
                "distance_error_from_ground_truth_cm"
            ].to_numpy()
        )

        summary_rows.append(
            {
                "sigma_px":
                    float(sigma_px),

                "trials":
                    len(group),

                "mean_actual_corner_shift_px":
                    group[
                        "mean_corner_shift_px"
                    ].mean(),

                "mean_abs_distance_change_cm":
                    np.mean(
                        abs_distance_change
                    ),

                "std_distance_change_cm":
                    np.std(
                        group[
                            "distance_change_from_baseline_cm"
                        ].to_numpy()
                    ),

                "max_abs_distance_change_cm":
                    np.max(
                        abs_distance_change
                    ),

                "mean_abs_yaw_change_deg":
                    np.mean(
                        abs_yaw_change
                    ),

                "std_yaw_change_deg":
                    np.std(
                        group[
                            "yaw_change_from_baseline_deg"
                        ].to_numpy()
                    ),

                "max_abs_yaw_change_deg":
                    np.max(
                        abs_yaw_change
                    ),

                "mean_abs_ground_truth_distance_error_cm":
                    np.mean(
                        ground_truth_error
                    ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def print_summary(
        summary: pd.DataFrame,
) -> None:

    print(
        "\n========== CORNER NOISE SENSITIVITY =========="
    )

    for _, row in summary.iterrows():

        print(
            f"\nNoise sigma: "
            f"{row['sigma_px']:.2f} px"
        )

        print(
            f"  Mean actual corner shift: "
            f"{row['mean_actual_corner_shift_px']:.3f} px"
        )

        print(
            f"  Mean |distance change|: "
            f"{row['mean_abs_distance_change_cm']:.3f} cm"
        )

        print(
            f"  Max  |distance change|: "
            f"{row['max_abs_distance_change_cm']:.3f} cm"
        )

        print(
            f"  Mean |yaw change|     : "
            f"{row['mean_abs_yaw_change_deg']:.4f} deg"
        )

        print(
            f"  Max  |yaw change|     : "
            f"{row['max_abs_yaw_change_deg']:.4f} deg"
        )

        print(
            f"  Mean distance error "
            f"from GT: "
            f"{row['mean_abs_ground_truth_distance_error_cm']:.3f} cm"
        )

    print(
        "\n==============================================\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {IMAGE_PATH}"
        )

    calibration = load_calibration_results(
        CALIBRATION_PATH
    )

    K_ref = calibration[
        "K_cv"
    ].astype(np.float64)

    baseline_corners = (
        detect_single_plate_corners(
            image,
            IMAGE_PATH,
        )
    )

    print(
        "\nDetected baseline corners:"
    )

    for i, corner in enumerate(
            baseline_corners
    ):
        print(
            f"{i}: {corner}"
        )

    results, _ = run_noise_experiment(
        baseline_corners=baseline_corners,
        K=K_ref,
    )

    summary = summarize_results(
        results
    )
    if SAVE_VISUALIZATIONS:
        save_representative_visualizations(
            image=image,
            baseline_corners=baseline_corners,
            dataframe=results,
            output_dir=OUTPUT_DIR,
        )
    print_summary(
        summary
    )

    raw_csv_path = (
        OUTPUT_DIR
        / "corner_noise_trials.csv"
    )

    summary_csv_path = (
        OUTPUT_DIR
        / "corner_noise_summary.csv"
    )

    results.to_csv(
        raw_csv_path,
        index=False,
    )

    summary.to_csv(
        summary_csv_path,
        index=False,
    )

    print(
        f"Saved raw trials to: "
        f"{raw_csv_path}"
    )

    print(
        f"Saved summary to: "
        f"{summary_csv_path}"
    )
def draw_noise_visualization(
        image: np.ndarray,
        original_corners: np.ndarray,
        noisy_corners: np.ndarray,
        sigma_px: float,
        save_path: Path,
        arrow_scale: float = VISUALIZATION_SCALE,
) -> None:
    """
    Visualize actual noisy corners and magnified displacement vectors.

    Important:
    The arrows are magnified only for visualization.
    Actual noisy corner locations are also drawn at their real positions.
    """

    vis = image.copy()

    original_corners = np.asarray(
        original_corners,
        dtype=np.float64,
    )

    noisy_corners = np.asarray(
        noisy_corners,
        dtype=np.float64,
    )

    shifts = noisy_corners - original_corners

    shift_distances = np.linalg.norm(
        shifts,
        axis=1,
    )

    # Draw original plate polygon.
    original_polygon = np.round(
        original_corners
    ).astype(np.int32)

    cv2.polylines(
        vis,
        [original_polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    # Draw actual noisy polygon.
    noisy_polygon = np.round(
        noisy_corners
    ).astype(np.int32)

    cv2.polylines(
        vis,
        [noisy_polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
    )

    for corner_index in range(4):
        original = original_corners[
            corner_index
        ]

        noisy = noisy_corners[
            corner_index
        ]

        shift = shifts[
            corner_index
        ]

        display_endpoint = (
            original
            + arrow_scale * shift
        )

        original_point = tuple(
            np.round(original).astype(int)
        )

        noisy_point = tuple(
            np.round(noisy).astype(int)
        )

        display_point = tuple(
            np.round(
                display_endpoint
            ).astype(int)
        )

        # Original corner.
        cv2.circle(
            vis,
            original_point,
            radius=8,
            color=(0, 255, 0),
            thickness=-1,
        )

        # Actual noisy corner.
        cv2.circle(
            vis,
            noisy_point,
            radius=6,
            color=(0, 0, 255),
            thickness=-1,
        )

        # Magnified displacement vector.
        cv2.arrowedLine(
            vis,
            original_point,
            display_point,
            color=(255, 0, 0),
            thickness=3,
            tipLength=0.15,
        )

        label_position = (
            display_point[0] + 10,
            display_point[1] - 10,
        )

        cv2.putText(
            vis,
            (
                f"C{corner_index}: "
                f"{shift_distances[corner_index]:.2f}px"
            ),
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    mean_shift = float(
        np.mean(
            shift_distances
        )
    )

    max_shift = float(
        np.max(
            shift_distances
        )
    )

    title_lines = [
        f"Noise sigma = {sigma_px:.2f} px",
        f"Mean actual corner shift = {mean_shift:.3f} px",
        f"Max actual corner shift = {max_shift:.3f} px",
        f"Displacement arrows magnified x{arrow_scale:.0f}",
        "Green = original, Red = actual noisy, Blue = magnified displacement",
    ]

    y = 50

    for line in title_lines:
        cv2.putText(
            vis,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 40

    cv2.imwrite(
        str(save_path),
        vis,
    )


def find_representative_trial(
        dataframe: pd.DataFrame,
        sigma_px: float,
) -> pd.Series:
    """
    Find the trial whose mean corner shift is closest
    to the mean corner shift for this sigma.

    This avoids visualizing an unusually easy or extreme trial.
    """

    group = dataframe[
        dataframe["sigma_px"] == sigma_px
    ].copy()

    if group.empty:
        raise ValueError(
            f"No trials found for sigma={sigma_px}"
        )

    target_shift = group[
        "mean_corner_shift_px"
    ].mean()

    difference = np.abs(
        group[
            "mean_corner_shift_px"
        ]
        - target_shift
    )

    representative_index = (
        difference.idxmin()
    )

    return dataframe.loc[
        representative_index
    ]

if __name__ == "__main__":
    main()