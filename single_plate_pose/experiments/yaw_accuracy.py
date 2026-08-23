from pathlib import Path
import csv
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# car/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the current project structure.
# This can be removed later if the project becomes a proper package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from car/.env
load_dotenv(PROJECT_ROOT / ".env")


from single_plate_pose.detection import (
    detect_plate_with_roboflow,
    save_roboflow_result,
    load_roboflow_result,
    find_predictions_recursive,
    prediction_score,
)
from corner_refinement import refine_yellow_inner_corners
from single_plate_pose.pose_estimation import estimate_plate_pose


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set."
    )


CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results.npz"
)

YAW_DIR = (
    PROJECT_ROOT
    / "cars_photos"
    / "yaw_experiment"
)

YAW_JSON_DIR = (
    PROJECT_ROOT
    / "roboflow_jsons"
    / "yaw_experiment"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "experiments"
    / "outputs"
    / "yaw_accuracy"
)

METHOD = "robust_mask_lines"
USE_DISTORTION = False

# Approximate manually measured yaw values.
# The 60-degree case is intentionally excluded from the official experiment.
ANGLE_CONFIG = {
    "angle_1": +28.0,
    "angle_2": +37.0,
    "angle_3": +41.0,
}

DISTANCES_M = [1.0, 3.0, 5.0]


# ============================================================
# VISUALIZATION
# ============================================================

GREEN = (40, 170, 40)
YELLOW = (0, 200, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)
ORANGE = (0, 170, 255)


def draw_quad(img, pts):
    pts = np.round(np.asarray(pts)).astype(np.int32)

    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=GREEN,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    for p in pts:
        cv2.circle(
            img,
            tuple(p),
            7,
            YELLOW,
            -1,
            cv2.LINE_AA,
        )


def draw_yaw_panel(
    img,
    angle_name,
    distance_label_m,
    gt_yaw_deg,
    estimated_yaw_deg,
    abs_error_deg,
    estimated_distance_m,
):
    overlay = img.copy()

    x1 = 35
    y1 = 35
    box_w = 650
    box_h = 310

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x1 + box_w, y1 + box_h),
        DARK,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        img,
        0.22,
        0,
        img,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        ("YAW ACCURACY", WHITE),
        (f"Angle case:          {angle_name}", WHITE),
        (f"Distance label:      {distance_label_m:.1f} m", WHITE),
        (f"GT yaw:              {gt_yaw_deg:+.2f} deg", YELLOW),
        (f"Estimated yaw:       {estimated_yaw_deg:+.2f} deg", ORANGE),
        (f"Absolute yaw error:  {abs_error_deg:.2f} deg", WHITE),
        (f"Estimated distance:  {estimated_distance_m:.3f} m", GREEN),
    ]

    y = y1 + 45

    for i, (text, color) in enumerate(lines):
        scale = 0.92 if i == 0 else 0.74

        cv2.putText(
            img,
            text,
            (x1 + 22, y),
            font,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 40


# ============================================================
# ROBOFLOW CACHE
# ============================================================

def get_yaw_predictions(
    image_path,
    angle_name,
    api_key,
):
    """
    Load cached Roboflow output for a yaw image, or run Roboflow and
    cache the result if it does not exist yet.

    Each angle gets its own cache folder because the source files have
    repeating names such as 1m.jpeg, 3m.jpeg and 5m.jpeg.
    """

    cache_dir = (
        YAW_JSON_DIR
        / angle_name
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        cache_dir
        / f"roboflow_result_{image_path.stem}.json"
    )

    if json_path.exists():
        print(
            "Loading yaw Roboflow result from JSON: "
            f"{json_path}"
        )

        raw_result = load_roboflow_result(
            json_path
        )

    else:
        print(
            "Running Roboflow for yaw experiment..."
        )

        _, raw_result = detect_plate_with_roboflow(
            str(image_path),
            api_key,
        )

        save_roboflow_result(
            raw_result,
            json_path,
        )

        print(
            f"Saved yaw Roboflow JSON: {json_path}"
        )

    predictions = find_predictions_recursive(
        raw_result
    )

    valid_predictions = [
        p for p in predictions
        if float(p.get("width", 0.0)) > 100
        and float(p.get("height", 0.0)) > 20
    ]

    if len(valid_predictions) == 0:
        raise RuntimeError(
            "No realistic license plate prediction found."
        )

    valid_predictions = sorted(
        valid_predictions,
        key=prediction_score,
        reverse=True,
    )

    return valid_predictions


# ============================================================
# EXPERIMENT
# ============================================================

def process_image(
    angle_name,
    image_path,
    distance_label_m,
    gt_yaw_deg,
):
    print("\n" + "=" * 72)
    print(f"Angle: {angle_name}")
    print(f"Image: {image_path.name}")
    print(f"Nominal distance: {distance_label_m:.1f} m")
    print(f"Approx. GT yaw: {gt_yaw_deg:+.2f} deg")
    print("=" * 72)

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------------
    # 1. Plate detection / cached Roboflow result
    # --------------------------------------------------------

    predictions = get_yaw_predictions(
        image_path=image_path,
        angle_name=angle_name,
        api_key=API_KEY,
    )

    best_prediction = max(
        predictions,
        key=lambda p: p.get("confidence", 0),
    )

    # --------------------------------------------------------
    # 2. Yellow-region corner refinement
    # --------------------------------------------------------

    yellow_points, _, _ = refine_yellow_inner_corners(
        img,
        best_prediction,
        debug=False,
        method=METHOD,
        debug_name=f"yaw_{angle_name}_{image_path.stem}",
    )

    # --------------------------------------------------------
    # 3. Pose estimation
    # --------------------------------------------------------

    pose = estimate_plate_pose(
        yellow_points,
        str(CALIBRATION_PATH),
        use_distortion=USE_DISTORTION,
    )

    estimated_yaw_deg = float(
        pose["yaw_deg"]
    )

    estimated_distance_m = (
        float(pose["distance_to_center_cm"])
        / 100.0
    )

    signed_yaw_error_deg = (
        estimated_yaw_deg
        - gt_yaw_deg
    )

    abs_yaw_error_deg = abs(
        signed_yaw_error_deg
    )

    center_m = (
        pose["plate_center_camera"]
        .reshape(-1)
        / 100.0
    )

    # --------------------------------------------------------
    # 4. Visualization
    # --------------------------------------------------------

    output = img.copy()

    draw_quad(
        output,
        yellow_points,
    )

    draw_yaw_panel(
        output,
        angle_name=angle_name,
        distance_label_m=distance_label_m,
        gt_yaw_deg=gt_yaw_deg,
        estimated_yaw_deg=estimated_yaw_deg,
        abs_error_deg=abs_yaw_error_deg,
        estimated_distance_m=estimated_distance_m,
    )

    print(
        f"Estimated yaw      : "
        f"{estimated_yaw_deg:+.3f} deg"
    )

    print(
        f"Signed yaw error   : "
        f"{signed_yaw_error_deg:+.3f} deg"
    )

    print(
        f"Absolute yaw error : "
        f"{abs_yaw_error_deg:.3f} deg"
    )

    print(
        f"Estimated distance : "
        f"{estimated_distance_m:.4f} m"
    )

    return {
        "image": output,
        "angle_name": angle_name,
        "distance_label_m": distance_label_m,
        "gt_yaw_deg": gt_yaw_deg,
        "estimated_yaw_deg": estimated_yaw_deg,
        "signed_yaw_error_deg": signed_yaw_error_deg,
        "abs_yaw_error_deg": abs_yaw_error_deg,
        "estimated_distance_m": estimated_distance_m,
        "center_x_m": float(center_m[0]),
        "center_y_m": float(center_m[1]),
        "center_z_m": float(center_m[2]),
    }


# ============================================================
# STATISTICS
# ============================================================

def compute_per_angle_statistics(results):
    per_angle = []

    for angle_name, gt_yaw_deg in ANGLE_CONFIG.items():
        rows = [
            r for r in results
            if r["angle_name"] == angle_name
        ]

        if len(rows) == 0:
            print(
                f"Skipping summary for {angle_name}: "
                "no successful images."
            )
            continue

        yaw_values = np.array(
            [r["estimated_yaw_deg"] for r in rows],
            dtype=np.float64,
        )

        abs_errors = np.array(
            [r["abs_yaw_error_deg"] for r in rows],
            dtype=np.float64,
        )

        mean_yaw = float(
            np.mean(yaw_values)
        )

        std_yaw = float(
            np.std(yaw_values)
        )

        yaw_range = float(
            np.max(yaw_values)
            - np.min(yaw_values)
        )

        mae = float(
            np.mean(abs_errors)
        )

        bias = float(
            mean_yaw
            - gt_yaw_deg
        )

        per_angle.append({
            "angle_name": angle_name,
            "gt_yaw_deg": gt_yaw_deg,
            "mean_estimated_yaw_deg": mean_yaw,
            "mae_deg": mae,
            "bias_deg": bias,
            "std_across_distance_deg": std_yaw,
            "range_across_distance_deg": yaw_range,
        })

    return per_angle


# ============================================================
# OUTPUT
# ============================================================

def save_results(results, per_angle):
    # --------------------------------------------------------
    # CSV — all successful images
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "yaw_all_images_results.csv"
    )

    fieldnames = [
        "angle_name",
        "distance_label_m",
        "gt_yaw_deg",
        "estimated_yaw_deg",
        "signed_yaw_error_deg",
        "abs_yaw_error_deg",
        "estimated_distance_m",
        "center_x_m",
        "center_y_m",
        "center_z_m",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for r in results:
            writer.writerow({
                key: r[key]
                for key in fieldnames
            })

    # --------------------------------------------------------
    # CSV — one row per angle
    # --------------------------------------------------------

    angle_csv_path = (
        OUTPUT_DIR
        / "yaw_per_angle_summary.csv"
    )

    angle_fields = [
        "angle_name",
        "gt_yaw_deg",
        "mean_estimated_yaw_deg",
        "mae_deg",
        "bias_deg",
        "std_across_distance_deg",
        "range_across_distance_deg",
    ]

    with open(
        angle_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=angle_fields,
        )

        writer.writeheader()
        writer.writerows(per_angle)

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    all_abs_errors = np.array(
        [r["abs_yaw_error_deg"] for r in results],
        dtype=np.float64,
    )

    all_signed_errors = np.array(
        [r["signed_yaw_error_deg"] for r in results],
        dtype=np.float64,
    )

    overall_mae = float(
        np.mean(all_abs_errors)
    )

    overall_bias = float(
        np.mean(all_signed_errors)
    )

    overall_max_error = float(
        np.max(all_abs_errors)
    )

    mean_distance_std = float(
        np.mean([
            s["std_across_distance_deg"]
            for s in per_angle
        ])
    )

    mean_distance_range = float(
        np.mean([
            s["range_across_distance_deg"]
            for s in per_angle
        ])
    )

    # --------------------------------------------------------
    # TXT summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "yaw_experiment_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "YAW EXPERIMENT — OFFICIAL ANGLES\n"
            "================================\n\n"
        )

        for s in per_angle:
            f.write(
                f"{s['angle_name']} | "
                f"GT {s['gt_yaw_deg']:+.2f} deg\n"
                f"  Mean estimated yaw : "
                f"{s['mean_estimated_yaw_deg']:+.3f} deg\n"
                f"  MAE                : "
                f"{s['mae_deg']:.3f} deg\n"
                f"  Bias               : "
                f"{s['bias_deg']:+.3f} deg\n"
                f"  Std across distance: "
                f"{s['std_across_distance_deg']:.3f} deg\n"
                f"  Range 1m-5m        : "
                f"{s['range_across_distance_deg']:.3f} deg\n\n"
            )

        f.write(
            "GLOBAL SUMMARY\n"
            "==============\n"
            f"Overall MAE              : "
            f"{overall_mae:.3f} deg\n"
            f"Overall signed bias      : "
            f"{overall_bias:+.3f} deg\n"
            f"Maximum absolute error   : "
            f"{overall_max_error:.3f} deg\n"
            f"Mean per-angle yaw std   : "
            f"{mean_distance_std:.3f} deg\n"
            f"Mean per-angle yaw range : "
            f"{mean_distance_range:.3f} deg\n"
        )

    return {
        "csv_path": csv_path,
        "angle_csv_path": angle_csv_path,
        "summary_path": summary_path,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "overall_max_error": overall_max_error,
        "mean_distance_std": mean_distance_std,
        "mean_distance_range": mean_distance_range,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for angle_name, gt_yaw_deg in ANGLE_CONFIG.items():
        angle_dir = (
            YAW_DIR
            / angle_name
        )

        angle_output_dir = (
            OUTPUT_DIR
            / angle_name
        )

        angle_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for distance_m in DISTANCES_M:
            image_path = (
                angle_dir
                / f"{int(distance_m)}m.jpeg"
            )

            try:
                result = process_image(
                    angle_name=angle_name,
                    image_path=image_path,
                    distance_label_m=distance_m,
                    gt_yaw_deg=gt_yaw_deg,
                )

                results.append(
                    result
                )

                output_path = (
                    angle_output_dir
                    / f"{image_path.stem}_yaw_debug.png"
                )

                cv2.imwrite(
                    str(output_path),
                    result["image"],
                )

                print(
                    f"Saved: {output_path}"
                )

            except Exception as e:
                print(
                    f"FAILED {angle_name} / "
                    f"{distance_m:.0f}m: {e}"
                )
                continue

    if len(results) == 0:
        raise RuntimeError(
            "Yaw experiment produced no successful results."
        )

    per_angle = compute_per_angle_statistics(
        results
    )

    if len(per_angle) == 0:
        raise RuntimeError(
            "No per-angle statistics could be computed."
        )

    output_info = save_results(
        results,
        per_angle,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 82)
    print("YAW EXPERIMENT — OFFICIAL ANGLES")
    print("=" * 82)

    for s in per_angle:
        print(
            f"{s['angle_name']} | "
            f"GT {s['gt_yaw_deg']:+.1f} deg | "
            f"Mean est {s['mean_estimated_yaw_deg']:+.3f} deg | "
            f"MAE {s['mae_deg']:.3f} deg | "
            f"std {s['std_across_distance_deg']:.3f} deg | "
            f"range {s['range_across_distance_deg']:.3f} deg"
        )

    print("\nGLOBAL:")
    print(
        f"Overall MAE            : "
        f"{output_info['overall_mae']:.3f} deg"
    )
    print(
        f"Overall bias           : "
        f"{output_info['overall_bias']:+.3f} deg"
    )
    print(
        f"Max absolute error     : "
        f"{output_info['overall_max_error']:.3f} deg"
    )
    print(
        f"Mean yaw std vs dist   : "
        f"{output_info['mean_distance_std']:.3f} deg"
    )
    print(
        f"Mean yaw range vs dist : "
        f"{output_info['mean_distance_range']:.3f} deg"
    )

    print("\nFiles:")
    print(output_info["csv_path"])
    print(output_info["angle_csv_path"])
    print(output_info["summary_path"])


if __name__ == "__main__":
    main()
