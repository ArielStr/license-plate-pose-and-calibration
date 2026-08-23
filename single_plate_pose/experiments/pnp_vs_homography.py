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


from single_plate_pose.detection import get_predictions
from corner_refinement import refine_yellow_inner_corners
from single_plate_pose.pose_estimation import (
    estimate_plate_pose,
    estimate_plate_pose_from_homography,
)
from calibration.calibration_utils import load_calibration_results


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

DEPTH_DIR = (
    PROJECT_ROOT
    / "cars_photos"
    / "depth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "experiments"
    / "outputs"
    / "pnp_vs_homography"
)

METHOD = "robust_mask_lines"
USE_DISTORTION = False


TEST_CASES = [
    {
        "image": DEPTH_DIR / "1_meter.jpeg",
        "gt_m": 1.0,
    },
    {
        "image": DEPTH_DIR / "3_meter.jpeg",
        "gt_m": 3.0,
    },
    {
        "image": DEPTH_DIR / "5_meter.jpeg",
        "gt_m": 5.0,
    },
]


# ============================================================
# VISUALIZATION
# ============================================================

NAVY = (90, 45, 10)
GREEN = (40, 170, 40)
YELLOW = (0, 200, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)


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


def draw_distance_panel(
    img,
    gt_m,
    estimated_m,
    error_cm,
):
    """
    Draw a clean presentation-style box in the top-left.
    """

    overlay = img.copy()

    x1 = 35
    y1 = 35
    box_w = 560
    box_h = 230

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

    cv2.putText(
        img,
        "DISTANCE ACCURACY",
        (x1 + 25, y1 + 50),
        font,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"GT distance:        {gt_m:.2f} m",
        (x1 + 25, y1 + 105),
        font,
        0.85,
        YELLOW,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"Estimated distance: {estimated_m:.3f} m",
        (x1 + 25, y1 + 150),
        font,
        0.85,
        GREEN,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"Absolute error:     {error_cm:.2f} cm",
        (x1 + 25, y1 + 195),
        font,
        0.85,
        WHITE,
        2,
        cv2.LINE_AA,
    )


def resize_for_grid(img, target_w=800, target_h=600):
    """
    Resize image while preserving aspect ratio and place it on a white canvas.
    """

    h, w = img.shape[:2]

    scale = min(
        target_w / w,
        target_h / h,
    )

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (target_h, target_w, 3),
        255,
        dtype=np.uint8,
    )

    x = (target_w - nw) // 2
    y = (target_h - nh) // 2

    canvas[
        y:y + nh,
        x:x + nw
    ] = resized

    return canvas


def make_comparison_image(results):
    """
    Create one wide image:
        1m | 3m | 5m
    """

    cards = []

    for result in results:
        card = resize_for_grid(
            result["image"]
        )

        header_h = 100

        canvas = np.full(
            (
                card.shape[0] + header_h,
                card.shape[1],
                3,
            ),
            255,
            dtype=np.uint8,
        )

        canvas[header_h:] = card

        gt = result["gt_m"]
        est = result["estimated_m"]
        err = result["error_cm"]

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            canvas,
            f"GT: {gt:.1f} m",
            (35, 40),
            font,
            0.9,
            NAVY,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"PnP: {est:.3f} m   |   Error: {err:.1f} cm",
            (35, 78),
            font,
            0.68,
            GREEN,
            2,
            cv2.LINE_AA,
        )

        cards.append(
            canvas
        )

    return np.hstack(
        cards
    )


# ============================================================
# EXPERIMENT
# ============================================================

def process_image(
    image_path,
    gt_m,
    K,
):
    print("\n==========================================")
    print(f"Image: {image_path.name}")
    print(f"GT distance: {gt_m:.3f} m")
    print("==========================================")

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------------
    # 1. Plate detection
    # --------------------------------------------------------

    predictions = get_predictions(
        str(image_path),
        API_KEY,
    )

    if len(predictions) == 0:
        raise RuntimeError(
            f"No plate detected in {image_path.name}"
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
        debug_name=f"pnp_vs_homography_{image_path.stem}",
    )

    # --------------------------------------------------------
    # 3. Pose with PnP / IPPE
    # --------------------------------------------------------

    pnp_pose = estimate_plate_pose(
        yellow_points,
        str(CALIBRATION_PATH),
        use_distortion=USE_DISTORTION,
    )

    # --------------------------------------------------------
    # 4. Pose with classical homography decomposition
    # --------------------------------------------------------

    hom_pose = estimate_plate_pose_from_homography(
        image_points=yellow_points,
        K=K,
        debug=True,
    )

    pnp_m = (
        pnp_pose["distance_to_center_cm"]
        / 100.0
    )

    hom_m = (
        hom_pose["distance_to_center_cm"]
        / 100.0
    )

    pnp_error_cm = (
        pnp_m - gt_m
    ) * 100.0

    hom_error_cm = (
        hom_m - gt_m
    ) * 100.0

    dbg = hom_pose["debug"]

    print("\n--- GT DISTANCE COMPARISON ---")
    print(f"GT                 : {gt_m:.4f} m")
    print(f"PnP/IPPE           : {pnp_m:.4f} m")
    print(f"PnP signed error   : {pnp_error_cm:+.2f} cm")
    print(f"Homography         : {hom_m:.4f} m")
    print(f"H signed error     : {hom_error_cm:+.2f} cm")
    print(
        f"|PnP - H|          : "
        f"{abs(pnp_m - hom_m) * 100.0:.2f} cm"
    )

    print("\n--- HOMOGRAPHY ORTHONORMALITY ---")
    print(f"||b1||             : {dbg['norm_b1']:.8f}")
    print(f"||b2||             : {dbg['norm_b2']:.8f}")
    print(
        f"scale mismatch     : "
        f"{dbg['relative_norm_mismatch_percent']:.4f}%"
    )
    print(f"||r1_raw||         : {dbg['norm_r1_raw']:.8f}")
    print(f"||r2_raw||         : {dbg['norm_r2_raw']:.8f}")
    print(f"r1 dot r2          : {dbg['dot_r1_r2_raw']:.8e}")
    print(
        f"angle(r1,r2)       : "
        f"{dbg['angle_r1_r2_deg_raw']:.6f} deg"
    )
    print(
        f"raw ||R^T R-I||_F : "
        f"{dbg['orthogonality_error_raw']:.8e}"
    )
    print(f"raw det(R)         : {dbg['det_R_raw']:.8f}")

    # --------------------------------------------------------
    # 5. Visualization
    # --------------------------------------------------------

    output = img.copy()

    draw_quad(
        output,
        yellow_points,
    )

    # The visualization keeps PnP as the primary estimator,
    # since that is the production method used in Part 1.
    draw_distance_panel(
        output,
        gt_m=gt_m,
        estimated_m=pnp_m,
        error_cm=abs(pnp_error_cm),
    )

    return {
        "image": output,
        "gt_m": gt_m,
        "estimated_m": pnp_m,
        "error_cm": abs(pnp_error_cm),
        "signed_error_cm": pnp_error_cm,
        "yaw_deg": pnp_pose["yaw_deg"],
        "points": yellow_points,

        "pnp_m": pnp_m,
        "pnp_error_cm": abs(pnp_error_cm),
        "pnp_signed_error_cm": pnp_error_cm,

        "homography_m": hom_m,
        "homography_error_cm": abs(hom_error_cm),
        "homography_signed_error_cm": hom_error_cm,

        "method_difference_cm": abs(
            pnp_m - hom_m
        ) * 100.0,

        "h_scale_mismatch_percent":
            dbg["relative_norm_mismatch_percent"],
        "h_angle_r1_r2_deg":
            dbg["angle_r1_r2_deg_raw"],
        "h_orthogonality_error":
            dbg["orthogonality_error_raw"],
        "h_det_R_raw":
            dbg["det_R_raw"],
    }


# ============================================================
# OUTPUT
# ============================================================

def save_csv(results):
    csv_path = (
        OUTPUT_DIR
        / "pnp_vs_homography.csv"
    )

    fieldnames = [
        "gt_m",
        "pnp_m",
        "pnp_abs_error_cm",
        "pnp_signed_error_cm",
        "homography_m",
        "homography_abs_error_cm",
        "homography_signed_error_cm",
        "pnp_vs_homography_diff_cm",
        "h_scale_mismatch_percent",
        "h_angle_r1_r2_deg",
        "h_orthogonality_error",
        "h_det_R_raw",
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
                "gt_m": r["gt_m"],
                "pnp_m": r["pnp_m"],
                "pnp_abs_error_cm": r["pnp_error_cm"],
                "pnp_signed_error_cm": r["pnp_signed_error_cm"],
                "homography_m": r["homography_m"],
                "homography_abs_error_cm": r["homography_error_cm"],
                "homography_signed_error_cm":
                    r["homography_signed_error_cm"],
                "pnp_vs_homography_diff_cm":
                    r["method_difference_cm"],
                "h_scale_mismatch_percent":
                    r["h_scale_mismatch_percent"],
                "h_angle_r1_r2_deg":
                    r["h_angle_r1_r2_deg"],
                "h_orthogonality_error":
                    r["h_orthogonality_error"],
                "h_det_R_raw":
                    r["h_det_R_raw"],
            })

    return csv_path


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load calibration once for the complete experiment.
    calib = load_calibration_results(
        str(CALIBRATION_PATH)
    )

    K = calib["K_cv"].astype(
        np.float64
    )

    results = []

    for case in TEST_CASES:
        result = process_image(
            image_path=case["image"],
            gt_m=case["gt_m"],
            K=K,
        )

        results.append(
            result
        )

        output_path = (
            OUTPUT_DIR
            / f"{case['image'].stem}_debug.png"
        )

        cv2.imwrite(
            str(output_path),
            result["image"],
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Combined visualization
    # --------------------------------------------------------

    comparison = make_comparison_image(
        results
    )

    comparison_path = (
        OUTPUT_DIR
        / "pnp_vs_homography_comparison.png"
    )

    cv2.imwrite(
        str(comparison_path),
        comparison,
    )

    # --------------------------------------------------------
    # CSV comparison table
    # --------------------------------------------------------

    csv_path = save_csv(
        results
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "pnp_vs_homography_summary.txt"
    )

    mean_pnp_error = float(
        np.mean([
            r["pnp_error_cm"]
            for r in results
        ])
    )

    mean_h_error = float(
        np.mean([
            r["homography_error_cm"]
            for r in results
        ])
    )

    mean_method_diff = float(
        np.mean([
            r["method_difference_cm"]
            for r in results
        ])
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PnP/IPPE VS CLASSICAL HOMOGRAPHY\n"
            "================================\n\n"
        )

        for r in results:
            f.write(
                f"GT = {r['gt_m']:.3f} m\n"
                f"PnP = {r['pnp_m']:.4f} m\n"
                f"PnP signed error = "
                f"{r['pnp_signed_error_cm']:+.2f} cm\n"
                f"Homography = {r['homography_m']:.4f} m\n"
                f"Homography signed error = "
                f"{r['homography_signed_error_cm']:+.2f} cm\n"
                f"|PnP - Homography| = "
                f"{r['method_difference_cm']:.2f} cm\n"
                f"Homography scale mismatch = "
                f"{r['h_scale_mismatch_percent']:.4f}%\n"
                f"angle(r1,r2) = "
                f"{r['h_angle_r1_r2_deg']:.6f} deg\n"
                f"||R^T R - I||_F = "
                f"{r['h_orthogonality_error']:.8e}\n"
                f"det(R_raw) = "
                f"{r['h_det_R_raw']:.8f}\n"
                "\n"
            )

        f.write(
            "SUMMARY\n"
            "=======\n"
            f"Mean PnP absolute error       : "
            f"{mean_pnp_error:.2f} cm\n"
            f"Mean homography absolute error: "
            f"{mean_h_error:.2f} cm\n"
            f"Mean |PnP - Homography|       : "
            f"{mean_method_diff:.2f} cm\n"
        )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("PnP/IPPE vs HOMOGRAPHY — GT SUMMARY")
    print("==========================================")

    for r in results:
        print(
            f"GT {r['gt_m']:.1f} m | "
            f"PnP {r['pnp_m']:.4f} m "
            f"(err {r['pnp_error_cm']:.2f} cm) | "
            f"H {r['homography_m']:.4f} m "
            f"(err {r['homography_error_cm']:.2f} cm) | "
            f"diff {r['method_difference_cm']:.2f} cm"
        )

    print(
        f"\nMean PnP abs error : "
        f"{mean_pnp_error:.2f} cm"
    )

    print(
        f"Mean H abs error   : "
        f"{mean_h_error:.2f} cm"
    )

    print(
        f"Mean method diff   : "
        f"{mean_method_diff:.2f} cm"
    )

    print("\nFiles:")
    print(comparison_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
