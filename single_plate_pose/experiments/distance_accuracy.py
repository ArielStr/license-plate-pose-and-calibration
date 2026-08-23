from pathlib import Path
import os
import sys
from dotenv import load_dotenv
import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from car/.env
load_dotenv(PROJECT_ROOT / ".env")


from single_plate_pose.detection import get_predictions
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
    / "distance_accuracy"
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
    """Draw a clean presentation-style box in the top-left."""

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
    """Resize while preserving aspect ratio and place on white canvas."""

    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

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

    canvas[y:y + nh, x:x + nw] = resized

    return canvas


def make_comparison_image(results):
    """Create one wide image: 1m | 3m | 5m."""

    cards = []

    for result in results:
        card = resize_for_grid(result["image"])

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
            f"Estimated: {est:.3f} m   |   Error: {err:.1f} cm",
            (35, 78),
            font,
            0.68,
            GREEN,
            2,
            cv2.LINE_AA,
        )

        cards.append(canvas)

    return np.hstack(cards)


# ============================================================
# EXPERIMENT
# ============================================================

def process_image(image_path, gt_m):

    print("\n==========================================")
    print(f"Image: {image_path.name}")
    print(f"GT distance: {gt_m:.3f} m")
    print("==========================================")

    img = cv2.imread(str(image_path))

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
        debug_name=f"distance_test_{image_path.stem}",
    )

    # --------------------------------------------------------
    # 3. Pose estimation
    # --------------------------------------------------------

    pose = estimate_plate_pose(
        yellow_points,
        str(CALIBRATION_PATH),
        use_distortion=USE_DISTORTION,
    )

    estimated_m = (
        pose["distance_to_center_cm"] / 100.0
    )

    yaw_deg = pose["yaw_deg"]

    error_m = estimated_m - gt_m
    error_cm = error_m * 100.0

    # --------------------------------------------------------
    # 4. Visualization
    # --------------------------------------------------------

    output = img.copy()

    draw_quad(
        output,
        yellow_points,
    )

    draw_distance_panel(
        output,
        gt_m=gt_m,
        estimated_m=estimated_m,
        error_cm=abs(error_cm),
    )

    print(f"Estimated distance : {estimated_m:.4f} m")
    print(f"Absolute error     : {abs(error_cm):.2f} cm")
    print(f"Signed error       : {error_cm:+.2f} cm")
    print(f"Yaw                : {yaw_deg:+.3f} deg")

    return {
        "image": output,
        "gt_m": gt_m,
        "estimated_m": estimated_m,
        "error_cm": abs(error_cm),
        "signed_error_cm": error_cm,
        "yaw_deg": yaw_deg,
        "points": yellow_points,
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

    for case in TEST_CASES:

        result = process_image(
            case["image"],
            case["gt_m"],
        )

        results.append(result)

        output_path = (
            OUTPUT_DIR
            / f"{case['image'].stem}_distance_debug.png"
        )

        cv2.imwrite(
            str(output_path),
            result["image"],
        )

        print(f"Saved: {output_path}")

    # --------------------------------------------------------
    # Combined visualization
    # --------------------------------------------------------

    comparison = make_comparison_image(results)

    comparison_path = (
        OUTPUT_DIR
        / "distance_accuracy_1m_3m_5m_comparison.png"
    )

    cv2.imwrite(
        str(comparison_path),
        comparison,
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "distance_accuracy_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        for r in results:

            f.write(
                f"GT = {r['gt_m']:.3f} m\n"
                f"Estimated = {r['estimated_m']:.4f} m\n"
                f"Signed error = {r['signed_error_cm']:+.2f} cm\n"
                f"Absolute error = {r['error_cm']:.2f} cm\n"
                f"Yaw = {r['yaw_deg']:+.3f} deg\n"
                "\n"
            )

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")

    for r in results:

        print(
            f"GT {r['gt_m']:.1f} m"
            f" -> estimated {r['estimated_m']:.3f} m"
            f" | error {r['error_cm']:.2f} cm"
        )

    print("\nOutput files:")
    print(comparison_path)
    print(summary_path)


if __name__ == "__main__":
    main()