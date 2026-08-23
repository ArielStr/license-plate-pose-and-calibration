from pathlib import Path
import math
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

ANGLES_DIR = (
    PROJECT_ROOT
    / "cars_photos"
    / "angles"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "experiments"
    / "outputs"
    / "lateral_displacement"
)

METHOD = "robust_mask_lines"
USE_DISTORTION = False

DEPTH_M = 3.0


TEST_CASES = [
    {
        "image": ANGLES_DIR / "L_0.5_3.jpeg",
        "side": "L",
        "x_m": 0.5,
    },
    {
        "image": ANGLES_DIR / "L_1_3.jpeg",
        "side": "L",
        "x_m": 1.0,
    },
    {
        "image": ANGLES_DIR / "R_0.5_3.jpeg",
        "side": "R",
        "x_m": 0.5,
    },
    {
        "image": ANGLES_DIR / "R_1_3.jpeg",
        "side": "R",
        "x_m": 1.0,
    },
    {
        "image": ANGLES_DIR / "R_1.5_3.jpeg",
        "side": "R",
        "x_m": 1.5,
    },
]


# ============================================================
# COLORS
# ============================================================

GREEN = (40, 180, 40)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)


# ============================================================
# HELPERS
# ============================================================

def compute_gt_distance(depth_m, lateral_m):
    """
    Ground-truth Euclidean distance from the camera center to the plate
    when the forward depth and lateral displacement are known.
    """
    return math.sqrt(depth_m ** 2 + lateral_m ** 2)


def draw_quad(img, pts):
    pts = np.round(np.asarray(pts)).astype(np.int32)

    cv2.polylines(
        img,
        [pts],
        True,
        GREEN,
        3,
        cv2.LINE_AA,
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


def draw_result_panel(
    img,
    side,
    lateral_m,
    depth_m,
    gt_m,
    estimated_m,
    error_cm,
):
    overlay = img.copy()

    x1 = 25
    y1 = 25
    box_w = 600
    box_h = 280

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
        "LATERAL DISPLACEMENT TEST",
        (x1 + 25, y1 + 45),
        font,
        0.85,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    side_text = "Left" if side == "L" else "Right"

    lines = [
        (
            f"Lateral shift: {side_text} {lateral_m:.1f} m",
            WHITE,
        ),
        (
            f"Depth:         {depth_m:.2f} m",
            WHITE,
        ),
        (
            f"GT distance:   {gt_m:.3f} m",
            YELLOW,
        ),
        (
            f"Estimated:     {estimated_m:.3f} m",
            GREEN,
        ),
        (
            f"Abs. error:    {error_cm:.2f} cm",
            WHITE,
        ),
    ]

    y = y1 + 95

    for text, color in lines:
        cv2.putText(
            img,
            text,
            (x1 + 25, y),
            font,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 38


def resize_for_grid(img, target_w=700, target_h=600):
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


def make_comparison_grid(results):
    cards = []

    for result in results:
        card_img = resize_for_grid(
            result["image"],
            target_w=700,
            target_h=600,
        )

        header_h = 110

        canvas = np.full(
            (
                card_img.shape[0] + header_h,
                card_img.shape[1],
                3,
            ),
            255,
            dtype=np.uint8,
        )

        canvas[header_h:] = card_img

        side_name = (
            "LEFT"
            if result["side"] == "L"
            else "RIGHT"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            canvas,
            f"{side_name} {result['x_m']:.1f} m",
            (25, 40),
            font,
            0.85,
            (30, 60, 110),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                f"GT {result['gt_m']:.3f} m"
                f" | Est. {result['estimated_m']:.3f} m"
                f" | Err. {result['error_cm']:.1f} cm"
            ),
            (25, 80),
            font,
            0.58,
            GREEN,
            2,
            cv2.LINE_AA,
        )

        cards.append(canvas)

    # First row: the two LEFT measurements
    left_cards = [
        card
        for card, result in zip(cards, results)
        if result["side"] == "L"
    ]

    # Second row: the three RIGHT measurements
    right_cards = [
        card
        for card, result in zip(cards, results)
        if result["side"] == "R"
    ]

    card_h, card_w = cards[0].shape[:2]

    # Make both rows three cards wide.
    # Keep one blank card in the LEFT row to preserve the original layout.
    blank = np.full(
        (card_h, card_w, 3),
        255,
        dtype=np.uint8,
    )

    row_left = np.hstack([
        blank,
        left_cards[0],
        left_cards[1],
    ])

    row_right = np.hstack(
        right_cards
    )

    return np.vstack([
        row_left,
        row_right,
    ])


# ============================================================
# EXPERIMENT
# ============================================================

def process_image(
    image_path,
    side,
    lateral_m,
):
    gt_m = compute_gt_distance(
        DEPTH_M,
        lateral_m,
    )

    print("\n==========================================")
    print(f"Image: {image_path.name}")
    print(f"Side: {side}")
    print(f"Lateral shift: {lateral_m:.3f} m")
    print(f"Depth: {DEPTH_M:.3f} m")
    print(f"GT distance: {gt_m:.4f} m")
    print("==========================================")

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------------
    # 1. Detect plate
    # --------------------------------------------------------

    predictions = get_predictions(
        str(image_path),
        API_KEY,
    )

    if len(predictions) == 0:
        raise RuntimeError(
            f"No plate detected in {image_path.name}"
        )

    best_pred = max(
        predictions,
        key=lambda p: p.get(
            "confidence",
            0
        ),
    )

    # --------------------------------------------------------
    # 2. Refine yellow-region corners
    # --------------------------------------------------------

    yellow_points, _, _ = (
        refine_yellow_inner_corners(
            img,
            best_pred,
            debug=False,
            method=METHOD,
            debug_name=f"lateral_{image_path.stem}",
        )
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
        pose["distance_to_center_cm"]
        / 100.0
    )

    yaw_deg = pose["yaw_deg"]

    signed_error_m = (
        estimated_m - gt_m
    )

    signed_error_cm = (
        signed_error_m * 100.0
    )

    abs_error_cm = abs(
        signed_error_cm
    )

    # --------------------------------------------------------
    # 4. Visualization
    # --------------------------------------------------------

    output = img.copy()

    draw_quad(
        output,
        yellow_points,
    )

    draw_result_panel(
        output,
        side=side,
        lateral_m=lateral_m,
        depth_m=DEPTH_M,
        gt_m=gt_m,
        estimated_m=estimated_m,
        error_cm=abs_error_cm,
    )

    print(
        f"Estimated distance : "
        f"{estimated_m:.4f} m"
    )

    print(
        f"Signed error       : "
        f"{signed_error_cm:+.2f} cm"
    )

    print(
        f"Absolute error     : "
        f"{abs_error_cm:.2f} cm"
    )

    print(
        f"Yaw                : "
        f"{yaw_deg:+.3f} deg"
    )

    return {
        "image": output,
        "side": side,
        "x_m": lateral_m,
        "depth_m": DEPTH_M,
        "gt_m": gt_m,
        "estimated_m": estimated_m,
        "signed_error_cm": signed_error_cm,
        "error_cm": abs_error_cm,
        "yaw_deg": yaw_deg,
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
            image_path=case["image"],
            side=case["side"],
            lateral_m=case["x_m"],
        )

        results.append(
            result
        )

        output_path = (
            OUTPUT_DIR
            / (
                f"{case['image'].stem}"
                f"_lateral_distance_debug.png"
            )
        )

        cv2.imwrite(
            str(output_path),
            result["image"],
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Combined comparison image
    # --------------------------------------------------------

    comparison = (
        make_comparison_grid(
            results
        )
    )

    comparison_path = (
        OUTPUT_DIR
        / "lateral_distance_accuracy_comparison.png"
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
        / "lateral_distance_accuracy_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            f.write(
                f"{r['side']} "
                f"{r['x_m']:.1f} m\n"
                f"  Depth = {r['depth_m']:.3f} m\n"
                f"  GT distance = {r['gt_m']:.4f} m\n"
                f"  Estimated = {r['estimated_m']:.4f} m\n"
                f"  Signed error = "
                f"{r['signed_error_cm']:+.2f} cm\n"
                f"  Absolute error = "
                f"{r['error_cm']:.2f} cm\n"
                f"  Yaw = {r['yaw_deg']:+.3f} deg\n"
                "\n"
            )

    # --------------------------------------------------------
    # Final terminal summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")

    for r in results:
        print(
            f"{r['side']} {r['x_m']:.1f} m"
            f" | GT {r['gt_m']:.3f} m"
            f" | Est {r['estimated_m']:.3f} m"
            f" | Error {r['error_cm']:.2f} cm"
            f" | Yaw {r['yaw_deg']:+.2f} deg"
        )

    print("\nExperiment outputs:")
    print(comparison_path)
    print(summary_path)


if __name__ == "__main__":
    main()
