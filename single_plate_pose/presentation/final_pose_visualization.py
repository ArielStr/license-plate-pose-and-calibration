from pathlib import Path
import json
import sys

import cv2
import numpy as np


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


from corner_refinement import refine_yellow_inner_corners
from single_plate_pose.pose_estimation import estimate_plate_pose


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "multi_cars"
    / "IMG_4349.jpeg"
)

JSON_PATH = (
    PROJECT_ROOT
    / "roboflow_jsons"
    / "roboflow_result_IMG_4349.json"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "presentation"
    / "outputs"
    / "final_pose"
)

METHOD = "robust_mask_lines"
MAX_PLATES = None
USE_DISTORTION = False


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

QUAD_COLOR = (0, 255, 0)
CORNER_COLOR = (0, 255, 255)
DIST_COLOR = (0, 220, 0)
YAW_COLOR = (0, 180, 255)
BOX_BG = (20, 20, 20)


# ============================================================
# HELPERS
# ============================================================

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data[0]["predictions"]["predictions"]


def draw_quad(img, pts):
    pts = np.round(
        np.asarray(pts, np.float32)
    ).astype(np.int32)

    cv2.polylines(
        img,
        [pts],
        True,
        QUAD_COLOR,
        2,
        cv2.LINE_AA,
    )

    for x, y in pts:
        cv2.circle(
            img,
            (int(x), int(y)),
            5,
            CORNER_COLOR,
            -1,
            cv2.LINE_AA,
        )


def draw_result_box(
    img,
    x,
    y,
    plate_idx,
    distance_m,
    yaw_deg,
    center_m=None,
    scale=0.75,
):
    lines = [
        (
            f"Car {plate_idx}",
            (255, 255, 255),
        ),
        (
            f"Distance: {distance_m:.2f} m",
            DIST_COLOR,
        ),
        (
            f"Yaw: {yaw_deg:+.1f} deg",
            YAW_COLOR,
        ),
    ]

    if center_m is not None:
        lines.append(
            (
                (
                    f"P: [{center_m[0]:.2f}, "
                    f"{center_m[1]:.2f}, "
                    f"{center_m[2]:.2f}] m"
                ),
                (230, 230, 230),
            )
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    pad = 10
    gap = 7

    sizes = [
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )[0]
        for text, _ in lines
    ]

    box_w = max(
        width
        for width, height in sizes
    ) + 2 * pad

    box_h = (
        sum(
            height
            for width, height in sizes
        )
        + gap * (len(lines) - 1)
        + 2 * pad
    )

    img_h, img_w = img.shape[:2]

    x = max(
        0,
        min(
            int(x),
            img_w - box_w,
        ),
    )

    y = max(
        box_h,
        min(
            int(y),
            img_h,
        ),
    )

    top = y - box_h

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, top),
        (x + box_w, y),
        BOX_BG,
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

    current_y = top + pad

    for (
        (text, color),
        (_, text_h),
    ) in zip(lines, sizes):

        current_y += text_h

        cv2.putText(
            img,
            text,
            (x + pad, current_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        current_y += gap


def crop_around_points(
    image,
    pts,
    margin_x=0.75,
    margin_y=2.0,
):
    pts = np.asarray(
        pts,
        np.float32,
    )

    x1, y1 = np.min(
        pts,
        axis=0,
    )

    x2, y2 = np.max(
        pts,
        axis=0,
    )

    box_w = max(
        1,
        x2 - x1,
    )

    box_h = max(
        1,
        y2 - y1,
    )

    x1 = max(
        0,
        int(
            x1
            - margin_x * box_w
        ),
    )

    x2 = min(
        image.shape[1],
        int(
            x2
            + margin_x * box_w
        ),
    )

    y1 = max(
        0,
        int(
            y1
            - margin_y * box_h
        ),
    )

    y2 = min(
        image.shape[0],
        int(
            y2
            + margin_y * box_h
        ),
    )

    return (
        image[y1:y2, x1:x2].copy(),
        x1,
        y1,
    )


def make_grid(images, cols=3):
    if not images:
        return None

    cols = min(
        cols,
        len(images),
    )

    rows = int(
        np.ceil(
            len(images) / cols
        )
    )

    target_w = max(
        image.shape[1]
        for image in images
    )

    target_h = max(
        image.shape[0]
        for image in images
    )

    normalized = []

    for image in images:
        canvas = np.full(
            (
                target_h,
                target_w,
                3,
            ),
            255,
            np.uint8,
        )

        h, w = image.shape[:2]

        scale = min(
            target_w / w,
            target_h / h,
        )

        new_w = int(
            w * scale
        )

        new_h = int(
            h * scale
        )

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )

        x = (
            target_w
            - new_w
        ) // 2

        y = (
            target_h
            - new_h
        ) // 2

        canvas[
            y:y + new_h,
            x:x + new_w
        ] = resized

        normalized.append(
            canvas
        )

    while len(normalized) < rows * cols:
        normalized.append(
            np.full(
                (
                    target_h,
                    target_w,
                    3,
                ),
                255,
                np.uint8,
            )
        )

    row_images = []

    for row_idx in range(rows):
        row_images.append(
            np.hstack(
                normalized[
                    row_idx * cols:
                    (row_idx + 1) * cols
                ]
            )
        )

    return np.vstack(
        row_images
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    img = cv2.imread(
        str(IMAGE_PATH)
    )

    if img is None:
        raise ValueError(
            f"Could not read image: {IMAGE_PATH}"
        )

    predictions = load_predictions(
        JSON_PATH
    )

    if MAX_PLATES is not None:
        predictions = predictions[
            :MAX_PLATES
        ]

    full_overlay = img.copy()
    crop_results = []
    summary = []

    print(
        f"Found {len(predictions)} "
        "plate prediction(s)"
    )

    for i, pred in enumerate(
        predictions,
        start=1,
    ):
        try:
            points, _, _ = (
                refine_yellow_inner_corners(
                    img,
                    pred,
                    debug=False,
                    method=METHOD,
                    debug_name=(
                        f"presentation_pose_plate_{i}"
                    ),
                )
            )

            pose = estimate_plate_pose(
                points,
                str(CALIBRATION_PATH),
                use_distortion=USE_DISTORTION,
            )

            distance_m = (
                pose["distance_to_center_cm"]
                / 100.0
            )

            yaw_deg = pose["yaw_deg"]

            center_m = (
                pose["plate_center_camera"]
                .reshape(-1)
                / 100.0
            )

            # ------------------------------------------------
            # Full-image visualization
            # ------------------------------------------------

            draw_quad(
                full_overlay,
                points,
            )

            pts_np = np.asarray(
                points,
                np.float32,
            )

            text_x = int(
                np.mean(
                    pts_np[:, 0]
                )
            )

            text_y = int(
                np.min(
                    pts_np[:, 1]
                ) - 20
            )

            draw_result_box(
                full_overlay,
                text_x,
                text_y,
                i,
                distance_m,
                yaw_deg,
                center_m,
                scale=0.65,
            )

            # ------------------------------------------------
            # Presentation crop
            # ------------------------------------------------

            crop, offset_x, offset_y = (
                crop_around_points(
                    img,
                    points,
                )
            )

            local_pts = np.asarray(
                points,
                np.float32,
            ).copy()

            local_pts[:, 0] -= offset_x
            local_pts[:, 1] -= offset_y

            draw_quad(
                crop,
                local_pts,
            )

            draw_result_box(
                crop,
                10,
                min(
                    crop.shape[0] - 10,
                    120,
                ),
                i,
                distance_m,
                yaw_deg,
                center_m,
                scale=0.72,
            )

            crop_path = (
                OUTPUT_DIR
                / (
                    f"{IMAGE_PATH.stem}"
                    f"_car_{i}_pose_callout.png"
                )
            )

            cv2.imwrite(
                str(crop_path),
                crop,
            )

            crop_results.append(
                crop
            )

            summary.append(
                (
                    i,
                    distance_m,
                    yaw_deg,
                    center_m,
                )
            )

            print(
                f"Saved: {crop_path}"
            )

        except Exception as exc:
            print(
                f"Plate {i} failed: {exc}"
            )

    full_path = (
        OUTPUT_DIR
        / (
            f"{IMAGE_PATH.stem}"
            "_full_pose_overlay.png"
        )
    )

    cv2.imwrite(
        str(full_path),
        full_overlay,
    )

    grid = make_grid(
        crop_results
    )

    grid_path = None

    if grid is not None:
        grid_path = (
            OUTPUT_DIR
            / (
                f"{IMAGE_PATH.stem}"
                "_pose_callouts_grid.png"
            )
        )

        cv2.imwrite(
            str(grid_path),
            grid,
        )

        print(
            f"Saved: {grid_path}"
        )

    summary_path = (
        OUTPUT_DIR
        / (
            f"{IMAGE_PATH.stem}"
            "_pose_results_summary.txt"
        )
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        for (
            i,
            distance_m,
            yaw_deg,
            center_m,
        ) in summary:

            f.write(
                f"Car {i}\n"
                f"  Distance to plate center: "
                f"{distance_m:.3f} m\n"
                f"  Yaw: "
                f"{yaw_deg:+.3f} deg\n"
                f"  Plate center in camera frame: "
                f"[{center_m[0]:.3f}, "
                f"{center_m[1]:.3f}, "
                f"{center_m[2]:.3f}] m\n\n"
            )

    print(
        f"\nSaved full overlay: {full_path}"
    )

    if grid_path is not None:
        print(
            f"Saved grid: {grid_path}"
        )

    print(
        f"Saved summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
