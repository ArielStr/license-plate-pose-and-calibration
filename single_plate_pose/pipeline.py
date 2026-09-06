from pathlib import Path
import os
import sys

import cv2
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# car/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


from corner_refinement import refine_yellow_inner_corners
from single_plate_pose.detection import get_predictions
from single_plate_pose.pose_estimation import estimate_plate_pose
from single_plate_pose.visualization import draw_pose_on_image


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv(
    "ROBOFLOW_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set."
    )


CALIBRATION_PATH = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results.npz"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "multi_cars"
    / "IMG_4425.jpeg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "outputs"
)

MAX_PLATES = None
METHOD = "yellow_exit_ransac"
USE_DISTORTION = False


# ============================================================
# MAIN PIPELINE
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

    predictions = get_predictions(
        str(IMAGE_PATH),
        API_KEY,
    )

    if MAX_PLATES is not None:
        predictions = predictions[
            :MAX_PLATES
        ]

    print(
        f"Found {len(predictions)} "
        "plate predictions"
    )

    output = img.copy()

    for i, pred in enumerate(
        predictions,
        start=1,
    ):
        print(
            "\n=============================="
        )
        print(
            f"PLATE {i}"
        )
        print(
            "=============================="
        )

        try:
            yellow_points, _, _ = (
                refine_yellow_inner_corners(
                    img,
                    pred,
                    debug=False,
                    method=METHOD,
                    debug_name=(
                        f"single_image_pose_plate_{i}"
                    ),
                )
            )

            pose = estimate_plate_pose(
                yellow_points,
                str(CALIBRATION_PATH),
                use_distortion=USE_DISTORTION,
            )

            output = draw_pose_on_image(
                output,
                yellow_points,
                pose,
                plate_index=i,
            )

        except Exception as exc:
            print(
                f"Failed on plate {i}: {exc}"
            )

    output_path = (
        OUTPUT_DIR
        / f"{IMAGE_PATH.stem}_pose_overlay.jpg"
    )

    cv2.imwrite(
        str(output_path),
        output,
    )

    print(
        "\nSaved pose overlay to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
