from pathlib import Path
import os
import sys

import cv2
from dotenv import load_dotenv
from multi_plate_new.visualization.multi_plate import (
    draw_refined_corners,
    draw_homography_plates_grid,
    draw_refined_plates_grid
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from single_plate_pose.detection import (
    detect_plate_with_roboflow,
    save_roboflow_result,
    load_roboflow_result,
)
from multi_plate_new.detection import (
    extract_plate_detections,
    non_max_suppression,
)
from multi_plate_new.refinement import refine_all_detections_yellow
from multi_plate_new.homographies import compute_all_homographies
from multi_plate_new.intrinsics import estimate_intrinsics_from_homographies

DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY")
DEFAULT_METHOD = "robust_mask_lines"

ROBOFLOW_JSON_DIR = PROJECT_ROOT / "roboflow_jsons"


def json_path_for_image(image_path):
    image_path = Path(image_path)
    return ROBOFLOW_JSON_DIR / f"roboflow_result_{image_path.stem}.json"


def load_or_run_roboflow(image_path, api_key, use_cached_json=True):
    image_path = Path(image_path)
    ROBOFLOW_JSON_DIR.mkdir(parents=True, exist_ok=True)

    json_path = json_path_for_image(image_path)

    if use_cached_json and json_path.exists():
        print(f"Loading cached Roboflow result: {json_path}")
        return load_roboflow_result(json_path)

    print("Running Roboflow detection...")
    _, raw_result = detect_plate_with_roboflow(str(image_path), api_key)
    save_roboflow_result(raw_result, json_path)
    print(f"Saved Roboflow result: {json_path}")

    return raw_result

def calibrate_from_image(
    image_path,
    api_key=None,
    *,
    use_cached_json=True,
    method=DEFAULT_METHOD,
    min_confidence=0.35,
    min_area=300.0,
    iou_threshold=0.5,
    min_width=80.0,
    min_height=20.0,
    debug_refinement=False,
    save_visualizations=True,
):
    """
    Estimate camera intrinsics K from one image containing at least
    three visible planar license plates.

    Pipeline:
        detection
        -> refinement
        -> homographies
        -> save visual diagnostics
        -> Zhang calibration
        -> K

    Visual diagnostics are saved BEFORE Zhang calibration, so they
    are still available even if intrinsic recovery fails.
    """

    image_path = Path(image_path)

    if api_key is None:
        api_key = DEFAULT_API_KEY

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not available.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # ---------------------------------------------------------
    # 1. Detection
    # ---------------------------------------------------------

    raw_result = load_or_run_roboflow(
        image_path=image_path,
        api_key=api_key,
        use_cached_json=use_cached_json,
    )

    detections = extract_plate_detections(
        raw_result,
        min_confidence=min_confidence,
        min_area=min_area,
    )

    detections = non_max_suppression(
        detections,
        iou_threshold=iou_threshold,
    )

    detections = [
        det
        for det in detections
        if det.width >= min_width
        and det.height >= min_height
    ]

    if len(detections) < 3:
        raise RuntimeError(
            "At least three valid plate detections are required."
        )

    # ---------------------------------------------------------
    # 2. Yellow-region refinement
    # ---------------------------------------------------------

    refinement_results = refine_all_detections_yellow(
        image,
        detections,
        debug=debug_refinement,
        method=method,
    )

    successful_refinements = [
        result
        for result in refinement_results
        if result.get("success", False)
    ]

    if len(successful_refinements) < 3:
        raise RuntimeError(
            "At least three successful plate refinements are required."
        )

    # ---------------------------------------------------------
    # 3. Homographies
    # ---------------------------------------------------------

    homographies = compute_all_homographies(
        refinement_results
    )

    if len(homographies) < 3:
        raise RuntimeError(
            "At least three valid homographies are required."
        )

    # ---------------------------------------------------------
    # 4. Save visual diagnostics BEFORE Zhang calibration
    # ---------------------------------------------------------

    if save_visualizations:
        output_dir = (
            PROJECT_ROOT
            / "multi_plate_new"
            / "outputs"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        draw_refined_corners(
            image,
            refinement_results,
            save_path=str(
                output_dir
                / "refined_multi_plate_corners.jpg"
            ),
        )

        draw_refined_plates_grid(
            image,
            refinement_results,
            save_path=str(
                output_dir
                / "refined_plates_grid.jpg"
            ),
            cols=3,
            show=False,
        )

        draw_homography_plates_grid(
            image,
            refinement_results,
            save_path=str(
                output_dir
                / "homography_input_plates_grid.jpg"
            ),
            show=False,
        )

        print(
            "\nSaved visualization outputs to:",
            output_dir,
        )

    # ---------------------------------------------------------
    # 5. Zhang intrinsic calibration
    # ---------------------------------------------------------

    intrinsics_result = (
        estimate_intrinsics_from_homographies(
            homographies
        )
    )

    # ---------------------------------------------------------
    # 6. Return successful result
    # ---------------------------------------------------------

    return {
        "image_path": image_path,
        "image": image,
        "raw_roboflow_result": raw_result,

        "detections": detections,
        "num_detections": len(detections),

        "refinement_results": refinement_results,
        "successful_refinements": successful_refinements,
        "num_successful_refinements": len(
            successful_refinements
        ),

        "homographies": homographies,

        "K_est": intrinsics_result["K"],
        "B_est": intrinsics_result["B"],
        "V": intrinsics_result["V"],

        "constraint_singular_values": (
            intrinsics_result["singular_values"]
        ),

        "constraint_condition_number": (
            intrinsics_result["condition_number"]
        ),
    }



def main():
    image_path = (
        PROJECT_ROOT
        / "cars_photos"
        / "multi_cars"
        / "IMG_4349.jpeg"
    )

    result = calibrate_from_image(
        image_path=image_path,
        use_cached_json=True,
        method="robust_mask_lines",
        debug_refinement=False,
        save_visualizations=True,
    )

    print(
        "\n========== MULTI-PLATE CALIBRATION =========="
    )

    print(
        "Detections:",
        result["num_detections"],
    )

    print(
        "Successful refinements:",
        result["num_successful_refinements"],
    )

    print(
        "Homographies:",
        len(result["homographies"]),
    )

    print("\nEstimated K:")
    print(result["K_est"])

    print("\nConstraint condition number:")
    print(
        f"{result['constraint_condition_number']:.6e}"
    )


if __name__ == "__main__":
    main()
