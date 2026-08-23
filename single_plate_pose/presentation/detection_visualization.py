from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# car/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "single_plate_pose"
    / "presentation"
    / "outputs"
    / "detection"
)


# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

POLYGON_COLOR = (0, 255, 0)

POLYGON_THICKNESS = 2
POINT_RADIUS = 2

CONFIDENCE_FONT_SCALE = 0.7
CONFIDENCE_THICKNESS = 2


# ============================================================
# LOAD ROBOFLOW PREDICTIONS
# ============================================================

def load_predictions(json_path: Path):
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Expected JSON structure:
    #
    # [
    #     {
    #         "annotated_image": ...,
    #         "predictions": {
    #             "image": {...},
    #             "predictions": [...]
    #         }
    #     }
    # ]

    return data[0]["predictions"]["predictions"]


# ============================================================
# DRAW ONE PREDICTION
# ============================================================

def draw_prediction(
    image,
    prediction,
    plate_index,
):
    confidence = prediction[
        "confidence"
    ]

    points = prediction.get(
        "points",
        [],
    )

    if len(points) == 0:
        print(
            f"Plate {plate_index}: "
            "no polygon points"
        )

        return image

    polygon_points = np.array(
        [
            [
                int(point["x"]),
                int(point["y"]),
            ]
            for point in points
        ],
        dtype=np.int32,
    )

    polygon = polygon_points.reshape(
        (-1, 1, 2)
    )

    # --------------------------------------------------------
    # Draw segmentation polygon
    # --------------------------------------------------------

    cv2.polylines(
        image,
        [polygon],
        isClosed=True,
        color=POLYGON_COLOR,
        thickness=POLYGON_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw actual Roboflow polygon points
    # --------------------------------------------------------

    for x, y in polygon_points:
        cv2.circle(
            image,
            (x, y),
            radius=POINT_RADIUS,
            color=POLYGON_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Confidence label
    # --------------------------------------------------------

    min_x = int(
        np.min(
            polygon_points[:, 0]
        )
    )

    min_y = int(
        np.min(
            polygon_points[:, 1]
        )
    )

    label = (
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        image,
        label,
        (
            min_x,
            max(
                min_y - 10,
                25,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        CONFIDENCE_FONT_SCALE,
        POLYGON_COLOR,
        CONFIDENCE_THICKNESS,
        cv2.LINE_AA,
    )

    print(
        f"Plate {plate_index}: "
        f"confidence={confidence:.3f}, "
        f"polygon_points={len(points)}"
    )

    return image


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

    predictions = load_predictions(
        JSON_PATH
    )

    print(
        f"\nFound {len(predictions)} "
        "plate prediction(s)\n"
    )

    output = image.copy()

    for i, prediction in enumerate(
        predictions,
        start=1,
    ):
        output = draw_prediction(
            output,
            prediction,
            i,
        )

    output_path = (
        OUTPUT_DIR
        / (
            f"{IMAGE_PATH.stem}"
            "_roboflow_segmentation.jpg"
        )
    )

    cv2.imwrite(
        str(output_path),
        output,
    )

    print(
        "\nSaved visualization to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
