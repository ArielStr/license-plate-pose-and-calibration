from pathlib import Path
import json

from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# car/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOFLOW_JSON_DIR = (
    PROJECT_ROOT
    / "roboflow_jsons"
)


# ============================================================
# ROBOFLOW HELPERS
# ============================================================

def find_predictions_recursive(obj):
    found = []

    if isinstance(obj, dict):
        if all(
            key in obj
            for key in ["x", "y", "width", "height"]
        ):
            found.append(obj)

        for value in obj.values():
            found.extend(
                find_predictions_recursive(value)
            )

    elif isinstance(obj, list):
        for item in obj:
            found.extend(
                find_predictions_recursive(item)
            )

    return found


def detect_plate_with_roboflow(
    image_path,
    api_key,
):
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )

    print(f"Sending to Roboflow: {image_path}")

    result = client.run_workflow(
        workspace_name="ariel-stoenescu",
        workflow_id="general-segmentation-api-5",
        images={"image": image_path},
        parameters={"classes": "LicensePlate"},
        use_cache=True,
    )

    print("\n========== RAW ROBOFLOW RESULT ==========")
    print(json.dumps(result, indent=2))
    print("=========================================\n")

    predictions = find_predictions_recursive(result)

    print(f"Recursive predictions found: {len(predictions)}")

    if len(predictions) == 0:
        raise RuntimeError(
            "No license plate detected by Roboflow workflow."
        )

    best_pred = max(
        predictions,
        key=lambda p: float(
            p.get("confidence", 0.0)
        ),
    )

    return best_pred, result


# ============================================================
# CACHE
# ============================================================

def save_roboflow_result(
    result,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )


def load_roboflow_result(
    input_path,
):
    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def json_path_for_image(
    image_path,
):
    image_path = Path(
        image_path
    )

    return (
        ROBOFLOW_JSON_DIR
        / f"roboflow_result_{image_path.stem}.json"
    )


# ============================================================
# PREDICTION SELECTION
# ============================================================

def prediction_score(prediction):
    confidence = float(
        prediction.get(
            "confidence",
            0.0,
        )
    )

    width = float(
        prediction.get(
            "width",
            0.0,
        )
    )

    height = float(
        prediction.get(
            "height",
            0.0,
        )
    )

    return (
        confidence
        * width
        * height
    )


def get_predictions(
    image_path,
    api_key,
):
    ROBOFLOW_JSON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = json_path_for_image(
        image_path
    )

    if json_path.exists():
        print(
            "Loading Roboflow result from JSON: "
            f"{json_path}"
        )

        raw_result = load_roboflow_result(
            json_path
        )

    else:
        print(
            "Running Roboflow..."
        )

        _, raw_result = detect_plate_with_roboflow(
            image_path,
            api_key,
        )

        save_roboflow_result(
            raw_result,
            json_path,
        )

    predictions = find_predictions_recursive(
        raw_result
    )

    valid_predictions = [
        prediction
        for prediction in predictions
        if float(
            prediction.get(
                "width",
                0.0,
            )
        ) > 100
        and float(
            prediction.get(
                "height",
                0.0,
            )
        ) > 20
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


def get_prediction(
    image_path,
    api_key,
):
    predictions = get_predictions(
        image_path,
        api_key,
    )

    return predictions[0]
