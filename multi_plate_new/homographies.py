import cv2
import numpy as np

PLATE_W_CM = 45.0
PLATE_H_CM = 10.0


def compute_plate_homography(image_points, plate_w_cm=PLATE_W_CM, plate_h_cm=PLATE_H_CM):
    object_points = np.array(
        [
            [0.0, 0.0],
            [plate_w_cm, 0.0],
            [plate_w_cm, plate_h_cm],
            [0.0, plate_h_cm],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(
        object_points,
        np.asarray(image_points, dtype=np.float32),
    )

    if H is None:
        raise RuntimeError("Could not compute plate homography.")

    return H


def compute_all_homographies(refinement_results):
    homographies = []

    for result in refinement_results:
        if not result.get("success", False):
            continue

        H = compute_plate_homography(result["image_points"])

        homographies.append({
            "index": result["index"],
            "image_points": result["image_points"],
            "H": H,
        })

    return homographies


def analyze_homography(H):
    _, singular_values, _ = np.linalg.svd(H)

    return {
        "det": float(np.linalg.det(H)),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "h31": float(H[2, 0]),
        "h32": float(H[2, 1]),
        "singular_values": singular_values,
    }


def print_homography_analysis(homographies):
    print("\n========== HOMOGRAPHY ANALYSIS ==========\n")

    for item in homographies:
        analysis = analyze_homography(item["H"])
        print(f"Plate {item['index']}")
        print(f"det(H): {analysis['det']:.6e}")
        print(f"cond(H): {analysis['condition_number']:.6e}")
        print(f"h31: {analysis['h31']:.6e}")
        print(f"h32: {analysis['h32']:.6e}")
        print(f"singular values: {analysis['singular_values']}")
        print()
