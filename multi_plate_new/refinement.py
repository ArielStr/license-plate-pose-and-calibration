from __future__ import annotations
from pathlib import Path
import cv2

from corner_refinement import refine_yellow_inner_corners
from multi_plate_new.detection import PlateDetection


def crop_plate(image, detection, padding_x_ratio=0.02, padding_y_ratio=0.15):
    x1, y1, x2, y2 = detection.bbox_xyxy
    w, h = x2 - x1, y2 - y1

    pad_x = int(w * padding_x_ratio)
    pad_y = int(h * padding_y_ratio)

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(image.shape[1], int(x2 + pad_x))
    y2 = min(image.shape[0], int(y2 + pad_y))

    return image[y1:y2, x1:x2].copy(), (x1, y1)


def save_plate_crops(image, detections, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_infos = []

    for idx, det in enumerate(detections):
        crop, origin = crop_plate(image, det)
        path = output_dir / f"plate_{idx}.jpg"
        cv2.imwrite(str(path), crop)

        crop_infos.append({
            "index": idx,
            "path": str(path),
            "crop": crop,
            "origin": origin,
            "detection": det,
        })

    return crop_infos


def detection_to_prediction_dict(det: PlateDetection):
    points = None
    if det.points is not None:
        points = [{"x": float(p[0]), "y": float(p[1])} for p in det.points]

    return {
        "x": float(det.x),
        "y": float(det.y),
        "width": float(det.width),
        "height": float(det.height),
        "confidence": float(det.confidence),
        "class": det.class_name,
        "points": points,
    }


def refine_all_detections_yellow(image, detections, debug=False, method="robust_mask_lines"):
    results = []

    for idx, det in enumerate(detections):
        pred = detection_to_prediction_dict(det)

        try:
            image_points, bbox, mask = refine_yellow_inner_corners(
                image,
                pred,
                debug=debug,
                method=method,
                debug_name=f"plate_{idx:02d}",
            )

            results.append({
                "index": idx,
                "success": True,
                "image_points": image_points,
                "bbox": bbox,
                "mask": mask,
                "detection": det,
            })
            print(f"Plate {idx}: SUCCESS")

        except Exception as exc:
            results.append({
                "index": idx,
                "success": False,
                "detection": det,
                "error": str(exc),
            })
            print(f"Plate {idx}: FAILED - {exc}")

    return results
