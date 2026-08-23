from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from single_plate_pose.detection import find_predictions_recursive


@dataclass
class PlateDetection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    points: np.ndarray | None = None

    @property
    def bbox_xyxy(self):
        x1 = self.x - self.width / 2.0
        y1 = self.y - self.height / 2.0
        x2 = self.x + self.width / 2.0
        y2 = self.y + self.height / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


def extract_plate_detections(raw_result, min_confidence=0.4, min_area=500):
    predictions = find_predictions_recursive(raw_result)
    detections = []

    for pred in predictions:
        if not all(k in pred for k in ("x", "y", "width", "height")):
            continue

        confidence = float(pred.get("confidence", 0.0))
        width = float(pred["width"])
        height = float(pred["height"])

        if confidence < min_confidence or width * height < min_area:
            continue

        raw_points = pred.get("points")
        points = None
        if raw_points is not None:
            points = np.array(
                [[p["x"], p["y"]] for p in raw_points],
                dtype=np.float32,
            )

        detections.append(
            PlateDetection(
                x=float(pred["x"]),
                y=float(pred["y"]),
                width=width,
                height=height,
                confidence=confidence,
                class_name=str(pred.get("class", pred.get("class_name", ""))),
                points=points,
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def non_max_suppression(detections, iou_threshold=0.5):
    kept = []

    for det in detections:
        duplicate = any(
            bbox_iou(det.bbox_xyxy, kept_det.bbox_xyxy) > iou_threshold
            for kept_det in kept
        )
        if not duplicate:
            kept.append(det)

    return kept
