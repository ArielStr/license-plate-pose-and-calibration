import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def bbox_from_prediction_points(pred, img_shape, padding_ratio=0.25):
    img_h, img_w = img_shape[:2]
    points = pred.get("points", [])

    if len(points) == 0:
        raise RuntimeError("Prediction has no segmentation points.")

    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)

    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

    plate_w = max_x - min_x
    plate_h = max_y - min_y

    pad_x = padding_ratio * plate_w
    pad_y = padding_ratio * plate_h

    x1 = max(0, int(min_x - pad_x))
    y1 = max(0, int(min_y - pad_y))
    x2 = min(img_w - 1, int(max_x + pad_x))
    y2 = min(img_h - 1, int(max_y + pad_y))

    return x1, y1, x2, y2


def crop_from_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("Empty crop.")

    return crop, x1, y1

def bbox_from_prediction_points_xy_padding(
        pred,
        img_shape,
        padding_x_ratio=0.02,
        padding_y_ratio=0.15
):
    img_h, img_w = img_shape[:2]

    cx = float(pred["x"])
    cy = float(pred["y"])
    w = float(pred["width"])
    h = float(pred["height"])

    x1 = int(cx - w / 2 - w * padding_x_ratio)
    y1 = int(cy - h / 2 - h * padding_y_ratio)
    x2 = int(cx + w / 2 + w * padding_x_ratio)
    y2 = int(cy + h / 2 + h * padding_y_ratio)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return x1, y1, x2, y2