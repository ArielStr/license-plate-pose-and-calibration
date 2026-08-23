import cv2
import numpy as np


def estimate_dynamic_yellow_hsv(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    roi = hsv[y1:y2, x1:x2]
    pixels = roi.reshape(-1, 3)

    candidates = pixels[
        (pixels[:, 0] >= 8) &
        (pixels[:, 0] <= 55) &
        (pixels[:, 1] >= 25) &
        (pixels[:, 2] >= 25)
    ]

    if len(candidates) < 50:
        return None

    median_h, median_s, median_v = np.median(candidates, axis=0)

    lower = np.array([
        max(0, median_h - 14),
        max(20, median_s - 70),
        max(15, median_v - 90)
    ], dtype=np.uint8)

    upper = np.array([
        min(179, median_h + 14),
        255,
        255
    ], dtype=np.uint8)

    return lower, upper, (median_h, median_s, median_v), len(candidates)

def choose_geometry_config(pred_w):
    open_kernel_size = None
    open_iterations = 0

    if pred_w > 800:
        mode = "close"
        close_kernel_size = (21, 5)
        close_iterations = 2
        fill_kernel_size = (9, 9)
        fill_iterations = 1

    elif pred_w > 350:
        mode = "medium"
        open_kernel_size = (3, 3)
        open_iterations = 1
        close_kernel_size = (11, 1)
        close_iterations = 1
        fill_kernel_size = None
        fill_iterations = 0

    else:
        mode = "far"
        open_kernel_size = (3, 3)
        open_iterations = 1
        close_kernel_size = (7, 1)
        close_iterations = 1
        fill_kernel_size = None
        fill_iterations = 0

    return {
        "mode": mode,
        "open_kernel": open_kernel_size,
        "open_iterations": open_iterations,
        "close_kernel": close_kernel_size,
        "close_iterations": close_iterations,
        "fill_kernel": fill_kernel_size,
        "fill_iterations": fill_iterations,
    }