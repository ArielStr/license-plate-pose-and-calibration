import cv2
import numpy as np

from ..common import order_points


def add_offset(points_crop, x_offset, y_offset):
    points_full = points_crop.astype(np.float32).copy()
    points_full[:, 0] += x_offset
    points_full[:, 1] += y_offset
    return points_full


def get_min_area_rect_points(contour, x_offset, y_offset):
    rect = cv2.minAreaRect(contour)
    box_crop = cv2.boxPoints(rect).astype(np.float32)
    points_full = add_offset(box_crop, x_offset, y_offset)

    return order_points(points_full), rect