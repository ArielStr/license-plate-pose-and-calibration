import cv2
import numpy as np

from ..common import order_points
from .min_area_rect import add_offset


def get_approx_quad_points(
        contour,
        x_offset,
        y_offset,
        eps_values=None,
        debug_info=None
):
    if eps_values is None:
        eps_values = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]

    peri = cv2.arcLength(contour, True)

    for eps in eps_values:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        pts_crop = approx.reshape(-1, 2).astype(np.float32)

        if debug_info is not None:
            debug_info["eps_results"].append({
                "eps": eps,
                "num_vertices": len(pts_crop),
                "points_crop": pts_crop
            })

        print(f"approxPolyDP eps={eps} -> {len(pts_crop)} vertices")

        if len(pts_crop) == 4:
            pts_full = add_offset(pts_crop, x_offset, y_offset)
            print(f"Corner method candidate: contour_quad eps={eps}")
            return order_points(pts_full)

    print("Corner method candidate: contour_quad failed")
    return None