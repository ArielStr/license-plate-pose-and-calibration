import cv2
import numpy as np

from .common import (
    bbox_from_prediction_points_xy_padding,
    crop_from_bbox
)
from corner_refinement.debug.debug_visualization import _draw_yellow_refinement_debug

from .methods.min_area_rect import get_min_area_rect_points
from .methods.contour_quad import get_approx_quad_points
from .methods.mask_lines import corners_from_contour_lines
from .methods.robust_mask_lines import corners_from_robust_mask_lines
from .yellow_utils import (
    estimate_dynamic_yellow_hsv,
    choose_geometry_config
)

def refine_yellow_inner_corners(
        img,
        pred,
        debug=True,
        method="min_area_rect",
        debug_name="yellow_inner_corners"
):
    bbox = bbox_from_prediction_points_xy_padding(
        pred,
        img.shape,
        padding_x_ratio=0.02,
        padding_y_ratio=0.15
    )
    crop, x_offset, y_offset = crop_from_bbox(img, bbox)
    pred_w = float(pred["width"])

    config = choose_geometry_config(pred_w)
    open_kernel_size = config["open_kernel"]
    open_iterations = config["open_iterations"]
    close_kernel_size = config["close_kernel"]
    close_iterations = config["close_iterations"]
    fill_kernel_size = config["fill_kernel"]
    fill_iterations = config["fill_iterations"]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    dynamic_result = estimate_dynamic_yellow_hsv(crop)

    if dynamic_result is not None:
        lower_yellow, upper_yellow, median_hsv, num_candidates = dynamic_result
    else:
        if pred_w > 800:
            lower_yellow = np.array([18, 80, 40])
            upper_yellow = np.array([38, 255, 255])
        elif pred_w > 350:
            lower_yellow = np.array([18, 80, 30])
            upper_yellow = np.array([42, 255, 255])
        else:
            lower_yellow = np.array([14, 95, 35])
            upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    if open_kernel_size is not None:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=open_iterations)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)

    if fill_kernel_size is not None:
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, fill_kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill, iterations=fill_iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:
        # label 0 הוא הרקע, לכן מתחילים מ-1
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 255
        mask = clean_mask
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise RuntimeError("No yellow region found.")

    largest = max(contours, key=cv2.contourArea)

    rect_points, rect = get_min_area_rect_points(
        largest,
        x_offset,
        y_offset
    )

    quad_points = None
    line_points = None
    robust_line_points = None
    if method in ["contour_quad", "mask_lines"]:
        quad_points = get_approx_quad_points(
            largest,
            x_offset,
            y_offset
        )

    if method == "mask_lines":
        line_points = corners_from_contour_lines(
            largest,
            x_offset,
            y_offset,
            crop=crop,
            mask=mask,
            debug=debug,
            debug_name=debug_name
        )

        if line_points is not None:
            print("Corner method candidate: mask_lines")
        else:
            print("Corner method candidate: mask_lines failed")

    if method == "robust_mask_lines":
        robust_line_points = corners_from_robust_mask_lines(
            largest,
            x_offset,
            y_offset,
            crop=crop,
            mask=mask,
            debug=debug,
            debug_name=debug_name
        )

        if robust_line_points is not None:
            print("Corner method candidate: robust_mask_lines")
        else:
            print("Corner method candidate: robust_mask_lines failed")

    w = rect[1][0]
    h = rect[1][1]
    ratio = max(w, h) / min(w, h)

    print("\n========== YELLOW REGION STATS ==========")
    print(f"Contour area: {cv2.contourArea(largest):.1f} px")
    print(f"Width : {max(w, h):.2f} px")
    print(f"Height: {min(w, h):.2f} px")
    print(f"Aspect ratio: {ratio:.4f}")
    print("=========================================\n")
    if method == "min_area_rect":
        image_points = rect_points
    elif method == "contour_quad":
        if quad_points is None:
            raise RuntimeError(
                "contour_quad failed: approxPolyDP did not find 4 corners"
            )
        image_points = quad_points
    elif method == "mask_lines":
        if line_points is None:
            raise RuntimeError("mask_lines failed")
        image_points = line_points
    elif method == "robust_mask_lines":
        if robust_line_points is None:
            raise RuntimeError("robust_mask_lines failed")
        image_points = robust_line_points
    else:
        raise ValueError(f"Unknown method: {method}")

    print("\nSelected points:")
    for i, p in enumerate(image_points):
        print(f"{i}: {p}")

    if debug:
        _draw_yellow_refinement_debug(
            crop=crop,
            contour=largest,
            rect_points=rect_points,
            quad_points=quad_points,
            line_points=line_points,
            selected_points=image_points,
            x_offset=x_offset,
            y_offset=y_offset,
            debug_name=debug_name,
            method=method
        )



    return image_points, bbox, mask


