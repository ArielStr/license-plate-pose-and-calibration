import cv2
import numpy as np
from ..common import order_points
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEBUG_OUTPUT_DIR = (
    PROJECT_ROOT
    / "debug"
    / "corner_refinement"
    / "mask_lines"
)
def fit_line_from_points(points):
    points = points.astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    return float(vx), float(vy), float(x0), float(y0)


def intersect_lines(line1, line2):
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    A = np.array([
        [vx1, -vx2],
        [vy1, -vy2]
    ], dtype=np.float64)

    b = np.array([
        x2 - x1,
        y2 - y1
    ], dtype=np.float64)

    if abs(np.linalg.det(A)) < 1e-8:
        return None

    t, _ = np.linalg.solve(A, b)

    return np.array([
        x1 + t * vx1,
        y1 + t * vy1
    ], dtype=np.float32)


def point_to_segment_distance(p, a, b):
    p = p.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    ab = b - a
    denom = np.dot(ab, ab)

    if denom < 1e-8:
        return np.linalg.norm(p - a)

    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    return np.linalg.norm(p - closest)


def corners_from_contour_lines(
        contour,
        x_offset,
        y_offset,
        crop=None,
        mask=None,
        debug=True,
        debug_name="mask_lines"
):
    pts = contour.reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_points(box)

    tl, tr, br, bl = box

    top_pts = []
    right_pts = []
    bottom_pts = []
    left_pts = []

    for p in pts:
        d_top = point_to_segment_distance(p, tl, tr)
        d_right = point_to_segment_distance(p, tr, br)
        d_bottom = point_to_segment_distance(p, bl, br)
        d_left = point_to_segment_distance(p, tl, bl)

        side = np.argmin([d_top, d_right, d_bottom, d_left])

        if side == 0:
            top_pts.append(p)
        elif side == 1:
            right_pts.append(p)
        elif side == 2:
            bottom_pts.append(p)
        else:
            left_pts.append(p)

    groups = {
        "top": top_pts,
        "right": right_pts,
        "bottom": bottom_pts,
        "left": left_pts
    }

    if debug:
        print("\n========== MASK_LINES DEBUG ==========")
        print(f"Total contour points: {len(pts)}")
        print(f"top points    : {len(top_pts)}")
        print(f"right points  : {len(right_pts)}")
        print(f"bottom points : {len(bottom_pts)}")
        print(f"left points   : {len(left_pts)}")
        print("Initial minAreaRect box:")
        print(box)
        print("======================================\n")

    MIN_POINTS_PER_SIDE = 3

    if any(len(g) < MIN_POINTS_PER_SIDE for g in groups.values()):
        print("mask_lines failed: not enough points on one or more sides")

        if debug and crop is not None:
            draw_mask_lines_debug(
                crop=crop,
                mask=mask,
                contour=contour,
                box=box,
                top_pts=np.array(top_pts, dtype=np.float32),
                right_pts=np.array(right_pts, dtype=np.float32),
                bottom_pts=np.array(bottom_pts, dtype=np.float32),
                left_pts=np.array(left_pts, dtype=np.float32),
                lines=None,
                corners_crop=None,
                debug_name=debug_name
            )

        return None

    top_line = fit_line_from_points(np.array(top_pts))
    right_line = fit_line_from_points(np.array(right_pts))
    bottom_line = fit_line_from_points(np.array(bottom_pts))
    left_line = fit_line_from_points(np.array(left_pts))

    lines = {
        "top": top_line,
        "right": right_line,
        "bottom": bottom_line,
        "left": left_line
    }

    if debug:
        print("Fitted lines:")
        print(f"top    : {top_line}")
        print(f"right  : {right_line}")
        print(f"bottom : {bottom_line}")
        print(f"left   : {left_line}")

        draw_mask_lines_debug(
            crop=crop,
            mask=mask,
            contour=contour,
            box=box,
            top_pts=np.array(top_pts, dtype=np.float32),
            right_pts=np.array(right_pts, dtype=np.float32),
            bottom_pts=np.array(bottom_pts, dtype=np.float32),
            left_pts=np.array(left_pts, dtype=np.float32),
            lines=None,
            corners_crop=None,
            debug_name=debug_name
        )

    tl_p = intersect_lines(top_line, left_line)
    tr_p = intersect_lines(top_line, right_line)
    br_p = intersect_lines(bottom_line, right_line)
    bl_p = intersect_lines(bottom_line, left_line)

    if debug:
        print("\nLine intersections:")
        print(f"tl: {tl_p}")
        print(f"tr: {tr_p}")
        print(f"br: {br_p}")
        print(f"bl: {bl_p}")
        print("======================================\n")

    if any(p is None for p in [tl_p, tr_p, br_p, bl_p]):
        print("mask_lines failed: one or more line intersections are None")
        return None

    corners_crop = np.array([tl_p, tr_p, br_p, bl_p], dtype=np.float32)

    if debug and crop is not None:
        draw_mask_lines_debug(
            crop=crop,
            mask=mask,
            contour=contour,
            box=box,
            top_pts=np.array(top_pts, dtype=np.float32),
            right_pts=np.array(right_pts, dtype=np.float32),
            bottom_pts=np.array(bottom_pts, dtype=np.float32),
            left_pts=np.array(left_pts, dtype=np.float32),
            lines=lines,
            corners_crop=corners_crop,
            debug_name=debug_name
        )

    corners_full = corners_crop.copy()
    corners_full[:, 0] += x_offset
    corners_full[:, 1] += y_offset

    return order_points(corners_full)

def draw_mask_lines_debug(
        crop,
        mask,
        contour,
        box,
        top_pts,
        right_pts,
        bottom_pts,
        left_pts,
        lines,
        corners_crop,
        debug_name="mask_lines"
):
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # 1. contour + initial minAreaRect box
    img_box = crop.copy()
    cv2.drawContours(img_box, [contour], -1, (0, 0, 255), 2)

    box_int = box.astype(np.int32)
    cv2.polylines(img_box, [box_int], True, (255, 0, 0), 2)

    img_box = make_panel(
        img_box,
        "1 initial box: red=contour, blue=minAreaRect"
    )

    if mask is not None:
        img_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        img_mask = np.zeros_like(crop)

    img_mask = make_panel(
        img_mask,
        "2 yellow mask"
    )

    # 2. side assignment
    img_groups = crop.copy()
    cv2.drawContours(img_groups, [contour], -1, (180, 180, 180), 1)

    side_groups = [
        ("top", top_pts, (255, 0, 0)),
        ("right", right_pts, (0, 255, 255)),
        ("bottom", bottom_pts, (0, 255, 0)),
        ("left", left_pts, (0, 0, 255)),
    ]
    box_int = box.astype(np.int32)
    cv2.polylines(
        img_groups,
        [box_int],
        True,
        (255, 255, 255),
        2
    )
    for name, pts, color in side_groups:
        for p in pts:
            x, y = p.astype(int)
            cv2.circle(img_groups, (x, y), 3, color, -1)

    y_text = 55
    for name, pts, color in side_groups:
        cv2.putText(
            img_groups,
            f"{name}: {len(pts)} pts",
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
        y_text += 22

    img_groups = make_panel(img_groups, "3 side assignment")

    # 3. fitted lines
    img_lines = crop.copy()
    cv2.drawContours(img_lines, [contour], -1, (180, 180, 180), 1)

    def draw_infinite_line(img, line, color):
        vx, vy, x0, y0 = line
        h, w = img.shape[:2]

        scale = max(h, w)

        x1 = int(x0 - vx * scale)
        y1 = int(y0 - vy * scale)
        x2 = int(x0 + vx * scale)
        y2 = int(y0 + vy * scale)

        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    if lines is not None:
        draw_infinite_line(img_lines, lines["top"], (255, 0, 0))
        draw_infinite_line(img_lines, lines["right"], (0, 255, 255))
        draw_infinite_line(img_lines, lines["bottom"], (0, 255, 0))
        draw_infinite_line(img_lines, lines["left"], (0, 0, 255))
    else:
        cv2.putText(
            img_lines,
            "lines failed",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    img_lines = make_panel(img_lines, "4 fitted lines")

    # 4. final corners from intersections
    img_corners = crop.copy()
    cv2.drawContours(img_corners, [contour], -1, (180, 180, 180), 1)

    if corners_crop is not None:
        corners_int = corners_crop.astype(np.int32)
        cv2.polylines(img_corners, [corners_int], True, (0, 255, 0), 3)

        for i, (x, y) in enumerate(corners_int):
            cv2.circle(img_corners, (int(x), int(y)), 7, (0, 255, 0), -1)
            cv2.putText(
                img_corners,
                str(i),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    else:
        cv2.putText(
            img_corners,
            "corners failed",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    img_corners = make_panel(img_corners, "5 final corners")

    # Build 2x2 grid
    empty_panel = np.zeros_like(img_corners)
    cv2.putText(
        empty_panel,
        "mask_lines debug",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    row1 = np.hstack([img_box, img_mask])
    row2 = np.hstack([img_groups, img_lines])
    row3 = np.hstack([img_corners, empty_panel])

    grid = np.vstack([row1, row2, row3])

    output_path = DEBUG_OUTPUT_DIR / f"{debug_name}_mask_lines_grid.jpg"
    cv2.imwrite(str(output_path), grid)

DEBUG_SCALE = 3
BORDER = 25
HEADER_H = 42

def make_panel(img, title):
    img = cv2.copyMakeBorder(
        img,
        BORDER,
        BORDER,
        BORDER,
        BORDER,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    img = cv2.resize(
        img,
        None,
        fx=DEBUG_SCALE,
        fy=DEBUG_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    header = np.zeros((HEADER_H, img.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        header,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return np.vstack([header, img])