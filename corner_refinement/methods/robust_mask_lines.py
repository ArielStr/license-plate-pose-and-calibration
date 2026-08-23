import cv2
import numpy as np


def _fit_line_ransac(points, num_iters=150, threshold=2.0, min_inliers=20):
    """
    Robustly fit a 2D line to points using simple RANSAC.
    Returns OpenCV-style line: vx, vy, x0, y0
    """
    points = np.asarray(points, dtype=np.float32)

    if len(points) < min_inliers:
        return None

    best_inliers = None
    best_count = 0

    for _ in range(num_iters):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]

        direction = p2 - p1
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            continue

        direction = direction / norm

        # Distance from point to line
        diffs = points - p1
        cross = np.abs(diffs[:, 0] * direction[1] - diffs[:, 1] * direction[0])

        inliers = cross < threshold
        count = np.count_nonzero(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    inlier_points = points[best_inliers]

    line = cv2.fitLine(
        inlier_points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    return line.reshape(-1)


def _intersect_lines(line1, line2):
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

    det = np.linalg.det(A)

    if abs(det) < 1e-8:
        return None

    t, _ = np.linalg.solve(A, b)

    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

def _convert_groups(groups):
    out = {}

    for name, pts in groups.items():
        if len(pts) == 0:
            out[name] = np.empty((0, 2), dtype=np.float32)
        else:
            out[name] = np.array(pts, dtype=np.float32)

    return out
def _edge_points_from_external_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)

    xs = pts[:, 0]
    ys = pts[:, 1]

    top = []
    bottom = []
    left = []
    right = []

    for x in np.unique(xs):
        y_vals = ys[xs == x]
        top.append([x, np.min(y_vals)])
        bottom.append([x, np.max(y_vals)])

    for y in np.unique(ys):
        x_vals = xs[ys == y]
        left.append([np.min(x_vals), y])
        right.append([np.max(x_vals), y])

    return (
        np.asarray(top, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        np.asarray(bottom, dtype=np.float32),
        np.asarray(left, dtype=np.float32),
    )
def _corners_from_edge_extractor(mask, extractor):
    edge_points = extractor(mask)

    if edge_points is None:
        return None

    top, right, bottom, left = edge_points

    top = _trim_by_axis(top, axis=0, trim_ratio=0.08)
    bottom = _trim_by_axis(bottom, axis=0, trim_ratio=0.08)
    left = _trim_by_axis(left, axis=1, trim_ratio=0.08)
    right = _trim_by_axis(right, axis=1, trim_ratio=0.08)

    top_line = _fit_line_ransac(top, threshold=2.0)
    right_line = _fit_line_ransac(right, threshold=2.0)
    bottom_line = _fit_line_ransac(bottom, threshold=2.0)
    left_line = _fit_line_ransac(left, threshold=2.0)

    if any(line is None for line in [top_line, right_line, bottom_line, left_line]):
        return None

    tl = _intersect_lines(top_line, left_line)
    tr = _intersect_lines(top_line, right_line)
    br = _intersect_lines(bottom_line, right_line)
    bl = _intersect_lines(bottom_line, left_line)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    return np.array([tl, tr, br, bl], dtype=np.float32)
def _edge_points_from_oriented_external_contour(
        mask,
        num_bins_long=140,
        num_bins_short=45,
        min_points_per_bin=1,
):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float32)

    if len(pts) < 10:
        return None

    center = np.mean(pts, axis=0)
    centered = pts - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    u = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    u = u / (np.linalg.norm(u) + 1e-8)

    if u[0] < 0:
        u = -u

    v = np.array([-u[1], u[0]], dtype=np.float32)
    if v[1] < 0:
        v = -v

    uv = np.column_stack([
        centered @ u,
        centered @ v,
    ])

    us = uv[:, 0]
    vs = uv[:, 1]

    def pick_extreme_points(values, other_values, bins, choose_min=True):
        chosen = []

        for i in range(len(bins) - 1):
            a = bins[i]
            b = bins[i + 1]

            if i == len(bins) - 2:
                keep = (values >= a) & (values <= b)
            else:
                keep = (values >= a) & (values < b)

            idxs = np.where(keep)[0]
            if len(idxs) < min_points_per_bin:
                continue

            local_other = other_values[idxs]
            local_idx = idxs[
                np.argmin(local_other) if choose_min else np.argmax(local_other)
            ]
            chosen.append(pts[local_idx])

        if len(chosen) == 0:
            return None

        return np.asarray(chosen, dtype=np.float32)

    u_bins = np.linspace(np.min(us), np.max(us), num_bins_long + 1)
    v_bins = np.linspace(np.min(vs), np.max(vs), num_bins_short + 1)

    top = pick_extreme_points(us, vs, u_bins, choose_min=True)
    bottom = pick_extreme_points(us, vs, u_bins, choose_min=False)
    left = pick_extreme_points(vs, us, v_bins, choose_min=True)
    right = pick_extreme_points(vs, us, v_bins, choose_min=False)

    if any(x is None for x in [top, right, bottom, left]):
        return None

    return (
        top,
        right,
        bottom,
        left,
    )
def _order_points_tl_tr_br_bl(pts):
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)
def _trim_by_axis(points, axis, trim_ratio=0.08):
    """
    Remove extreme points along x/y axis.
    Helps ignore screws / strange local artifacts near corners.
    axis=0 trims by x, axis=1 trims by y
    """
    if len(points) < 10:
        return points

    values = points[:, axis]
    lo = np.quantile(values, trim_ratio)
    hi = np.quantile(values, 1.0 - trim_ratio)

    keep = (values >= lo) & (values <= hi)
    return points[keep]


def corners_from_robust_mask_lines(
        contour,
        x_offset,
        y_offset,
        crop=None,
        mask=None,
        debug=False,
        debug_name="robust_mask_lines"
):
    if mask is None:
        raise ValueError("robust_mask_lines requires mask")

    points = _corners_from_edge_extractor(
        mask,
        _edge_points_from_oriented_external_contour,
    )

    method_used = "oriented_external_ransac"

    if points is None:
        points = _corners_from_edge_extractor(
            mask,
            _edge_points_from_external_contour,
        )
        method_used = "external_contour_ransac"

    if points is None:
        return None

    # points = _order_points_tl_tr_br_bl(points)

    points[:, 0] += x_offset
    points[:, 1] += y_offset

    print(f"\nRobust mask lines method used: {method_used}")
    print("\nRETURN POINTS")
    for i, p in enumerate(points):
        print(i, p)

    return points

def debug_robust_mask_lines(mask):
    edge_points = _edge_points_from_oriented_external_contour(mask)

    if edge_points is None:
        return None

    top, right, bottom, left = edge_points

    top = _trim_by_axis(top, axis=0, trim_ratio=0.08)
    bottom = _trim_by_axis(bottom, axis=0, trim_ratio=0.08)
    left = _trim_by_axis(left, axis=1, trim_ratio=0.08)
    right = _trim_by_axis(right, axis=1, trim_ratio=0.08)

    top_line = _fit_line_ransac(top, threshold=2.0)
    bottom_line = _fit_line_ransac(bottom, threshold=2.0)

    if top_line is None or bottom_line is None:
        return None

    right_line = _fit_line_ransac(right, threshold=2.0)
    left_line = _fit_line_ransac(left, threshold=2.0)

    if any(line is None for line in [right_line, left_line]):
        return None

    if any(line is None for line in [top_line, right_line, bottom_line, left_line]):
        return None

    return {
        "edge_points": {
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
        },
        "lines": {
            "top": top_line,
            "right": right_line,
            "bottom": bottom_line,
            "left": left_line,
        }
    }

def debug_robust_mask_lines_full(mask):
    """
    Debug-only function.
    Returns all intermediate stages of robust_mask_lines
    without changing the production pipeline.
    """
    result = {
        "raw_edge_points": None,
        "trimmed_edge_points": None,
        "lines": None,
        "corners": None,
        "failure_stage": None,
    }

    edge_points = _edge_points_from_oriented_external_contour(mask)
    if edge_points is None:
        result["failure_stage"] = "edge_points_from_mask"
        return result

    top_raw, right_raw, bottom_raw, left_raw = edge_points

    result["raw_edge_points"] = {
        "top": top_raw,
        "right": right_raw,
        "bottom": bottom_raw,
        "left": left_raw,
    }

    top = _trim_by_axis(top_raw, axis=0, trim_ratio=0.08)
    bottom = _trim_by_axis(bottom_raw, axis=0, trim_ratio=0.08)
    left = _trim_by_axis(left_raw, axis=1, trim_ratio=0.08)
    right = _trim_by_axis(right_raw, axis=1, trim_ratio=0.08)

    result["trimmed_edge_points"] = {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
    }

    top_line = _fit_line_ransac(top, threshold=2.0)
    right_line = _fit_line_ransac(right, threshold=2.0)
    bottom_line = _fit_line_ransac(bottom, threshold=2.0)
    left_line = _fit_line_ransac(left, threshold=2.0)

    result["lines"] = {
        "top": top_line,
        "right": right_line,
        "bottom": bottom_line,
        "left": left_line,
    }

    if any(line is None for line in [top_line, right_line, bottom_line, left_line]):
        result["failure_stage"] = "fit_line_ransac"
        return result

    tl = _intersect_lines(top_line, left_line)
    tr = _intersect_lines(top_line, right_line)
    br = _intersect_lines(bottom_line, right_line)
    bl = _intersect_lines(bottom_line, left_line)
    pts = np.array([tl, tr, br, bl], dtype=np.float32)

    print("\nDEBUG POINTS")
    for i, p in enumerate(pts):
        print(i, p)
    result["corners"] = {
        "tl": tl,
        "tr": tr,
        "br": br,
        "bl": bl,
    }

    if any(p is None for p in [tl, tr, br, bl]):
        result["failure_stage"] = "intersect_lines"
        return result

    result["failure_stage"] = None
    return result