from __future__ import annotations

import cv2
import numpy as np


def quad_pair_distance_px(
    quad_a: np.ndarray,
    quad_b: np.ndarray,
) -> float:
    a = np.asarray(quad_a, dtype=np.float64)
    b = np.asarray(quad_b, dtype=np.float64)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def edge_support_score(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    samples_per_edge: int = 80,
    search_radius_px: int = 2,
) -> float:
    """
    Lightweight image score for future experiments.

    It does NOT decide V0 consensus yet; it only gives us a reusable primitive.
    Score is normalized gradient magnitude sampled near each quad edge.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    p = np.asarray(corners, dtype=np.float64)
    values = []

    h, w = gray.shape[:2]

    for i in range(4):
        a = p[i]
        b = p[(i + 1) % 4]

        ts = np.linspace(0.0, 1.0, samples_per_edge)

        for t in ts:
            x, y = (1.0 - t) * a + t * b
            xi = int(round(x))
            yi = int(round(y))

            x1 = max(0, xi - search_radius_px)
            x2 = min(w, xi + search_radius_px + 1)
            y1 = max(0, yi - search_radius_px)
            y2 = min(h, yi + search_radius_px + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            values.append(float(np.max(mag[y1:y2, x1:x2])))

    if not values:
        return 0.0

    score = float(np.mean(values))
    return score
