from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from single_plate_pose.detection import get_predictions
from corner_refinement import refine_yellow_inner_corners


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "depth"
    / "3_meter.jpeg"
)

RESULTS_CSV = (
    PROJECT_ROOT
    / "debug"
    / "single_plate_pose"
    / "corner_noise_sensitivity"
    / "corner_noise_trials.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "debug"
    / "presentation"
    / "corner_noise_visualization"
)

METHOD = "robust_mask_lines"

# Use an environment variable instead of storing the API key in this file.
# Windows PowerShell:
#   $env:ROBOFLOW_API_KEY="your_key"
import os

API_KEY = os.getenv("ROBOFLOW_API_KEY")

# The main presentation cases.
SIGMAS_TO_SHOW = [3.0, 5.0]

# Crop padding relative to plate width / height.
PAD_X = 0.45
PAD_Y = 1.10

# Output card size.
CARD_W = 1000
CARD_H = 520


# ============================================================
# COLORS
# ============================================================

GREEN = (35, 185, 55)
RED = (45, 45, 230)
NAVY = (90, 50, 15)
GRAY = (95, 95, 95)
WHITE = (255, 255, 255)
LIGHT = (247, 248, 250)


# ============================================================
# BASELINE CORNERS
# ============================================================

def detect_baseline_corners(
    image: np.ndarray,
    image_path: Path,
) -> np.ndarray:

    if not API_KEY:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set.\n"
            "Set it before running the script."
        )

    predictions = get_predictions(
        str(image_path),
        API_KEY,
    )

    if not predictions:
        raise RuntimeError(
            "No license plate prediction found."
        )

    # Same assumption as the original sensitivity experiment:
    # one dominant plate in this image.
    prediction = predictions[0]

    points, _, _ = refine_yellow_inner_corners(
        image,
        prediction,
        debug=False,
        method=METHOD,
        debug_name="corner_noise_presentation_baseline",
    )

    return np.asarray(
        points,
        dtype=np.float64,
    )


# ============================================================
# REPRESENTATIVE TRIAL
# ============================================================

def find_representative_trial(
    dataframe: pd.DataFrame,
    sigma_px: float,
) -> pd.Series:
    """
    Select the trial whose actual mean corner displacement is
    closest to the mean displacement for this sigma.
    """

    group = dataframe[
        np.isclose(
            dataframe["sigma_px"].to_numpy(dtype=float),
            sigma_px,
        )
    ].copy()

    if group.empty:
        raise ValueError(
            f"No trials found for sigma={sigma_px}"
        )

    target = float(
        group["mean_corner_shift_px"].mean()
    )

    idx = (
        group["mean_corner_shift_px"]
        .sub(target)
        .abs()
        .idxmin()
    )

    return dataframe.loc[idx]


def noise_from_row(row: pd.Series) -> np.ndarray:

    return np.array(
        [
            [row["corner_0_dx"], row["corner_0_dy"]],
            [row["corner_1_dx"], row["corner_1_dy"]],
            [row["corner_2_dx"], row["corner_2_dy"]],
            [row["corner_3_dx"], row["corner_3_dy"]],
        ],
        dtype=np.float64,
    )


# ============================================================
# CROP / DRAW HELPERS
# ============================================================

def crop_bounds(
    image: np.ndarray,
    baseline: np.ndarray,
):
    x1, y1 = np.min(baseline, axis=0)
    x2, y2 = np.max(baseline, axis=0)

    plate_w = max(1.0, x2 - x1)
    plate_h = max(1.0, y2 - y1)

    x1 = int(max(0, x1 - PAD_X * plate_w))
    x2 = int(min(image.shape[1], x2 + PAD_X * plate_w))

    y1 = int(max(0, y1 - PAD_Y * plate_h))
    y2 = int(min(image.shape[0], y2 + PAD_Y * plate_h))

    return x1, y1, x2, y2


def shift_points_to_crop(
    pts: np.ndarray,
    x1: int,
    y1: int,
) -> np.ndarray:
    out = np.asarray(
        pts,
        dtype=np.float64,
    ).copy()

    out[:, 0] -= x1
    out[:, 1] -= y1

    return out


def draw_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color,
    thickness: int,
    radius: int,
):
    p = np.round(
        points
    ).astype(np.int32)

    cv2.polylines(
        image,
        [p],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    for x, y in p:
        cv2.circle(
            image,
            (int(x), int(y)),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def resize_on_canvas(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:

    h, w = image.shape[:2]

    scale = min(
        target_w / w,
        target_h / h,
    )

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (target_h, target_w, 3),
        255,
        dtype=np.uint8,
    )

    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2

    canvas[
        oy:oy + nh,
        ox:ox + nw,
    ] = resized

    return canvas


def add_card_header(
    card: np.ndarray,
    title: str,
    subtitle: str,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        card,
        title,
        (35, 48),
        font,
        1.15,
        NAVY,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        card,
        subtitle,
        (35, 86),
        font,
        0.68,
        GRAY,
        2,
        cv2.LINE_AA,
    )

def draw_corner_insets(
    card: np.ndarray,
    crop: np.ndarray,
    baseline_local: np.ndarray,
    noisy_local: np.ndarray,
    inset_size: int = 120,
    source_half_size: int = 12,
):

    positions = [
        (20, 125),
        (CARD_W - inset_size - 20, 125),
        (CARD_W - inset_size - 20, CARD_H - inset_size - 20),
        (20, CARD_H - inset_size - 20),
    ]

    h, w = crop.shape[:2]

    for i in range(4):

        # Center the zoom between original and noisy locations
        center = (
            baseline_local[i] + noisy_local[i]
        ) / 2.0

        cx = int(round(center[0]))
        cy = int(round(center[1]))

        x1 = max(0, cx - source_half_size)
        x2 = min(w, cx + source_half_size)

        y1 = max(0, cy - source_half_size)
        y2 = min(h, cy + source_half_size)

        # IMPORTANT:
        # Take a CLEAN patch before drawing anything.
        patch = crop[y1:y2, x1:x2].copy()

        if patch.size == 0:
            continue

        patch_h, patch_w = patch.shape[:2]

        scale_x = inset_size / patch_w
        scale_y = inset_size / patch_h

        # Zoom the clean image
        patch_big = cv2.resize(
            patch,
            (inset_size, inset_size),
            interpolation=cv2.INTER_CUBIC,
        )

        # Convert coordinates into the enlarged inset
        b = baseline_local[i] - np.array([x1, y1])
        n = noisy_local[i] - np.array([x1, y1])

        baseline_big = (
            int(round(b[0] * scale_x)),
            int(round(b[1] * scale_y)),
        )

        noisy_big = (
            int(round(n[0] * scale_x)),
            int(round(n[1] * scale_y)),
        )

        # Thin connector = actual displacement
        cv2.line(
            patch_big,
            baseline_big,
            noisy_big,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ORIGINAL = green hollow circle
        cv2.circle(
            patch_big,
            baseline_big,
            6,
            GREEN,
            2,
            cv2.LINE_AA,
        )

        # NOISY = red filled circle
        cv2.circle(
            patch_big,
            noisy_big,
            5,
            RED,
            -1,
            cv2.LINE_AA,
        )

        # Border
        cv2.rectangle(
            patch_big,
            (0, 0),
            (inset_size - 1, inset_size - 1),
            NAVY,
            2,
        )

        # Corner name
        cv2.putText(
            patch_big,
            f"C{i}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        px, py = positions[i]

        card[
            py:py + inset_size,
            px:px + inset_size,
        ] = patch_big
def make_zoom_card(
    image: np.ndarray,
    baseline: np.ndarray,
    noisy: np.ndarray | None,
    title: str,
    subtitle: str,
) -> np.ndarray:

    x1, y1, x2, y2 = crop_bounds(
        image,
        baseline,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    clean_crop = crop.copy()

    baseline_local = shift_points_to_crop(
        baseline,
        x1,
        y1,
    )

    draw_polygon(
        crop,
        baseline_local,
        GREEN,
        thickness=4,
        radius=8,
    )

    if noisy is not None:
        noisy_local = shift_points_to_crop(
            noisy,
            x1,
            y1,
        )

        draw_polygon(
            crop,
            noisy_local,
            RED,
            thickness=3,
            radius=7,
        )

    HEADER_H = 110
    image_h = CARD_H - HEADER_H

    crop_canvas = resize_on_canvas(
        crop,
        CARD_W,
        image_h,
    )

    card = np.full(
        (CARD_H, CARD_W, 3),
        255,
        dtype=np.uint8,
    )

    card[HEADER_H:] = crop_canvas
    if noisy is not None:
        # We need the local corner coordinates in the original crop.
        draw_corner_insets(
            card=card,
            crop=clean_crop,
            baseline_local=baseline_local,
            noisy_local=noisy_local,
            inset_size=160,
            source_half_size=12,
        )
    add_card_header(
        card,
        title,
        subtitle,
    )

    return card


# ============================================================
# FULL CONTEXT IMAGE
# ============================================================

def make_context_image(
    image: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:

    vis = image.copy()

    draw_polygon(
        vis,
        baseline,
        GREEN,
        thickness=5,
        radius=10,
    )

    return vis


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {IMAGE_PATH}"
        )

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find results CSV:\n{RESULTS_CSV}"
        )

    df = pd.read_csv(
        RESULTS_CSV
    )

    baseline = detect_baseline_corners(
        image,
        IMAGE_PATH,
    )

    # --------------------------------------------------------
    # 1. Full context image
    # --------------------------------------------------------

    context = make_context_image(
        image,
        baseline,
    )

    context_path = (
        OUTPUT_DIR
        / "corner_noise_context_full.jpg"
    )

    cv2.imwrite(
        str(context_path),
        context,
    )

    # --------------------------------------------------------
    # 2. Original card
    # --------------------------------------------------------

    cards = []

    original_card = make_zoom_card(
        image=image,
        baseline=baseline,
        noisy=None,
        title="ORIGINAL",
        subtitle="Baseline detected corners",
    )

    original_path = (
        OUTPUT_DIR
        / "corner_noise_original_zoom.png"
    )

    cv2.imwrite(
        str(original_path),
        original_card,
    )

    cards.append(
        original_card
    )

    # --------------------------------------------------------
    # 3. Representative sigma cards
    # --------------------------------------------------------

    for sigma in SIGMAS_TO_SHOW:

        row = find_representative_trial(
            df,
            sigma,
        )

        noise = noise_from_row(
            row
        )

        noisy = baseline + noise

        actual_shift = np.linalg.norm(
            noise,
            axis=1,
        )

        mean_shift = float(
            np.mean(actual_shift)
        )

        max_shift = float(
            np.max(actual_shift)
        )

        card = make_zoom_card(
            image=image,
            baseline=baseline,
            noisy=noisy,
            title=f"SIGMA = {sigma:.0f} px",
            subtitle=f"Mean actual corner shift: {mean_shift:.2f} px"
        )

        sigma_name = str(
            sigma
        ).replace(
            ".",
            "p",
        )

        save_path = (
            OUTPUT_DIR
            / f"corner_noise_sigma_{sigma_name}_zoom.png"
        )

        cv2.imwrite(
            str(save_path),
            card,
        )

        cards.append(
            card
        )

    # --------------------------------------------------------
    # 4. Combined presentation strip
    # --------------------------------------------------------

    strip = np.hstack(
        cards
    )

    strip_path = (
        OUTPUT_DIR
        / "corner_noise_original_3px_5px_strip.png"
    )

    cv2.imwrite(
        str(strip_path),
        strip,
    )

    print("\nSaved presentation visualizations:")
    print(f"  {context_path}")
    print(f"  {original_path}")
    print(f"  {strip_path}")

    for sigma in SIGMAS_TO_SHOW:
        sigma_name = str(
            sigma
        ).replace(
            ".",
            "p",
        )
        print(
            "  "
            + str(
                OUTPUT_DIR
                / f"corner_noise_sigma_{sigma_name}_zoom.png"
            )
        )


if __name__ == "__main__":
    main()
