from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "cars_photos"
    / "chessboard_video.MOV"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "calibration"
    / "calibration_results_video.npz"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "debug"
    / "calibration"
    / "video_chessboard"
)

CHESSBOARD_SIZE = (7, 7)      # inner corners: columns, rows
SQUARE_SIZE_CM = 2.8

# Sample every N frames from the video.
# At 30 FPS, 5 means one candidate every ~0.17 sec.
FRAME_STEP = 5

# Maximum number of accepted chessboard views used for calibration.
# The script stops after this many valid and sufficiently different views.
MAX_VALID_VIEWS = 35

# Need enough different views for a stable calibration.
MIN_VALID_VIEWS = 15

# Reject almost-duplicate chessboard poses.
# Comparison is done in normalized image coordinates.
MIN_CENTER_SHIFT_NORM = 0.035
MIN_SCALE_CHANGE = 0.06
MIN_ANGLE_CHANGE_DEG = 4.0

# Chessboard detection / subpixel refinement.
SUBPIX_WINDOW = (11, 11)
SUBPIX_MAX_ITERS = 30
SUBPIX_EPS = 0.001

# Save a debug image for every accepted view.
SAVE_DEBUG_IMAGES = True

# Show accepted detections while running.
SHOW_DEBUG_WINDOW = False
DEBUG_WAIT_MS = 120


# ============================================================
# OBJECT POINTS
# ============================================================

def create_object_points() -> np.ndarray:

    objp = np.zeros(
        (
            CHESSBOARD_SIZE[0]
            * CHESSBOARD_SIZE[1],
            3,
        ),
        dtype=np.float32,
    )

    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1],
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_CM

    return objp


# ============================================================
# VIEW DIVERSITY
# ============================================================

def chessboard_view_descriptor(
    corners: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    """
    Build a simple descriptor for deciding whether a detected
    chessboard view is meaningfully different from previous ones.

    image_size = (width, height)
    """

    pts = corners.reshape(-1, 2).astype(np.float64)

    width, height = image_size

    center = pts.mean(axis=0)

    center_norm = np.array(
        [
            center[0] / width,
            center[1] / height,
        ],
        dtype=np.float64,
    )

    x_span = (
        pts[:, 0].max()
        - pts[:, 0].min()
    )

    y_span = (
        pts[:, 1].max()
        - pts[:, 1].min()
    )

    area_norm = (
        x_span
        * y_span
        / float(width * height)
    )

    # Estimate chessboard orientation from the first row.
    row0 = pts[:CHESSBOARD_SIZE[0]]

    direction = (
        row0[-1]
        - row0[0]
    )

    angle_deg = float(
        np.degrees(
            np.arctan2(
                direction[1],
                direction[0],
            )
        )
    )

    return {
        "center_norm": center_norm,
        "area_norm": float(area_norm),
        "angle_deg": angle_deg,
    }


def angle_difference_deg(
    a: float,
    b: float,
) -> float:

    diff = abs(a - b) % 180.0

    if diff > 90.0:
        diff = 180.0 - diff

    return float(diff)


def is_sufficiently_different_view(
    descriptor: dict,
    accepted_descriptors: list[dict],
) -> bool:

    if not accepted_descriptors:
        return True

    for previous in accepted_descriptors:

        center_shift = float(
            np.linalg.norm(
                descriptor["center_norm"]
                - previous["center_norm"]
            )
        )

        prev_area = max(
            previous["area_norm"],
            1e-12,
        )

        scale_change = abs(
            descriptor["area_norm"]
            - previous["area_norm"]
        ) / prev_area

        angle_change = angle_difference_deg(
            descriptor["angle_deg"],
            previous["angle_deg"],
        )

        # If it is close in ALL three aspects, treat it as a duplicate.
        if (
            center_shift < MIN_CENTER_SHIFT_NORM
            and scale_change < MIN_SCALE_CHANGE
            and angle_change < MIN_ANGLE_CHANGE_DEG
        ):
            return False

    return True


# ============================================================
# REPROJECTION ERROR
# ============================================================

def compute_reprojection_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    K,
    dist_coeffs,
) -> np.ndarray:

    errors = []

    for i in range(
        len(object_points)
    ):

        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            K,
            dist_coeffs,
        )

        error = (
            cv2.norm(
                image_points[i],
                projected,
                cv2.NORM_L2,
            )
            / len(projected)
        )

        errors.append(
            float(error)
        )

    return np.asarray(
        errors,
        dtype=np.float64,
    )


# ============================================================
# MAIN
# ============================================================

def calibrate_camera_from_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find video:\n"
            f"{VIDEO_PATH}"
        )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{VIDEO_PATH}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    image_size = (
        width,
        height,
    )

    print(
        "\n"
        "========== VIDEO CHESSBOARD CALIBRATION =========="
    )

    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    print(
        f"Frame step: {FRAME_STEP}"
    )

    objp = create_object_points()

    object_points = []
    image_points = []

    accepted_frame_indices = []
    accepted_descriptors = []

    frame_index = 0
    sampled_count = 0
    detected_count = 0
    rejected_duplicate_count = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if (
            frame_index
            % FRAME_STEP
            != 0
        ):
            frame_index += 1
            continue

        sampled_count += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        found, corners = (
            cv2.findChessboardCorners(
                gray,
                CHESSBOARD_SIZE,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                ),
            )
        )

        if not found:

            print(
                f"Frame {frame_index:5d}"
                f" | chessboard not found"
            )

            frame_index += 1
            continue

        detected_count += 1

        corners_refined = (
            cv2.cornerSubPix(
                gray,
                corners,
                winSize=SUBPIX_WINDOW,
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    SUBPIX_MAX_ITERS,
                    SUBPIX_EPS,
                ),
            )
        )

        descriptor = (
            chessboard_view_descriptor(
                corners_refined,
                image_size,
            )
        )

        if not is_sufficiently_different_view(
            descriptor,
            accepted_descriptors,
        ):

            rejected_duplicate_count += 1

            print(
                f"Frame {frame_index:5d}"
                f" | detected"
                f" | REJECT duplicate view"
            )

            frame_index += 1
            continue

        object_points.append(
            objp.copy()
        )

        image_points.append(
            corners_refined
        )

        accepted_frame_indices.append(
            frame_index
        )

        accepted_descriptors.append(
            descriptor
        )

        print(
            f"Frame {frame_index:5d}"
            f" | detected"
            f" | ACCEPT"
            f" | views={len(object_points):2d}"
            f" | center="
            f"("
            f"{descriptor['center_norm'][0]:.3f}, "
            f"{descriptor['center_norm'][1]:.3f}"
            f")"
            f" | area="
            f"{descriptor['area_norm']:.4f}"
            f" | angle="
            f"{descriptor['angle_deg']:.1f} deg"
        )

        if SAVE_DEBUG_IMAGES:

            debug = frame.copy()

            cv2.drawChessboardCorners(
                debug,
                CHESSBOARD_SIZE,
                corners_refined,
                True,
            )

            debug_path = (
                DEBUG_DIR
                / (
                    f"accepted_"
                    f"{len(object_points):02d}_"
                    f"frame_{frame_index:06d}.jpg"
                )
            )

            cv2.imwrite(
                str(debug_path),
                debug,
            )

        if SHOW_DEBUG_WINDOW:

            debug_window = frame.copy()

            cv2.drawChessboardCorners(
                debug_window,
                CHESSBOARD_SIZE,
                corners_refined,
                True,
            )

            cv2.namedWindow(
                "Detected corners",
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                "Detected corners",
                900,
                650,
            )

            cv2.imshow(
                "Detected corners",
                debug_window,
            )

            key = cv2.waitKey(
                DEBUG_WAIT_MS
            ) & 0xFF

            if key == 27:
                break

        if (
            len(object_points)
            >= MAX_VALID_VIEWS
        ):
            print(
                "\nReached MAX_VALID_VIEWS."
            )
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Validate enough calibration views
    # --------------------------------------------------------

    print(
        "\n"
        "========== COLLECTION SUMMARY =========="
    )

    print(
        f"Sampled frames: "
        f"{sampled_count}"
    )

    print(
        f"Chessboard detections: "
        f"{detected_count}"
    )

    print(
        f"Rejected near-duplicates: "
        f"{rejected_duplicate_count}"
    )

    print(
        f"Accepted calibration views: "
        f"{len(object_points)}"
    )

    if (
        len(object_points)
        < MIN_VALID_VIEWS
    ):
        raise RuntimeError(
            "Not enough diverse calibration views.\n"
            f"Accepted: {len(object_points)}\n"
            f"Required: {MIN_VALID_VIEWS}\n"
            "Move/rotate/tilt the chessboard more "
            "and make sure it appears in different "
            "parts of the frame."
        )

    # --------------------------------------------------------
    # Calibrate
    # --------------------------------------------------------

    (
        rms,
        K,
        dist_coeffs,
        rvecs,
        tvecs,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    errors = compute_reprojection_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        K,
        dist_coeffs,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        "========== CALIBRATION RESULTS =========="
    )

    print(
        "\nRMS error from OpenCV:"
    )

    print(
        f"{rms:.6f}"
    )

    print(
        "\nCamera matrix K:"
    )

    print(
        K
    )

    print(
        "\nDistortion coefficients:"
    )

    print(
        dist_coeffs.reshape(-1)
    )

    print(
        "\nReprojection error per accepted view:"
    )

    for frame_idx, error in zip(
        accepted_frame_indices,
        errors,
    ):

        print(
            f"Frame {frame_idx:5d}: "
            f"{error:.4f} px"
        )

    print(
        "\nMean reprojection error:"
    )

    print(
        f"{errors.mean():.4f} px"
    )

    print(
        "Median reprojection error:"
    )

    print(
        f"{np.median(errors):.4f} px"
    )

    print(
        "Max reprojection error:"
    )

    print(
        f"{errors.max():.4f} px"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        OUTPUT_FILE,
        K_cv=K,
        dist_coeffs=dist_coeffs,
        image_size=np.asarray(
            image_size,
            dtype=np.int32,
        ),
        reprojection_errors=errors,
        rms=np.asarray(
            rms,
            dtype=np.float64,
        ),
        accepted_frame_indices=np.asarray(
            accepted_frame_indices,
            dtype=np.int32,
        ),
    )

    print(
        f"\nSaved video calibration to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nSaved accepted-view debug images to:\n"
        f"{DEBUG_DIR}"
    )


if __name__ == "__main__":
    calibrate_camera_from_video()
