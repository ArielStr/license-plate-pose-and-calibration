import cv2
import numpy as np

from calibration.calibration_utils import load_calibration_results


# ============================================================
# PLATE GEOMETRY
# ============================================================

YELLOW_W_CM = 45.0
YELLOW_H_CM = 10.0


# ============================================================
# CLASSICAL HOMOGRAPHY DECOMPOSITION
# ============================================================

def estimate_plate_pose_from_homography(
    image_points,
    K,
    plate_w_cm=YELLOW_W_CM,
    plate_h_cm=YELLOW_H_CM,
    debug=True,
):
    """
    Recover planar pose from a homography using:

        H -> K^-1 H -> (R, t)

    This intentionally uses raw image points without undistortion.
    It is mainly used for comparison against solvePnP/IPPE.
    """

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    )

    K = np.asarray(
        K,
        dtype=np.float64,
    )

    object_points_2d = np.array(
        [
            [0.0, 0.0],
            [plate_w_cm, 0.0],
            [plate_w_cm, plate_h_cm],
            [0.0, plate_h_cm],
        ],
        dtype=np.float64,
    )

    H, _ = cv2.findHomography(
        object_points_2d,
        image_points,
        method=0,
    )

    if H is None:
        raise RuntimeError(
            "findHomography failed."
        )

    K_inv = np.linalg.inv(
        K
    )

    B = K_inv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    norm_b1 = float(
        np.linalg.norm(b1)
    )

    norm_b2 = float(
        np.linalg.norm(b2)
    )

    if (
        norm_b1 < 1e-12
        or norm_b2 < 1e-12
    ):
        raise RuntimeError(
            "Degenerate homography decomposition."
        )

    lambda_from_r1 = (
        1.0 / norm_b1
    )

    lambda_from_r2 = (
        1.0 / norm_b2
    )

    # Robust classical scale estimate.
    scale = (
        2.0
        / (norm_b1 + norm_b2)
    )

    r1_raw = scale * b1
    r2_raw = scale * b2

    tvec = (
        scale * b3
    ).reshape(3, 1)

    # Homography is defined up to sign.
    if tvec[2, 0] < 0:
        scale = -scale
        r1_raw = scale * b1
        r2_raw = scale * b2

        tvec = (
            scale * b3
        ).reshape(3, 1)

    r3_raw = np.cross(
        r1_raw,
        r2_raw,
    )

    R_raw = np.column_stack(
        [
            r1_raw,
            r2_raw,
            r3_raw,
        ]
    )

    # --------------------------------------------------------
    # Orthonormality diagnostics before correction
    # --------------------------------------------------------

    norm_r1_raw = float(
        np.linalg.norm(
            r1_raw
        )
    )

    norm_r2_raw = float(
        np.linalg.norm(
            r2_raw
        )
    )

    norm_r3_raw = float(
        np.linalg.norm(
            r3_raw
        )
    )

    dot_r1_r2_raw = float(
        np.dot(
            r1_raw,
            r2_raw,
        )
    )

    cos_angle_raw = np.clip(
        dot_r1_r2_raw
        / (
            norm_r1_raw
            * norm_r2_raw
        ),
        -1.0,
        1.0,
    )

    angle_r1_r2_deg_raw = float(
        np.degrees(
            np.arccos(
                cos_angle_raw
            )
        )
    )

    RtR_raw = (
        R_raw.T
        @ R_raw
    )

    orthogonality_error_raw = float(
        np.linalg.norm(
            RtR_raw
            - np.eye(3),
            ord="fro",
        )
    )

    det_R_raw = float(
        np.linalg.det(
            R_raw
        )
    )

    # --------------------------------------------------------
    # Project to nearest valid rotation
    # --------------------------------------------------------

    U, singular_values, Vt = (
        np.linalg.svd(
            R_raw
        )
    )

    correction = np.eye(
        3
    )

    correction[2, 2] = np.sign(
        np.linalg.det(
            U @ Vt
        )
    )

    R = (
        U
        @ correction
        @ Vt
    )

    RtR_corrected = (
        R.T
        @ R
    )

    orthogonality_error_corrected = float(
        np.linalg.norm(
            RtR_corrected
            - np.eye(3),
            ord="fro",
        )
    )

    det_R_corrected = float(
        np.linalg.det(
            R
        )
    )

    plate_center_obj = np.array(
        [
            [plate_w_cm / 2.0],
            [plate_h_cm / 2.0],
            [0.0],
        ],
        dtype=np.float64,
    )

    plate_center_camera = (
        R
        @ plate_center_obj
        + tvec
    )

    distance_to_origin_cm = float(
        np.linalg.norm(
            tvec
        )
    )

    distance_to_center_cm = float(
        np.linalg.norm(
            plate_center_camera
        )
    )

    normal_camera = (
        R
        @ np.array(
            [
                [0.0],
                [0.0],
                [1.0],
            ],
            dtype=np.float64,
        )
    ).reshape(-1)

    yaw_rad = np.arctan2(
        normal_camera[0],
        normal_camera[2],
    )

    yaw_deg = float(
        np.degrees(
            yaw_rad
        )
    )

    if debug:
        print(
            "\n========== HOMOGRAPHY POSE DEBUG =========="
        )

        print("\nH:")
        print(H)

        print("\nB = K^-1 H:")
        print(B)

        print(
            "\nColumn norms BEFORE scale:"
        )
        print(
            f"||b1|| = {norm_b1:.8f}"
        )
        print(
            f"||b2|| = {norm_b2:.8f}"
        )
        print(
            "difference = "
            f"{abs(norm_b1 - norm_b2):.8f}"
        )
        print(
            "relative norm mismatch = "
            f"{abs(norm_b1 - norm_b2) / ((norm_b1 + norm_b2) / 2.0) * 100.0:.4f}%"
        )

        print("\nScale estimates:")
        print(
            f"lambda from b1 = {lambda_from_r1:.8f}"
        )
        print(
            f"lambda from b2 = {lambda_from_r2:.8f}"
        )
        print(
            "lambda used (average norm) = "
            f"{scale:.8f}"
        )

        print(
            "\nRaw rotation columns AFTER scale:"
        )
        print(
            f"||r1_raw|| = {norm_r1_raw:.8f}"
        )
        print(
            f"||r2_raw|| = {norm_r2_raw:.8f}"
        )
        print(
            f"||r3_raw|| = {norm_r3_raw:.8f}"
        )
        print(
            "r1_raw dot r2_raw = "
            f"{dot_r1_r2_raw:.8e}"
        )
        print(
            "angle(r1_raw, r2_raw) = "
            f"{angle_r1_r2_deg_raw:.6f} deg"
        )

        print("\nR_raw^T R_raw:")
        print(RtR_raw)

        print(
            "raw orthogonality error "
            f"||R^T R - I||_F = {orthogonality_error_raw:.8e}"
        )
        print(
            f"det(R_raw) = {det_R_raw:.8f}"
        )

        print(
            "\nSVD singular values of R_raw:"
        )
        print(
            singular_values
        )

        print(
            "\nCorrected rotation R:"
        )
        print(R)

        print(
            "corrected orthogonality error "
            f"||R^T R - I||_F = {orthogonality_error_corrected:.8e}"
        )
        print(
            f"det(R) = {det_R_corrected:.8f}"
        )

        print(
            "\nRecovered translation t [cm]:"
        )
        print(
            tvec.reshape(-1)
        )

        print(
            "Distance to plate origin: "
            f"{distance_to_origin_cm:.3f} cm"
        )
        print(
            "Distance to plate center: "
            f"{distance_to_center_cm:.3f} cm"
        )
        print(
            f"Yaw: {yaw_deg:.6f} deg"
        )

        print(
            "============================================\n"
        )

    return {
        "method": "homography_decomposition",
        "H": H,
        "K": K,
        "B": B,
        "scale": float(scale),
        "lambda_from_r1": float(
            lambda_from_r1
        ),
        "lambda_from_r2": float(
            lambda_from_r2
        ),
        "R_raw": R_raw,
        "R": R,
        "tvec": tvec,
        "plate_center_camera":
            plate_center_camera,
        "normal_camera":
            normal_camera,
        "yaw_deg":
            yaw_deg,
        "distance_to_origin_cm":
            distance_to_origin_cm,
        "distance_to_center_cm":
            distance_to_center_cm,
        "debug": {
            "norm_b1":
                norm_b1,
            "norm_b2":
                norm_b2,
            "relative_norm_mismatch_percent":
                float(
                    abs(
                        norm_b1
                        - norm_b2
                    )
                    / (
                        (
                            norm_b1
                            + norm_b2
                        )
                        / 2.0
                    )
                    * 100.0
                ),
            "norm_r1_raw":
                norm_r1_raw,
            "norm_r2_raw":
                norm_r2_raw,
            "norm_r3_raw":
                norm_r3_raw,
            "dot_r1_r2_raw":
                dot_r1_r2_raw,
            "angle_r1_r2_deg_raw":
                angle_r1_r2_deg_raw,
            "RtR_raw":
                RtR_raw,
            "orthogonality_error_raw":
                orthogonality_error_raw,
            "det_R_raw":
                det_R_raw,
            "svd_singular_values":
                singular_values,
            "orthogonality_error_corrected":
                orthogonality_error_corrected,
            "det_R_corrected":
                det_R_corrected,
        },
    }


# ============================================================
# PNP / IPPE
# ============================================================

def estimate_plate_pose_from_K(
    image_points,
    K,
    dist_coeffs=None,
    plate_w_cm=YELLOW_W_CM,
    plate_h_cm=YELLOW_H_CM,
):
    """
    Estimate plate pose directly from a known intrinsic matrix K.
    """

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    )

    K = np.asarray(
        K,
        dtype=np.float64,
    )

    object_points_2d = np.array(
        [
            [0, 0],
            [plate_w_cm, 0],
            [plate_w_cm, plate_h_cm],
            [0, plate_h_cm],
        ],
        dtype=np.float64,
    )

    object_points_3d = np.array(
        [
            [0, 0, 0],
            [plate_w_cm, 0, 0],
            [plate_w_cm, plate_h_cm, 0],
            [0, plate_h_cm, 0],
        ],
        dtype=np.float64,
    )

    H, _ = cv2.findHomography(
        object_points_2d,
        image_points,
    )

    if dist_coeffs is None:
        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )
    else:
        dist_coeffs = np.asarray(
            dist_coeffs,
            dtype=np.float64,
        )

    success, rvec, tvec = cv2.solvePnP(
        object_points_3d,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        raise RuntimeError(
            "solvePnP failed."
        )

    R, _ = cv2.Rodrigues(
        rvec
    )

    plate_center_obj = np.array(
        [
            [plate_w_cm / 2.0],
            [plate_h_cm / 2.0],
            [0.0],
        ],
        dtype=np.float64,
    )

    plate_center_camera = (
        R
        @ plate_center_obj
        + tvec
    )

    distance_to_origin_cm = float(
        np.linalg.norm(
            tvec
        )
    )

    distance_to_center_cm = float(
        np.linalg.norm(
            plate_center_camera
        )
    )

    normal_camera = (
        R
        @ np.array(
            [
                [0.0],
                [0.0],
                [1.0],
            ],
            dtype=np.float64,
        )
    ).reshape(-1)

    yaw_rad = np.arctan2(
        normal_camera[0],
        normal_camera[2],
    )

    yaw_deg = float(
        np.degrees(
            yaw_rad
        )
    )

    return {
        "H": H,
        "K": K,
        "dist_coeffs":
            dist_coeffs,
        "R":
            R,
        "rvec":
            rvec,
        "tvec":
            tvec,
        "plate_center_camera":
            plate_center_camera,
        "normal_camera":
            normal_camera,
        "yaw_deg":
            yaw_deg,
        "distance_to_origin_cm":
            distance_to_origin_cm,
        "distance_to_center_cm":
            distance_to_center_cm,
    }


def estimate_plate_pose(
    image_points,
    calibration_path,
    use_distortion=True,
    plate_w_cm=YELLOW_W_CM,
    plate_h_cm=YELLOW_H_CM,
):
    """
    Load K and distortion from a calibration file, then estimate
    the plate pose using solvePnP/IPPE.
    """

    calib = load_calibration_results(
        calibration_path
    )

    K = calib[
        "K_cv"
    ].astype(
        np.float64
    )

    dist_coeffs = calib[
        "dist_coeffs"
    ].astype(
        np.float64
    )

    if not use_distortion:
        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

    pose = estimate_plate_pose_from_K(
        image_points=image_points,
        K=K,
        dist_coeffs=dist_coeffs,
        plate_w_cm=plate_w_cm,
        plate_h_cm=plate_h_cm,
    )

    print_pose_summary(
        pose["H"],
        pose["K"],
        pose["dist_coeffs"],
        pose["R"],
        pose["tvec"],
        pose["plate_center_camera"],
        pose["normal_camera"],
        pose["yaw_deg"],
        pose["distance_to_origin_cm"],
        pose["distance_to_center_cm"],
    )

    return pose


# ============================================================
# DEBUG SUMMARY
# ============================================================

def print_pose_summary(
    H,
    K,
    dist_coeffs,
    R,
    tvec,
    plate_center_camera,
    normal_camera,
    yaw_deg,
    distance_to_origin_cm,
    distance_to_center_cm,
):
    print(
        "\nHomography H: plate plane -> image"
    )
    print(H)

    print(
        "\nLoaded calibrated K:"
    )
    print(K)

    print(
        "\nUsing dist_coeffs:"
    )
    print(
        dist_coeffs.reshape(-1)
    )

    print(
        "\nRotation R:"
    )
    print(R)

    print(
        "\nTranslation tvec in centimeters:"
    )
    print(
        tvec.reshape(-1)
    )

    print(
        "\nDistance to plate origin: "
        f"{distance_to_origin_cm:.2f} cm"
    )

    print(
        "Distance to plate center: "
        f"{distance_to_center_cm:.2f} cm"
    )

    print(
        "\nPlate center in camera coordinates:"
    )
    print(
        plate_center_camera.reshape(-1)
    )

    print(
        "\nPlate normal in camera coordinates:"
    )
    print(
        normal_camera
    )

    print(
        "\nApproximate yaw angle: "
        f"{yaw_deg:.2f} degrees"
    )
