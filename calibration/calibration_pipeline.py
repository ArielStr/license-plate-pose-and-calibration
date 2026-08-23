import cv2
import numpy as np
import glob
import os


CHESSBOARD_SIZE = (7, 7)      # inner corners: columns, rows
SQUARE_SIZE_CM = 2.8          # change to your printed square size
IMAGES_DIR = "calibration_images"
OUTPUT_FILE = "calibration_results.npz"


def create_object_points():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)

    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_CM

    return objp


def calibrate_camera():
    objp = create_object_points()

    object_points = []
    image_points = []

    image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(IMAGES_DIR, "*.JPG")) + \
                  glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")

    image_size = None

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"Could not read {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"Chessboard not found: {path}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        object_points.append(objp)
        image_points.append(corners_refined)

        debug = img.copy()
        cv2.drawChessboardCorners(debug, CHESSBOARD_SIZE, corners_refined, found)

        cv2.namedWindow("Detected corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected corners", 1000, 700)
        cv2.imshow("Detected corners", debug)
        cv2.waitKey(150)

        print(f"Detected: {path}")

    cv2.destroyAllWindows()

    if len(object_points) < 10:
        raise RuntimeError("Not enough valid calibration images. Aim for at least 15-20.")

    rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    errors = []

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            K,
            dist_coeffs
        )

        error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(error)

    errors = np.array(errors)

    print("\n========== Calibration Results ==========")
    print("RMS error from OpenCV:")
    print(rms)

    print("\nCamera matrix K:")
    print(K)

    print("\nDistortion coefficients:")
    print(dist_coeffs.reshape(-1))

    print("\nReprojection error per image:")
    for path, err in zip(image_paths, errors):
        print(f"{os.path.basename(path)}: {err:.4f} px")

    print("\nMean reprojection error:")
    print(f"{errors.mean():.4f} px")

    print("Median reprojection error:")
    print(f"{np.median(errors):.4f} px")

    print("Max reprojection error:")
    print(f"{errors.max():.4f} px")

    np.savez(
        OUTPUT_FILE,
        K_cv=K,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        reprojection_errors=errors,
        rms=rms
    )

    print(f"\nSaved calibration to {OUTPUT_FILE}")


if __name__ == "__main__":
    calibrate_camera()