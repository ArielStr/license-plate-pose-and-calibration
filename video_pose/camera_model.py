from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibration.calibration_utils import load_calibration_results


@dataclass(frozen=True)
class CameraModel:
    """
    Camera intrinsics used by the video pose pipeline.

    The calibration is loaded once before video processing starts,
    instead of reloading the .npz file for every detected plate/frame.
    """

    K: np.ndarray
    dist_coeffs: np.ndarray

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64)

        if K.shape != (3, 3):
            raise ValueError(
                f"K must have shape (3, 3), received {K.shape}"
            )

        if dist_coeffs.ndim == 1:
            dist_coeffs = dist_coeffs.reshape(-1, 1)

        if dist_coeffs.ndim != 2:
            raise ValueError(
                "dist_coeffs must be a 1D or 2D NumPy array"
            )

        if not np.all(np.isfinite(K)):
            raise ValueError("K contains non-finite values")

        if not np.all(np.isfinite(dist_coeffs)):
            raise ValueError("dist_coeffs contains non-finite values")

        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist_coeffs", dist_coeffs)


def load_camera_model(
    calibration_path: str | Path,
    *,
    use_distortion: bool = True,
) -> CameraModel:
    """
    Load the existing calibration format used by single_plate_pose.

    Expected keys:
        K_cv
        dist_coeffs
    """
    calibration = load_calibration_results(
        str(calibration_path)
    )

    K = np.asarray(
        calibration["K_cv"],
        dtype=np.float64,
    )

    dist_coeffs = np.asarray(
        calibration["dist_coeffs"],
        dtype=np.float64,
    )

    if not use_distortion:
        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

    return CameraModel(
        K=K,
        dist_coeffs=dist_coeffs,
    )
