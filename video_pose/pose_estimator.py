from __future__ import annotations

from dataclasses import dataclass

from single_plate_pose.pose_estimation import (
    YELLOW_H_CM,
    YELLOW_W_CM,
    estimate_plate_pose_from_K,
)

from video_pose.camera_model import CameraModel
from video_pose.models import (
    PlateDetection,
    PlateFrameResult,
    PlatePose,
)


@dataclass(frozen=True)
class PlatePoseEstimatorConfig:
    """
    Physical plate dimensions used by solvePnP.

    Units remain centimeters because the existing single-image PnP
    implementation uses centimeters for its object points.
    """

    plate_width_cm: float = YELLOW_W_CM
    plate_height_cm: float = YELLOW_H_CM


class PlatePoseEstimator:
    """
    Adapter between the video product model and the existing,
    already-tested single-image solvePnP/IPPE implementation.

    Camera calibration is supplied once at construction time.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        config: PlatePoseEstimatorConfig | None = None,
    ) -> None:
        self.camera_model = camera_model
        self.config = (
            config
            if config is not None
            else PlatePoseEstimatorConfig()
        )

    def estimate(
        self,
        detection: PlateDetection,
    ) -> PlatePose:
        """
        Estimate the 3D pose of one accepted refined plate detection.
        """
        if not detection.accepted:
            raise ValueError(
                "Cannot estimate pose for a rejected PlateDetection"
            )

        raw_pose = estimate_plate_pose_from_K(
            image_points=detection.corners,
            K=self.camera_model.K,
            dist_coeffs=self.camera_model.dist_coeffs,
            plate_w_cm=self.config.plate_width_cm,
            plate_h_cm=self.config.plate_height_cm,
        )

        return PlatePose(
            distance_m=(
                raw_pose["distance_to_center_cm"]
                / 100.0
            ),
            yaw_deg=raw_pose["yaw_deg"],
            rvec=raw_pose["rvec"],
            tvec=raw_pose["tvec"],
        )

    def estimate_result(
        self,
        detection: PlateDetection,
    ) -> PlateFrameResult:
        """
        Convenience wrapper returning the product-level result object.
        """
        return PlateFrameResult(
            detection=detection,
            pose=self.estimate(detection),
        )


def estimate_frame_results(
    detections: list[PlateDetection],
    estimator: PlatePoseEstimator,
    *,
    accepted_only: bool = True,
) -> list[PlateFrameResult]:
    """
    Estimate pose for all usable plate detections in one frame.

    Individual solvePnP failures are isolated: a bad plate does not
    terminate processing of the rest of the frame.
    """
    results: list[PlateFrameResult] = []

    for detection in detections:
        if accepted_only and not detection.accepted:
            continue

        try:
            result = estimator.estimate_result(
                detection
            )

        except Exception as exc:
            print(
                f"Frame {detection.frame_index}: "
                f"pose failed for track={detection.track_id}: "
                f"{exc}"
            )
            continue

        results.append(result)

    return results
