from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlateDetection:
    """
    Represents one refined license-plate detection in one video frame.

    This object intentionally contains only image-space information.
    It does not contain calibration-specific state such as homographies,
    and it does not contain the final 3D pose.
    """

    frame_index: int
    timestamp_sec: float

    # Bounding box in image coordinates:
    # [x_min, y_min, x_max, y_max]
    bbox: np.ndarray

    # Refined plate corners in a consistent order:
    # top-left, top-right, bottom-right, bottom-left
    # Shape: (4, 2)
    corners: np.ndarray

    detection_confidence: float

    # Filled/updated by the geometric-quality filtering stage.
    quality_score: float = 0.0
    accepted: bool = True
    rejection_reason: Optional[str] = None

    # Filled by the tracking stage.
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64)
        self.corners = np.asarray(self.corners, dtype=np.float64)

        if self.bbox.shape != (4,):
            raise ValueError(
                f"bbox must have shape (4,), received {self.bbox.shape}"
            )

        if self.corners.shape != (4, 2):
            raise ValueError(
                f"corners must have shape (4, 2), received {self.corners.shape}"
            )

        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

        if self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")

        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(
                "detection_confidence must be between 0 and 1"
            )

    @property
    def width(self) -> float:
        x_min, _, x_max, _ = self.bbox
        return float(max(0.0, x_max - x_min))

    @property
    def height(self) -> float:
        _, y_min, _, y_max = self.bbox
        return float(max(0.0, y_max - y_min))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        x_min, y_min, x_max, y_max = self.bbox
        return np.array(
            [
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
            ],
            dtype=np.float64,
        )

    def reject(self, reason: str) -> None:
        if not reason:
            raise ValueError("A rejection reason must be provided")

        self.accepted = False
        self.rejection_reason = reason

    def accept(self) -> None:
        self.accepted = True
        self.rejection_reason = None


@dataclass(frozen=True)
class PlatePose:
    """
    Pose estimated from one refined plate detection.

    distance_m and yaw_deg are the values shown to the user.
    rvec/tvec are kept so later stages can visualize or smooth the
    full pose without recomputing PnP.
    """

    distance_m: float
    yaw_deg: float
    rvec: np.ndarray
    tvec: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rvec",
            np.asarray(self.rvec, dtype=np.float64).reshape(3, 1),
        )
        object.__setattr__(
            self,
            "tvec",
            np.asarray(self.tvec, dtype=np.float64).reshape(3, 1),
        )


@dataclass
class PlateFrameResult:
    """
    Final per-frame product object for one plate.

    V1 may contain a pose immediately after single-frame PnP.
    Later versions can add raw/smoothed pose fields without changing
    PlateDetection itself.
    """

    detection: PlateDetection
    pose: Optional[PlatePose] = None
