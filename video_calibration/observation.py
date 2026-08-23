from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlateObservation:
    """
    Represents one detected license plate in one video frame.

    This object is passed between the different stages of the
    video-calibration pipeline:

        detection
            -> filtering
            -> observation selection
            -> calibration
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

    # Homography from the physical plate plane to the image.
    # Shape: (3, 3)
    homography: np.ndarray

    detection_confidence: float

    # Filled or updated by the quality-filtering stage.
    quality_score: float = 0.0

    # An observation may be rejected without being removed from the list.
    # This makes debugging and visualization easier.
    accepted: bool = True
    rejection_reason: Optional[str] = None

    # Reserved for the tracking stage.
    # In the MVP this will usually remain None.
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Convert inputs to consistent NumPy representations and validate
        their basic shapes.
        """
        self.bbox = np.asarray(self.bbox, dtype=np.float64)
        self.corners = np.asarray(self.corners, dtype=np.float64)
        self.homography = np.asarray(self.homography, dtype=np.float64)

        if self.bbox.shape != (4,):
            raise ValueError(
                f"bbox must have shape (4,), received {self.bbox.shape}"
            )

        if self.corners.shape != (4, 2):
            raise ValueError(
                f"corners must have shape (4, 2), received {self.corners.shape}"
            )

        if self.homography.shape != (3, 3):
            raise ValueError(
                "homography must have shape (3, 3), "
                f"received {self.homography.shape}"
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
        """Bounding-box width in pixels."""
        x_min, _, x_max, _ = self.bbox
        return float(max(0.0, x_max - x_min))

    @property
    def height(self) -> float:
        """Bounding-box height in pixels."""
        _, y_min, _, y_max = self.bbox
        return float(max(0.0, y_max - y_min))

    @property
    def area(self) -> float:
        """Bounding-box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        """Bounding-box center as [x, y]."""
        x_min, y_min, x_max, y_max = self.bbox

        return np.array(
            [
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
            ],
            dtype=np.float64,
        )

    def reject(self, reason: str) -> None:
        """
        Mark the observation as rejected while preserving it for debugging.
        """
        if not reason:
            raise ValueError("A rejection reason must be provided")

        self.accepted = False
        self.rejection_reason = reason

    def accept(self) -> None:
        """Mark the observation as accepted."""
        self.accepted = True
        self.rejection_reason = None

