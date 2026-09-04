from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np

from video_pose.models import PlateDetection


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    detection: PlateDetection


@dataclass(frozen=True)
class TrackWindow:
    track_id: int
    observations: tuple[TrackObservation, ...]

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def center_index(self) -> int:
        return len(self.observations) // 2

    @property
    def center(self) -> TrackObservation:
        return self.observations[self.center_index]


class SlidingTrackWindowBuilder:
    """
    Keeps a fixed-size observation buffer per Track ID.

    V0 purpose:
    build clean 7-frame windows for experiments without changing the
    production tracking or corner-refinement pipeline.
    """

    def __init__(self, window_size: int = 7) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")
        self.window_size = int(window_size)
        self._buffers: dict[int, deque[TrackObservation]] = {}

    def add(
        self,
        *,
        image: np.ndarray,
        detection: PlateDetection,
    ) -> TrackWindow | None:
        track_id = detection.track_id
        if track_id is None:
            return None

        buffer = self._buffers.setdefault(
            int(track_id),
            deque(maxlen=self.window_size),
        )

        buffer.append(
            TrackObservation(
                frame_index=int(detection.frame_index),
                timestamp_sec=float(detection.timestamp_sec),
                image=image.copy(),
                detection=detection,
            )
        )

        if len(buffer) < self.window_size:
            return None

        return TrackWindow(
            track_id=int(track_id),
            observations=tuple(buffer),
        )
