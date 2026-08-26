from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from video_calibration.observation import PlateObservation


@dataclass
class TrackState:
    """State of one plate track across video frames."""

    track_id: int
    observations: list[PlateObservation] = field(default_factory=list)

    @property
    def last_observation(self) -> PlateObservation:
        if not self.observations:
            raise RuntimeError(f"Track {self.track_id} contains no observations.")
        return self.observations[-1]

    @property
    def first_frame_index(self) -> int:
        return self.observations[0].frame_index

    @property
    def last_frame_index(self) -> int:
        return self.last_observation.frame_index

    @property
    def length(self) -> int:
        return len(self.observations)

    def append(self, observation: PlateObservation) -> None:
        observation.track_id = self.track_id
        self.observations.append(observation)


@dataclass(frozen=True)
class PlateTrackerConfig:
    """Thresholds for associating plate observations between nearby frames."""

    min_iou: float = 0.10
    max_normalized_center_distance: float = 1.50
    max_area_ratio: float = 2.50
    max_frame_gap: int = 3

    iou_weight: float = 0.60
    center_distance_weight: float = 0.30
    area_change_weight: float = 0.10
    max_match_cost: float = 1.20


def bbox_xyxy(observation: PlateObservation) -> np.ndarray:
    return np.asarray(observation.bbox, dtype=np.float64)


def bbox_area(bbox: np.ndarray) -> float:
    x_min, y_min, x_max, y_max = bbox
    width = max(0.0, float(x_max - x_min))
    height = max(0.0, float(y_max - y_min))
    return width * height


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    return np.array(
        [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
        dtype=np.float64,
    )


def bbox_diagonal(bbox: np.ndarray) -> float:
    x_min, y_min, x_max, y_max = bbox
    return float(np.hypot(x_max - x_min, y_max - y_min))


def bbox_iou(first_bbox: np.ndarray, second_bbox: np.ndarray) -> float:
    first_x_min, first_y_min, first_x_max, first_y_max = first_bbox
    second_x_min, second_y_min, second_x_max, second_y_max = second_bbox

    intersection_x_min = max(first_x_min, second_x_min)
    intersection_y_min = max(first_y_min, second_y_min)
    intersection_x_max = min(first_x_max, second_x_max)
    intersection_y_max = min(first_y_max, second_y_max)

    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height

    first_area = bbox_area(first_bbox)
    second_area = bbox_area(second_bbox)
    union_area = first_area + second_area - intersection_area

    if union_area <= 1e-12:
        return 0.0

    return float(intersection_area / union_area)


def normalized_center_distance(
    previous_bbox: np.ndarray,
    current_bbox: np.ndarray,
) -> float:
    previous_center = bbox_center(previous_bbox)
    current_center = bbox_center(current_bbox)
    distance = float(np.linalg.norm(current_center - previous_center))
    diagonal = bbox_diagonal(previous_bbox)

    if diagonal <= 1e-12:
        return float("inf")

    return distance / diagonal


def bbox_area_ratio(first_bbox: np.ndarray, second_bbox: np.ndarray) -> float:
    first_area = bbox_area(first_bbox)
    second_area = bbox_area(second_bbox)
    smaller_area = min(first_area, second_area)
    larger_area = max(first_area, second_area)

    if smaller_area <= 1e-12:
        return float("inf")

    return larger_area / smaller_area


def association_cost(
    previous_observation: PlateObservation,
    current_observation: PlateObservation,
    config: PlateTrackerConfig,
) -> float | None:
    """Return a lower-is-better association cost, or None if implausible."""

    previous_bbox = bbox_xyxy(previous_observation)
    current_bbox = bbox_xyxy(current_observation)

    iou = bbox_iou(previous_bbox, current_bbox)
    center_distance = normalized_center_distance(previous_bbox, current_bbox)
    area_ratio = bbox_area_ratio(previous_bbox, current_bbox)

    if (
        iou < config.min_iou
        and center_distance > config.max_normalized_center_distance
    ):
        return None

    if area_ratio > config.max_area_ratio:
        return None

    iou_cost = 1.0 - iou
    center_cost = min(
        center_distance / max(config.max_normalized_center_distance, 1e-12),
        1.0,
    )
    area_cost = min(
        (area_ratio - 1.0) / max(config.max_area_ratio - 1.0, 1e-12),
        1.0,
    )

    cost = (
        config.iou_weight * iou_cost
        + config.center_distance_weight * center_cost
        + config.area_change_weight * area_cost
    )

    if cost > config.max_match_cost:
        return None

    return float(cost)


def _active_tracks(
    tracks: list[TrackState],
    current_frame_index: int,
    config: PlateTrackerConfig,
) -> list[TrackState]:
    return [
        track
        for track in tracks
        if (current_frame_index - track.last_frame_index) <= config.max_frame_gap
    ]


def associate_frame_observations(
    observations: list[PlateObservation],
    tracks: list[TrackState],
    *,
    next_track_id: int,
    config: PlateTrackerConfig | None = None,
) -> tuple[list[TrackState], int]:
    """Associate one frame's observations to existing tracks."""

    if config is None:
        config = PlateTrackerConfig()

    if not observations:
        return tracks, next_track_id

    frame_indices = {observation.frame_index for observation in observations}
    if len(frame_indices) != 1:
        raise ValueError(
            "associate_frame_observations() expects observations from exactly one frame."
        )

    current_frame_index = observations[0].frame_index
    active_tracks = _active_tracks(tracks, current_frame_index, config)

    candidates: list[tuple[float, int, int]] = []

    for track_index, track in enumerate(active_tracks):
        previous_observation = track.last_observation

        for observation_index, observation in enumerate(observations):
            cost = association_cost(previous_observation, observation, config)
            if cost is not None:
                candidates.append((cost, track_index, observation_index))

    candidates.sort(key=lambda item: item[0])

    matched_track_indices: set[int] = set()
    matched_observation_indices: set[int] = set()

    for _, active_track_index, observation_index in candidates:
        if active_track_index in matched_track_indices:
            continue
        if observation_index in matched_observation_indices:
            continue

        track = active_tracks[active_track_index]
        observation = observations[observation_index]
        track.append(observation)

        matched_track_indices.add(active_track_index)
        matched_observation_indices.add(observation_index)

    for observation_index, observation in enumerate(observations):
        if observation_index in matched_observation_indices:
            continue

        track = TrackState(track_id=next_track_id)
        track.append(observation)
        tracks.append(track)
        next_track_id += 1

    return tracks, next_track_id


def build_tracks(
    observations: Iterable[PlateObservation],
    *,
    config: PlateTrackerConfig | None = None,
) -> list[TrackState]:
    """Build tracks from observations spanning multiple frames."""

    if config is None:
        config = PlateTrackerConfig()

    observations = list(observations)
    if not observations:
        return []

    observations_by_frame: dict[int, list[PlateObservation]] = {}

    for observation in observations:
        observations_by_frame.setdefault(observation.frame_index, []).append(observation)

    tracks: list[TrackState] = []
    next_track_id = 0

    for frame_index in sorted(observations_by_frame):
        tracks, next_track_id = associate_frame_observations(
            observations_by_frame[frame_index],
            tracks,
            next_track_id=next_track_id,
            config=config,
        )

    return tracks


def get_long_tracks(
    tracks: Iterable[TrackState],
    *,
    min_length: int = 5,
) -> list[TrackState]:
    return [track for track in tracks if track.length >= min_length]


def print_track_summary(tracks: Iterable[TrackState]) -> None:
    tracks = list(tracks)

    print("\n========== PLATE TRACKS ==========")
    print(f"Total tracks: {len(tracks)}")

    for track in sorted(tracks, key=lambda item: (-item.length, item.track_id)):
        print(
            f"Track {track.track_id:3d} | "
            f"length={track.length:3d} | "
            f"frames={track.first_frame_index}->{track.last_frame_index}"
        )

    print("==================================\n")
