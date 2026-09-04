from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from video_pose.models import PlateFrameResult, PlatePose


@dataclass(frozen=True)
class AdaptivePoseFilterConfig:
    history_size: int = 5

    # How much a normal new measurement may affect the displayed pose.
    alpha_max: float = 0.45

    # Even a strong outlier gets a small influence so the filter can
    # eventually follow a real sustained change.
    alpha_min: float = 0.05

    # Robust-deviation scale. Larger values reject outliers more strongly.
    deviation_strength: float = 0.70

    # Prevent a nearly constant history from producing an unrealistically
    # tiny MAD and rejecting every small movement.
    distance_mad_floor_m: float = 0.03
    yaw_mad_floor_deg: float = 0.75

    # Need a few observations before adaptive outlier weighting is meaningful.
    min_history_for_adaptive_weight: int = 3


@dataclass
class _TrackPoseState:
    filtered_distance_m: float
    filtered_yaw_deg: float
    raw_distances: deque[float] = field(default_factory=deque)
    raw_yaws: deque[float] = field(default_factory=deque)


@dataclass(frozen=True)
class AdaptivePoseDebug:
    track_id: int
    raw_distance_m: float
    filtered_distance_m: float
    raw_yaw_deg: float
    filtered_yaw_deg: float
    alpha_distance: float
    alpha_yaw: float
    distance_deviation: float
    yaw_deviation: float


class AdaptiveTemporalPoseFilter:
    """
    V1 temporal stabilization.

    Corners and PnP are NOT changed. Each frame first gets its normal raw PnP
    pose. This class only stabilizes the displayed distance/yaw per Track ID.

    The EMA weight is adaptive:
      - measurement consistent with recent raw observations -> alpha near max
      - measurement that is a robust local outlier -> alpha near min

    Distance and yaw are weighted independently.
    """

    def __init__(self, config: AdaptivePoseFilterConfig | None = None) -> None:
        self.config = config or AdaptivePoseFilterConfig()
        self._states: dict[int, _TrackPoseState] = {}
        self.last_debug: list[AdaptivePoseDebug] = []

    @staticmethod
    def _mad(values: list[float], center: float) -> float:
        return float(median([abs(value - center) for value in values]))

    def _adaptive_alpha(
        self,
        value: float,
        history: deque[float],
        *,
        mad_floor: float,
    ) -> tuple[float, float]:
        cfg = self.config

        if len(history) < cfg.min_history_for_adaptive_weight:
            return cfg.alpha_max, 0.0

        values = list(history)
        center = float(median(values))
        mad = max(self._mad(values, center), mad_floor)

        # 1.4826 converts MAD to a robust sigma estimate for Gaussian noise.
        robust_sigma = max(1.4826 * mad, mad_floor)
        deviation = abs(value - center) / robust_sigma

        weight = 1.0 / (1.0 + cfg.deviation_strength * deviation * deviation)

        alpha = cfg.alpha_min + (
            cfg.alpha_max - cfg.alpha_min
        ) * weight

        return float(alpha), float(deviation)

    def filter_result(self, result: PlateFrameResult) -> PlateFrameResult:
        if result.pose is None:
            return result

        track_id = result.detection.track_id
        if track_id is None:
            return result

        raw_pose = result.pose
        state = self._states.get(track_id)

        if state is None:
            state = _TrackPoseState(
                filtered_distance_m=float(raw_pose.distance_m),
                filtered_yaw_deg=float(raw_pose.yaw_deg),
                raw_distances=deque(maxlen=self.config.history_size),
                raw_yaws=deque(maxlen=self.config.history_size),
            )
            state.raw_distances.append(float(raw_pose.distance_m))
            state.raw_yaws.append(float(raw_pose.yaw_deg))
            self._states[track_id] = state

            alpha_distance = 1.0
            alpha_yaw = 1.0
            distance_deviation = 0.0
            yaw_deviation = 0.0
        else:
            alpha_distance, distance_deviation = self._adaptive_alpha(
                float(raw_pose.distance_m),
                state.raw_distances,
                mad_floor=self.config.distance_mad_floor_m,
            )
            alpha_yaw, yaw_deviation = self._adaptive_alpha(
                float(raw_pose.yaw_deg),
                state.raw_yaws,
                mad_floor=self.config.yaw_mad_floor_deg,
            )

            state.filtered_distance_m = (
                alpha_distance * float(raw_pose.distance_m)
                + (1.0 - alpha_distance) * state.filtered_distance_m
            )
            state.filtered_yaw_deg = (
                alpha_yaw * float(raw_pose.yaw_deg)
                + (1.0 - alpha_yaw) * state.filtered_yaw_deg
            )

            # Important: the robust reference uses RAW history, not filtered
            # history, so the outlier detector can notice real sustained motion.
            state.raw_distances.append(float(raw_pose.distance_m))
            state.raw_yaws.append(float(raw_pose.yaw_deg))

        filtered_pose = PlatePose(
            distance_m=state.filtered_distance_m,
            yaw_deg=state.filtered_yaw_deg,
            # V1 does not claim to smooth the full 6-DoF pose.
            # Keep the raw PnP vectors untouched.
            rvec=raw_pose.rvec,
            tvec=raw_pose.tvec,
        )

        self.last_debug.append(
            AdaptivePoseDebug(
                track_id=int(track_id),
                raw_distance_m=float(raw_pose.distance_m),
                filtered_distance_m=float(filtered_pose.distance_m),
                raw_yaw_deg=float(raw_pose.yaw_deg),
                filtered_yaw_deg=float(filtered_pose.yaw_deg),
                alpha_distance=float(alpha_distance),
                alpha_yaw=float(alpha_yaw),
                distance_deviation=float(distance_deviation),
                yaw_deviation=float(yaw_deviation),
            )
        )

        return PlateFrameResult(
            detection=result.detection,
            pose=filtered_pose,
        )

    def filter_frame_results(
        self,
        results: list[PlateFrameResult],
    ) -> list[PlateFrameResult]:
        self.last_debug = []
        return [self.filter_result(result) for result in results]
